"""
model_core.py
--------------
Module centralisant la logique IA — modèle DINOv2 (classification multitâche).

Utilisé par:
- disease_api.py (FastAPI backend)
- app locale de debug

Objectifs:
- Charger le modèle et les artefacts UNE SEULE FOIS
- Garantir exactement les mêmes prédictions partout
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from huggingface_hub import hf_hub_download
from model_def import DINOv2Multitask, CosineLinear

# Sur Streamlit Cloud / Spaces CPU, pas de GPU disponible.
DEVICE = torch.device("cpu")

# Référentiel Hugging Face contenant les artefacts du modèle DINOv2
MODEL_REPO = "mohamedsamake8322/maladie-plantes-dinov2"

# Dernier chemin de checkpoint chargé (utile pour diagnostiquer quel fichier a été utilisé)
LOADED_MODEL_PATH: str | None = None


def load_model_and_metadata() -> Tuple[torch.nn.Module, Dict[Any, str], torch.device]:
    """
    Charge le modèle DINOv2 (poids EMA) et le mapping idx→classe depuis Hugging Face.
    Retourne: (model, idx_to_class, device)
    """
    global LOADED_MODEL_PATH

    ckpt_path = Path(hf_hub_download(repo_id=MODEL_REPO, filename="dinov2_v3_infer.pt"))
    idx_path  = Path(hf_hub_download(repo_id=MODEL_REPO, filename="idx_to_class.json"))
    LOADED_MODEL_PATH = str(ckpt_path)

    d = torch.load(ckpt_path, map_location=DEVICE)
    cfg = d["config"]

    model = DINOv2Multitask(
        backbone_name=cfg["backbone"],
        num_classes=cfg["num_classes"],
        num_crops=cfg["num_crops"],
        num_categories=cfg["num_categories"],
        embed_dim=cfg["embed_dim"],
        total_blocks=cfg["total_blocks"],
    )
    if cfg.get("cosine_head", False):
        final = model.head_main[-1]
        model.head_main[-1] = CosineLinear(final.in_features, final.out_features)

    model.load_state_dict({k: v.float() for k, v in d["ema_state"].items()})
    model = model.to(DEVICE).eval()

    idx_to_class = json.load(open(idx_path, encoding="utf-8"))

    return model, idx_to_class, DEVICE


def preprocess_image_pil(image: Image.Image, size: int = 518) -> torch.Tensor:
    """Prétraitement commun (PIL -> tensor normalisé). DINOv2 attend 518x518."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size, size))
    img_array = np.array(image).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array.astype("float32")).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.float()


def _class_name_for(idx_to_class: Dict[Any, Any], label: int) -> str:
    if label in idx_to_class:
        return idx_to_class[label]
    if str(label) in idx_to_class:
        return idx_to_class[str(label)]
    return f"class_{label}"


def infer_on_image(
    model: torch.nn.Module,
    idx_to_class: Dict[Any, str],
    image: Image.Image,
    device: torch.device,
    top_k: int = 5,
    image_size: int = 518,
) -> Dict[str, Any]:
    """
    Pipeline d'inférence DINOv2.

    Retourne un dict avec:
    - predicted_label, predicted_disease, predicted_similarity (= confiance softmax)
    - is_unknown (toujours False pour l'instant : pas de calibration température ni seuil validé)
    - topk_prototypes: [{rank, label, disease, similarity}, ...]   (conservé pour compat avec l'ancien format)
    - topk_neighbors: [] (pas de FAISS pour DINOv2)
    - crop, category: prédictions des têtes auxiliaires (bonus, absent du modèle Swin)
    """
    img_tensor = preprocess_image_pil(image, size=image_size).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        probs = out["main"].softmax(dim=1).cpu().numpy()[0]

    order = np.argsort(probs)[::-1][:top_k]
    topk = [
        {
            "rank": r + 1,
            "label": int(i),
            "disease": _class_name_for(idx_to_class, int(i)),
            "similarity": float(probs[i]),
        }
        for r, i in enumerate(order)
    ]

    result = {
        "predicted_label": int(order[0]),
        "predicted_disease": topk[0]["disease"],
        "predicted_similarity": topk[0]["similarity"],
        "is_unknown": False,  # à revoir une fois la calibration/le seuil validés
        "topk_prototypes": topk,
        "topk_neighbors": [],
    }

    if "crop" in out:
        result["crop"] = int(out["crop"].argmax(dim=1).item())
    if "category" in out:
        result["category"] = int(out["category"].argmax(dim=1).item())

    return result


def infer_batch(
    model: torch.nn.Module,
    images: List[Image.Image],
    device: torch.device,
    image_size: int = 518,
) -> np.ndarray:
    """Inférence en batch. Retourne les probabilités softmax (N, num_classes)."""
    if not images:
        return np.empty((0, 0), dtype="float32")

    tensors = [preprocess_image_pil(img, size=image_size) for img in images]
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        probs = model(batch)["main"].softmax(dim=1).cpu().numpy()
    return probs


__all__ = [
    "DEVICE",
    "MODEL_REPO",
    "load_model_and_metadata",
    "preprocess_image_pil",
    "infer_on_image",
    "infer_batch",
]