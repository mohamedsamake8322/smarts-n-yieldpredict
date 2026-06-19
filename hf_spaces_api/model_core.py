"""
model_core.py
--------------
Module centralisant toute la logique IA pour le modèle de production
(Phase 2 - Swin Base, metric learning).

Utilisé par:
- disease_api.py (FastAPI backend)
- 04_app_streamlit.py (interface de debug Streamlit)

Objectifs:
- Charger le modèle et les artefacts UNE SEULE FOIS (chemins cohérents)
- Garantir exactement les mêmes prédictions partout
- Eviter la duplication de code (prétraitement, prototypes, FAISS, etc.)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import sys
sys.path.append(".")

from huggingface_hub import hf_hub_download
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS optionnel
    faiss = None

# Import from local metric_training_core (same HF Spaces directory)
from metric_training_core import DiagnosticModel  # type: ignore

# Dossiers de base (communs API + Streamlit)
MODELS_PATH_PHASE2 = Path("./outputs/phase2_swin_base_production/models")
DATASET_ROOT_LOCAL = Path("./dataset_final")

# Sur Streamlit Cloud, il n'y a pas de GPU disponible.
# Pour éviter les allocations inutiles et garder une empreinte mémoire prévisible,
# on force l'utilisation du CPU pour l'inférence.
DEVICE = torch.device("cpu")

# Référentiel Hugging Face contenant les artefacts du modèle
MODEL_REPO = "mohamedsamake8322/plant-diseaseS-swin-faiss"


def load_phase2_model_and_metadata(
    models_path: Path = MODELS_PATH_PHASE2,
) -> Tuple[torch.nn.Module, Optional[Any], Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], torch.device]:
    """
    Charge le modèle phase 2 (senedisease_macro_f1.pt), la metadata, l'index FAISS (optionnel)
    et les prototypes. Retourne:
    - model
    - index (FAISS ou None)
    - metadata (dict)
    - prototypes (np.ndarray ou None)
    - prototype_labels (np.ndarray ou None)
    - device
    """
    metric_model_path = models_path / "senedisease_macro_f1.pt"
    metadata_path = models_path / "metadata.pkl"

    # 1) On tente d'utiliser les fichiers locaux (utile en dev hors-ligne)
    if not metric_model_path.exists():
        try:
            metric_model_path = Path(
                hf_hub_download(repo_id=MODEL_REPO, filename="senedisease_macro_f1.pt")
            )
        except Exception as e:
            raise RuntimeError(
                f"Impossible de trouver le checkpoint modèle localement dans {models_path} "
                f"et le téléchargement depuis Hugging Face a échoué: {e}"
            ) from e

    metadata = {}
    if not metadata_path.exists():
        for metadata_filename in ["metadata.pkl", "metadata.json"]:
            try:
                metadata_path = Path(
                    hf_hub_download(repo_id=MODEL_REPO, filename=metadata_filename)
                )
                break
            except Exception:
                metadata_path = Path("")

    if metadata_path and metadata_path.exists():
        import json
        import pickle

        if metadata_path.suffix == ".pkl":
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        else:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
    else:
        print("⚠️ metadata absent, fallback vers metadata vide")

    # Pour limiter l'empreinte mémoire en production (Streamlit Cloud),
    # on supprime explicitement les champs lourds dont on n'a pas besoin
    # pour l'inférence (embeddings bruts, métriques détaillées, etc.).
    for heavy_key in [
        "embeddings",
        "train_embeddings",
        "val_embeddings",
        "test_embeddings",
        "logits",
        "train_metrics",
        "val_metrics",
    ]:
        if heavy_key in metadata:
            try:
                tmp = metadata.pop(heavy_key, None)
                del tmp
            except Exception:
                metadata[heavy_key] = None

    checkpoint = torch.load(metric_model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get("config", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        cfg = {}
        state_dict = checkpoint

    model_name = cfg.get("model_name", "swin_base_patch4_window7_224")
    embedding_dim = cfg.get("embedding_dim", metadata.get("embedding_dim", 768))
    image_size = cfg.get("image_size", metadata.get("image_size", 224))

    if isinstance(state_dict, dict):
        model = DiagnosticModel(
            model_name=model_name, embedding_dim=embedding_dim, image_size=image_size
        )
        model.load_state_dict(state_dict)
        del checkpoint
    else:
        model = state_dict
        image_size = cfg.get("image_size", image_size)

    model = model.to(DEVICE)
    model.eval()

    # FAISS index
    index = None
    faiss_index_path = metadata.get("faiss_index_path") or str(
        models_path / "faiss_index.bin"
    )
    faiss_index_file = Path(faiss_index_path)

    # Si l'index FAISS n'existe pas localement, on essaie de le récupérer depuis Hugging Face
    if faiss is not None and not faiss_index_file.exists():
        try:
            faiss_index_file = Path(
                hf_hub_download(repo_id=MODEL_REPO, filename="faiss_index.bin")
            )
        except Exception:
            faiss_index_file = Path("")
    if faiss is not None and faiss_index_file.exists():
        try:
            # Index FAISS chargé en CPU uniquement (pas de GPU sur Streamlit Cloud)
            index = faiss.read_index(str(faiss_index_file))
        except Exception:
            index = None

    # Prototypes
    prototypes = None
    prototype_labels = None
    if "prototypes" in metadata and "prototype_labels" in metadata:
        prototypes = np.asarray(metadata["prototypes"], dtype="float32")
        prototype_labels = np.asarray(metadata["prototype_labels"], dtype=int)

    return model, index, metadata, prototypes, prototype_labels, DEVICE


def preprocess_image_pil(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Prétraitement commun (PIL -> tensor normalisé)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size, size))
    img_array = np.array(image).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array.astype("float32")).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.float()


def map_image_path_to_local(raw_path: str) -> str:
    """
    Mappe un chemin absolu Colab (/content/drive/MyDrive/dataset_final/...)
    vers le dataset local ./dataset_final/... si necessaire.
    """
    p = Path(raw_path)
    if p.exists():
        return str(p)
    parts = p.parts
    if "dataset_final" in parts:
        idx = parts.index("dataset_final")
        rel = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path(".")
        candidate = DATASET_ROOT_LOCAL / rel
        if candidate.exists():
            return str(candidate)
    return raw_path


def _class_name_for(idx_to_class: Dict[Any, Any], label: int) -> str:
    if label in idx_to_class:
        return idx_to_class[label]
    if str(label) in idx_to_class:
        return idx_to_class[str(label)]
    return f"class_{label}"


def infer_on_image(
    model: torch.nn.Module,
    index: Optional[Any],
    metadata: Dict[str, Any],
    prototypes: Optional[np.ndarray],
    prototype_labels: Optional[np.ndarray],
    image: Image.Image,
    device: torch.device,
    top_k: int = 5,
    unknown_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Pipeline d'inférence commun.

    Retourne un dict avec:
    - predicted_label
    - predicted_disease
    - predicted_similarity
    - is_unknown
    - topk_prototypes: [{rank, label, disease, similarity}, ...]
    - topk_neighbors: [{rank, label, disease, similarity, image_path}, ...]
    """
    image_paths = metadata.get("image_paths", [])
    labels = metadata.get("labels", [])
    idx_to_class = metadata.get("idx_to_class", {})
    image_size = metadata.get("image_size", 224)

    # Embedding
    img_tensor = preprocess_image_pil(
        image, size=image_size
    ).to(device)
    with torch.no_grad():
        emb = model(img_tensor).cpu().numpy().astype("float32")  # (1, D)

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    # Prototypes
    proto_ranking: List[Dict[str, Any]] = []
    predicted_label: Optional[int] = None
    predicted_score: Optional[float] = None

    if prototypes is not None and prototype_labels is not None:
        sims = prototypes @ emb_norm.T  # (C, 1)
        sims = sims.squeeze(axis=1)
        order = np.argsort(sims)[::-1]
        for rank, ci in enumerate(order[: max(top_k, 5)]):
            class_id = int(prototype_labels[ci])
            proto_ranking.append(
                {
                    "rank": rank + 1,
                    "label": class_id,
                    "disease": _class_name_for(idx_to_class, class_id),
                    "similarity": float(sims[ci]),
                }
            )
        if proto_ranking:
            predicted_label = proto_ranking[0]["label"]
            predicted_score = proto_ranking[0]["similarity"]

    is_unknown = (
        predicted_score is not None
        and float(predicted_score) < float(unknown_threshold)
    )

    # Voisins FAISS
    neighbors: List[Dict[str, Any]] = []
    if index is not None:
        distances, indices = index.search(emb_norm.astype("float32"), k=top_k)
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            label_i = labels[idx]
            neighbors.append(
                {
                    "rank": rank + 1,
                    "label": int(label_i),
                    "disease": _class_name_for(idx_to_class, label_i),
                    "similarity": float(dist),
                    "image_path": map_image_path_to_local(image_paths[idx]),
                }
            )

    return {
        "predicted_label": int(predicted_label) if predicted_label is not None else None,
        "predicted_disease": "UNKNOWN DISEASE"
        if is_unknown
        else (_class_name_for(idx_to_class, predicted_label) if predicted_label is not None else None),
        "predicted_similarity": predicted_score,
        "is_unknown": is_unknown,
        "topk_prototypes": proto_ranking[:top_k],
        "topk_neighbors": neighbors,
    }


def infer_batch(
    model: torch.nn.Module,
    images: List[Image.Image],
    device: torch.device,
    image_size: int,
) -> np.ndarray:
    """
    Inférence en batch pour de futures extensions multi‑images.
    Retourne un tableau numpy d'embeddings normalisés (N, D).
    """
    tensors = []
    for img in images:
        tensors.append(preprocess_image_pil(img, size=image_size))

    if not tensors:
        return np.empty((0, 0), dtype="float32")

    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        emb = model(batch).cpu().numpy().astype("float32")

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb_norm


__all__ = [
    "MODELS_PATH_PHASE2",
    "DATASET_ROOT_LOCAL",
    "DEVICE",
    "load_phase2_model_and_metadata",
    "preprocess_image_pil",
    "map_image_path_to_local",
    "infer_on_image",
    "infer_batch",
]

