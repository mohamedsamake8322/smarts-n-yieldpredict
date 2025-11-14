import os
from typing import List, Dict, Tuple, Any
from PIL import Image, ImageEnhance
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class DiseaseDetector:
    """
    Détecteur de maladies basé sur PyTorch.
    - Charge un modèle `.pth` (TorchScript ou module sérialisé)
    - Prétraite les images et renvoie top-k prédictions formatées
    """

    def __init__(
        self,
        model_path: str = "plant_disease_model.pth",
        input_size: int = 224,
        classes: Dict[str, str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.class_labels = classes or {}
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_torch_model(self.model_path)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_torch_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        # Essaye TorchScript d'abord
        try:
            return torch.jit.load(model_path, map_location=self.device)
        except Exception:
            pass
        # Essaye un objet modèle sérialisé complet
        try:
            loaded = torch.load(model_path, map_location=self.device)
            # Si c'est directement un nn.Module
            if hasattr(loaded, "forward"):
                return loaded
            # Sinon, on ne peut pas reconstruire sans architecture
            raise RuntimeError("Le fichier chargé n'est pas un module PyTorch exécutable.")
        except Exception as e:
            raise RuntimeError(f"Échec du chargement du modèle PyTorch: {e}")

    def preprocess_image(self, image_pil: Image.Image) -> torch.Tensor:
        try:
            image_pil = image_pil.convert("RGB")
            tensor = self.transform(image_pil).unsqueeze(0)
            return tensor.to(self.device)
        except Exception as e:
            st.error(f"🚨 Erreur dans le prétraitement : {e}")
            # retourne un tenseur neutre pour éviter crash
            dummy = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32, device=self.device)
            return dummy

    def predict(self, image_pil: Image.Image, confidence_threshold: float = 0.7, topk: int = 3) -> List[Dict[str, Any]]:
        try:
            with torch.inference_mode():
                batch = self.preprocess_image(image_pil)
                logits = self.model(batch)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = F.softmax(logits, dim=1)
                probs_cpu = probs.squeeze(0).detach().cpu().numpy()

            sorted_indices = np.argsort(probs_cpu)[::-1]

            results: List[Dict[str, Any]] = []
            for idx in sorted_indices[:topk]:
                confidence = float(probs_cpu[idx]) * 100.0
                if confidence < confidence_threshold * 100.0:
                    continue
                # mapping d'index: accepte clées str ou int
                label = self.class_labels.get(str(idx)) or self.class_labels.get(idx) or f"Classe inconnue {idx}"
                severity, urgency = self._assess_disease_severity(label, confidence)
                results.append({
                    "disease": label,
                    "confidence": round(confidence, 2),
                    "severity": severity,
                    "urgency": urgency,
                })

            # Si filtrage trop strict, retourne quand même topk bruts
            if not results:
                for idx in sorted_indices[:topk]:
                    confidence = float(probs_cpu[idx]) * 100.0
                    label = self.class_labels.get(str(idx)) or self.class_labels.get(idx) or f"Classe inconnue {idx}"
                    severity, urgency = self._assess_disease_severity(label, confidence)
                    results.append({
                        "disease": label,
                        "confidence": round(confidence, 2),
                        "severity": severity,
                        "urgency": urgency,
                    })

            return results

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
            return []

    def _assess_disease_severity(self, label: str, confidence: float) -> Tuple[str, str]:
        high_severity_keywords = ["blight", "rot", "wilt", "rust"]
        moderate_keywords = ["spot", "mildew"]

        severity = "Faible"
        urgency = "Faible"

        if any(word in label.lower() for word in high_severity_keywords):
            severity = "Élevée"
            urgency = "Haute"
        elif any(word in label.lower() for word in moderate_keywords):
            severity = "Modérée"
            urgency = "Moyenne"

        if confidence > 90:
            urgency = "Haute" if severity == "Élevée" else "Moyenne"

        return severity, urgency

    def enhance_image(self, image_pil: Image.Image) -> Image.Image:
        try:
            image_pil = ImageEnhance.Contrast(image_pil).enhance(1.3)
            image_pil = ImageEnhance.Sharpness(image_pil).enhance(1.2)
            image_pil = ImageEnhance.Color(image_pil).enhance(1.15)
            return image_pil
        except Exception as e:
            st.warning(f"⚠️ Amélioration d’image échouée : {e}")
            return image_pil
