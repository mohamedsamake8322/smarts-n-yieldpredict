import os
import requests # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess  # type: ignore
from typing import List, Dict, Tuple, Any
import datetime
from PIL import Image, ImageEnhance # type: ignore
import cv2 # type: ignore
from utils.config_model import MODEL_URL, MODEL_PATH
from utils.config_model import MODEL_PATH, check_model_presence
from tensorflow import keras

check_model_presence()
model = keras.models.load_model(MODEL_PATH, compile=False)

class DiseaseDetector:
    def __init__(self):
        import streamlit as st
        st.warning("🚨 Début __init__")

        self.models = {}
        self.preprocessors = {}
        self.class_labels = {}

        st.warning("🔍 Vérification du modèle...")
        if not os.path.exists(MODEL_PATH):
            st.info("📦 Téléchargement du modèle...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with requests.get(MODEL_URL, stream=True) as r:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            st.success("✅ Modèle téléchargé")

        try:
            st.info("🧠 Chargement du modèle .keras...")
            self.models["efficientnet_resnet"] = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ Modèle chargé avec succès")
        except Exception as e:
            st.error(f"❌ Échec du chargement du modèle : {e}")
            return

        self.preprocessors["efficientnet_resnet"] = efficientnet_preprocess
        self.class_labels["efficientnet_resnet"] = [
            "Healthy",
            "Tomato_Late_blight",
            "Tomato_Early_blight",
            "Tomato_Bacterial_spot",
            "Tomato_Septoria_leaf_spot",
            "Potato_Late_blight",
            "Potato_Early_blight",
            "Corn_Common_rust",
            "Corn_Northern_Leaf_Blight",
            "Wheat_Leaf_rust",
            "Wheat_Yellow_rust",
            "Rice_Blast",
            "Rice_Brown_spot",
            "Pepper_Bacterial_spot",
            "Grape_Black_rot",
            "Grape_Powdery_mildew",
        ]

        st.success("🚀 DiseaseDetector initialisé 🎉")


    def preprocess_image(self, image_pil: Image.Image) -> np.ndarray:
        """
        Préprocessing de l'image pour EfficientNet-ResNet.
        """
        try:
            img_resized = image_pil.resize((380, 380)).convert("RGB")
            img_array = np.expand_dims(np.array(img_resized), axis=0)
            return efficientnet_preprocess(img_array)
        except Exception as e:
            print(f"🚨 Erreur lors du preprocessing: {e}")
            return np.zeros((1, 380, 380, 3))


    def predict_disease(
        self, image_pil: Image.Image, confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Prédiction de maladie sur une image avec EfficientNet-ResNet.
        """
        try:
            model = self.models.get("efficientnet_resnet", None)
            if model is None:
                raise ValueError(
                    "🚨 Modèle non chargé: Vérifie que efficientnet_resnet.keras est bien disponible."
                )

            class_labels = self.class_labels["efficientnet_resnet"]
            processed_img = self.preprocess_image(image_pil)
            predictions = model.predict(processed_img, verbose=0)[0]

            sorted_indices = np.argsort(predictions)[::-1]
            results = []

            for idx in sorted_indices:
                confidence = float(predictions[idx]) * 100
                disease_name = class_labels[idx]

                if confidence < confidence_threshold * 100:
                    break

                results.append(
                    {
                        "disease": disease_name,
                        "confidence": confidence,
                        "severity": self._assess_disease_severity(
                            disease_name, confidence
                        ),
                        "model_used": "efficientnet_resnet",
                    }
                )

            return results

        except Exception as e:
            print(f"🚨 Erreur lors de la prédiction: {e}")
            return []

    def _assess_disease_severity(self, disease_name: str, confidence: float) -> str:
        """
        Évalue la sévérité d'une maladie en fonction du niveau de confiance.
        """
        if confidence > 90:
            return "Élevée"
        elif confidence > 75:
            return "Modérée"
        else:
            return "Faible"

def _heuristic_disease_detection(
    self, image_pil: Image.Image, crop_filter: List[str] = None
) -> List[Dict]:
    """
    Détection basée sur EfficientNet-ResNet au lieu des heuristiques visuelles.
    """
    try:
        model = self.models.get("efficientnet_resnet", None)
        if model is None:
            raise ValueError(
                "🚨 Modèle non chargé: Vérifie que efficientnet_resnet.keras est bien disponible."
            )

        class_labels = self.class_labels["efficientnet_resnet"]
        processed_img = self.preprocess_image(image_pil)
        predictions = model.predict(processed_img, verbose=0)[0]

        sorted_indices = np.argsort(predictions)[::-1]
        results = []

        for idx in sorted_indices:
            confidence = float(predictions[idx]) * 100
            disease_name = class_labels[idx]

            if crop_filter and not self._disease_matches_crops(
                disease_name, crop_filter
            ):
                continue

            severity = self._assess_disease_severity(disease_name, confidence)

            results.append(
                {
                    "disease": disease_name,
                    "confidence": confidence,
                    "severity": severity,
                    "model_used": "efficientnet_resnet",
                }
            )

        # ✅ Si aucun résultat ne dépasse le seuil, prendre la meilleure prédiction
        if not results and sorted_indices:
            top_idx = sorted_indices[0]
            confidence = float(predictions[top_idx]) * 100
            disease_name = class_labels[top_idx]

            severity = self._assess_disease_severity(disease_name, confidence)

            results.append(
                {
                    "disease": disease_name,
                    "confidence": confidence,
                    "severity": severity,
                    "model_used": "efficientnet_resnet",
                }
            )

        return results

    except Exception as e:
        print(f"🚨 Erreur lors de la détection heuristique: {e}")
        return []
def _analyze_image_features(self, img_cv: np.ndarray) -> Dict[str, float]:
    """
    Analyse les caractéristiques de l'image pour un passage optimisé au modèle EfficientNet-ResNet.
    """
    try:
        # ✅ Conversion en niveaux de gris pour une analyse robuste
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # ✅ Extraction des statistiques de texture
        texture_variance = np.var(gray) / 10000  # Normalisation

        # ✅ Calcul du contraste global
        contrast = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000

        # ✅ Évaluation préliminaire de la santé via la texture
        overall_health = max(0.0, min(1.0, 1.0 - texture_variance))

        return {
            "texture_variance": texture_variance,
            "contrast": contrast,
            "overall_health": overall_health,
        }

    except Exception as e:
        print(f"🚨 Erreur dans l'analyse des caractéristiques: {e}")
        return {"texture_variance": 0.0, "contrast": 0.0, "overall_health": 0.5}


def _disease_matches_crops(self, disease_name: str, crop_filter: List[str]) -> bool:
    """
    Vérifie si une maladie correspond aux cultures filtrées
    """
    if not crop_filter:
        return True

    # Map diseases to crops
    disease_crop_mapping = {
        "Tomato": ["tomato"],
        "Potato": ["potato", "pomme de terre"],
        "Corn": ["corn", "maïs"],
        "Wheat": ["wheat", "blé"],
        "Rice": ["rice", "riz"],
        "Pepper": ["pepper", "poivron"],
        "Grape": ["grape", "raisin"],
    }

    for crop_prefix, crop_names in disease_crop_mapping.items():
        if disease_name.startswith(crop_prefix):
            for filter_crop in crop_filter:
                if filter_crop.lower() in [c.lower() for c in crop_names]:
                    return True

    # If disease doesn't match any specific crop, allow it
    return True


def _assess_disease_severity(
    self, disease_name: str, confidence: float
) -> Tuple[str, str]:
    """
    Évalue la sévérité et l'urgence d'une maladie en fonction de son type et du niveau de confiance.
    """
    # ✅ Maladies à forte sévérité
    high_severity_diseases = [
        "Late_blight",
        "Black_rot",
        "Blast",
        "Wilt",
        "Crown_Rot",
        "Root_Rot",
    ]

    # ✅ Maladies à sévérité modérée
    moderate_severity_diseases = [
        "Early_blight",
        "Bacterial_spot",
        "Leaf_rust",
        "Common_rust",
        "Brown_spot",
        "Powdery_mildew",
    ]

    # ✅ Cas sain
    if disease_name == "Healthy":
        return "Aucune", "Aucune"

    # ✅ Détermination initiale
    severity = "Faible"
    urgency = "Faible"

    # ✅ Vérification des maladies graves
    for high_disease in high_severity_diseases:
        if high_disease in disease_name:
            severity = "Élevée"
            urgency = "Haute"
            break

    # ✅ Vérification des maladies modérées
    if severity == "Faible":
        for mod_disease in moderate_severity_diseases:
            if mod_disease in disease_name:
                severity = "Modérée"
                urgency = "Moyenne"
                break

    # ✅ Ajustement selon la confiance du modèle
    if confidence > 90:
        if urgency == "Moyenne":
            urgency = "Haute"
        elif urgency == "Faible":
            urgency = "Moyenne"

    return severity, urgency

def get_model_info(self) -> Dict[str, Any]:
    """
    Retourne les informations sur le modèle EfficientNet-ResNet.
    """
    model = self.models.get("efficientnet_resnet", None)
    if model is None:
        return {
            "status": "error",
            "message": "🚨 Modèle non chargé: Vérifie efficientnet_resnet.keras",
        }

    return {
        "model_name": "efficientnet_resnet",
        "input_size": (380, 380),
        "num_classes": len(self.class_labels["efficientnet_resnet"]),
        "status": "loaded",
    }


def benchmark_model(self, test_images: List[Image.Image]) -> Dict[str, Any]:
    """
    Benchmark du modèle EfficientNet-ResNet sur un ensemble d'images test.
    """
    model = self.models.get("efficientnet_resnet", None)
    if model is None:
        return {
            "status": "error",
            "message": "🚨 Modèle non chargé: Vérifie efficientnet_resnet.keras",
        }

    print("Benchmarking EfficientNet-ResNet...")

    start_time = datetime.now()
    predictions = []

    for img in test_images:
        pred = self.predict_disease(img)
        predictions.append(pred[0] if pred else None)

    end_time = datetime.now()

    # ✅ Calcul des métriques
    processing_time = (end_time - start_time).total_seconds()
    avg_time_per_image = processing_time / len(test_images)

    valid_predictions = [p for p in predictions if p is not None]
    avg_confidence = (
        np.mean([p["confidence"] for p in valid_predictions])
        if valid_predictions
        else 0
    )

    return {
        "total_time": processing_time,
        "avg_time_per_image": avg_time_per_image,
        "avg_confidence": avg_confidence,
        "success_rate": len(valid_predictions) / len(test_images) * 100,
    }


def preprocess_image(
    image_pil: Image.Image, target_size: Tuple[int, int] = (380, 380)
) -> np.ndarray:
    """
    Fonction de preprocessing adaptée à EfficientNet-ResNet
    """
    try:
        # ✅ Redimensionnement en conservant l’aspect ratio
        image_pil.thumbnail(target_size, Image.Resampling.LANCZOS)

        # ✅ Création d'une image RGB avec fond blanc
        new_image = Image.new("RGB", target_size, (255, 255, 255))

        # ✅ Centrage de l’image
        x = (target_size[0] - image_pil.width) // 2
        y = (target_size[1] - image_pil.height) // 2
        new_image.paste(image_pil, (x, y))

        # ✅ Conversion en tableau NumPy
        img_array = np.array(new_image, dtype=np.float32)

        # ✅ Ajout de la dimension batch
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Appliquer le prétraitement officiel EfficientNet
        img_array = efficientnet_preprocess(img_array)

        return img_array

    except Exception as e:
        print(f"🚨 Erreur dans le preprocessing: {e}")
        return np.zeros((1, 380, 380, 3))


def enhance_image_quality(image_pil: Image.Image) -> Image.Image:
    """
    Améliore la qualité de l'image avant analyse par EfficientNet-ResNet
    """
    try:
        # ✅ Augmentation du contraste
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(1.3)

        # ✅ Augmentation de la netteté
        enhancer = ImageEnhance.Sharpness(image_pil)
        image_pil = enhancer.enhance(1.2)

        # ✅ Amélioration des couleurs
        enhancer = ImageEnhance.Color(image_pil)
        image_pil = enhancer.enhance(1.15)

        return image_pil

    except Exception as e:
        print(f"🚨 Erreur dans l'amélioration de l'image: {e}")
        return image_pil
