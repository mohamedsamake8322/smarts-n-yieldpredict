"""
Service pour le modèle de vision par ordinateur
Détection des plantes et maladies à partir d'images
Support pour modèles entraînés (TensorFlow, PyTorch, ONNX)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

class VisionModelService:
    """Service pour charger et utiliser le modèle de vision entraîné"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_type = None  # 'tensorflow', 'pytorch', 'onnx', 'huggingface'
        
        # Chemins possibles pour le modèle
        self.model_paths = {
            'tensorflow': Path("models/vision_model_tf"),
            'pytorch': Path("models/vision_model_pytorch.pth"),
            'onnx': Path("models/vision_model.onnx"),
            'huggingface': Path("models/vision_model_hf")
        }
        
        self._detect_and_load_model()
    
    def _detect_and_load_model(self):
        """Détecte et charge le modèle disponible"""
        # Vérifier TensorFlow
        if self.model_paths['tensorflow'].exists():
            self._load_tensorflow_model()
        # Vérifier PyTorch
        elif self.model_paths['pytorch'].exists():
            self._load_pytorch_model()
        # Vérifier ONNX
        elif self.model_paths['onnx'].exists():
            self._load_onnx_model()
        # Vérifier HuggingFace
        elif self.model_paths['huggingface'].exists():
            self._load_huggingface_model()
        else:
            logger.warning("⚠️ Aucun modèle de vision trouvé. Mode simulation activé.")
            self.model_loaded = False
    
    def _load_tensorflow_model(self):
        """Charge un modèle TensorFlow/Keras"""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_paths['tensorflow']))
            self.model_type = 'tensorflow'
            self.model_loaded = True
            logger.info("✅ Modèle TensorFlow chargé avec succès")
        except ImportError:
            logger.warning("⚠️ TensorFlow non installé. Installez-le avec: pip install tensorflow")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle TensorFlow: {str(e)}")
    
    def _load_pytorch_model(self):
        """Charge un modèle PyTorch"""
        try:
            import torch
            self.model = torch.load(str(self.model_paths['pytorch']), map_location='cpu')
            self.model.eval()  # Mode évaluation
            self.model_type = 'pytorch'
            self.model_loaded = True
            logger.info("✅ Modèle PyTorch chargé avec succès")
        except ImportError:
            logger.warning("⚠️ PyTorch non installé. Installez-le avec: pip install torch torchvision")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle PyTorch: {str(e)}")
    
    def _load_onnx_model(self):
        """Charge un modèle ONNX"""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(self.model_paths['onnx']))
            self.model_type = 'onnx'
            self.model_loaded = True
            logger.info("✅ Modèle ONNX chargé avec succès")
        except ImportError:
            logger.warning("⚠️ ONNX Runtime non installé. Installez-le avec: pip install onnxruntime")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle ONNX: {str(e)}")
    
    def _load_huggingface_model(self):
        """Charge un modèle HuggingFace Transformers"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            processor = AutoImageProcessor.from_pretrained(str(self.model_paths['huggingface']))
            self.model = AutoModelForImageClassification.from_pretrained(str(self.model_paths['huggingface']))
            self.processor = processor
            self.model_type = 'huggingface'
            self.model_loaded = True
            logger.info("✅ Modèle HuggingFace chargé avec succès")
        except ImportError:
            logger.warning("⚠️ Transformers non installé. Installez-le avec: pip install transformers")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle HuggingFace: {str(e)}")
    
    def is_ready(self) -> bool:
        """Vérifie si le modèle est prêt"""
        return self.model_loaded and self.model is not None
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Prétraite l'image pour le modèle
        Redimensionne, normalise, etc.
        """
        # Redimensionnement standard (224x224 pour la plupart des modèles)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        
        # Normalisation (0-1)
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        # Ajout de la dimension batch
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_data: bytes) -> Dict:
        """
        Effectue la prédiction sur l'image
        
        Returns:
            Dict avec les prédictions (plante, maladies, confiance, etc.)
        """
        if not self.is_ready():
            return self._simulate_prediction()
        
        try:
            # Charger l'image
            image = Image.open(io.BytesIO(image_data))
            processed_image = self.preprocess_image(image)
            
            # Prédiction selon le type de modèle
            if self.model_type == 'tensorflow':
                predictions = self.model.predict(processed_image)
            elif self.model_type == 'pytorch':
                import torch
                with torch.no_grad():
                    tensor_input = torch.from_numpy(processed_image).float()
                    predictions = self.model(tensor_input).numpy()
            elif self.model_type == 'onnx':
                input_name = self.model.get_inputs()[0].name
                predictions = self.model.run(None, {input_name: processed_image.astype(np.float32)})[0]
            elif self.model_type == 'huggingface':
                inputs = self.processor(image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits.softmax(dim=-1).numpy()
            
            # Interpréter les résultats
            return self._interpret_predictions(predictions)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            return self._simulate_prediction()
    
    def _interpret_predictions(self, predictions: np.ndarray) -> Dict:
        """
        Interprète les prédictions du modèle
        À adapter selon votre architecture de sortie
        """
        # Exemple de structure de sortie
        # Adaptez selon votre modèle entraîné
        
        # Si le modèle retourne des probabilités pour chaque classe
        if len(predictions.shape) > 1:
            predictions = predictions[0]  # Prendre le premier élément du batch
        
        # Classes possibles (à remplacer par vos vraies classes)
        plant_classes = ["Tomate", "Maïs", "Riz", "Manioc", "Banane", "Cacao", "Café", "Arachide"]
        disease_classes = ["Sain", "Mildiou", "Oïdium", "Rouille", "Anthracnose", "Cicadelle"]
        
        # Exemple simple : prendre les indices avec les plus hautes probabilités
        top_indices = np.argsort(predictions)[::-1][:3]
        
        # Simulation pour l'instant - à remplacer par votre logique
        return {
            "plant": {
                "name": plant_classes[top_indices[0] % len(plant_classes)],
                "confidence": float(predictions[top_indices[0]])
            },
            "diseases": [
                {
                    "name": disease_classes[top_indices[1] % len(disease_classes)],
                    "confidence": float(predictions[top_indices[1]]),
                    "severity": "moderate"
                }
            ],
            "raw_predictions": predictions.tolist()
        }
    
    def _simulate_prediction(self) -> Dict:
        """Mode simulation si le modèle n'est pas disponible"""
        return {
            "plant": {
                "name": "Tomate",
                "scientific_name": "Solanum lycopersicum",
                "confidence": 0.85
            },
            "diseases": [
                {
                    "name": "Mildiou",
                    "confidence": 0.75,
                    "severity": "moderate",
                    "description": "Taches brunes sur les feuilles avec duvet blanc"
                }
            ],
            "deficiencies": [
                {
                    "type": "Azote",
                    "confidence": 0.60,
                    "severity": "light"
                }
            ],
            "stress": {
                "water": 0.55,
                "temperature": 0.30
            }
        }








