"""
Swin Transformer Model for Plant Disease Classification

Loads the trained Swin Transformer model and provides inference capabilities.
Uses the same architecture and preprocessing as metric_training_core (timm).
"""

import os
import torch
import numpy as np
import json
import pickle
from PIL import Image
import faiss
from training_pipelines.metric_training_core import DiagnosticModel
from config import SWIN_MODEL_PATH, SWIN_FAISS_INDEX, SWIN_METADATA, SWIN_METADATA_FULL

# Nom du modèle timm (identique au training) - PAS le nom HuggingFace
TIMM_MODEL_NAME = "swin_base_patch4_window7_224"

class SwinDiseaseClassifier:
    def __init__(self, model_path=None, faiss_index_path=None, metadata_path=None, strict: bool = True):
        """Initialize the Swin disease classifier.

        Args:
            model_path: Path to the trained model weights
            faiss_index_path: Path to the FAISS index
            metadata_path: Path to the metadata JSON
            strict: If True, raise errors when the model or index is missing (no mock fallback).
        """
        self.strict = strict
        self.model_path = model_path or SWIN_MODEL_PATH
        self.faiss_index_path = faiss_index_path or SWIN_FAISS_INDEX
        self.metadata_path = metadata_path or SWIN_METADATA

        # Load model components
        self.model = None
        self.image_size = 224
        self.faiss_index = None
        self.metadata = None
        self.class_names = None

        self._load_model()
        self._load_faiss_index()
        self._load_metadata()

    def _load_model(self):
        """Load the Swin Transformer model (timm, même architecture que le training)."""
        try:
            # First check if trained weights exist
            if os.path.exists(self.model_path):
                print(f"Loading trained weights from {self.model_path}")

                # Récupérer config depuis le checkpoint si possible
                checkpoint = torch.load(self.model_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    cfg = checkpoint.get("config", {})
                    model_name = cfg.get("model_name", TIMM_MODEL_NAME)
                    embedding_dim = cfg.get("embedding_dim", 768)
                    image_size = cfg.get("image_size", 224)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                else:
                    model_name = TIMM_MODEL_NAME
                    embedding_dim = 768
                    image_size = 224
                    state_dict = checkpoint

                # Forcer le nom timm (évite swin-base-patch4-window7-224 HuggingFace)
                if "swin" in str(model_name).lower() and "-" in str(model_name):
                    model_name = TIMM_MODEL_NAME

                # DiagnosticModel utilise timm (swin_base_patch4_window7_224)
                self.model = DiagnosticModel(
                    model_name=model_name,
                    embedding_dim=embedding_dim,
                    image_size=image_size,
                )

                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                self.image_size = image_size
                print(f"✅ Loaded trained weights from {self.model_path}")
            else:
                msg = f"Trained weights not found at {self.model_path}"
                if self.strict:
                    raise FileNotFoundError(msg)
                print(f"⚠️  {msg}")
                print("⚠️  Swin classifier will use mock predictions")
                self.model = None
                return

            self.model.eval()

            # A100 optimizations (improvement #5)
            if torch.cuda.is_available():
                self.model = self.model.cuda()

                # Enable cuDNN benchmark for faster inference
                torch.backends.cudnn.benchmark = True

                # Enable TF32 for A100 GPUs (faster but slightly less precise)
                if torch.cuda.get_device_capability()[0] >= 8:  # Ampere architecture (A100, A6000, etc.)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("✅ A100 optimizations enabled: TF32, cuDNN benchmark")

                self.scaler = torch.cuda.amp.GradScaler()
                print("✅ Swin model moved to GPU with A100 optimizations")
            else:
                self.scaler = None
                print("✅ Swin model on CPU")

        except Exception as e:
            if self.strict:
                raise
            print(f"❌ Error loading Swin model: {e}")
            print("Swin classifier will use mock predictions")
            self.model = None

    def _load_faiss_index(self):
        """Load the FAISS index for nearest neighbor search."""
        try:
            if os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                print(f"✅ Loaded FAISS index from {self.faiss_index_path}")
            else:
                msg = f"FAISS index not found at {self.faiss_index_path}"
                if self.strict:
                    raise FileNotFoundError(msg)
                print(f"⚠️  {msg}")
        except Exception as e:
            if self.strict:
                raise
            print(f"❌ Error loading FAISS index: {e}")

    def _load_metadata(self):
        """Load metadata (labels, idx_to_class pour mapper FAISS -> classe)."""
        try:
            # Préférer metadata.pkl (complet avec labels pour FAISS)
            if os.path.exists(SWIN_METADATA_FULL):
                with open(SWIN_METADATA_FULL, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✅ Loaded full metadata from {SWIN_METADATA_FULL}")
            elif os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata from {self.metadata_path}")
            else:
                print(f"⚠️  Metadata not found, using default class names")
                self.metadata = {}

            self.labels = self.metadata.get('labels', [])
            self.idx_to_class = self.metadata.get('idx_to_class', {})
            if isinstance(self.idx_to_class, dict):
                for k in list(self.idx_to_class.keys()):
                    if isinstance(k, str) and k.isdigit():
                        self.idx_to_class[int(k)] = self.idx_to_class.pop(k)
            self.class_names = self.metadata.get('class_names', [])
            if not self.class_names and self.idx_to_class:
                max_idx = max(k for k in self.idx_to_class if isinstance(k, int))
                self.class_names = [self.idx_to_class.get(i, f'class_{i}') for i in range(max_idx + 1)]

        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            self.labels = []
            self.idx_to_class = {}
            self.class_names = []

    def _preprocess_image(self, image_path):
        """
        Preprocessing identique au training (metric_training_core) :
        resize 224, normalize ImageNet. Pas HuggingFace.
        """
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image.resize((self.image_size, self.image_size))).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def extract_features(self, image_path):
        """
        Extract features from image using Swin model.
        """
        if not self.model:
            if self.strict:
                raise RuntimeError("Swin model not loaded - cannot extract features")
            return np.random.rand(768).astype(np.float32)

        try:
            tensor = self._preprocess_image(image_path)
            if torch.cuda.is_available():
                tensor = tensor.cuda()

            with torch.no_grad():
                if torch.cuda.is_available() and hasattr(self, 'scaler') and self.scaler:
                    with torch.cuda.amp.autocast():
                        emb = self.model(tensor)
                else:
                    emb = self.model(tensor)

                features = emb.cpu().numpy().astype(np.float32)
                features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.random.rand(768).astype(np.float32)

    def classify_image(self, image_path, top_k=3):
        """
        Classify disease in image.

        Args:
            image_path: Path to the image
            top_k: Number of top predictions to return

        Returns:
            list: List of predictions with class names and confidences
        """
        if not self.model or not self.faiss_index:
            if self.strict:
                raise RuntimeError("Swin model or FAISS index not available - cannot classify")
            # Fallback: return mock predictions
            print("⚠️  Swin model or FAISS index not available, using mock predictions")
            mock_diseases = [
                "bean bruchid", "bean rust", "bean blight", "bean anthracnose",
                "bean bacterial blight", "bean mosaic virus", "bean powdery mildew"
            ]
            predictions = []
            for i in range(min(top_k, len(mock_diseases))):
                confidence = max(0.1, 0.9 - i * 0.2)  # Decreasing confidence
                predictions.append({
                    'disease': mock_diseases[i],
                    'confidence': confidence,
                    'class_id': i
                })
            return predictions

        try:
            # Extract features (déjà L2-normalisées)
            features = self.extract_features(image_path)

            # FAISS IndexFlatIP : retourne similarité (produit scalaire), plus élevé = plus similaire
            D, I = self.faiss_index.search(features.astype(np.float32), top_k)

            # Mapper index FAISS -> class_id -> class_name via metadata
            def _label_to_name(label: int) -> str:
                if label in self.idx_to_class:
                    return self.idx_to_class[label]
                if str(label) in self.idx_to_class:
                    return self.idx_to_class[str(label)]
                if self.class_names and 0 <= label < len(self.class_names):
                    return self.class_names[label]
                return f'class_{label}'

            predictions = []
            for i, (idx, sim) in enumerate(zip(I[0], D[0])):
                if idx < 0:
                    continue
                if self.labels and idx < len(self.labels):
                    class_id = int(self.labels[idx])
                elif self.class_names and idx < len(self.class_names):
                    class_id = idx
                else:
                    class_id = idx
                class_name = _label_to_name(class_id)
                conf = float(np.clip(sim, 0.0, 1.0))
                predictions.append({
                    'disease': class_name,
                    'confidence': conf,
                    'class_id': class_id
                })

            return predictions[:top_k] if predictions else []

        except Exception as e:
            print(f"Error during classification: {e}")
            # Return mock predictions on error
            return [
                {'disease': 'classification_error', 'confidence': 0.0, 'class_id': -1}
            ]

    def get_class_info(self, class_id):
        """
        Get information about a specific class.

        Args:
            class_id: Class ID

        Returns:
            dict: Class information
        """
        if not self.metadata or 'class_info' not in self.metadata:
            return {'name': f'class_{class_id}', 'description': 'No information available'}

        class_info = self.metadata['class_info'].get(str(class_id), {})
        return class_info