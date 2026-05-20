"""
Swin Transformer Model for Plant Disease Classification

Loads the trained Swin Transformer model and provides inference capabilities.
"""

import os
import torch
import numpy as np
import json
import pickle
from PIL import Image
import faiss
from transformers import AutoFeatureExtractor
from training_pipelines.metric_training_core import DiagnosticModel
from config import SWIN_MODEL_PATH, SWIN_FAISS_INDEX, SWIN_METADATA, SWIN_METADATA_FULL

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
        self.feature_extractor = None
        self.faiss_index = None
        self.metadata = None
        self.class_names = None

        self._load_model()
        self._load_faiss_index()
        self._load_metadata()

    def _load_model(self):
        """Load the Swin Transformer model."""
        try:
            # First check if trained weights exist
            if os.path.exists(self.model_path):
                print(f"Loading trained weights from {self.model_path}")
                model_name = "microsoft/swin-base-patch4-window7-224"

                # Use the same architecture as the training pipeline (DiagnosticModel)
                self.model = DiagnosticModel(
                    model_name=model_name,
                    embedding_dim=768,
                    image_size=224,
                )

                # Load feature extractor for preprocessing
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

                # Load trained weights (supports both raw state dict and wrapped checkpoint)
                state_dict = torch.load(self.model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded trained weights from {self.model_path}")
            else:
                msg = f"Trained weights not found at {self.model_path}"
                if self.strict:
                    raise FileNotFoundError(msg)
                print(f"⚠️  {msg}")
                print("⚠️  Swin classifier will use mock predictions")
                self.model = None
                self.feature_extractor = None
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

                # Mixed precision setup for faster inference
                self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

                print("✅ Swin model moved to GPU with A100 optimizations")
            else:
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
        """Load metadata and class names."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata from {self.metadata_path}")
            elif os.path.exists(SWIN_METADATA_FULL):
                # Try loading full metadata pickle
                with open(SWIN_METADATA_FULL, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✅ Loaded full metadata from {SWIN_METADATA_FULL}")
            else:
                print(f"⚠️  Metadata not found, using default class names")
                self.metadata = {'class_names': [f'class_{i}' for i in range(109)]}

            self.class_names = self.metadata.get('class_names', [])

        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            self.class_names = [f'class_{i}' for i in range(109)]

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input.

        Args:
            image_path: Path to the image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        return inputs

    def extract_features(self, image_path):
        """
        Extract features from image using Swin model.

        Args:
            image_path: Path to the image

        Returns:
            np.ndarray: Feature vector
        """
        if not self.model or not self.feature_extractor:
            if self.strict:
                raise RuntimeError("Swin model not loaded - cannot extract features")
            # Return random features as fallback
            return np.random.rand(768).astype(np.float32)

        try:
            inputs = self.preprocess_image(image_path)
            # The DiagnosticModel expects a tensor input (pixel values)
            pixel_values = inputs.get("pixel_values") if isinstance(inputs, dict) else inputs
            if pixel_values is None:
                pixel_values = next(iter(inputs.values()))

            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()

            with torch.no_grad():
                # Use mixed precision for faster inference on A100
                if torch.cuda.is_available() and hasattr(self, 'scaler') and self.scaler:
                    with torch.cuda.amp.autocast():
                        emb = self.model(pixel_values)
                else:
                    emb = self.model(pixel_values)

                features = emb.cpu().numpy()

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
            # Extract features
            features = self.extract_features(image_path)

            # Search in FAISS index
            D, I = self.faiss_index.search(features, top_k)

            # Convert distances to confidence scores (higher distance = lower confidence)
            # Normalize distances to [0, 1] range
            max_dist = np.max(D)
            min_dist = np.min(D)
            if max_dist > min_dist:
                confidences = 1.0 - (D[0] - min_dist) / (max_dist - min_dist)
            else:
                confidences = np.ones(top_k) * 0.5

            # Get class names
            predictions = []
            for i, (idx, conf) in enumerate(zip(I[0], confidences)):
                class_name = self.class_names[idx] if idx < len(self.class_names) else f'class_{idx}'
                predictions.append({
                    'disease': class_name,
                    'confidence': float(conf),
                    'class_id': int(idx)
                })

            return predictions

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