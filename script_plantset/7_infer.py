#!/usr/bin/env python3
"""
infer.py - Prédictions avec le modèle multimodal
Fait des prédictions sur de nouvelles images
Retourne maladie prédite + texte descriptif (symptômes + traitements)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
import numpy as np

# Imports locaux
from model_builder import create_multimodal_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalInference:
    """
    Interface d'inférence pour modèle multimodal vision-langage
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 num_classes: int,
                 vision_backbone: str = "resnet50",
                 text_model: str = "microsoft/DialoGPT-medium",
                 vision_dim: int = 512,
                 text_dim: int = 512,
                 fusion_dim: int = 256,
                 image_size: int = 224,
                 max_text_length: int = 512,
                 device: str = "auto",
                 class_names: Optional[List[str]] = None):
        """
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
            num_classes: Nombre de classes
            vision_backbone: Architecture du backbone visuel
            text_model: Modèle de texte HuggingFace
            vision_dim: Dimension des features visuelles
            text_dim: Dimension des features textuelles
            fusion_dim: Dimension de la fusion
            image_size: Taille des images
            max_text_length: Longueur maximale du texte
            device: Device (auto, cuda, cpu, tpu)
            class_names: Noms des classes
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.device = self._setup_device(device)
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Charger le tokenizer
        self.tokenizer = self._load_tokenizer(text_model)
        
        # Créer le modèle
        self.model = create_multimodal_model(
            num_classes=num_classes,
            vision_backbone=vision_backbone,
            text_model=text_model,
            vision_dim=vision_dim,
            text_dim=text_dim,
            fusion_dim=fusion_dim
        )
        
        # Charger le checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Créer les transforms
        self.image_transform = self._create_image_transforms()
        
        logger.info(f"Inference initialisé sur {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Configure le device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch, 'xla') and torch.xla.is_available():
                return torch.device("xla")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_tokenizer(self, model_name: str):
        """Charge le tokenizer pour le texte"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.warning(f"Erreur chargement tokenizer {model_name}: {e}")
            # Fallback vers un tokenizer simple
            return self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """Crée un tokenizer simple en fallback"""
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {}
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"
                self.unk_token = "[UNK]"
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.unk_token_id = 2
            
            def encode(self, text, max_length=512, padding=True, truncation=True):
                words = text.lower().split()[:max_length-2]
                token_ids = [self.pad_token_id]
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab) + 3
                    token_ids.append(self.vocab[word])
                token_ids.append(self.eos_token_id)
                
                if padding and len(token_ids) < max_length:
                    token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
                
                return token_ids
            
            def __call__(self, text, max_length=512, padding=True, truncation=True, return_tensors=None):
                token_ids = self.encode(text, max_length, padding, truncation)
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor(token_ids).unsqueeze(0)}
                return {"input_ids": token_ids}
        
        return SimpleTokenizer()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Charge le checkpoint du modèle"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Modèle chargé depuis: {checkpoint_path}")
    
    def _create_image_transforms(self) -> T.Compose:
        """Crée les transformations d'images"""
        return T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Prétraite une image pour l'inférence
        """
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Appliquer les transforms
        image_tensor = self.image_transform(image).unsqueeze(0)  # [1, 3, H, W]
        
        return image_tensor.to(self.device)
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Prétraite un texte pour l'inférence
        """
        if hasattr(self.tokenizer, 'encode'):
            # Tokenizer simple
            token_ids = self.tokenizer.encode(
                text,
                max_length=self.max_text_length,
                padding=True,
                truncation=True
            )
            text_tensor = torch.tensor(token_ids).unsqueeze(0)  # [1, seq_len]
        else:
            # Tokenizer HuggingFace
            encoded = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_tensor = encoded['input_ids']  # [1, seq_len]
        
        return text_tensor.to(self.device)
    
    def predict_single(self, 
                      image_path: Union[str, Path, Image.Image],
                      text: str = "",
                      top_k: int = 5) -> Dict[str, Any]:
        """
        Prédiction sur une seule image
        """
        # Prétraiter les inputs
        image_tensor = self.preprocess_image(image_path)
        text_tensor = self.preprocess_text(text)
        
        # Inférence
        with torch.no_grad():
            outputs = self.model(image_tensor, text_tensor)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
        
        # Extraire les prédictions top-k
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        predictions = []
        for i in range(top_indices.size(1)):
            class_idx = top_indices[0, i].item()
            confidence = top_probs[0, i].item()
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
            
            predictions.append({
                'class_name': class_name,
                'class_index': class_idx,
                'confidence': confidence,
                'confidence_percent': confidence * 100
            })
        
        # Prédiction principale
        predicted_class = predictions[0]
        
        return {
            'predicted_class': predicted_class,
            'top_predictions': predictions,
            'all_probabilities': probabilities[0].cpu().numpy().tolist(),
            'features': {
                'vision_features': outputs['vision_features'][0].cpu().numpy().tolist(),
                'text_features': outputs['text_features'][0].cpu().numpy().tolist(),
                'fused_features': outputs['fused_features'][0].cpu().numpy().tolist()
            }
        }
    
    def predict_batch(self, 
                     image_paths: List[Union[str, Path, Image.Image]],
                     texts: List[str],
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Prédiction sur un batch d'images
        """
        results = []
        
        for image_path, text in zip(image_paths, texts):
            try:
                result = self.predict_single(image_path, text, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur prédiction {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': str(image_path)
                })
        
        return results
    
    def generate_description(self, 
                           image_path: Union[str, Path, Image.Image],
                           text: str = "",
                           max_length: int = 200) -> str:
        """
        Génère une description textuelle de la maladie
        Note: Cette fonction est simplifiée. Dans un vrai système,
        on utiliserait un générateur de texte entraîné.
        """
        # Obtenir la prédiction
        prediction = self.predict_single(image_path, text, top_k=1)
        class_name = prediction['predicted_class']['class_name']
        confidence = prediction['predicted_class']['confidence']
        
        # Générer une description basique
        description = f"Maladie détectée: {class_name} (confiance: {confidence:.2%})\n\n"
        
        # Ajouter des informations basiques selon la classe
        if "healthy" in class_name.lower():
            description += "Cette plante semble en bonne santé. Aucun symptôme de maladie détecté."
        elif "blight" in class_name.lower():
            description += "Symptômes de flétrissure détectés. Cette maladie peut causer des taches brunes et le flétrissement des feuilles."
        elif "spot" in class_name.lower():
            description += "Taches sur les feuilles détectées. Surveillez l'évolution et appliquez un traitement approprié."
        elif "mosaic" in class_name.lower():
            description += "Virus de la mosaïque détecté. Cette maladie virale peut affecter la croissance de la plante."
        else:
            description += f"Maladie de type {class_name} détectée. Consultez un expert pour un diagnostic précis et un traitement approprié."
        
        return description
    
    def save_predictions(self, 
                        predictions: List[Dict[str, Any]], 
                        output_path: str = "predictions.json"):
        """Sauvegarde les prédictions dans un fichier JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        logger.info(f"Prédictions sauvegardées: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Inférence avec modèle multimodal")
    
    # Modèle
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Chemin vers le checkpoint du modèle")
    parser.add_argument("--num-classes", type=int, required=True,
                       help="Nombre de classes")
    parser.add_argument("--class-names", type=str, default=None,
                       help="Fichier JSON avec noms des classes")
    
    # Modèle (doit correspondre à l'entraînement)
    parser.add_argument("--vision-backbone", type=str, default="resnet50",
                       help="Backbone visuel")
    parser.add_argument("--text-model", type=str, default="microsoft/DialoGPT-medium",
                       help="Modèle de texte")
    parser.add_argument("--vision-dim", type=int, default=512,
                       help="Dimension des features visuelles")
    parser.add_argument("--text-dim", type=int, default=512,
                       help="Dimension des features textuelles")
    parser.add_argument("--fusion-dim", type=int, default=256,
                       help="Dimension de la fusion")
    
    # Inférence
    parser.add_argument("--image", type=str, required=True,
                       help="Chemin vers l'image à analyser")
    parser.add_argument("--text", type=str, default="",
                       help="Texte descriptif optionnel")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Nombre de prédictions top-k")
    parser.add_argument("--output", type=str, default="prediction.json",
                       help="Fichier de sortie")
    parser.add_argument("--generate-description", action="store_true",
                       help="Générer une description textuelle")
    
    # Options
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu, tpu)")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Taille des images")
    parser.add_argument("--text-length", type=int, default=512,
                       help="Longueur du texte")
    
    args = parser.parse_args()
    
    # Charger les noms de classes si fournis
    class_names = None
    if args.class_names and Path(args.class_names).exists():
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
    
    # Créer l'interface d'inférence
    logger.info("Initialisation de l'inférence...")
    inference = MultimodalInference(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        vision_backbone=args.vision_backbone,
        text_model=args.text_model,
        vision_dim=args.vision_dim,
        text_dim=args.text_dim,
        fusion_dim=args.fusion_dim,
        image_size=args.image_size,
        max_text_length=args.text_length,
        device=args.device,
        class_names=class_names
    )
    
    # Faire la prédiction
    logger.info(f"Analyse de l'image: {args.image}")
    prediction = inference.predict_single(
        image_path=args.image,
        text=args.text,
        top_k=args.top_k
    )
    
    # Afficher les résultats
    print("\n" + "="*50)
    print("RÉSULTATS DE PRÉDICTION")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Texte: {args.text}")
    print(f"\nPrédiction principale:")
    print(f"  Classe: {prediction['predicted_class']['class_name']}")
    print(f"  Confiance: {prediction['predicted_class']['confidence_percent']:.2f}%")
    
    print(f"\nTop-{args.top_k} prédictions:")
    for i, pred in enumerate(prediction['top_predictions']):
        print(f"  {i+1}. {pred['class_name']}: {pred['confidence_percent']:.2f}%")
    
    # Générer description si demandé
    if args.generate_description:
        description = inference.generate_description(args.image, args.text)
        print(f"\nDescription générée:")
        print(description)
        prediction['generated_description'] = description
    
    # Sauvegarder les résultats
    inference.save_predictions([prediction], args.output)
    
    print(f"\nRésultats sauvegardés: {args.output}")
    print("="*50)

if __name__ == "__main__":
    main()
