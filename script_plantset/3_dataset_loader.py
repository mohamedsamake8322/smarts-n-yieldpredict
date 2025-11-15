#!/usr/bin/env python3
"""
dataset_loader.py - Chargement du dataset multimodal dans PyTorch
Charge le dataset multimodal avec DataLoader scalable
Prépare images (torchvision.transforms) et textes (tokenizer HuggingFace)
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoProcessor
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    """
    Dataset multimodal pour images + textes + labels
    Optimisé pour l'entraînement de modèles vision-langage
    """
    
    def __init__(self,
                 jsonl_file: str,
                 root_dir: str,
                 split: str = "train",
                 image_size: int = 224,
                 text_model_name: str = "microsoft/DialoGPT-medium",
                 max_text_length: int = 512,
                 augment: bool = True,
                 normalize_images: bool = True):
        """
        Args:
            jsonl_file: Chemin vers le fichier multimodal_dataset.jsonl
            root_dir: Répertoire racine des images
            split: Split à charger (train/val/test)
            image_size: Taille des images (carré)
            text_model_name: Nom du modèle de tokenisation
            max_text_length: Longueur maximale du texte
            augment: Appliquer des augmentations (train seulement)
            normalize_images: Normaliser les images
        """
        self.jsonl_file = Path(jsonl_file)
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.augment = augment and (split == "train")
        self.normalize_images = normalize_images
        
        # Charger le tokenizer
        self.tokenizer = self._load_tokenizer(text_model_name)
        
        # Charger les données
        self.samples = self._load_samples()
        
        # Créer les transforms
        self.image_transform = self._create_image_transforms()
        
        logger.info(f"Chargé {len(self.samples)} échantillons pour split '{split}'")
    
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
                # Tokenisation simple par mots
                words = text.lower().split()[:max_length-2]  # -2 pour [CLS] et [SEP]
                token_ids = [self.pad_token_id]  # [CLS]
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab) + 3  # +3 pour les tokens spéciaux
                    token_ids.append(self.vocab[word])
                token_ids.append(self.eos_token_id)  # [SEP]
                
                if padding and len(token_ids) < max_length:
                    token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
                
                return token_ids
            
            def __call__(self, text, max_length=512, padding=True, truncation=True, return_tensors=None):
                token_ids = self.encode(text, max_length, padding, truncation)
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor(token_ids).unsqueeze(0)}
                return {"input_ids": token_ids}
        
        return SimpleTokenizer()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Charge les échantillons depuis le JSONL"""
        samples = []
        
        if not self.jsonl_file.exists():
            raise FileNotFoundError(f"Fichier JSONL introuvable: {self.jsonl_file}")
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line.strip())
                    if data.get('split') == self.split:
                        # Vérifier que l'image existe
                        image_path = self.root_dir / data['image_path']
                        if image_path.exists():
                            samples.append(data)
                        else:
                            logger.warning(f"Image introuvable: {image_path}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Erreur JSON ligne {line_num}: {e}")
                    continue
        
        return samples
    
    def _create_image_transforms(self) -> T.Compose:
        """Crée les transformations d'images"""
        transforms_list = []
        
        # Redimensionnement
        if self.augment:
            # Augmentations pour l'entraînement
            transforms_list.extend([
                T.Resize((self.image_size + 32, self.image_size + 32)),
                T.RandomCrop((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            # Pas d'augmentation pour val/test
            transforms_list.append(T.Resize((self.image_size, self.image_size)))
        
        # Conversion en tenseur
        transforms_list.append(T.ToTensor())
        
        # Normalisation
        if self.normalize_images:
            transforms_list.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return T.Compose(transforms_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retourne un échantillon du dataset"""
        sample = self.samples[idx]
        
        # Charger et transformer l'image
        image_path = self.root_dir / sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image)
        except Exception as e:
            logger.warning(f"Erreur chargement image {image_path}: {e}")
            # Image de fallback
            image_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # Tokeniser le texte
        text = sample.get('text', '')
        try:
            if hasattr(self.tokenizer, 'encode'):
                # Tokenizer simple
                text_tokens = self.tokenizer.encode(
                    text, 
                    max_length=self.max_text_length,
                    padding=True,
                    truncation=True
                )
                text_tensor = torch.tensor(text_tokens)
            else:
                # Tokenizer HuggingFace
                text_encoded = self.tokenizer(
                    text,
                    max_length=self.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_tensor = text_encoded['input_ids'].squeeze(0)
        except Exception as e:
            logger.warning(f"Erreur tokenisation: {e}")
            text_tensor = torch.zeros(self.max_text_length, dtype=torch.long)
        
        # Label (sera mappé en indices par le DataLoader)
        label = sample['label']
        
        return {
            'image': image_tensor,
            'text': text_tensor,
            'label': label,
            'image_path': sample['image_path']
        }

class MultimodalDataModule:
    """
    Module de données pour gérer train/val/test splits
    """
    
    def __init__(self,
                 jsonl_file: str,
                 root_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: int = 224,
                 text_model_name: str = "microsoft/DialoGPT-medium",
                 max_text_length: int = 512,
                 pin_memory: bool = True):
        """
        Args:
            jsonl_file: Chemin vers multimodal_dataset.jsonl
            root_dir: Répertoire racine des images
            batch_size: Taille des batches
            num_workers: Nombre de workers pour DataLoader
            image_size: Taille des images
            text_model_name: Modèle de tokenisation
            max_text_length: Longueur maximale du texte
            pin_memory: Utiliser pin_memory pour GPU
        """
        self.jsonl_file = jsonl_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.text_model_name = text_model_name
        self.max_text_length = max_text_length
        self.pin_memory = pin_memory
        
        # Créer les datasets
        self.train_dataset = MultimodalDataset(
            jsonl_file=jsonl_file,
            root_dir=root_dir,
            split="train",
            image_size=image_size,
            text_model_name=text_model_name,
            max_text_length=max_text_length,
            augment=True
        )
        
        self.val_dataset = MultimodalDataset(
            jsonl_file=jsonl_file,
            root_dir=root_dir,
            split="val",
            image_size=image_size,
            text_model_name=text_model_name,
            max_text_length=max_text_length,
            augment=False
        )
        
        # Créer les DataLoaders
        self.train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        
        # Créer le mapping des labels
        self.label_to_idx, self.idx_to_label = self._create_label_mapping()
        
        logger.info(f"DataModule créé: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
        logger.info(f"Classes: {len(self.label_to_idx)}")
    
    def _create_dataloader(self, dataset: MultimodalDataset, shuffle: bool) -> DataLoader:
        """Crée un DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,  # Drop last batch seulement pour train
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Fonction de collation pour les batches"""
        images = torch.stack([item['image'] for item in batch])
        texts = torch.stack([item['text'] for item in batch])
        labels = [item['label'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        
        # Convertir les labels en indices
        label_indices = torch.tensor([self.label_to_idx.get(label, 0) for label in labels])
        
        return {
            'images': images,
            'texts': texts,
            'labels': label_indices,
            'label_names': labels,
            'image_paths': image_paths
        }
    
    def _create_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Crée le mapping des labels vers indices"""
        all_labels = set()
        
        # Collecter tous les labels
        for dataset in [self.train_dataset, self.val_dataset]:
            for sample in dataset.samples:
                all_labels.add(sample['label'])
        
        # Créer les mappings
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return label_to_idx, idx_to_label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calcule les poids de classes pour l'équilibrage"""
        from collections import Counter
        
        # Compter les occurrences
        train_labels = [sample['label'] for sample in self.train_dataset.samples]
        label_counts = Counter(train_labels)
        
        # Calculer les poids inverses
        total_samples = len(train_labels)
        num_classes = len(self.label_to_idx)
        
        weights = []
        for label in sorted(self.label_to_idx.keys()):
            count = label_counts.get(label, 1)  # Éviter division par zéro
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

def create_data_module(jsonl_file: str,
                      root_dir: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: int = 224,
                      text_model_name: str = "microsoft/DialoGPT-medium",
                      **kwargs) -> MultimodalDataModule:
    """
    Fonction utilitaire pour créer un MultimodalDataModule
    """
    return MultimodalDataModule(
        jsonl_file=jsonl_file,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        text_model_name=text_model_name,
        **kwargs
    )

if __name__ == "__main__":
    # Test du dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du MultimodalDataset")
    parser.add_argument("--jsonl-file", type=str, required=True, help="Fichier multimodal_dataset.jsonl")
    parser.add_argument("--root-dir", type=str, required=True, help="Répertoire racine des images")
    parser.add_argument("--batch-size", type=int, default=4, help="Taille du batch")
    
    args = parser.parse_args()
    
    # Créer le data module
    data_module = create_data_module(
        jsonl_file=args.jsonl_file,
        root_dir=args.root_dir,
        batch_size=args.batch_size
    )
    
    # Tester un batch
    print("Test du DataLoader...")
    for i, batch in enumerate(data_module.train_loader):
        print(f"Batch {i}:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Texts shape: {batch['texts'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Label names: {batch['label_names']}")
        if i >= 2:  # Tester seulement 3 batches
            break
    
    print("Test terminé!")
