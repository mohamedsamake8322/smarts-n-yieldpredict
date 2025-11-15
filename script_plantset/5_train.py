#!/usr/bin/env python3
"""
train.py - Entraînement du modèle multimodal vision-langage
Entraîne le modèle sur le dataset multimodal avec mixed precision
Support TPU, logging, checkpoints, et métriques complètes
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

# Imports locaux
from dataset_loader import create_data_module
from model_builder import create_multimodal_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalTrainer:
    """
    Entraîneur pour modèle multimodal vision-langage
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_classes: int,
                 device: str = "auto",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 use_mixed_precision: bool = True,
                 gradient_clip_val: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            model: Modèle multimodal à entraîner
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            num_classes: Nombre de classes
            device: Device (auto, cuda, cpu, tpu)
            learning_rate: Taux d'apprentissage
            weight_decay: Décroissance des poids
            use_mixed_precision: Utiliser la précision mixte
            gradient_clip_val: Valeur de clipping des gradients
            class_weights: Poids des classes pour l'équilibrage
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_val = gradient_clip_val
        
        # Device
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Sera mis à jour
            eta_min=learning_rate * 0.01
        )
        
        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Mixed precision
        if use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Métriques
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        logger.info(f"Trainer initialisé sur {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
    
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Déplacer vers device
            images = batch['images'].to(self.device, non_blocking=True)
            texts = batch['texts'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass avec mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, texts)
                    loss = self.criterion(outputs['logits'], labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass normal
                outputs = self.model(images, texts)
                loss = self.criterion(outputs['logits'], labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # Update
                self.optimizer.step()
            
            # Métriques
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculer les métriques moyennes
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Valide le modèle pour une époque"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch in pbar:
                # Déplacer vers device
                images = batch['images'].to(self.device, non_blocking=True)
                texts = batch['texts'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, texts)
                        loss = self.criterion(outputs['logits'], labels)
                else:
                    outputs = self.model(images, texts)
                    loss = self.criterion(outputs['logits'], labels)
                
                # Métriques
                total_loss += loss.item()
                pred = outputs['logits'].argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculer les métriques
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, 
                       checkpoint_dir: str = "checkpoints") -> str:
        """Sauvegarde un checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Sauvegarder checkpoint normal
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarder best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Meilleur modèle sauvegardé: {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Charge un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Checkpoint chargé: {checkpoint_path}")
        logger.info(f"Reprise à l'époque {start_epoch}")
        
        return start_epoch
    
    def train(self, 
              num_epochs: int,
              checkpoint_dir: str = "checkpoints",
              log_dir: str = "logs",
              save_every: int = 5,
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Entraîne le modèle
        """
        # Créer les répertoires
        Path(checkpoint_dir).mkdir(exist_ok=True)
        Path(log_dir).mkdir(exist_ok=True)
        
        # TensorBoard
        writer = SummaryWriter(log_dir)
        
        # Early stopping
        best_epoch = 0
        patience_counter = 0
        
        logger.info(f"Début de l'entraînement pour {num_epochs} époques")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Entraînement
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Enregistrer les métriques
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Logging
            epoch_time = time.time() - start_time
            logger.info(f"Epoque {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                       f"Val F1: {val_metrics['f1']:.2f}%, "
                       f"Time: {epoch_time:.1f}s")
            
            # TensorBoard
            writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Sauvegarder checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, checkpoint_dir)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping à l'époque {epoch+1} "
                           f"(patience: {early_stopping_patience})")
                break
        
        # Sauvegarder le modèle final
        self.save_checkpoint(epoch, False, checkpoint_dir)
        
        # Fermer TensorBoard
        writer.close()
        
        logger.info(f"Entraînement terminé. Meilleure précision: {self.best_val_acc:.2f}% "
                   f"à l'époque {best_epoch+1}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }

def main():
    parser = argparse.ArgumentParser(description="Entraînement modèle multimodal")
    
    # Données
    parser.add_argument("--jsonl-file", type=str, required=True,
                       help="Fichier multimodal_dataset.jsonl")
    parser.add_argument("--root-dir", type=str, required=True,
                       help="Répertoire racine des images")
    parser.add_argument("--num-classes", type=int, required=True,
                       help="Nombre de classes")
    
    # Modèle
    parser.add_argument("--vision-backbone", type=str, default="resnet50",
                       help="Backbone visuel (resnet50, efficientnet_b0, vit_base_patch16)")
    parser.add_argument("--text-model", type=str, default="microsoft/DialoGPT-medium",
                       help="Modèle de texte HuggingFace")
    parser.add_argument("--vision-dim", type=int, default=512,
                       help="Dimension des features visuelles")
    parser.add_argument("--text-dim", type=int, default=512,
                       help="Dimension des features textuelles")
    parser.add_argument("--fusion-dim", type=int, default=256,
                       help="Dimension de la fusion")
    
    # Entraînement
    parser.add_argument("--epochs", type=int, default=100,
                       help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Taille des batches")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Taux d'apprentissage")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Décroissance des poids")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Taille des images")
    parser.add_argument("--text-length", type=int, default=512,
                       help="Longueur du texte")
    
    # Options
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu, tpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Nombre de workers DataLoader")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Utiliser la précision mixte")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Valeur de clipping des gradients")
    parser.add_argument("--freeze-vision", action="store_true",
                       help="Geler l'encodeur visuel")
    parser.add_argument("--freeze-text", action="store_true",
                       help="Geler l'encodeur textuel")
    
    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Répertoire des checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Répertoire des logs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Chemin vers checkpoint à reprendre")
    parser.add_argument("--save-every", type=int, default=5,
                       help="Sauvegarder tous les N époques")
    parser.add_argument("--early-stopping", type=int, default=10,
                       help="Patience pour early stopping")
    
    args = parser.parse_args()
    
    # Créer le data module
    logger.info("Chargement des données...")
    data_module = create_data_module(
        jsonl_file=args.jsonl_file,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        text_model_name=args.text_model,
        max_text_length=args.text_length
    )
    
    # Créer le modèle
    logger.info("Création du modèle...")
    model = create_multimodal_model(
        num_classes=args.num_classes,
        vision_backbone=args.vision_backbone,
        text_model=args.text_model,
        vision_dim=args.vision_dim,
        text_dim=args.text_dim,
        fusion_dim=args.fusion_dim,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text
    )
    
    # Obtenir les poids de classes
    class_weights = data_module.get_class_weights()
    
    # Créer le trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        num_classes=args.num_classes,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_mixed_precision=args.mixed_precision,
        gradient_clip_val=args.gradient_clip,
        class_weights=class_weights
    )
    
    # Reprendre depuis checkpoint si spécifié
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Entraîner
    logger.info("Début de l'entraînement...")
    metrics = trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping
    )
    
    logger.info("Entraînement terminé!")

if __name__ == "__main__":
    main()
