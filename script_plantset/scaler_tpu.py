#!/usr/bin/env python3
"""
scaler_tpu.py - Support TPU pour entraînement massif
Adapte l'entraînement pour TPU v4, v5, v6 avec torch_xla
Support multi-TPU et dataset sharding
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp

# Imports locaux
from dataset_loader import create_data_module
from model_builder import create_multimodal_model
from train import MultimodalTrainer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TPUTrainer:
    """
    Entraîneur optimisé pour TPU
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_classes: int,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 gradient_clip_val: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 use_mixed_precision: bool = True):
        """
        Args:
            model: Modèle multimodal à entraîner
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            num_classes: Nombre de classes
            learning_rate: Taux d'apprentissage
            weight_decay: Décroissance des poids
            gradient_clip_val: Valeur de clipping des gradients
            class_weights: Poids des classes
            use_mixed_precision: Utiliser la précision mixte
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_val = gradient_clip_val
        
        # Device TPU
        self.device = xm.xla_device()
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Sera mis à jour
            eta_min=learning_rate * 0.01
        )
        
        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Mixed precision
        if use_mixed_precision:
            self.scaler = xm.amp.GradScaler()
        else:
            self.scaler = None
        
        # Métriques
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        logger.info(f"TPU Trainer initialisé sur {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Entraîne le modèle pour une époque sur TPU"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Parallel loader pour TPU
        train_loader = pl.MpDeviceLoader(self.train_loader, self.device)
        
        for batch_idx, batch in enumerate(train_loader):
            # Déplacer vers device TPU
            images = batch['images']
            texts = batch['texts']
            labels = batch['labels']
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass avec mixed precision
            if self.scaler is not None:
                with xm.amp.autocast():
                    outputs = self.model(images, texts)
                    loss = self.criterion(outputs['logits'], labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    xm.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
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
                    xm.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                # Update
                self.optimizer.step()
            
            # Métriques
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            # Logging périodique
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, "
                           f"Loss: {loss.item():.4f}, "
                           f"Acc: {100.*correct/total:.2f}%")
        
        # Synchroniser les métriques entre TPUs
        total_loss = xm.mesh_reduce('train_loss', total_loss, sum)
        correct = xm.mesh_reduce('train_correct', correct, sum)
        total = xm.mesh_reduce('train_total', total, sum)
        
        # Calculer les métriques moyennes
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Valide le modèle pour une époque sur TPU"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Parallel loader pour TPU
        val_loader = pl.MpDeviceLoader(self.val_loader, self.device)
        
        with torch.no_grad():
            for batch in val_loader:
                # Déplacer vers device TPU
                images = batch['images']
                texts = batch['texts']
                labels = batch['labels']
                
                # Forward pass
                if self.scaler is not None:
                    with xm.amp.autocast():
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
        
        # Synchroniser les métriques entre TPUs
        total_loss = xm.mesh_reduce('val_loss', total_loss, sum)
        
        # Calculer les métriques
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculer accuracy localement (pas de synchronisation nécessaire)
        from sklearn.metrics import accuracy_score, f1_score
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
        """Sauvegarde un checkpoint (seulement sur le processus principal)"""
        if xm.is_master_ordinal():
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
        return ""
    
    def train(self, 
              num_epochs: int,
              checkpoint_dir: str = "checkpoints",
              save_every: int = 5,
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Entraîne le modèle sur TPU
        """
        # Créer le répertoire de checkpoints (seulement sur le processus principal)
        if xm.is_master_ordinal():
            Path(checkpoint_dir).mkdir(exist_ok=True)
        
        # Early stopping
        best_epoch = 0
        patience_counter = 0
        
        logger.info(f"Début de l'entraînement TPU pour {num_epochs} époques")
        
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
            
            # Logging (seulement sur le processus principal)
            if xm.is_master_ordinal():
                epoch_time = time.time() - start_time
                logger.info(f"Epoque {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                           f"Val F1: {val_metrics['f1']:.2f}%, "
                           f"Time: {epoch_time:.1f}s")
            
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
                if xm.is_master_ordinal():
                    logger.info(f"Early stopping à l'époque {epoch+1} "
                               f"(patience: {early_stopping_patience})")
                break
            
            # Synchroniser tous les processus
            xm.rendezvous("epoch_complete")
        
        # Sauvegarder le modèle final
        self.save_checkpoint(epoch, False, checkpoint_dir)
        
        if xm.is_master_ordinal():
            logger.info(f"Entraînement TPU terminé. Meilleure précision: {self.best_val_acc:.2f}% "
                       f"à l'époque {best_epoch+1}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }

def _mp_fn(index, args):
    """
    Fonction multiprocessing pour TPU
    """
    # Configuration des arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--vision-backbone", type=str, default="resnet50")
    parser.add_argument("--text-model", type=str, default="microsoft/DialoGPT-medium")
    parser.add_argument("--vision-dim", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--text-length", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    
    # Parse args depuis la chaîne
    import sys
    sys.argv = args.split()
    parsed_args = parser.parse_args()
    
    # Créer le data module
    data_module = create_data_module(
        jsonl_file=parsed_args.jsonl_file,
        root_dir=parsed_args.root_dir,
        batch_size=parsed_args.batch_size,
        num_workers=4,  # Réduit pour TPU
        image_size=parsed_args.image_size,
        text_model_name=parsed_args.text_model,
        max_text_length=parsed_args.text_length
    )
    
    # Créer le modèle
    model = create_multimodal_model(
        num_classes=parsed_args.num_classes,
        vision_backbone=parsed_args.vision_backbone,
        text_model=parsed_args.text_model,
        vision_dim=parsed_args.vision_dim,
        text_dim=parsed_args.text_dim,
        fusion_dim=parsed_args.fusion_dim
    )
    
    # Obtenir les poids de classes
    class_weights = data_module.get_class_weights()
    
    # Créer le trainer TPU
    trainer = TPUTrainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        num_classes=parsed_args.num_classes,
        learning_rate=parsed_args.learning_rate,
        weight_decay=parsed_args.weight_decay,
        gradient_clip_val=parsed_args.gradient_clip,
        class_weights=class_weights,
        use_mixed_precision=parsed_args.mixed_precision
    )
    
    # Entraîner
    trainer.train(
        num_epochs=parsed_args.epochs,
        checkpoint_dir=parsed_args.checkpoint_dir,
        save_every=parsed_args.save_every,
        early_stopping_patience=parsed_args.early_stopping
    )

def main():
    parser = argparse.ArgumentParser(description="Entraînement TPU pour modèle multimodal")
    
    # Données
    parser.add_argument("--jsonl-file", type=str, required=True,
                       help="Fichier multimodal_dataset.jsonl")
    parser.add_argument("--root-dir", type=str, required=True,
                       help="Répertoire racine des images")
    parser.add_argument("--num-classes", type=int, required=True,
                       help="Nombre de classes")
    
    # Modèle
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
    
    # Options TPU
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
    parser.add_argument("--save-every", type=int, default=5,
                       help="Sauvegarder tous les N époques")
    parser.add_argument("--early-stopping", type=int, default=10,
                       help="Patience pour early stopping")
    
    # TPU
    parser.add_argument("--tpu-cores", type=int, default=8,
                       help="Nombre de cœurs TPU")
    
    args = parser.parse_args()
    
    # Vérifier que torch_xla est disponible
    try:
        import torch_xla
        logger.info(f"torch_xla version: {torch_xla.__version__}")
    except ImportError:
        logger.error("torch_xla non disponible. Installez-le avec: pip install torch_xla")
        return 1
    
    # Convertir les arguments en chaîne pour le multiprocessing
    args_str = " ".join([f"--{k} {v}" for k, v in vars(args).items() if v is not None])
    
    # Lancer l'entraînement sur TPU
    logger.info(f"Lancement de l'entraînement TPU avec {args.tpu_cores} cœurs")
    xmp.spawn(_mp_fn, args=(args_str,), nprocs=args.tpu_cores)
    
    logger.info("Entraînement TPU terminé!")

if __name__ == "__main__":
    import time
    main()








