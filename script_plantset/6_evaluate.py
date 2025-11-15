#!/usr/bin/env python3
"""
evaluate.py - Évaluation du modèle multimodal
Évalue le modèle (top-1 accuracy, F1, BLEU/ROUGE pour descriptions)
Calcule métriques complètes et génère rapports détaillés
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Imports locaux
from dataset_loader import create_data_module
from model_builder import create_multimodal_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalEvaluator:
    """
    Évaluateur pour modèle multimodal vision-langage
    """
    
    def __init__(self,
                 model: nn.Module,
                 data_loader: DataLoader,
                 device: str = "auto",
                 class_names: Optional[List[str]] = None):
        """
        Args:
            model: Modèle multimodal à évaluer
            data_loader: DataLoader pour l'évaluation
            device: Device (auto, cuda, cpu, tpu)
            class_names: Noms des classes pour les rapports
        """
        self.model = model
        self.data_loader = data_loader
        self.class_names = class_names or [f"Class_{i}" for i in range(model.num_classes)]
        self.device = self._setup_device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialiser les métriques de texte
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_smoother = SmoothingFunction().method4
        
        logger.info(f"Évaluateur initialisé sur {self.device}")
    
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
    
    def evaluate_classification(self) -> Dict[str, Any]:
        """
        Évalue les performances de classification
        """
        logger.info("Évaluation de la classification...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_losses = []
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Évaluation"):
                # Déplacer vers device
                images = batch['images'].to(self.device, non_blocking=True)
                texts = batch['texts'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images, texts)
                logits = outputs['logits']
                
                # Prédictions
                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)
                
                # Loss
                losses = criterion(logits, labels)
                
                # Collecter les résultats
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_losses.extend(losses.cpu().numpy())
        
        # Calculer les métriques
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        # AUC (si binaire ou multiclasse)
        try:
            if len(np.unique(all_labels)) == 2:
                auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        except:
            auc = None
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Rapport de classification
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Métriques par classe
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                per_class_metrics[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
        
        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'auc': float(auc) if auc is not None else None,
            'mean_loss': float(np.mean(all_losses)),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def evaluate_text_generation(self, reference_texts: List[str]) -> Dict[str, Any]:
        """
        Évalue la génération de texte (si applicable)
        Args:
            reference_texts: Textes de référence pour comparaison
        """
        logger.info("Évaluation de la génération de texte...")
        
        generated_texts = []
        all_text_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Génération texte"):
                # Déplacer vers device
                images = batch['images'].to(self.device, non_blocking=True)
                texts = batch['texts'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images, texts)
                text_embeddings = outputs.get('text_embedding', outputs['text_features'])
                
                # Pour l'instant, on utilise les embeddings comme "génération"
                # Dans un vrai système, on aurait un générateur de texte
                all_text_embeddings.extend(text_embeddings.cpu().numpy())
                
                # Simuler des textes générés (à remplacer par un vrai générateur)
                generated_texts.extend([f"Generated text for class {i}" for i in range(len(batch['images']))])
        
        # Calculer BLEU et ROUGE si on a des références
        if reference_texts and len(reference_texts) == len(generated_texts):
            bleu_scores = []
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for gen_text, ref_text in zip(generated_texts, reference_texts):
                # BLEU
                ref_tokens = ref_text.split()
                gen_tokens = gen_text.split()
                bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.bleu_smoother)
                bleu_scores.append(bleu)
                
                # ROUGE
                rouge_scores_dict = self.rouge_scorer.score(ref_text, gen_text)
                for metric in rouge_scores:
                    rouge_scores[metric].append(rouge_scores_dict[metric].fmeasure)
            
            return {
                'bleu_mean': float(np.mean(bleu_scores)),
                'bleu_std': float(np.std(bleu_scores)),
                'rouge1_mean': float(np.mean(rouge_scores['rouge1'])),
                'rouge1_std': float(np.std(rouge_scores['rouge1'])),
                'rouge2_mean': float(np.mean(rouge_scores['rouge2'])),
                'rouge2_std': float(np.std(rouge_scores['rouge2'])),
                'rougeL_mean': float(np.mean(rouge_scores['rougeL'])),
                'rougeL_std': float(np.std(rouge_scores['rougeL'])),
                'text_embeddings': all_text_embeddings
            }
        else:
            return {
                'text_embeddings': all_text_embeddings,
                'note': 'Pas de textes de référence pour BLEU/ROUGE'
            }
    
    def plot_confusion_matrix(self, cm: np.ndarray, output_path: str = "confusion_matrix.png"):
        """Trace la matrice de confusion"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies Classes')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Matrice de confusion sauvegardée: {output_path}")
    
    def plot_metrics_by_class(self, per_class_metrics: Dict[str, Any], 
                             output_path: str = "metrics_by_class.png"):
        """Trace les métriques par classe"""
        classes = list(per_class_metrics.keys())
        precision = [per_class_metrics[cls]['precision'] for cls in classes]
        recall = [per_class_metrics[cls]['recall'] for cls in classes]
        f1 = [per_class_metrics[cls]['f1'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(15, 8))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Métriques par Classe')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Métriques par classe sauvegardées: {output_path}")
    
    def generate_report(self, results: Dict[str, Any], output_path: str = "evaluation_report.json"):
        """Génère un rapport d'évaluation complet"""
        report = {
            'evaluation_timestamp': str(torch.datetime.now()),
            'model_info': {
                'num_classes': len(self.class_names),
                'device': str(self.device)
            },
            'classification_metrics': {
                'accuracy': results['accuracy'],
                'f1_macro': results['f1_macro'],
                'f1_weighted': results['f1_weighted'],
                'auc': results.get('auc'),
                'mean_loss': results['mean_loss']
            },
            'per_class_metrics': results['per_class_metrics'],
            'confusion_matrix': results['confusion_matrix']
        }
        
        if 'text_generation' in results:
            report['text_generation_metrics'] = results['text_generation']
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rapport d'évaluation sauvegardé: {output_path}")
        return report
    
    def evaluate(self, reference_texts: Optional[List[str]] = None,
                output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Évaluation complète du modèle
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Début de l'évaluation complète...")
        
        # Évaluation de classification
        classification_results = self.evaluate_classification()
        
        # Évaluation de génération de texte
        text_results = self.evaluate_text_generation(reference_texts)
        
        # Combiner les résultats
        all_results = {
            **classification_results,
            'text_generation': text_results
        }
        
        # Générer les visualisations
        cm = np.array(classification_results['confusion_matrix'])
        self.plot_confusion_matrix(cm, str(output_dir / "confusion_matrix.png"))
        self.plot_metrics_by_class(classification_results['per_class_metrics'],
                                 str(output_dir / "metrics_by_class.png"))
        
        # Générer le rapport
        report = self.generate_report(all_results, str(output_dir / "evaluation_report.json"))
        
        # Afficher un résumé
        logger.info("\n" + "="*50)
        logger.info("RÉSULTATS D'ÉVALUATION")
        logger.info("="*50)
        logger.info(f"Accuracy: {classification_results['accuracy']:.4f}")
        logger.info(f"F1 Macro: {classification_results['f1_macro']:.4f}")
        logger.info(f"F1 Weighted: {classification_results['f1_weighted']:.4f}")
        if classification_results.get('auc'):
            logger.info(f"AUC: {classification_results['auc']:.4f}")
        logger.info(f"Mean Loss: {classification_results['mean_loss']:.4f}")
        logger.info("="*50)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle multimodal")
    
    # Modèle
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Chemin vers le checkpoint du modèle")
    parser.add_argument("--jsonl-file", type=str, required=True,
                       help="Fichier multimodal_dataset.jsonl")
    parser.add_argument("--root-dir", type=str, required=True,
                       help="Répertoire racine des images")
    parser.add_argument("--num-classes", type=int, required=True,
                       help="Nombre de classes")
    
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
    
    # Évaluation
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Taille des batches")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Taille des images")
    parser.add_argument("--text-length", type=int, default=512,
                       help="Longueur du texte")
    parser.add_argument("--split", type=str, default="val",
                       help="Split à évaluer (train/val/test)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu, tpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Nombre de workers DataLoader")
    
    # Sortie
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Répertoire de sortie")
    parser.add_argument("--reference-texts", type=str, default=None,
                       help="Fichier JSON avec textes de référence pour BLEU/ROUGE")
    
    args = parser.parse_args()
    
    # Charger les textes de référence si fournis
    reference_texts = None
    if args.reference_texts and Path(args.reference_texts).exists():
        with open(args.reference_texts, 'r', encoding='utf-8') as f:
            reference_texts = json.load(f)
    
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
    
    # Sélectionner le bon split
    if args.split == "train":
        data_loader = data_module.train_loader
    elif args.split == "val":
        data_loader = data_module.val_loader
    else:
        raise ValueError(f"Split non supporté: {args.split}")
    
    # Créer le modèle
    logger.info("Création du modèle...")
    model = create_multimodal_model(
        num_classes=args.num_classes,
        vision_backbone=args.vision_backbone,
        text_model=args.text_model,
        vision_dim=args.vision_dim,
        text_dim=args.text_dim,
        fusion_dim=args.fusion_dim
    )
    
    # Charger le checkpoint
    logger.info(f"Chargement du checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Créer l'évaluateur
    evaluator = MultimodalEvaluator(
        model=model,
        data_loader=data_loader,
        device=args.device,
        class_names=list(data_module.idx_to_label.values())
    )
    
    # Évaluer
    results = evaluator.evaluate(
        reference_texts=reference_texts,
        output_dir=args.output_dir
    )
    
    logger.info("Évaluation terminée!")

if __name__ == "__main__":
    main()
