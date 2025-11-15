#!/usr/bin/env python3
"""
data_cleaner.py - Nettoyage et préparation du dataset multimodal
Vérifie et nettoie le dataset (images corrompues, tailles, formats)
Normalise les noms de classes/dossiers et génère un JSONL propre
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageOps
import hashlib
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCleaner:
    """Classe pour nettoyer et préparer le dataset multimodal"""
    
    def __init__(self, 
                 data_dir: str,
                 min_size: int = 256,
                 target_size: Optional[int] = None,
                 supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                 quality_threshold: float = 0.8):
        """
        Args:
            data_dir: Répertoire racine du dataset
            min_size: Taille minimale des images (côté le plus petit)
            target_size: Taille cible pour redimensionnement (None = pas de resize)
            supported_formats: Formats d'image supportés
            quality_threshold: Seuil de qualité pour détecter images corrompues
        """
        self.data_dir = Path(data_dir)
        self.min_size = min_size
        self.target_size = target_size
        self.supported_formats = supported_formats
        self.quality_threshold = quality_threshold
        
        # Statistiques
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': 0,
            'too_small_images': 0,
            'unsupported_format': 0,
            'classes_found': set()
        }
    
    def is_image_valid(self, image_path: Path) -> Tuple[bool, str]:
        """
        Vérifie si une image est valide (non corrompue, bonne taille, format supporté)
        Returns: (is_valid, reason)
        """
        try:
            # Vérifier l'extension
            if image_path.suffix.lower() not in self.supported_formats:
                return False, f"Format non supporté: {image_path.suffix}"
            
            # Ouvrir l'image
            with Image.open(image_path) as img:
                # Vérifier la taille
                width, height = img.size
                min_dimension = min(width, height)
                
                if min_dimension < self.min_size:
                    return False, f"Trop petite: {width}x{height} (min: {self.min_size})"
                
                # Vérifier la qualité (tentative de redimensionnement)
                try:
                    img.verify()
                except Exception:
                    return False, "Image corrompue (verify failed)"
                
                # Test de chargement complet
                img.load()
                
                return True, "OK"
                
        except Exception as e:
            return False, f"Erreur: {str(e)}"
    
    def resize_image_if_needed(self, image_path: Path, output_path: Path) -> bool:
        """
        Redimensionne l'image si nécessaire
        Returns: True si redimensionné, False sinon
        """
        if self.target_size is None:
            return False
            
        try:
            with Image.open(image_path) as img:
                # Convertir en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionner en gardant l'aspect ratio
                img.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)
                
                # Sauvegarder
                img.save(output_path, 'JPEG', quality=95, optimize=True)
                return True
                
        except Exception as e:
            logger.error(f"Erreur redimensionnement {image_path}: {e}")
            return False
    
    def clean_dataset(self, 
                     train_dir: str = "train", 
                     val_dir: str = "val",
                     output_file: str = "dataset_clean.jsonl") -> Dict:
        """
        Nettoie le dataset complet
        """
        train_path = self.data_dir / train_dir
        val_path = self.data_dir / val_dir
        
        if not train_path.exists():
            raise FileNotFoundError(f"Répertoire train introuvable: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Répertoire val introuvable: {val_path}")
        
        output_path = self.data_dir / output_file
        cleaned_data = []
        
        # Traiter train et val
        for split in ['train', 'val']:
            split_path = self.data_dir / split
            logger.info(f"Traitement du split: {split}")
            
            # Parcourir les classes
            for class_dir in tqdm(sorted(split_path.iterdir()), desc=f"Classes {split}"):
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                self.stats['classes_found'].add(class_name)
                
                # Parcourir les images de cette classe
                for image_path in class_dir.iterdir():
                    if not image_path.is_file():
                        continue
                    
                    self.stats['total_images'] += 1
                    
                    # Vérifier la validité
                    is_valid, reason = self.is_image_valid(image_path)
                    
                    if not is_valid:
                        logger.debug(f"Image invalide {image_path}: {reason}")
                        if "corrompue" in reason:
                            self.stats['corrupted_images'] += 1
                        elif "Trop petite" in reason:
                            self.stats['too_small_images'] += 1
                        elif "Format non supporté" in reason:
                            self.stats['unsupported_format'] += 1
                        continue
                    
                    # Redimensionner si nécessaire
                    final_image_path = image_path
                    if self.target_size is not None:
                        resized_path = image_path.parent / f"resized_{image_path.name}"
                        if self.resize_image_if_needed(image_path, resized_path):
                            final_image_path = resized_path
                    
                    # Ajouter à la liste nettoyée
                    relative_path = final_image_path.relative_to(self.data_dir)
                    cleaned_data.append({
                        "image_path": str(relative_path).replace("\\", "/"),
                        "label": class_name,
                        "split": split,
                        "original_size": Image.open(final_image_path).size
                    })
                    
                    self.stats['valid_images'] += 1
        
        # Sauvegarder le dataset nettoyé
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset nettoyé sauvegardé: {output_path}")
        logger.info(f"Total: {len(cleaned_data)} images valides")
        
        return self.get_stats()
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques de nettoyage"""
        self.stats['classes_found'] = len(self.stats['classes_found'])
        return self.stats.copy()
    
    def print_stats(self):
        """Affiche les statistiques de nettoyage"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("STATISTIQUES DE NETTOYAGE")
        print("="*50)
        print(f"Images totales: {stats['total_images']}")
        print(f"Images valides: {stats['valid_images']}")
        print(f"Images corrompues: {stats['corrupted_images']}")
        print(f"Images trop petites: {stats['too_small_images']}")
        print(f"Formats non supportés: {stats['unsupported_format']}")
        print(f"Classes trouvées: {stats['classes_found']}")
        print(f"Taux de réussite: {stats['valid_images']/max(stats['total_images'], 1)*100:.1f}%")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Nettoyage du dataset multimodal")
    parser.add_argument("--data-dir", type=str, required=True, 
                       help="Répertoire racine du dataset")
    parser.add_argument("--min-size", type=int, default=256,
                       help="Taille minimale des images (défaut: 256)")
    parser.add_argument("--target-size", type=int, default=None,
                       help="Taille cible pour redimensionnement (défaut: pas de resize)")
    parser.add_argument("--output-file", type=str, default="dataset_clean.jsonl",
                       help="Nom du fichier de sortie (défaut: dataset_clean.jsonl)")
    parser.add_argument("--verbose", action="store_true",
                       help="Mode verbeux")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Créer le nettoyeur
    cleaner = DatasetCleaner(
        data_dir=args.data_dir,
        min_size=args.min_size,
        target_size=args.target_size
    )
    
    # Nettoyer le dataset
    try:
        stats = cleaner.clean_dataset(output_file=args.output_file)
        cleaner.print_stats()
        
        # Sauvegarder les stats
        stats_file = Path(args.data_dir) / "cleaning_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistiques sauvegardées: {stats_file}")
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
