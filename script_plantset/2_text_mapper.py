#!/usr/bin/env python3
"""
text_mapper.py - Association d'images avec descriptions textuelles
Associe chaque image à son texte descriptif depuis maladies_enrichies.json
Crée un dataset multimodal : image_path + label + texte
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextMapper:
    """Classe pour mapper les images avec leurs descriptions textuelles"""
    
    def __init__(self, 
                 diseases_json_path: str,
                 dataset_jsonl_path: str,
                 language: str = "fr"):
        """
        Args:
            diseases_json_path: Chemin vers maladies_enrichies.json
            dataset_jsonl_path: Chemin vers dataset_clean.jsonl
            language: Langue des descriptions (fr, en, etc.)
        """
        self.diseases_json_path = Path(diseases_json_path)
        self.dataset_jsonl_path = Path(dataset_jsonl_path)
        self.language = language
        self.diseases_data = {}
        self.class_mapping = {}
        
    def load_diseases_data(self) -> Dict[str, Any]:
        """Charge les données des maladies depuis le JSON"""
        if not self.diseases_json_path.exists():
            logger.warning(f"Fichier maladies introuvable: {self.diseases_json_path}")
            return {}
        
        try:
            with open(self.diseases_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Chargé {len(data)} entrées de maladies")
            return data
        except Exception as e:
            logger.error(f"Erreur chargement maladies: {e}")
            return {}
    
    def normalize_class_name(self, class_name: str) -> str:
        """
        Normalise le nom de classe pour correspondre aux clés du JSON
        """
        # Nettoyer le nom de classe
        normalized = class_name.strip()
        
        # Remplacer les underscores par des espaces
        normalized = normalized.replace('_', ' ')
        
        # Supprimer les préfixes communs
        prefixes_to_remove = ['Apple___', 'Tomato___', 'Potato___', 'Corn___', 'Grape___']
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Nettoyer les espaces multiples
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def find_disease_info(self, class_name: str) -> Dict[str, Any]:
        """
        Trouve les informations de maladie pour une classe donnée
        """
        normalized_name = self.normalize_class_name(class_name)
        
        # Recherche exacte d'abord
        if normalized_name in self.diseases_data:
            return self.diseases_data[normalized_name]
        
        # Recherche partielle
        for disease_name, disease_info in self.diseases_data.items():
            if normalized_name.lower() in disease_name.lower() or disease_name.lower() in normalized_name.lower():
                return disease_info
        
        # Recherche par mots-clés
        keywords = normalized_name.lower().split()
        best_match = None
        best_score = 0
        
        for disease_name, disease_info in self.diseases_data.items():
            disease_keywords = disease_name.lower().split()
            score = len(set(keywords) & set(disease_keywords))
            if score > best_score:
                best_score = score
                best_match = disease_info
        
        return best_match or {}
    
    def extract_text_description(self, disease_info: Dict[str, Any]) -> str:
        """
        Extrait et formate la description textuelle d'une maladie
        """
        if not disease_info:
            return "Description non disponible"
        
        parts = []
        
        # Description principale
        description = disease_info.get("description", {})
        if isinstance(description, dict):
            desc_text = description.get(self.language, 
                                     description.get("en", 
                                     next(iter(description.values()), "")))
        else:
            desc_text = str(description)
        
        if desc_text:
            parts.append(f"Description: {desc_text}")
        
        # Symptômes
        symptoms = disease_info.get("symptômes", disease_info.get("symptoms", {}))
        if isinstance(symptoms, dict):
            sympt_text = symptoms.get(self.language,
                                    symptoms.get("en",
                                    next(iter(symptoms.values()), "")))
        else:
            sympt_text = str(symptoms)
        
        if sympt_text:
            parts.append(f"Symptômes: {sympt_text}")
        
        # Traitement
        treatment = disease_info.get("traitement", disease_info.get("treatment", {}))
        if isinstance(treatment, dict):
            treat_text = treatment.get(self.language,
                                     treatment.get("en",
                                     next(iter(treatment.values()), "")))
        else:
            treat_text = str(treatment)
        
        if treat_text:
            parts.append(f"Traitement: {treat_text}")
        
        # Prévention
        prevention = disease_info.get("prévention", disease_info.get("prevention", {}))
        if isinstance(prevention, dict):
            prev_text = prevention.get(self.language,
                                     prevention.get("en",
                                     next(iter(prevention.values()), "")))
        else:
            prev_text = str(prevention)
        
        if prev_text:
            parts.append(f"Prévention: {prev_text}")
        
        # Causes
        causes = disease_info.get("causes", disease_info.get("cause", {}))
        if isinstance(causes, dict):
            cause_text = causes.get(self.language,
                                  causes.get("en",
                                  next(iter(causes.values()), "")))
        else:
            cause_text = str(causes)
        
        if cause_text:
            parts.append(f"Causes: {cause_text}")
        
        return " | ".join(parts) if parts else "Description non disponible"
    
    def create_multimodal_dataset(self, output_file: str = "multimodal_dataset.jsonl") -> Dict[str, int]:
        """
        Crée le dataset multimodal avec images + textes
        """
        # Charger les données de maladies
        self.diseases_data = self.load_diseases_data()
        
        if not self.dataset_jsonl_path.exists():
            raise FileNotFoundError(f"Dataset JSONL introuvable: {self.dataset_jsonl_path}")
        
        # Lire le dataset nettoyé
        dataset_items = []
        with open(self.dataset_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset_items.append(json.loads(line.strip()))
        
        logger.info(f"Chargé {len(dataset_items)} images du dataset nettoyé")
        
        # Créer le mapping multimodal
        multimodal_data = []
        stats = {
            'total_images': len(dataset_items),
            'mapped_images': 0,
            'unmapped_images': 0,
            'classes_mapped': set(),
            'classes_unmapped': set()
        }
        
        for item in tqdm(dataset_items, desc="Mapping textuel"):
            class_name = item['label']
            disease_info = self.find_disease_info(class_name)
            
            if disease_info:
                text_description = self.extract_text_description(disease_info)
                stats['mapped_images'] += 1
                stats['classes_mapped'].add(class_name)
            else:
                text_description = f"Maladie: {class_name}. Description non disponible dans la base de données."
                stats['unmapped_images'] += 1
                stats['classes_unmapped'].add(class_name)
            
            # Créer l'entrée multimodale
            multimodal_item = {
                "image_path": item['image_path'],
                "label": class_name,
                "split": item['split'],
                "text": text_description,
                "original_size": item.get('original_size', None)
            }
            
            multimodal_data.append(multimodal_item)
        
        # Sauvegarder le dataset multimodal
        output_path = self.dataset_jsonl_path.parent / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in multimodal_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset multimodal sauvegardé: {output_path}")
        
        # Finaliser les stats
        stats['classes_mapped'] = len(stats['classes_mapped'])
        stats['classes_unmapped'] = len(stats['classes_unmapped'])
        
        return stats
    
    def print_stats(self, stats: Dict[str, int]):
        """Affiche les statistiques de mapping"""
        print("\n" + "="*50)
        print("STATISTIQUES DE MAPPING TEXTUEL")
        print("="*50)
        print(f"Images totales: {stats['total_images']}")
        print(f"Images mappées: {stats['mapped_images']}")
        print(f"Images non mappées: {stats['unmapped_images']}")
        print(f"Classes mappées: {stats['classes_mapped']}")
        print(f"Classes non mappées: {stats['classes_unmapped']}")
        print(f"Taux de mapping: {stats['mapped_images']/max(stats['total_images'], 1)*100:.1f}%")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Mapping textuel du dataset multimodal")
    parser.add_argument("--diseases-json", type=str, required=True,
                       help="Chemin vers maladies_enrichies.json")
    parser.add_argument("--dataset-jsonl", type=str, required=True,
                       help="Chemin vers dataset_clean.jsonl")
    parser.add_argument("--output-file", type=str, default="multimodal_dataset.jsonl",
                       help="Nom du fichier de sortie (défaut: multimodal_dataset.jsonl)")
    parser.add_argument("--language", type=str, default="fr",
                       help="Langue des descriptions (défaut: fr)")
    parser.add_argument("--verbose", action="store_true",
                       help="Mode verbeux")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Créer le mapper
    mapper = TextMapper(
        diseases_json_path=args.diseases_json,
        dataset_jsonl_path=args.dataset_jsonl,
        language=args.language
    )
    
    # Créer le dataset multimodal
    try:
        stats = mapper.create_multimodal_dataset(output_file=args.output_file)
        mapper.print_stats(stats)
        
        # Sauvegarder les stats
        stats_file = Path(args.dataset_jsonl).parent / "mapping_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistiques sauvegardées: {stats_file}")
        
    except Exception as e:
        logger.error(f"Erreur lors du mapping: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
