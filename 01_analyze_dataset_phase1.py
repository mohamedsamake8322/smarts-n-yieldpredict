"""
PHASE 1: ANALYSE DATASET
Analyse complète du dataset_final:
- Nombre de classes et images
- Distribution par classe
- Qualité des images
- Détection des problèmes
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict
import pandas as pd

DATASET_PATH = Path(r"C:\smarts-n-yieldpredict.git\dataset_final")
OUTPUT_DIR = Path(r"C:\smarts-n-yieldpredict.git\phase1_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Extensions d'images valides
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

class DatasetAnalyzer:
    def __init__(self, dataset_path, fast=False):
        self.dataset_path = dataset_path
        self.fast = fast
        self.stats = {
            'total_classes': 0,
            'total_images': 0,
            'classes_breakdown': {},
            'corrupted_images': [],
            'small_images': [],  # < 100x100
            'large_images': [],   # > 2000x2000
            'image_formats': defaultdict(int),
            'image_resolutions': defaultdict(int),
        }
    
    def analyze(self):
        """Analyse complète du dataset

        If self.fast is True, do NOT open image files (much faster).
        """
        print("🔍 Analyse du dataset en cours...")
        print(f"📁 Chemin: {self.dataset_path}")
        print(f"📁 Existe: {self.dataset_path.exists()}\n")

        class_count = 0
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_count += 1
            class_name = class_dir.name
            print(f"  [{class_count}] Analyse {class_name}...", end='\r')

            if self.fast:
                # Fast path: only count files and infer formats from suffix
                files = [f for f in class_dir.rglob('*') if f.suffix.lower() in VALID_EXTENSIONS]
                images = {
                    'count': len(files),
                    'valid': len(files),
                    'corrupted': [],
                    'formats': defaultdict(int),
                    'resolutions': []
                }
                for f in files:
                    ext = f.suffix.lower().lstrip('.')
                    self.stats['image_formats'][ext] += 1
                # update total images
                if images['count'] > 0:
                    self.stats['classes_breakdown'][class_name] = images
                    self.stats['total_images'] += images['count']
                continue

            images = self._analyze_class(class_dir)

            if images['count'] > 0:
                self.stats['classes_breakdown'][class_name] = images
                self.stats['total_images'] += images['count']

        print()  # Nouvelle ligne après le \r
        self.stats['total_classes'] = len(self.stats['classes_breakdown'])
    
    def _analyze_class(self, class_dir):
        """Analyse une classe spécifique"""
        result = {
            'count': 0,
            'valid': 0,
            'corrupted': [],
            'formats': defaultdict(int),
            'resolutions': []
        }
        
        for file_path in class_dir.rglob('*'):
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            
            result['count'] += 1
            
            try:
                img = Image.open(file_path)
                width, height = img.size
                
                result['valid'] += 1
                result['formats'][img.format] += 1
                result['resolutions'].append((width, height))
                
                self.stats['image_resolutions'][f"{width}x{height}"] += 1
                self.stats['image_formats'][img.format] += 1
                
                # Détection des images trop petites ou trop grandes
                if width < 100 or height < 100:
                    self.stats['small_images'].append({
                        'class': class_dir.name,
                        'file': str(file_path),
                        'size': (width, height)
                    })
                elif width > 2000 or height > 2000:
                    self.stats['large_images'].append({
                        'class': class_dir.name,
                        'file': str(file_path),
                        'size': (width, height)
                    })
            
            except Exception as e:
                result['corrupted'].append({
                    'file': file_path.name,
                    'error': str(e)
                })
                self.stats['corrupted_images'].append({
                    'class': class_dir.name,
                    'file': str(file_path),
                    'error': str(e)
                })
        
        return result
    
    def print_report(self):
        """Affiche un rapport détaillé"""
        print("\n" + "="*80)
        print("📊 RAPPORT D'ANALYSE DU DATASET")
        print("="*80)
        
        print(f"\n✅ STATISTIQUES GLOBALES:")
        print(f"   • Nombre de classes: {self.stats['total_classes']}")
        print(f"   • Nombre total d'images: {self.stats['total_images']}")
        print(f"   • Images corrompues: {len(self.stats['corrupted_images'])}")
        
        # Distribution par classe
        print(f"\n📈 DISTRIBUTION PAR CLASSE:")
        breakdown = sorted(
            self.stats['classes_breakdown'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for class_name, data in breakdown[:20]:  # Top 20
            corrupted = len(data['corrupted'])
            status = f" ⚠️  ({corrupted} corrompues)" if corrupted > 0 else ""
            print(f"   {class_name:<40} {data['count']:>4} images{status}")
        
        if len(breakdown) > 20:
            print(f"   ... et {len(breakdown) - 20} autres classes")
        
        # Formats d'images
        print(f"\n🖼️  FORMATS D'IMAGES:")
        for fmt, count in sorted(self.stats['image_formats'].items(), 
                                 key=lambda x: x[1], reverse=True):
            pct = 100 * count / self.stats['total_images']
            print(f"   {fmt:<10} {count:>6} images ({pct:>5.1f}%)")
        
        # Résolutions
        print(f"\n📐 RÉSOLUTIONS (TOP 10):")
        sorted_res = sorted(self.stats['image_resolutions'].items(),
                           key=lambda x: x[1], reverse=True)
        for res, count in sorted_res[:10]:
            pct = 100 * count / self.stats['total_images']
            print(f"   {res:<20} {count:>6} images ({pct:>5.1f}%)")
        
        # Problèmes détectés
        if self.stats['corrupted_images']:
            print(f"\n⚠️  IMAGES CORROMPUES ({len(self.stats['corrupted_images'])}):")
            for item in self.stats['corrupted_images'][:10]:
                print(f"   {item['class']}/{item['file']}")
                print(f"      → {item['error']}")
            if len(self.stats['corrupted_images']) > 10:
                print(f"   ... et {len(self.stats['corrupted_images']) - 10} autres")
        
        if self.stats['small_images']:
            print(f"\n⚠️  IMAGES TROP PETITES: {len(self.stats['small_images'])}")
        
        if self.stats['large_images']:
            print(f"\n⚠️  IMAGES TROP GRANDES: {len(self.stats['large_images'])}")
        
        print("\n" + "="*80)
    
    def save_report(self):
        """Sauvegarde le rapport en JSON"""
        # Convertir les defaultdict
        stats = {
            'total_classes': self.stats['total_classes'],
            'total_images': self.stats['total_images'],
            'image_formats': dict(self.stats['image_formats']),
            'image_resolutions': dict(self.stats['image_resolutions']),
            'corrupted_count': len(self.stats['corrupted_images']),
            'small_images_count': len(self.stats['small_images']),
            'large_images_count': len(self.stats['large_images']),
            'corrupted_images': self.stats['corrupted_images'],
        }
        
        # Classes breakdown
        classes_data = []
        for class_name, data in sorted(
            self.stats['classes_breakdown'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ):
            classes_data.append({
                'class_name': class_name,
                'image_count': data['count'],
                'valid_count': data['valid'],
                'corrupted_count': len(data['corrupted']),
            })
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_stats': stats,
            'classes': classes_data
        }
        
        report_path = OUTPUT_DIR / "dataset_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Rapport sauvegardé: {report_path}")
        
        # Aussi en CSV pour Excel
        df = pd.DataFrame(classes_data)
        csv_path = OUTPUT_DIR / "dataset_classes.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV sauvegardé: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse dataset (phase 1)')
    parser.add_argument('--fast', action='store_true', help='Fast scan: do not open images')
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(DATASET_PATH, fast=args.fast)
    analyzer.analyze()
    analyzer.print_report()
    analyzer.save_report()

    print("\n✨ Analyse terminée!")
