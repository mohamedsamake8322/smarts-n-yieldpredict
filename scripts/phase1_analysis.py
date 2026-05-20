"""
COMPREHENSIVE DATASET ANALYSIS TOOL
Analyse complète selon framework d'excellence
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

class DatasetAnalyzer:
    def __init__(self, root_path):
        self.root = Path(root_path)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.classes = defaultdict(int)
        self.class_details = defaultdict(dict)
        self.image_hashes = {}  # pour détection doublons
        self.corrupted = []
        self.stats = {
            'total_images': 0,
            'total_classes': 0,
            'min_class': float('inf'),
            'max_class': 0,
            'avg_class': 0,
            'std_class': 0,
            'imbalance_ratio': 0,
            'problematic_classes': [],
            'low_data_classes': [],  # < 300
            'weak_classes': [],  # 300-500
            'balanced_classes': [],  # 500-3000
            'oversized_classes': [],  # > 3000
        }
    
    def scan_dataset(self, progress_interval=10):
        """Scan récursif du dataset"""
        print("🔍 PHASE 1: Scan global du dataset")
        print("=" * 80)
        
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_path = Path(dirpath).relative_to(self.root)
            image_files = [f for f in filenames if Path(f).suffix.lower() in self.image_extensions]
            
            if image_files:
                class_name = str(rel_path)
                self.classes[class_name] = len(image_files)
                self.class_details[class_name] = {
                    'path': dirpath,
                    'count': len(image_files),
                    'images': image_files,
                    'corrupted_count': 0,
                    'min_res': None,
                    'max_res': None,
                    'avg_res': None,
                    'file_size_mb': 0
                }
                
                count += 1
                if count % progress_interval == 0:
                    print(f"  Scanned: {count} classes...")
        
        print(f"✓ Total: {count} classes trouvées\n")
        return count > 0
    
    def check_image_quality(self):
        """Analyser résolution et fichiers corrompus"""
        print("🖼️  PHASE 2: Vérification qualité images")
        print("=" * 80)
        
        total_checked = 0
        for class_name, details in self.class_details.items():
            resolutions = []
            file_sizes = []
            corrupted = 0
            
            for img_file in details['images'][:100]:  # Vérifier max 100 par classe
                img_path = Path(details['path']) / img_file
                try:
                    img = Image.open(img_path)
                    resolutions.append(img.size)
                    file_sizes.append(img_path.stat().st_size / (1024*1024))  # MB
                    total_checked += 1
                except Exception as e:
                    corrupted += 1
                    self.corrupted.append(str(img_path))
            
            if resolutions:
                widths, heights = zip(*resolutions)
                details['min_res'] = (min(widths), min(heights))
                details['max_res'] = (max(widths), max(heights))
                details['avg_res'] = (sum(widths)//len(widths), sum(heights)//len(heights))
                details['file_size_mb'] = sum(file_sizes) / len(file_sizes) if file_sizes else 0
            
            details['corrupted_count'] = corrupted
            if corrupted > 0:
                print(f"  ⚠️  {class_name}: {corrupted} images corrompues")
        
        print(f"✓ Total images vérifiées: {total_checked}\n")
    
    def compute_statistics(self):
        """Calculer stats et identifier problèmes"""
        print("📊 PHASE 3: Statistiques & Analyse")
        print("=" * 80)
        
        if not self.classes:
            return False
        
        counts = list(self.classes.values())
        self.stats['total_images'] = sum(counts)
        self.stats['total_classes'] = len(counts)
        self.stats['min_class'] = min(counts)
        self.stats['max_class'] = max(counts)
        self.stats['avg_class'] = self.stats['total_images'] / self.stats['total_classes']
        self.stats['imbalance_ratio'] = self.stats['max_class'] / self.stats['min_class']
        
        # Classify classes
        for class_name, count in self.classes.items():
            if count < 100:
                self.stats['problematic_classes'].append((class_name, count))
            if count < 300:
                self.stats['low_data_classes'].append((class_name, count))
            elif count < 500:
                self.stats['weak_classes'].append((class_name, count))
            elif count <= 3000:
                self.stats['balanced_classes'].append((class_name, count))
            else:
                self.stats['oversized_classes'].append((class_name, count))
        
        # Print summary
        print(f"Total images: {self.stats['total_images']:,}")
        print(f"Total classes: {self.stats['total_classes']}")
        print(f"Min per class: {self.stats['min_class']}")
        print(f"Max per class: {self.stats['max_class']}")
        print(f"Avg per class: {self.stats['avg_class']:.1f}")
        print(f"Imbalance ratio: {self.stats['imbalance_ratio']:.1f}x")
        print()
        
        print(f"📌 Problematic (< 100): {len(self.stats['problematic_classes'])} classes")
        print(f"📌 Low data (100-300): {len(self.stats['low_data_classes'])} classes")
        print(f"📌 Weak (300-500): {len(self.stats['weak_classes'])} classes")
        print(f"📌 Balanced (500-3000): {len(self.stats['balanced_classes'])} classes")
        print(f"📌 Oversized (>3000): {len(self.stats['oversized_classes'])} classes")
        print()
        
        if len(self.stats['problematic_classes']) > 0:
            print("⚠️  PROBLEMATIC CLASSES (unusable):")
            for cls, count in sorted(self.stats['problematic_classes'], key=lambda x: x[1]):
                print(f"   - {cls}: {count} images")
        print()
        
        return True
    
    def detect_duplicates(self, sample_size=5):
        """Détecteur simple de doublons (hash)"""
        print("🔎 PHASE 4: Détection doublons (sample)")
        print("=" * 80)
        
        duplicate_candidates = []
        total_sampled = 0
        
        for class_name, details in self.class_details.items():
            for img_file in details['images'][:sample_size]:
                img_path = Path(details['path']) / img_file
                try:
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash in self.image_hashes:
                        duplicate_candidates.append((img_path, self.image_hashes[file_hash]))
                    else:
                        self.image_hashes[file_hash] = str(img_path)
                    
                    total_sampled += 1
                except Exception as e:
                    pass
        
        print(f"Sampled: {total_sampled} images")
        print(f"Duplicate candidates: {len(duplicate_candidates)}")
        if duplicate_candidates:
            for dup, original in duplicate_candidates[:5]:
                print(f"   - {dup} == {original}")
        print()
        
        return len(duplicate_candidates)
    
    def generate_report(self, output_csv, output_json):
        """Générer rapports CSV et JSON"""
        print("📄 PHASE 5: Génération rapports")
        print("=" * 80)
        
        # CSV report
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Classe', 'Nombre_images', 'Classification', 
                'Min_Res', 'Avg_Res', 'Max_Res',
                'Avg_File_Size_MB', 'Corrupted_Count'
            ])
            
            for class_name in sorted(self.classes.keys()):
                count = self.classes[class_name]
                details = self.class_details[class_name]
                
                # Classify
                if count < 100:
                    classification = 'UNUSABLE'
                elif count < 300:
                    classification = 'LOW_DATA'
                elif count < 500:
                    classification = 'WEAK'
                elif count <= 3000:
                    classification = 'BALANCED'
                else:
                    classification = 'OVERSIZED'
                
                writer.writerow([
                    class_name,
                    count,
                    classification,
                    str(details.get('min_res', 'N/A')),
                    str(details.get('avg_res', 'N/A')),
                    str(details.get('max_res', 'N/A')),
                    f"{details.get('file_size_mb', 0):.2f}",
                    details.get('corrupted_count', 0)
                ])
        
        # JSON report
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.stats,
                'classes': dict(self.classes),
                'corrupted_files': self.corrupted,
                'recommendations': self._generate_recommendations()
            }, f, indent=2)
        
        print(f"✓ CSV: {output_csv}")
        print(f"✓ JSON: {output_json}\n")
    
    def _generate_recommendations(self):
        """Recommandations d'action"""
        recs = []
        
        if len(self.stats['problematic_classes']) > 0:
            recs.append(f"URGENT: Supprimer {len(self.stats['problematic_classes'])} classes < 100 images")
        
        if len(self.stats['low_data_classes']) > 0:
            recs.append(f"Augmenter {len(self.stats['low_data_classes'])} classes faibles (100-300)")
        
        if self.stats['imbalance_ratio'] > 10:
            recs.append(f"Imbalance critique ({self.stats['imbalance_ratio']:.1f}x): réduire classes > 3000")
        
        if len(self.corrupted) > 0:
            recs.append(f"Nettoyer {len(self.corrupted)} images corrompues")
        
        return recs
    
    def run_full_analysis(self, output_dir):
        """Exécuter toute l'analyse"""
        self.scan_dataset()
        self.check_image_quality()
        self.compute_statistics()
        self.detect_duplicates()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_csv = output_dir / 'dataset_quality_report.csv'
        output_json = output_dir / 'dataset_analysis.json'
        
        self.generate_report(str(output_csv), str(output_json))
        
        print("🎉 ANALYSE COMPLÈTE TERMINÉE\n")
        print("Recommandations:")
        for i, rec in enumerate(self._generate_recommendations(), 1):
            print(f"  {i}. {rec}")

if __name__ == '__main__':
    root = r'C:\smarts-n-yieldpredict.git\Diseasedataset'
    output_dir = r'C:\smarts-n-yieldpredict.git\dataset_analysis'
    
    analyzer = DatasetAnalyzer(root)
    analyzer.run_full_analysis(output_dir)
