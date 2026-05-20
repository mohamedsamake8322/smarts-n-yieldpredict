"""
UPDATED STRATEGY: FLEXIBLE DATASET MANAGEMENT
Keep low-data classes for future completion, reduce oversized now
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

class FlexibleDatasetManager:
    def __init__(self, source_root, target_root):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.analysis_csv = self.target_root.parent / 'dataset_analysis' / 'dataset_quality_report.csv'
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def get_class_counts(self):
        """Récupérer les comptes par classe depuis l'analyse existante"""
        classes = {}
        print(f"DEBUG: Looking for CSV at: {self.analysis_csv}")
        print(f"DEBUG: File exists: {self.analysis_csv.exists()}")

        if self.analysis_csv.exists():
            import csv
            with open(self.analysis_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 4:
                        class_path = row[0]
                        try:
                            count = int(row[1])  # Nombre_Images is column 1
                            classes[class_path] = count
                        except ValueError:
                            print(f"Warning: Could not parse count for {class_path}: {row[1]}")

        print(f"DEBUG: Loaded {len(classes)} classes")
        return classes

    def categorize_classes(self, classes):
        """Catégoriser selon la nouvelle stratégie flexible"""
        categories = {
            'keep_for_later': [],    # < 100 images - garder pour compléter plus tard
            'augment_now': [],       # 100-500 images - augmenter maintenant
            'keep_as_is': [],        # 500-3000 images - parfait
            'reduce_now': []         # > 3000 images - réduire maintenant
        }

        for class_name, count in classes.items():
            if count < 100:
                categories['keep_for_later'].append((class_name, count))
            elif count < 500:
                categories['augment_now'].append((class_name, count))
            elif count <= 3000:
                categories['keep_as_is'].append((class_name, count))
            else:
                categories['reduce_now'].append((class_name, count))

        return categories

    def create_flexible_structure(self, categories):
        """Créer la structure flexible"""
        print("\nCREATING FLEXIBLE DATASET STRUCTURE")
        print("="*80)

        # Structure principale
        main_dataset = self.target_root / 'dataset_main'
        low_data_backup = self.target_root / 'low_data_backup'
        oversized_backup = self.target_root / 'oversized_backup'

        # Créer dossiers
        for path in [main_dataset, low_data_backup, oversized_backup]:
            path.mkdir(parents=True, exist_ok=True)

        print(f"Created: {main_dataset}")
        print(f"Created: {low_data_backup} (for future completion)")
        print(f"Created: {oversized_backup} (reduced versions)")

        return main_dataset, low_data_backup, oversized_backup

    def move_classes_by_category(self, categories, main_dataset, low_data_backup, oversized_backup):
        """Déplacer les classes selon leur catégorie"""
        print("\nMOVING CLASSES BY CATEGORY")
        print("="*80)

        # 1. Classes à garder pour plus tard (< 100 images)
        print(f"\nMoving {len(categories['keep_for_later'])} classes to low_data_backup:")
        for class_name, count in categories['keep_for_later']:
            src_path = self.source_root / class_name
            dst_path = low_data_backup / class_name.replace(os.sep, '_')  # Flatten path

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
                print(f"  Moved {class_name} ({count} images) to low_data_backup")
            else:
                print(f"  Warning: {class_name} not found")

        # 2. Classes à réduire (> 3000 images)
        print(f"\nReducing {len(categories['reduce_now'])} oversized classes:")
        for class_name, count in categories['reduce_now']:
            src_path = self.source_root / class_name
            dst_path = main_dataset / class_name

            if src_path.exists():
                # Créer destination
                dst_path.mkdir(parents=True, exist_ok=True)

                # Lister toutes les images
                images = []
                for ext in self.image_extensions:
                    images.extend(list(src_path.glob(f'*{ext}')))

                # Garder seulement 2000 images (random sample)
                import random
                selected_images = random.sample(images, min(2000, len(images)))

                # Copier vers destination
                for img in selected_images:
                    shutil.copy2(str(img), str(dst_path / img.name))

                # Sauvegarder le reste dans oversized_backup
                backup_path = oversized_backup / class_name.replace(os.sep, '_')
                backup_path.mkdir(parents=True, exist_ok=True)

                for img in images:
                    if img not in selected_images:
                        shutil.move(str(img), str(backup_path / img.name))

                print(f"  Reduced {class_name}: {len(selected_images)} kept, {len(images) - len(selected_images)} moved to backup")

        # 3. Classes à garder telles quelles (500-3000 images)
        print(f"\nMoving {len(categories['keep_as_is'])} balanced classes to main:")
        for class_name, count in categories['keep_as_is']:
            src_path = self.source_root / class_name
            dst_path = main_dataset / class_name

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
                print(f"  Moved {class_name} ({count} images) to main dataset")

        # 4. Classes à augmenter (100-500 images) - pour l'instant, les garder dans main
        print(f"\nMoving {len(categories['augment_now'])} classes to augment later:")
        for class_name, count in categories['augment_now']:
            src_path = self.source_root / class_name
            dst_path = main_dataset / class_name

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
                print(f"  Moved {class_name} ({count} images) to main (to augment)")

    def generate_flexible_plan(self, categories):
        """Générer le plan flexible"""
        plan = {
            "strategy": "FLEXIBLE_DATASET_MANAGEMENT",
            "description": "Keep low-data classes for future completion, reduce oversized now",
            "categories": {
                "keep_for_later": {
                    "count": len(categories['keep_for_later']),
                    "description": "Classes < 100 images - saved for future dataset additions",
                    "location": "low_data_backup/",
                    "action": "Wait for new datasets to reach 500+ images",
                    "classes": categories['keep_for_later']
                },
                "augment_now": {
                    "count": len(categories['augment_now']),
                    "description": "Classes 100-500 images - augment immediately",
                    "location": "dataset_main/",
                    "target_size": 1000,
                    "action": "Apply controlled augmentation (2-3x)",
                    "classes": categories['augment_now']
                },
                "keep_as_is": {
                    "count": len(categories['keep_as_is']),
                    "description": "Classes 500-3000 images - perfect balance",
                    "location": "dataset_main/",
                    "action": "No changes needed",
                    "classes": categories['keep_as_is']
                },
                "reduce_now": {
                    "count": len(categories['reduce_now']),
                    "description": "Classes > 3000 images - reduced to 2000",
                    "location": "dataset_main/ (2000 images), oversized_backup/ (extras)",
                    "action": "Random sampling to 2000 images",
                    "classes": categories['reduce_now']
                }
            },
            "next_steps": [
                "1. Add new datasets to complement low_data_backup classes",
                "2. Apply augmentation to augment_now classes (100-500 images)",
                "3. Train initial model on dataset_main (balanced classes)",
                "4. Gradually incorporate low_data classes as they reach threshold"
            ],
            "benefits": [
                "Flexible for future dataset additions",
                "No permanent data loss",
                "Immediate training possible on balanced subset",
                "Scalable approach"
            ]
        }

        return plan

    def execute_flexible_strategy(self):
        """Exécuter la stratégie flexible complète"""
        print("FLEXIBLE DATASET MANAGEMENT STRATEGY")
        print("="*80)
        print("Strategy: Keep low-data for future, reduce oversized now")
        print("="*80 + "\n")

        # 1. Charger les données d'analyse
        classes = self.get_class_counts()
        if not classes:
            print("ERROR: No analysis data found. Run phase1_analysis.py first.")
            return

        print(f"Loaded {len(classes)} classes from analysis")

        # 2. Catégoriser
        categories = self.categorize_classes(classes)
        print(f"Categorized: {len(categories['keep_for_later'])} keep, {len(categories['augment_now'])} augment, {len(categories['keep_as_is'])} keep, {len(categories['reduce_now'])} reduce")

        # 3. Créer structure
        main_dataset, low_data_backup, oversized_backup = self.create_flexible_structure(categories)

        # 4. Déplacer classes
        self.move_classes_by_category(categories, main_dataset, low_data_backup, oversized_backup)

        # 5. Générer plan
        plan = self.generate_flexible_plan(categories)

        # Sauvegarder plan
        import json
        plan_file = self.target_root / 'flexible_strategy_plan.json'
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)

        print(f"\nPlan saved: {plan_file}")

        # Afficher résumé final
        print("\n" + "="*80)
        print("FLEXIBLE STRATEGY EXECUTED")
        print("="*80)
        print(f"dataset_main/: {len(categories['augment_now']) + len(categories['keep_as_is'])} classes ready")
        print(f"low_data_backup/: {len(categories['keep_for_later'])} classes for future completion")
        print(f"oversized_backup/: {len(categories['reduce_now'])} reduced classes backup")
        print("\nReady for next steps!")

if __name__ == '__main__':
    source = r'C:\smarts-n-yieldpredict.git\Diseasedataset'
    target = r'C:\smarts-n-yieldpredict.git\dataset_flexible'

    manager = FlexibleDatasetManager(source, target)
    manager.execute_flexible_strategy()
