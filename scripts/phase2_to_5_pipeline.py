"""
PHASE 2-5: AUTOMATIC SORTING, CLEANING, RESTRUCTURING, AUGMENTATION
Pipeline complet de nettoyage et structuration du dataset
"""

import os
import shutil
import json
from pathlib import Path
from collections import defaultdict

class DatasetPipelineManager:
    def __init__(self, source_root, target_root):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.config = {
            'unusable_threshold': 100,        # < 100: à supprimer
            'low_data_threshold': 300,        # 100-300: à augmenter
            'weak_threshold': 500,            # 300-500: considérer augmentation
            'balanced_max': 3000,             # 500-3000: parfait
            'oversized_threshold': 3000,      # > 3000: à réduire
            'target_class_size': 1000,        # Cible idéale après augmentation
        }
        self.hierarchy = self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Construire mapppage hiérarchique level_1/level_2/level_3"""
        hierarchy = {}
        
        for dirpath, _, filenames in os.walk(self.source_root):
            rel_path = Path(dirpath).relative_to(self.source_root)
            image_count = sum(1 for f in filenames if Path(f).suffix.lower() in self.image_extensions)
            
            if image_count > 0:
                class_name = str(rel_path)
                parts = class_name.split(os.sep)
                
                # Level 1: Déterminer type (disease, pest, growth_stage)
                level_1 = parts[0]
                if 'chili' in level_1.lower():
                    if 'disease' in level_1.lower():
                        level_1_mapped = 'disease'
                    else:
                        level_1_mapped = 'growth_stage'
                elif 'pest' in level_1.lower():
                    level_1_mapped = 'pest'
                elif 'train' in level_1.lower() or 'val' in level_1.lower():
                    # Déterminer type basé sur le nom de classe
                    class_full_name = parts[-1].lower()
                    if any(x in class_full_name for x in ['disease', 'fungal', 'bacterial', 'viral', 'late_blight', 'early_blight', 'scab', 'rot', 'leaf', 'mold']):
                        level_1_mapped = 'disease'
                    else:
                        level_1_mapped = 'pest'
                else:
                    level_1_mapped = 'other'
                
                # Level 2: Catégorie (fungal, bacterial, viral pour disease; type insecte pour pest)
                if level_1_mapped == 'disease':
                    class_name_lower = parts[-1].lower()
                    if any(x in class_name_lower for x in ['fungal', 'scab', 'spot', 'mildew', 'cercospora']):
                        level_2_mapped = 'fungal'
                    elif any(x in class_name_lower for x in ['bacterial']):
                        level_2_mapped = 'bacterial'
                    elif any(x in class_name_lower for x in ['virus', 'curl', 'mosaic', 'haunglong']):
                        level_2_mapped = 'viral'
                    elif any(x in class_name_lower for x in ['deficiency', 'nutrition']):
                        level_2_mapped = 'nutritional'
                    else:
                        level_2_mapped = 'other_disease'
                elif level_1_mapped == 'pest':
                    # Classif simple par type d'insecte
                    class_name_lower = parts[-1].lower()
                    if any(x in class_name_lower for x in ['aphid', 'hopper', 'thrip', 'fly']):
                        level_2_mapped = 'sap_sucker'
                    elif any(x in class_name_lower for x in ['borer', 'worm', 'caterpillar']):
                        level_2_mapped = 'borer_worm'
                    elif any(x in class_name_lower for x in ['beetle', 'bug']):
                        level_2_mapped = 'beetle_bug'
                    elif any(x in class_name_lower for x in ['mite', 'spider']):
                        level_2_mapped = 'mite'
                    elif any(x in class_name_lower for x in ['grub', 'cricket']):
                        level_2_mapped = 'grub_cricket'
                    else:
                        level_2_mapped = 'other_pest'
                else:
                    level_2_mapped = 'other'
                
                # Level 3: Classe finale (nom complet)
                level_3_mapped = parts[-1]
                
                hierarchy[class_name] = {
                    'level_1': level_1_mapped,
                    'level_2': level_2_mapped,
                    'level_3': level_3_mapped,
                    'path': dirpath,
                    'count': image_count
                }
        
        return hierarchy
    
    def phase2_sort_classes(self):
        """PHASE 2: Trier les classes par statut"""
        print("\n" + "="*80)
        print("PHASE 2: AUTOMATIC SORTING BY STATUS")
        print("="*80 + "\n")
        
        sorted_structure = defaultdict(list)
        
        for class_name, info in self.hierarchy.items():
            count = info['count']
            
            if count < self.config['unusable_threshold']:
                status = 'UNUSABLE'
            elif count < self.config['low_data_threshold']:
                status = 'LOW_DATA'
            elif count < self.config['weak_threshold']:
                status = 'WEAK'
            elif count <= self.config['balanced_max']:
                status = 'BALANCED'
            else:
                status = 'OVERSIZED'
            
            sorted_structure[status].append((class_name, count))
        
        # Print summary
        for status in ['UNUSABLE', 'LOW_DATA', 'WEAK', 'BALANCED', 'OVERSIZED']:
            classes_list = sorted_structure[status]
            print(f"{status:12s} ({len(classes_list):3d} classes)")
            if len(classes_list) <= 10:
                for cls, count in sorted(classes_list, key=lambda x: x[1]):
                    print(f"   - {cls}: {count}")
            else:
                # Show extremes
                sorted_list = sorted(classes_list, key=lambda x: x[1])
                for cls, count in sorted_list[:3]:
                    print(f"   - {cls}: {count}")
                print(f"   ... ({len(sorted_list)-6} more)")
                for cls, count in sorted_list[-3:]:
                    print(f"   - {cls}: {count}")
            print()
        
        return sorted_structure
    
    def phase3_clean_dataset(self, dry_run=True):
        """PHASE 3: Nettoyer images corrompues, doublons suspects"""
        print("="*80)
        print("PHASE 3: DATASET CLEANING")
        print("="*80 + "\n")
        
        from PIL import Image
        
        corrupted_files = []
        removed_count = 0
        
        for class_name, info in self.hierarchy.items():
            dirpath = info['path']
            images = [f for f in os.listdir(dirpath) if Path(f).suffix.lower() in self.image_extensions]
            
            for img_file in images:
                img_path = Path(dirpath) / img_file
                try:
                    img = Image.open(img_path)
                    img.verify()  # Vérifie l'intégrité
                except Exception as e:
                    corrupted_files.append(str(img_path))
                    if not dry_run:
                        img_path.unlink()
                        removed_count += 1
        
        print(f"Corrupted files detected: {len(corrupted_files)}")
        if len(corrupted_files) <= 10:
            for f in corrupted_files:
                print(f"   - {f}")
        else:
            print(f"   (Too many to display, see JSON)")
        
        if dry_run:
            print(f"\n⚠️  DRY RUN: Would remove {len(corrupted_files)} files")
        else:
            print(f"✓ Removed {removed_count} corrupted files")
        
        return corrupted_files
    
    def phase4_restructure_hierarchical(self, dry_run=True):
        """PHASE 4: Restructurer en level_1/level_2/level_3"""
        print("="*80)
        print("PHASE 4: HIERARCHICAL RESTRUCTURING")
        print("="*80 + "\n")
        
        restructured_dir = self.target_root / 'dataset_structured'
        
        move_plan = defaultdict(list)
        
        for class_name, info in self.hierarchy.items():
            level_1 = info['level_1']
            level_2 = info['level_2']
            level_3 = info['level_3']
            
            target_dir = restructured_dir / level_1 / level_2 / level_3
            
            move_plan[str(target_dir)].append({
                'source': info['path'],
                'count': info['count']
            })
        
        print(f"Planned structure:")
        print(f"   Level_1 types: disease, pest, growth_stage, other")
        print(f"   Level_2 categories per type")
        print(f"   Level_3 final classes\n")
        
        total_moves = sum(len(v) for v in move_plan.values())
        print(f"Total class moves planned: {total_moves}\n")
        
        if not dry_run:
            for target_path, sources in move_plan.items():
                Path(target_path).mkdir(parents=True, exist_ok=True)
                for src_info in sources:
                    src = Path(src_info['source'])
                    # Move images
                    for img_file in os.listdir(src):
                        if Path(img_file).suffix.lower() in self.image_extensions:
                            shutil.move(str(src / img_file), str(Path(target_path) / img_file))
        
        print("✓ Restructuring simulation complete (use dry_run=False to execute)\n")
        
        return move_plan
    
    def phase5_augmentation_plan(self, sorted_structure):
        """PHASE 5: Planifier augmentation intelligente"""
        print("="*80)
        print("PHASE 5: AUGMENTATION PLANNING")
        print("="*80 + "\n")
        
        augmentation_plan = {
            'unusable': {
                'action': 'DELETE',
                'reason': 'Insufficient data (< 100 images)',
                'count': len(sorted_structure['UNUSABLE']),
                'details': sorted_structure['UNUSABLE']
            },
            'low_data': {
                'action': 'AUGMENT_AGGRESSIVE',
                'reason': 'Weak data (100-300 images)',
                'target_size': self.config['target_class_size'],
                'count': len(sorted_structure['LOW_DATA']),
                'augmentation_factor': 3,  # 3x à 5x
                'details': sorted_structure['LOW_DATA']
            },
            'weak': {
                'action': 'AUGMENT_MODERATE',
                'reason': 'Somewhat weak (300-500 images)',
                'target_size': self.config['target_class_size'],
                'count': len(sorted_structure['WEAK']),
                'augmentation_factor': 2,  # 2x à 3x
                'details': sorted_structure['WEAK']
            },
            'balanced': {
                'action': 'KEEP_AS_IS',
                'reason': 'Good balance (500-3000 images)',
                'count': len(sorted_structure['BALANCED']),
                'details': sorted_structure['BALANCED']
            },
            'oversized': {
                'action': 'DOWNSAMPLE',
                'reason': 'Excessive data (> 3000 images)',
                'target_size': 2000,
                'count': len(sorted_structure['OVERSIZED']),
                'details': sorted_structure['OVERSIZED']
            }
        }
        
        # Display plan
        for status, plan in augmentation_plan.items():
            print(f"{status.upper()}")
            print(f"   Action: {plan['action']}")
            print(f"   Reason: {plan['reason']}")
            print(f"   Classes: {plan['count']}")
            if 'augmentation_factor' in plan:
                print(f"   Augmentation: {plan['augmentation_factor']}x")
            if 'target_size' in plan:
                print(f"   Target size: {plan['target_size']}")
            print()
        
        return augmentation_plan
    
    def run_full_pipeline(self, dry_run=True):
        """Exécuter toutes les phases"""
        sorted_structure = self.phase2_sort_classes()
        corrupted = self.phase3_clean_dataset(dry_run=True)  # Always dry-run first
        restructured = self.phase4_restructure_hierarchical(dry_run=dry_run)
        augmentation = self.phase5_augmentation_plan(sorted_structure)
        
        # Sauvegarder le plan global
        plan_file = self.target_root / 'transformation_plan.json'
        self.target_root.mkdir(parents=True, exist_ok=True)
        
        with open(plan_file, 'w') as f:
            json.dump({
                'status': 'PLANNING_COMPLETE',
                'sorting': {k: [(name, count) for name, count in v] for k, v in sorted_structure.items()},
                'corrupted_files': corrupted,
                'augmentation_plan': augmentation
            }, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ PIPELINE PLANNING COMPLETE")
        print("="*80)
        print(f"Plan saved: {plan_file}\n")

if __name__ == '__main__':
    source = r'C:\smarts-n-yieldpredict.git\Diseasedataset'
    target = r'C:\smarts-n-yieldpredict.git\dataset_clean'
    
    manager = DatasetPipelineManager(source, target)
    manager.run_full_pipeline(dry_run=True)
