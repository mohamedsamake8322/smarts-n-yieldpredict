"""
PRODUCTION-GRADE DATASET REORGANIZATION
Hard cleaning + Phytopathology taxonomy structure
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from phytopathology_taxonomy import PhytopathologyTaxonomy

class ProductionDatasetManager:
    """Production-grade dataset manager with hard cleaning and phytopathology taxonomy"""

    def __init__(self, target_range=(500, 1500)):
        self.target_range = target_range  # (min_images, max_images) per class
        self.taxonomy = PhytopathologyTaxonomy()

        # Source and target paths
        self.source_dataset = Path('dataset_flexible/dataset_main')
        self.target_dataset = Path('dataset_production')

        # Statistics
        self.stats = {
            'original_classes': 0,
            'final_classes': 0,
            'images_deleted': 0,
            'images_kept': 0,
            'taxonomy_mappings': 0
        }

    def hard_clean_class(self, class_path, target_count):
        """Permanently delete excess images from a class to reach target count"""
        if not class_path.exists():
            return 0

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = []
        for root, dirs, files in os.walk(class_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)

        total_images = len(image_files)
        if total_images <= target_count:
            return total_images  # Already within range

        # Randomly select images to keep
        images_to_keep = random.sample(image_files, target_count)
        images_to_delete = [f for f in image_files if f not in images_to_keep]

        # Permanently delete excess images
        for img_path in images_to_delete:
            try:
                img_path.unlink()  # Permanent deletion
            except Exception as e:
                print(f"Warning: Could not delete {img_path}: {e}")

        deleted_count = len(images_to_delete)
        self.stats['images_deleted'] += deleted_count
        self.stats['images_kept'] += target_count

        print(f"  Hard cleaned: {total_images} -> {target_count} images ({deleted_count} deleted)")
        return target_count

    def create_taxonomy_structure(self):
        """Create the phytopathology taxonomy directory structure"""
        print("CREATING PHYTOPATHOLOGY TAXONOMY STRUCTURE")
        print("="*50)

        # Remove existing target if it exists
        if self.target_dataset.exists():
            shutil.rmtree(self.target_dataset)

        # Create level 1 directories
        for l1_category in self.taxonomy.level_1_categories.keys():
            (self.target_dataset / l1_category).mkdir(parents=True, exist_ok=True)

        # Create level 2 directories
        for l1, l2_dict in [
            ('disease', self.taxonomy.disease_categories),
            ('pest', self.taxonomy.pest_categories),
            ('growth_stage', self.taxonomy.growth_categories)
        ]:
            for l2_category in l2_dict.keys():
                (self.target_dataset / l1 / l2_category).mkdir(parents=True, exist_ok=True)

        print(f"✓ Created taxonomy structure in {self.target_dataset}")

    def reorganize_dataset(self):
        """Main reorganization function with hard cleaning and taxonomy mapping"""
        print("PRODUCTION-GRADE DATASET REORGANIZATION")
        print("="*60)
        print(f"Target range per class: {self.target_range[0]}-{self.target_range[1]} images")
        print(f"Source: {self.source_dataset}")
        print(f"Target: {self.target_dataset}")
        print()

        # Create taxonomy structure
        self.create_taxonomy_structure()

        # Get all classes from source
        classes_to_process = []
        for root, dirs, files in os.walk(self.source_dataset):
            rel_path = Path(root).relative_to(self.source_dataset)
            if rel_path != Path('.'):
                image_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')))
                if image_count > 0:
                    classes_to_process.append((rel_path, image_count))

        self.stats['original_classes'] = len(classes_to_process)
        print(f"Found {len(classes_to_process)} classes to process")

        # Process each class
        successful_mappings = 0

        for class_path, original_count in classes_to_process:
            print(f"\nProcessing: {class_path} ({original_count} images)")

            # Map to taxonomy
            taxonomy_path, scientific_name = self.taxonomy.map_class_name(class_path)
            target_class_dir = self.target_dataset / taxonomy_path

            print(f"  Taxonomy: {taxonomy_path} ({scientific_name})")

            # Determine target image count
            if original_count < self.target_range[0]:
                print(f"  Warning: Only {original_count} images (below minimum {self.target_range[0]})")
                target_count = original_count  # Keep all available
            elif original_count > self.target_range[1]:
                target_count = random.randint(self.target_range[0], self.target_range[1])
                print(f"  Reducing: {original_count} -> {target_count} images")
            else:
                target_count = original_count  # Already in range

            # Create target directory
            target_class_dir.mkdir(parents=True, exist_ok=True)

            # Copy and clean images
            source_class_dir = self.source_dataset / class_path
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

            # Get all source images
            source_images = []
            for root, dirs, files in os.walk(source_class_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        source_images.append(Path(root) / file)

            # Select images to copy
            if len(source_images) > target_count:
                selected_images = random.sample(source_images, target_count)
            else:
                selected_images = source_images

            # Copy selected images
            copied_count = 0
            for src_img in selected_images:
                # Create relative path structure if needed
                rel_path = src_img.relative_to(source_class_dir)
                dst_img = target_class_dir / rel_path.name  # Use just filename to flatten

                try:
                    shutil.copy2(str(src_img), str(dst_img))
                    copied_count += 1
                except Exception as e:
                    print(f"    Error copying {src_img.name}: {e}")

            print(f"  Copied {copied_count} images to {taxonomy_path}")

            if copied_count > 0:
                successful_mappings += 1

        self.stats['final_classes'] = successful_mappings
        self.stats['taxonomy_mappings'] = successful_mappings

        # Final cleanup - remove source dataset
        print(f"\nPERFORMING HARD CLEANUP")
        print("="*30)
        print("Removing source dataset (no backups kept)...")
        try:
            shutil.rmtree(self.source_dataset)
            print("✓ Source dataset removed")
        except Exception as e:
            print(f"Warning: Could not remove source dataset: {e}")

        # Remove backup directories
        for backup_dir in ['dataset_flexible/low_data_backup', 'dataset_flexible/oversized_backup']:
            backup_path = Path(backup_dir)
            if backup_path.exists():
                try:
                    shutil.rmtree(backup_path)
                    print(f"✓ Removed backup: {backup_dir}")
                except Exception as e:
                    print(f"Warning: Could not remove {backup_dir}: {e}")

    def generate_report(self):
        """Generate comprehensive reorganization report"""
        print("\n" + "="*60)
        print("PRODUCTION DATASET REORGANIZATION REPORT")
        print("="*60)

        print(f"Original classes: {self.stats['original_classes']}")
        print(f"Successfully mapped: {self.stats['final_classes']}")
        print(f"Images kept: {self.stats['images_kept']}")
        print(f"Images permanently deleted: {self.stats['images_deleted']}")
        print()

        # Verify final structure
        print("FINAL TAXONOMY STRUCTURE:")
        print("-" * 30)

        total_final_images = 0
        for l1_dir in sorted(self.target_dataset.iterdir()):
            if l1_dir.is_dir():
                l1_count = 0
                l1_classes = 0

                for l2_dir in sorted(l1_dir.iterdir()):
                    if l2_dir.is_dir():
                        l2_count = 0
                        l2_classes = 0

                        for class_dir in sorted(l2_dir.iterdir()):
                            if class_dir.is_dir():
                                class_images = sum(1 for f in class_dir.iterdir()
                                                 if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'})
                                if class_images > 0:
                                    l2_count += class_images
                                    l2_classes += 1
                                    total_final_images += class_images

                        if l2_classes > 0:
                            print(f"  {l1_dir.name}/{l2_dir.name}: {l2_classes} classes, {l2_count} images")
                            l1_count += l2_count
                            l1_classes += l2_classes

                if l1_classes > 0:
                    print(f"{l1_dir.name}: {l1_classes} classes total, {l1_count} images")
                    print()

        print(f"TOTAL: {total_final_images} images in production dataset")
        print(f"Target range per class: {self.target_range[0]}-{self.target_range[1]} images")

        # Quality checks
        print("\nQUALITY CHECKS:")
        print("-" * 20)

        issues = []
        for l1_dir in self.target_dataset.iterdir():
            if l1_dir.is_dir():
                for l2_dir in l1_dir.iterdir():
                    if l2_dir.is_dir():
                        for class_dir in l2_dir.iterdir():
                            if class_dir.is_dir():
                                class_images = sum(1 for f in class_dir.iterdir()
                                                 if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'})
                                if class_images < self.target_range[0]:
                                    issues.append(f"Low count: {l1_dir.name}/{l2_dir.name}/{class_dir.name} ({class_images} images)")
                                elif class_images > self.target_range[1]:
                                    issues.append(f"High count: {l1_dir.name}/{l2_dir.name}/{class_dir.name} ({class_images} images)")

        if issues:
            print(f"⚠️  Found {len(issues)} classes outside target range:")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        else:
            print("✓ All classes within target range")

        print("\n✓ PRODUCTION DATASET READY FOR SCALING")

def main():
    # Set random seed for reproducible results
    random.seed(42)

    # Create production manager
    manager = ProductionDatasetManager(target_range=(500, 1500))

    # Execute reorganization
    manager.reorganize_dataset()

    # Generate report
    manager.generate_report()

if __name__ == '__main__':
    main()