"""
CLEAN UP REMAINING CLASSES
Remove classes from source that have been successfully moved
"""

import os
import shutil
from pathlib import Path

def cleanup_remaining_classes():
    """Nettoyer les classes restantes qui ont été déplacées"""

    source = Path(r'C:\smarts-n-yieldpredict.git\Diseasedataset')
    target = Path(r'C:\smarts-n-yieldpredict.git\dataset_flexible')

    main_dataset = target / 'dataset_main'
    low_data_backup = target / 'low_data_backup'
    oversized_backup = target / 'oversized_backup'

    # Classes restantes à nettoyer
    remaining_classes = [
        'Chili Growth Stage Augmented Dataset\\Dry chili',
        'train\\Cicadellidae',
        'train\\Lycorma delicatula',
        'train\\Miridae'
    ]

    print("CLEANING UP REMAINING CLASSES")
    print("="*50)

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    for class_name in remaining_classes:
        src_path = source / class_name
        dst_path = main_dataset / class_name

        if not src_path.exists():
            print(f"  {class_name} - source doesn't exist")
            continue

        if not dst_path.exists():
            print(f"  {class_name} - destination doesn't exist")
            continue

        # Compter les images dans source et destination
        src_images = 0
        dst_images = 0

        for root, dirs, files in os.walk(src_path):
            src_images += sum(1 for f in files if Path(f).suffix.lower() in image_extensions)

        for root, dirs, files in os.walk(dst_path):
            dst_images += sum(1 for f in files if Path(f).suffix.lower() in image_extensions)

        print(f"  {class_name}:")
        print(f"    Source: {src_images} images")
        print(f"    Destination: {dst_images} images")

        # Décider quoi faire
        if dst_images >= src_images:
            # La destination a au moins autant d'images - supprimer la source
            try:
                shutil.rmtree(src_path)
                print(f"    Removed source (destination complete)")
            except Exception as e:
                print(f"    Error removing source: {e}")
        elif dst_images > 0 and src_images > dst_images:
            # La destination a des images mais moins que la source
            # Copier les images manquantes
            print(f"    Copying remaining {src_images - dst_images} images...")

            src_files = []
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        src_files.append(Path(root) / file)

            dst_files = set()
            for root, dirs, files in os.walk(dst_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        dst_files.add(Path(file).name)

            copied = 0
            for src_file in src_files:
                if src_file.name not in dst_files:
                    # Copier vers la destination
                    rel_path = src_file.relative_to(src_path)
                    dst_file = dst_path / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        shutil.copy2(str(src_file), str(dst_file))
                        copied += 1
                    except Exception as e:
                        print(f"      Error copying {src_file.name}: {e}")

            print(f"    Copied {copied} additional images")

            # Maintenant supprimer la source
            try:
                shutil.rmtree(src_path)
                print(f"    Removed source after copying")
            except Exception as e:
                print(f"    Error removing source: {e}")
        else:
            print(f"    Keeping source (destination incomplete)")

    print("\nFINAL CLEANUP VERIFICATION")
    print("="*50)

    # Vérification finale
    final_remaining = []
    for dirpath, dirnames, filenames in os.walk(source):
        rel_path = Path(dirpath).relative_to(source)
        image_count = sum(1 for f in filenames if Path(f).suffix.lower() in image_extensions)

        if image_count > 0:
            final_remaining.append(str(rel_path))

    if final_remaining:
        print(f"Still remaining: {len(final_remaining)} classes")
        for cls in final_remaining:
            print(f"  - {cls}")
    else:
        print("SUCCESS: All classes cleaned up!")

        # Statistiques finales
        print("\nFINAL ORGANIZATION SUMMARY")
        print("="*50)

        main_count = 0
        backup_count = 0
        oversized_count = 0

        for root, dirs, files in os.walk(main_dataset):
            if any(f.endswith(tuple(image_extensions)) for f in files):
                main_count += 1

        for root, dirs, files in os.walk(low_data_backup):
            if any(f.endswith(tuple(image_extensions)) for f in files):
                backup_count += 1

        for root, dirs, files in os.walk(oversized_backup):
            if any(f.endswith(tuple(image_extensions)) for f in files):
                oversized_count += 1

        print(f"Classes in dataset_main: {main_count}")
        print(f"Classes in low_data_backup: {backup_count}")
        print(f"Classes in oversized_backup: {oversized_count}")
        print(f"Total organized: {main_count + backup_count + oversized_count}")

        return True

    return False

if __name__ == '__main__':
    success = cleanup_remaining_classes()
    if success:
        print("\nFLEXIBLE STRATEGY COMPLETED SUCCESSFULLY!")
    else:
        print("\nCleanup completed but some classes remain")
