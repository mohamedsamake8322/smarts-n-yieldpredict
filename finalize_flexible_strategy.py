"""
FINALIZE FLEXIBLE STRATEGY
Handle remaining classes that weren't moved
"""

import os
import shutil
from pathlib import Path

def finalize_flexible_strategy():
    """Finaliser la stratégie flexible pour les classes restantes"""

    source = Path(r'C:\smarts-n-yieldpredict.git\Diseasedataset')
    target = Path(r'C:\smarts-n-yieldpredict.git\dataset_flexible')

    main_dataset = target / 'dataset_main'
    low_data_backup = target / 'low_data_backup'
    oversized_backup = target / 'oversized_backup'

    # Classes restantes à gérer
    remaining_classes = [
        'Chili Growth Stage Augmented Dataset\\Dry chili',
        'train\\Cicadellidae',
        'train\\Lycorma delicatula',
        'train\\Miridae'
    ]

    print("HANDLING REMAINING CLASSES")
    print("="*50)

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    for class_name in remaining_classes:
        src_path = source / class_name

        if not src_path.exists():
            print(f"  {class_name} - source path doesn't exist")
            continue

        # Compter les images
        image_count = 0
        for root, dirs, files in os.walk(src_path):
            image_count += sum(1 for f in files if Path(f).suffix.lower() in image_extensions)

        print(f"  {class_name} - {image_count} images")

        # Déterminer la destination
        if image_count < 100:
            dst_path = low_data_backup / class_name.replace(os.sep, '_')
        elif image_count <= 3000:
            dst_path = main_dataset / class_name
        else:
            dst_path = oversized_backup / class_name.replace(os.sep, '_')

        # Vérifier si la destination existe déjà
        if dst_path.exists():
            print(f"    Destination exists: {dst_path}")
            # Vérifier si c'est vide ou contient des images
            dst_images = 0
            for root, dirs, files in os.walk(dst_path):
                dst_images += sum(1 for f in files if Path(f).suffix.lower() in image_extensions)

            if dst_images > 0:
                print(f"    Destination has {dst_images} images - skipping")
                continue
            else:
                print(f"    Destination empty - removing and moving")
                shutil.rmtree(dst_path)

        # Créer le répertoire parent si nécessaire
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(src_path), str(dst_path))
            print(f"    Successfully moved to {dst_path}")
        except Exception as e:
            print(f"    Error moving: {e}")

    print("\nFINAL VERIFICATION")
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
        return False
    else:
        print("SUCCESS: All classes moved!")

        # Statistiques finales
        print("\nFINAL STATISTICS")
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

if __name__ == '__main__':
    success = finalize_flexible_strategy()
    if success:
        print("\nFLEXIBLE STRATEGY COMPLETED SUCCESSFULLY!")
    else:
        print("\nSome classes still remain - manual intervention needed")
