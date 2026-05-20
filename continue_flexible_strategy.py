"""
CONTINUE FLEXIBLE DATASET STRATEGY
Resume from where it stopped
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def continue_flexible_strategy():
    """Continuer la stratégie flexible"""

    source = Path(r'C:\smarts-n-yieldpredict.git\Diseasedataset')
    target = Path(r'C:\smarts-n-yieldpredict.git\dataset_flexible')

    main_dataset = target / 'dataset_main'
    low_data_backup = target / 'low_data_backup'
    oversized_backup = target / 'oversized_backup'

    print("CHECKING CURRENT STATE")
    print("="*50)

    # Vérifier ce qui reste à faire
    remaining_classes = []

    # Scanner les classes restantes dans source
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    for dirpath, dirnames, filenames in os.walk(source):
        rel_path = Path(dirpath).relative_to(source)
        image_count = sum(1 for f in filenames if Path(f).suffix.lower() in image_extensions)

        if image_count > 0:
            class_name = str(rel_path)
            remaining_classes.append((class_name, image_count))

    print(f"Classes remaining in source: {len(remaining_classes)}")

    if len(remaining_classes) == 0:
        print("All classes have been moved! Checking completion...")

        # Compter ce qui a été fait
        moved_to_main = 0
        moved_to_backup = 0
        moved_to_oversized = 0

        for root, dirs, files in os.walk(main_dataset):
            if any(f.endswith(tuple(image_extensions)) for f in files):
                moved_to_main += 1

        for root, dirs, files in os.walk(low_data_backup):
            if any(f.endswith(tuple(image_extensions)) for f in files):
                moved_to_backup += 1

        for root, dirs, files in os.walk(oversized_backup):
            if any(f.endswith(tuple(image_extensions)) for f in files):
                moved_to_oversized += 1

        print(f"Classes in dataset_main: {moved_to_main}")
        print(f"Classes in low_data_backup: {moved_to_backup}")
        print(f"Classes in oversized_backup: {moved_to_oversized}")
        print(f"Total moved: {moved_to_main + moved_to_backup + moved_to_oversized}")

        return True

    # Continuer le déplacement
    print("\nCONTINUING MOVES")
    print("="*50)

    for class_name, count in remaining_classes:
        src_path = source / class_name

        # Déterminer la destination basée sur le count
        if count < 100:
            dst_path = low_data_backup / class_name.replace(os.sep, '_')
        elif count <= 3000:
            dst_path = main_dataset / class_name
        else:
            # Cas spécial pour oversized - devrait déjà être traité
            dst_path = main_dataset / class_name

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if dst_path.exists():
                print(f"  Skipping {class_name} (already exists)")
            else:
                shutil.move(str(src_path), str(dst_path))
                print(f"  Moved {class_name} ({count} images)")
        except Exception as e:
            print(f"  Error moving {class_name}: {e}")

    print("\nCOMPLETION CHECK")
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
        for cls in final_remaining[:5]:
            print(f"  - {cls}")
    else:
        print("SUCCESS: All classes moved!")

    return len(final_remaining) == 0

if __name__ == '__main__':
    success = continue_flexible_strategy()
    if success:
        print("\nFLEXIBLE STRATEGY COMPLETED!")
    else:
        print("\nStrategy partially completed - may need another run")
