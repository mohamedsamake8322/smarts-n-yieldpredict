"""
FINAL DATASET SUMMARY
Generate comprehensive summary of flexible organization results
"""

import os
from pathlib import Path

def count_images_in_directory(directory):
    """Count image files in directory recursively"""
    count = 0
    if not directory.exists():
        return 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                count += 1
    return count

def count_classes_in_directory(directory):
    """Count class directories (those containing images)"""
    count = 0
    if not directory.exists():
        return 0

    for root, dirs, files in os.walk(directory):
        has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
                        for f in files)
        if has_images:
            count += 1
    return count

def main():
    dataset_flexible = Path('dataset_flexible')
    main_dataset = dataset_flexible / 'dataset_main'
    low_data_backup = dataset_flexible / 'low_data_backup'
    oversized_backup = dataset_flexible / 'oversized_backup'

    print('FLEXIBLE DATASET ORGANIZATION - FINAL RESULTS')
    print('='*60)

    # Main dataset stats
    main_classes = count_classes_in_directory(main_dataset)
    main_images = count_images_in_directory(main_dataset)
    print(f'Main Dataset (dataset_main):')
    print(f'  Classes: {main_classes}')
    print(f'  Total images: {main_images}')
    print()

    # Low data backup stats
    backup_classes = count_classes_in_directory(low_data_backup)
    backup_images = count_images_in_directory(low_data_backup)
    print(f'Low Data Backup (low_data_backup):')
    print(f'  Classes: {backup_classes}')
    print(f'  Total images: {backup_images}')
    print()

    # Oversized backup stats
    oversized_classes = count_classes_in_directory(oversized_backup)
    oversized_images = count_images_in_directory(oversized_backup)
    print(f'Oversized Backup (oversized_backup):')
    print(f'  Classes: {oversized_classes}')
    print(f'  Total images: {oversized_images}')
    print()

    # Summary
    total_classes = main_classes + backup_classes + oversized_classes
    total_images = main_images + backup_images + oversized_images

    print('STRATEGY SUMMARY:')
    print('='*30)
    print(f'- Total classes organized: {total_classes}')
    print(f'- Total images: {total_images}')
    print()
    print('Distribution:')
    print(f'- Balanced classes (100-3000 images): {main_classes} classes')
    print(f'- Low-data classes (<100 images): {backup_classes} classes (preserved for future)')
    print(f'- Oversized classes (>3000 images): {oversized_classes} classes (reduced to 2000)')
    print()
    print('Benefits:')
    print('- Preserved low-data classes for future dataset expansion')
    print('- Balanced training set ready for immediate model training')
    print('- Reduced oversized classes to prevent bias')
    print('- Maintained hierarchical structure for easy navigation')

if __name__ == '__main__':
    main()