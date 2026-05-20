import os
from pathlib import Path

target = Path(r'C:\Downloads\Plantdataset_organized')

if not target.exists():
    raise FileNotFoundError(f"Dossier non trouvé : {target}")

# calcul récursif

total_images = 0

# build structure
for crop in sorted(target.iterdir()):
    if not crop.is_dir():
        continue
    crop_images = 0
    print(crop.name)
    for primary in sorted(crop.iterdir()):
        if not primary.is_dir():
            continue
        primary_images = 0
        print(f"  {primary.name}")
        for sub in sorted(primary.iterdir()):
            if not sub.is_dir():
                continue
            count = sum(1 for f in sub.iterdir() if f.is_file())
            primary_images += count
            print(f"    {sub.name} ({count} images)")
        print(f"    --> {primary.name} total: {primary_images} images")
        crop_images += primary_images
    print(f"  == {crop.name} total: {crop_images} images")
    total_images += crop_images
    print()

print(f"TOTAL images in {target}: {total_images}")
