import os
import shutil
from pathlib import Path

dataset_path = r"C:\Users\moham\Videos\Moh"
small_classes_path = os.path.join(dataset_path, "Small_Classes")
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

print("🔍 IDENTIFICATION DES CLASSES AVEC MOINS DE 300 IMAGES\n")

# Créer le dossier Small_Classes s'il n'existe pas
if not os.path.exists(small_classes_path):
    os.makedirs(small_classes_path)
    print(f"📁 Dossier créé: {small_classes_path}\n")

small_classes = []

# Parcourir tous les dossiers
for item in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, item)

    # Ignorer les fichiers et le dossier Small_Classes lui-même
    if not os.path.isdir(folder_path) or item == "Small_Classes":
        continue

    # Compter les images
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        image_count += len([
            file for file in files
            if file.lower().endswith(image_extensions)
        ])

    if image_count < 300:
        small_classes.append((item, image_count))
        print(f"📦 Petite classe: {item} ({image_count} images)")

print(f"\n📊 Total classes identifiées: {len(small_classes)}\n")

# Déplacer les petites classes
moved_count = 0
for class_name, count in small_classes:
    src_path = os.path.join(dataset_path, class_name)
    dst_path = os.path.join(small_classes_path, class_name)

    try:
        shutil.move(src_path, dst_path)
        print(f"✅ Déplacé: {class_name} ({count} images)")
        moved_count += 1
    except Exception as e:
        print(f"❌ Erreur déplacement {class_name}: {e}")

print(f"\n🎯 RÉSUMÉ: {moved_count}/{len(small_classes)} classes déplacées vers Small_Classes")
print(f"📂 Dossier: {small_classes_path}")