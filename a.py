import os
import shutil

# Dossier principal contenant les sous-dossiers
source_folder = r"C:\Downloads\banana_black_leaf_streak_banana black sigatoka (14)-20260630T192415Z-3-001"

# Dossier où toutes les images seront regroupées
destination_folder = os.path.join(source_folder, "all_images")
os.makedirs(destination_folder, exist_ok=True)

# Extensions d'images acceptées
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

count = 0

for root, dirs, files in os.walk(source_folder):
    # Ne pas parcourir le dossier de destination
    if os.path.abspath(root) == os.path.abspath(destination_folder):
        continue

    for file in files:
        if file.lower().endswith(image_extensions):
            source_path = os.path.join(root, file)

            # Éviter les doublons de noms
            base_name, ext = os.path.splitext(file)
            destination_path = os.path.join(destination_folder, file)

            i = 1
            while os.path.exists(destination_path):
                destination_path = os.path.join(
                    destination_folder,
                    f"{base_name}_{i}{ext}"
                )
                i += 1

            shutil.copy2(source_path, destination_path)
            count += 1

print(f"Terminé ! {count} images copiées dans :")
print(destination_folder)