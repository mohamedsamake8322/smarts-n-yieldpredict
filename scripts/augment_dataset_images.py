import random
import uuid
import json
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# =========================
# CONFIGURATION
# =========================
logging.basicConfig(level=logging.WARNING)

random.seed(42)

INPUT_DATASET = r"C:\Downloads\Plantdataset_organized"
OUTPUT_DATASET = r"C:\Downloads\Plantdataset_balanced"

TARGET_PER_CLASS = 200  # Ajustable (200 recommandé)

# =========================
# TRANSFORMATIONS
# =========================
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.RandomResizedCrop((1024, 1024), scale=(0.8, 1.0)),
    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])

# =========================
# UTILITAIRES
# =========================
def load_valid_images(folder):
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images.extend(folder.glob(ext))

    valid = []
    for img_path in images:
        if img_path.name.startswith('._'):
            continue
        try:
            with Image.open(img_path) as img:
                img.convert('RGB')
            valid.append(img_path)
        except Exception:
            logging.warning(f"Image invalide ignorée: {img_path}")

    return valid

# =========================
# TRAITEMENT D'UNE CLASSE
# =========================
def process_class(input_folder, output_folder, global_metadata):
    output_folder.mkdir(parents=True, exist_ok=True)

    images = load_valid_images(input_folder)
    count = len(images)

    if count == 0:
        return

    # -------------------------
    # CAS 1 : Trop d'images → Downsampling
    # -------------------------
    if count > TARGET_PER_CLASS:
        selected_images = random.sample(images, TARGET_PER_CLASS)

        for img_path in selected_images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((1024, 1024))
                    img.save(output_folder / img_path.name, quality=95)

                global_metadata.append({
                    "filename": img_path.name,
                    "origin": str(img_path),
                    "augmented": False
                })

            except Exception as e:
                logging.error(f"Erreur copie: {img_path} | {e}")

    # -------------------------
    # CAS 2 : Pas assez → Augmentation
    # -------------------------
    else:
        # Copier originaux
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((1024, 1024))
                    img.save(output_folder / img_path.name, quality=95)

                global_metadata.append({
                    "filename": img_path.name,
                    "origin": str(img_path),
                    "augmented": False
                })

            except Exception as e:
                logging.error(f"Erreur copie: {img_path} | {e}")

        # Générer images augmentées
        needed = TARGET_PER_CLASS - count

        for i in range(needed):
            img_path = random.choice(images)

            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((1024, 1024))
                    augmented_img = augmentation_transforms(img)

                unique_id = uuid.uuid4().hex[:8]
                new_name = f"{img_path.stem}_aug_{unique_id}{img_path.suffix}"

                augmented_img.save(output_folder / new_name, quality=95)

                global_metadata.append({
                    "filename": new_name,
                    "origin": str(img_path),
                    "augmented": True
                })

            except Exception as e:
                logging.error(f"Erreur augmentation: {img_path} | {e}")

# =========================
# PIPELINE PRINCIPAL
# =========================
def balance_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        logging.error("Le dossier d'entrée n'existe pas.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    global_metadata = []

    for folder in input_root.rglob('*'):
        if folder.is_dir() and not any(sub.is_dir() for sub in folder.iterdir()):

            relative_path = folder.relative_to(input_root)
            output_folder = output_root / relative_path

            process_class(folder, output_folder, global_metadata)

    # Sauvegarde metadata globale
    with open(output_root / "dataset_metadata.json", "w") as f:
        json.dump(global_metadata, f, indent=4)

    print("✅ Dataset équilibré terminé.")

# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    balance_dataset(INPUT_DATASET, OUTPUT_DATASET)