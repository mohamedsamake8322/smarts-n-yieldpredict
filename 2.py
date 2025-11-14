import os
import shutil
import random

train_dir = r"C:\Users\moham\Pictures\New Plant Diseases Dataset(Augmented)\train"
valid_dir = r"C:\Users\moham\Pictures\New Plant Diseases Dataset(Augmented)\valid"

# Ratio d’images envoyées dans valid
val_ratio = 0.2

# Choisir si on déplace (True) ou copie (False)
MOVE_FILES = False

def safe_copy_or_move(src_file, dst_dir):
    """
    Copie ou déplace un fichier en évitant d'écraser les doublons.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    file = os.path.basename(src_file)
    dst_file = os.path.join(dst_dir, file)

    # éviter d'écraser un fichier déjà existant
    if os.path.exists(dst_file):
        base, ext = os.path.splitext(file)
        i = 1
        while os.path.exists(os.path.join(dst_dir, f"{base}_{i}{ext}")):
            i += 1
        dst_file = os.path.join(dst_dir, f"{base}_{i}{ext}")

    if MOVE_FILES:
        shutil.move(src_file, dst_file)
    else:
        shutil.copy2(src_file, dst_file)


def main():
    for class_name in os.listdir(train_dir):
        train_class_path = os.path.join(train_dir, class_name)
        valid_class_path = os.path.join(valid_dir, class_name)

        if os.path.isdir(train_class_path):
            if not os.path.exists(valid_class_path):
                print(f"Classe manquante détectée dans valid : {class_name}")
                os.makedirs(valid_class_path)

                # Prendre les images du train
                images = [os.path.join(train_class_path, f) for f in os.listdir(train_class_path) if os.path.isfile(os.path.join(train_class_path, f))]
                random.shuffle(images)

                # Sélectionner 20%
                val_count = max(1, int(len(images) * val_ratio))
                val_images = images[:val_count]

                for img in val_images:
                    safe_copy_or_move(img, valid_class_path)

            else:
                print(f"Classe déjà existante dans valid : {class_name}")

    print("Complétion de valid terminée ✅")


if __name__ == "__main__":
    main()
