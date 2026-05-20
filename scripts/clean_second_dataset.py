"""
Nettoyage et fusion du second dataset Plant_leave_diseases_dataset_with_augmentation
avec le premier dataset nettoyé
"""
import shutil
from pathlib import Path


# Mapping pour standardiser les noms du second dataset
SECOND_DATASET_MAPPING = {
    # APPLE
    "Apple___Apple_scab": "Apple_Scab",
    "Apple___Black_rot": "Apple_Black_Rot",
    "Apple___Cedar_apple_rust": "Apple_Cedar_Rust",
    "Apple___healthy": "Apple_Healthy",

    # BLUEBERRY
    "Blueberry___healthy": "Blueberry_Healthy",

    # CHERRY
    "Cherry___healthy": "Cherry_Healthy",
    "Cherry___Powdery_mildew": "Cherry_Powdery_Mildew",

    # CORN - FUSION AVEC EXISTANT
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn_Cercospora_Leaf_Spot",
    "Corn___Common_rust": "Corn_Common_Rust",
    "Corn___healthy": "Corn_Healthy",
    "Corn___Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",

    # GRAPE
    "Grape___Black_rot": "Grape_Black_Rot",
    "Grape___Esca_(Black_Measles)": "Grape_Esca",
    "Grape___healthy": "Grape_Healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape_Leaf_Blight",

    # ORANGE - FUSION AVEC EXISTANT
    "Orange___Haunglongbing_(Citrus_greening)": "Citrus_Greening",

    # PEACH
    "Peach___Bacterial_spot": "Peach_Bacterial_Spot",
    "Peach___healthy": "Peach_Healthy",

    # PEPPER - FUSION AVEC EXISTANT
    "Pepper,_bell___Bacterial_spot": "Pepper_Bacterial_Spot",
    "Pepper,_bell___healthy": "Pepper_Healthy",

    # POTATO
    "Potato___Early_blight": "Potato_Early_Blight",
    "Potato___healthy": "Potato_Healthy",
    "Potato___Late_blight": "Potato_Late_Blight",

    # RASPBERRY
    "Raspberry___healthy": "Raspberry_Healthy",

    # SOYBEAN
    "Soybean___healthy": "Soybean_Healthy",

    # SQUASH
    "Squash___Powdery_mildew": "Squash_Powdery_Mildew",

    # STRAWBERRY
    "Strawberry___healthy": "Strawberry_Healthy",
    "Strawberry___Leaf_scorch": "Strawberry_Leaf_Scorch",

    # TOMATO - FUSION AVEC EXISTANT
    "Tomato___Bacterial_spot": "Tomato_Bacterial_Spot",
    "Tomato___Early_blight": "Tomato_Early_Blight",
    "Tomato___healthy": "Tomato_Healthy",
    "Tomato___Late_blight": "Tomato_Late_Blight",
    "Tomato___Leaf_Mold": "Tomato_Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Generic_Red_Spider_Mite",
    "Tomato___Target_Spot": "Tomato_Target_Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato_Mosaic",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl",

    # AUTRES
    "Background_without_leaves": None,  # À ignorer
}


def clean_second_dataset(second_dataset_path, first_dataset_path):
    """
    Nettoyer le second dataset et le fusionner avec le premier
    """
    second_path = Path(second_dataset_path)
    first_path = Path(first_dataset_path)

    print("🚀 NETTOYAGE DU SECOND DATASET\n" + "=" * 80)

    # Statistiques
    processed = 0
    merged = 0
    ignored = 0

    for old_name, new_name in SECOND_DATASET_MAPPING.items():
        old_path = second_path / old_name

        if not old_path.exists():
            print(f"⚠️  {old_name} (non trouvé)")
            continue

        if new_name is None:
            # Ignorer ce dossier
            print(f"⏭️  {old_name} (ignoré)")
            ignored += 1
            continue

        # Vérifier si la classe existe déjà dans le premier dataset
        target_path = first_path / new_name

        if target_path.exists():
            # FUSION : ajouter les images au dossier existant
            print(f"🔗 Fusion: {old_name} → {new_name}")
            images = list(old_path.glob("*.*"))
            print(f"  Images à fusionner: {len(images)}")
            for img in images:
                try:
                    dst = target_path / img.name
                    if dst.exists():
                        # Renommer si conflit
                        import time
                        dst = target_path / f"{img.stem}_{int(time.time())}{img.suffix}"
                    shutil.move(str(img), str(dst))
                    merged += 1
                except Exception as e:
                    print(f"  Erreur: {e}")

            # Supprimer le dossier vide
            try:
                old_path.rmdir()
                print(f"  Dossier source supprimé")
            except Exception as e:
                print(f"  Impossible de supprimer {old_path}: {e}")

        else:
            # NOUVELLE CLASSE : renommer simplement
            print(f"✨ Nouveau: {old_name} → {new_name}")
            try:
                shutil.move(str(old_path), str(target_path))
                processed += 1
            except Exception as e:
                print(f"  Erreur: {e}")

        processed += 1

    print("\n" + "=" * 80)
    print(f"✅ TRAITEMENT TERMINÉ")
    print(f"Dossiers traités: {processed}")
    print(f"Fusions effectuées: {merged}")
    print(f"Dossiers ignorés: {ignored}")

    return processed, merged, ignored


if __name__ == "__main__":
    clean_second_dataset(
        r"C:\smarts-n-yieldpredict.git\Plant_leave_diseases_dataset_with_augmentation",
        r"C:\smarts-n-yieldpredict.git\Data traiter_cleaned"
    )
