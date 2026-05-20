"""
🎯 Dataset Subset Creator for Kaggle
=====================================
Crée un sous-dataset à partir de votre dataset complet sur Drive.
Utile pour tester avec 80GB+ sans tout télécharger.

Usage:
1. Téléchargez quelques classes depuis Drive
2. Créez un sous-dataset équilibré
3. Utilisez ce sous-dataset pour l'entraînement
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import shutil
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================
# Votre Drive folder ID (remplacez par le vôtre)
DRIVE_FOLDER_ID = "YOUR_DRIVE_FOLDER_ID"

# Classes à inclure dans le sous-dataset (commencez petit)
SUBSET_CLASSES = [
    # Maladies communes et équilibrées
    "Apple_Black_Rot", "Apple_Healthy", "Apple_Scab",
    "Banana_Healthy", "Banana_Sigatoka_Leaf_Spot",
    "Corn_Common_Rust", "Corn_Healthy", "Corn_Gray_Leaf_Spot",
    "Tomato_Early_Blight", "Tomato_Healthy", "Tomato_Late_Blight",
    "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
    "Grape_Black_Rot", "Grape_Healthy", "Grape_Downy_Mildew",
    "Rice_Bacterial_Blight", "Rice_Healthy", "Rice_Brown_Spot",
    # Ajoutez d'autres classes selon vos besoins
]

# Configuration
DOWNLOAD_DIR = Path('/kaggle/working/full_dataset')
SUBSET_DIR = Path('/kaggle/working/subset_dataset')
BATCH_SIZE = 10  # Télécharge 10 classes à la fois

# ============================================================================
# FONCTIONS
# ============================================================================
def download_classes_from_drive(drive_folder_id, classes_to_download, download_dir, batch_size=10):
    """Télécharge seulement certaines classes depuis Drive."""
    import gdown
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Téléchargement de {len(classes_to_download)} classes depuis Drive...")
    print(f"   Batch size: {batch_size}")
    
    downloaded = []
    
    for i in range(0, len(classes_to_download), batch_size):
        batch = classes_to_download[i:i+batch_size]
        print(f"\n📦 Batch {i//batch_size + 1}: {batch}")
        
        for class_name in batch:
            try:
                # Pour chaque classe, téléchargez le dossier
                # Note: gdown peut avoir du mal avec les sous-dossiers
                # Alternative: Téléchargez le ZIP de chaque classe si préparé
                
                # Simulation - remplacez par votre logique
                class_url = f"https://drive.google.com/uc?id=CLASS_{class_name}_ID"
                class_dir = download_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                # Ici vous devriez avoir l'ID spécifique de chaque classe
                # Pour l'instant, on simule
                print(f"   📥 Téléchargement {class_name}...")
                # gdown.download_folder(class_url, output=str(class_dir), quiet=True)
                
                downloaded.append(class_name)
                
            except Exception as e:
                print(f"   ❌ Erreur {class_name}: {e}")
    
    return downloaded

def create_balanced_subset(source_dir, subset_classes, output_dir, target_samples_per_class=500):
    """Crée un sous-dataset équilibré."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n⚖️  Création sous-dataset équilibré...")
    print(f"   Source: {source_dir}")
    print(f"   Destination: {output_dir}")
    print(f"   Target samples/class: {target_samples_per_class}")
    
    total_copied = 0
    
    for class_name in subset_classes:
        src_class_dir = source_dir / class_name
        dst_class_dir = output_dir / class_name
        
        if not src_class_dir.exists():
            print(f"   ⚠️  Classe {class_name} non trouvée, ignorée")
            continue
        
        # Liste des images
        images = list(src_class_dir.glob('*.[jJ][pP][gG]')) + list(src_class_dir.glob('*.[pP][nN][gG]'))
        
        if not images:
            print(f"   ⚠️  Aucune image dans {class_name}")
            continue
        
        # Équilibre : prends au plus target_samples_per_class
        selected_images = images[:target_samples_per_class]
        
        # Copie
        dst_class_dir.mkdir(exist_ok=True)
        for img_path in selected_images:
            shutil.copy2(img_path, dst_class_dir / img_path.name)
        
        print(f"   ✅ {class_name}: {len(selected_images)} images copiées")
        total_copied += len(selected_images)
    
    print(f"\n✅ Sous-dataset créé: {total_copied} images total")
    return total_copied

def analyze_subset(subset_dir):
    """Analyse le sous-dataset créé."""
    class_counts = {}
    
    for class_dir in subset_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.[jJ][pP][gG]'))) + len(list(class_dir.glob('*.[pP][nN][gG]')))
            class_counts[class_dir.name] = count
    
    print(f"\n📊 Analyse du sous-dataset:")
    print(f"   Classes: {len(class_counts)}")
    print(f"   Total images: {sum(class_counts.values())}")
    print(f"   Min/Max: {min(class_counts.values())} / {max(class_counts.values())}")
    
    return class_counts

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("🎯 Dataset Subset Creator for Kaggle")
    print("=" * 50)
    
    # Étape 1: Téléchargement
    if DRIVE_FOLDER_ID != "YOUR_DRIVE_FOLDER_ID":
        downloaded_classes = download_classes_from_drive(
            DRIVE_FOLDER_ID, SUBSET_CLASSES, DOWNLOAD_DIR, BATCH_SIZE
        )
        print(f"✅ Téléchargé: {len(downloaded_classes)} classes")
    else:
        print("⚠️  Configurez DRIVE_FOLDER_ID pour le téléchargement")
        # Simulation pour test
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        downloaded_classes = SUBSET_CLASSES
    
    # Étape 2: Création du sous-dataset
    total_images = create_balanced_subset(
        DOWNLOAD_DIR, downloaded_classes, SUBSET_DIR, target_samples_per_class=500
    )
    
    # Étape 3: Analyse
    class_counts = analyze_subset(SUBSET_DIR)
    
    # Sauvegarde de la configuration
    config = {
        "subset_classes": downloaded_classes,
        "class_counts": class_counts,
        "total_images": total_images,
        "source": str(DOWNLOAD_DIR),
        "output": str(SUBSET_DIR)
    }
    
    with open('/kaggle/working/subset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration sauvegardée: subset_config.json")
    print(f"\n🚀 Prêt pour l'entraînement avec {total_images} images !")
    
    # Instructions finales
    print(f"\n📋 Prochaines étapes:")
    print(f"   1. Utilisez SUBSET_DIR dans votre script d'entraînement")
    print(f"   2. DATA_DIR = Path('{SUBSET_DIR}')")
    print(f"   3. Lancez l'entraînement !")