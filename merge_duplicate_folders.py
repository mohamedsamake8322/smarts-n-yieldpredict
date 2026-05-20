import os
import shutil
import pandas as pd
from pathlib import Path
import re

def normalize_name(name):
    """
    Normalise un nom de dossier pour identifier les doublons
    """
    # Convertir en minuscules
    normalized = name.lower()

    # Remplacer les underscores et espaces par des espaces simples
    normalized = re.sub(r'[_]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Supprimer les parenthèses et leur contenu
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)

    # Normaliser les termes communs
    replacements = {
        'bacterial spot': 'bacterial spot',
        'leaf spot': 'leaf spot',
        'leaf blight': 'leaf blight',
        'late blight': 'late blight',
        'brown rust': 'brown rust',
        'yellow rust': 'yellow rust',
        'fusarium head blight': 'fusarium head blight',
        'leaf blight': 'leaf blight',
        'cercospora leaf spot': 'cercospora leaf spot',
        'brown blight': 'brown blight',
        'red leaf spot': 'red leaf spot',
        'gray leaf spot': 'gray leaf spot',
        'verticillium wilt': 'verticillium wilt',
        'healthy': 'healthy',
        'curl virus': 'curl virus',
        'mosaic virus': 'mosaic virus',
        'northern leaf blight': 'northern leaf blight',
        'black rot': 'black rot',
        'leaf scorch': 'leaf scorch',
        'powdery mildew': 'powdery mildew',
        'downy mildew': 'downy mildew',
        'anthracnose': 'anthracnose',
        'bacterial blight': 'bacterial blight',
        'septoria': 'septoria',
        'target spot': 'target spot',
        'spider mite': 'spider mite',
        'yellow leaf curl virus': 'yellow leaf curl virus',
        'apple scab': 'apple scab',
        'black rot': 'black rot',
        'cedar apple rust': 'cedar apple rust',
        'common rust': 'common rust',
        'early blight': 'early blight',
        'leaf blast': 'leaf blast',
        'hispa': 'hispa',
        'citrus greening': 'citrus greening',
        'smut': 'smut',
        'blast': 'blast',
        'mildew': 'mildew',
        'greening': 'greening',
        'canker': 'canker',
        'scab': 'scab',
        'rot': 'rot',
        'spot': 'spot',
        'blight': 'blight',
        'rust': 'rust',
        'wilt': 'wilt',
        'virus': 'virus',
        'mite': 'mite',
        'beetle': 'beetle',
        'aphid': 'aphid',
        'bug': 'bug',
        'worm': 'worm',
        'fly': 'fly',
        'borer': 'borer',
        'roller': 'roller',
        'planthopper': 'planthopper',
        'weevil': 'weevil',
        'whitefly': 'whitefly',
        'looper': 'looper',
        'moth': 'moth',
        'caterpillar': 'caterpillar',
        'grasshopper': 'grasshopper',
        'ant': 'ant',
        'bee': 'bee',
        'earwig': 'earwig',
        'slug': 'slug',
        'snail': 'snail',
        'wasp': 'wasp',
        'thrips': 'thrips',
        'flea beetle': 'flea beetle',
        'fall armyworm': 'fall armyworm',
        'root knot nematode': 'root knot nematode'
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized.strip()

def create_merge_mapping(dataset_path):
    """
    Crée un mapping de fusion basé sur les noms normalisés
    """
    merge_actions = {}

    if os.path.exists(dataset_path):
        subfolders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        # Grouper par nom normalisé
        normalized_groups = {}
        for folder in subfolders:
            normalized = normalize_name(folder)
            if normalized not in normalized_groups:
                normalized_groups[normalized] = []
            normalized_groups[normalized].append(folder)

        # Créer le mapping de fusion
        for normalized, folders in normalized_groups.items():
            if len(folders) > 1:
                # Trier par nombre d'images (descendant) pour choisir le dossier principal
                folder_info = []
                for folder in folders:
                    folder_path = os.path.join(dataset_path, folder)
                    image_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                    folder_info.append((folder, image_count))

                # Trier par nombre d'images descendant
                folder_info.sort(key=lambda x: x[1], reverse=True)

                # Le premier dossier avec le plus d'images devient le canonique
                canonical_folder = folder_info[0][0]

                # Les autres seront fusionnés dans le canonique
                for folder, _ in folder_info[1:]:
                    if canonical_folder not in merge_actions:
                        merge_actions[canonical_folder] = []
                    merge_actions[canonical_folder].append(folder)

    return merge_actions

def merge_duplicate_folders(dataset_path, empty_folders_path="empty_folders"):
    """
    Fusionne les dossiers dupliqués en déplaçant les fichiers
    """
    print("🔍 Analyse du dataset pour identifier les doublons...")
    merge_actions = create_merge_mapping(dataset_path)

    if not merge_actions:
        print("✅ Aucun doublon trouvé!")
        return

    print(f"📋 {len(merge_actions)} groupes de doublons identifiés pour fusion")

    # Créer le dossier pour les dossiers vides
    empty_folders_full_path = os.path.join(dataset_path, empty_folders_path)
    os.makedirs(empty_folders_full_path, exist_ok=True)

    total_moved = 0
    total_merged = 0

    for canonical_folder, duplicate_folders in merge_actions.items():
        canonical_path = os.path.join(dataset_path, canonical_folder)

        print(f"\n🔄 Fusion vers: {canonical_folder}")
        print(f"   Dossiers à fusionner: {duplicate_folders}")

        # Traiter chaque dossier dupliqué
        for duplicate_folder in duplicate_folders:
            duplicate_path = os.path.join(dataset_path, duplicate_folder)

            if not os.path.exists(duplicate_path):
                print(f"   ⚠️ Dossier manquant: {duplicate_folder}")
                continue

            # Compter les images avant déplacement
            images_before = len([f for f in os.listdir(canonical_path) if os.path.isfile(os.path.join(canonical_path, f))])
            duplicate_images = len([f for f in os.listdir(duplicate_path) if os.path.isfile(os.path.join(duplicate_path, f))])

            print(f"   📸 Déplacement de {duplicate_images} images depuis {duplicate_folder}")

            # Déplacer tous les fichiers
            moved = 0
            for filename in os.listdir(duplicate_path):
                src_file = os.path.join(duplicate_path, filename)
                dst_file = os.path.join(canonical_path, filename)

                if os.path.isfile(src_file):
                    # Gérer les conflits de noms de fichiers
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(dst_file):
                        dst_file = os.path.join(canonical_path, f"{base_name}_{counter}{ext}")
                        counter += 1

                    shutil.move(src_file, dst_file)
                    moved += 1

            # Vérifier si le dossier est maintenant vide
            remaining_files = os.listdir(duplicate_path)
            if not remaining_files:
                # Déplacer le dossier vide vers empty_folders
                empty_dst = os.path.join(empty_folders_full_path, duplicate_folder)
                shutil.move(duplicate_path, empty_dst)
                print(f"   🗂️ Dossier vide déplacé: {duplicate_folder} → {empty_folders_path}/")
            else:
                print(f"   ⚠️ Dossier {duplicate_folder} contient encore {len(remaining_files)} éléments")

            total_moved += moved
            total_merged += 1

            # Vérifier le résultat
            images_after = len([f for f in os.listdir(canonical_path) if os.path.isfile(os.path.join(canonical_path, f))])
            print(f"   ✅ {canonical_folder}: {images_before} → {images_after} images")

    print("
📊 RÉSULTATS DE LA FUSION:"    print(f"   • Groupes fusionnés: {len(merge_actions)}")
    print(f"   • Dossiers dupliqués traités: {total_merged}")
    print(f"   • Images déplacées: {total_moved}")
    print(f"   • Dossiers vides déplacés vers: {empty_folders_path}/")

    # Vérification finale
    print("
🔍 VÉRIFICATION FINALE:"    final_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d != empty_folders_path]
    print(f"   • Dossiers restants dans le dataset: {len(final_folders)}")

    empty_count = len([d for d in os.listdir(empty_folders_full_path) if os.path.isdir(os.path.join(empty_folders_full_path, d))])
    print(f"   • Dossiers vides archivés: {empty_count}")

    print("\n✅ Fusion des doublons terminée!")

def main():
    # Chemin du dataset (à adapter selon votre environnement)
    dataset_path = r"C:\path\to\your\Plantdataset"  # MODIFIEZ CE CHEMIN

    if not os.path.exists(dataset_path):
        print(f"❌ Chemin du dataset introuvable: {dataset_path}")
        print("Veuillez modifier la variable dataset_path dans le script")
        return

    print(f"🔧 Fusion des dossiers dupliqués dans: {dataset_path}")
    print("=" * 60)

    merge_duplicate_folders(dataset_path)

if __name__ == "__main__":
    main()