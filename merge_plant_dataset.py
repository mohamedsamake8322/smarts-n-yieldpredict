#!/usr/bin/env python3
"""
SCRIPT PRINCIPAL POUR LA FUSION DES DOUBLONS DANS LE DATASET PLANTDATASET

Ce script orchestre tout le processus de fusion des dossiers dupliqués:
1. Analyse des doublons
2. Création d'une sauvegarde
3. Fusion des dossiers dupliqués
4. Nettoyage des dossiers vides

UTILISATION:
    python merge_plant_dataset.py --path "C:\path\to\Plantdataset"

OPTIONS:
    --path: Chemin vers le dossier Plantdataset
    --no-backup: Ne pas créer de sauvegarde (dangereux!)
    --dry-run: Simuler la fusion sans effectuer les changements
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import pandas as pd
import re

def normalize_name(name):
    """Normalise un nom de dossier pour identifier les doublons"""
    normalized = name.lower()
    normalized = re.sub(r'[_]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)

    replacements = {
        'bacterial spot': 'bacterial spot',
        'leaf spot': 'leaf spot',
        'leaf blight': 'leaf blight',
        'late blight': 'late blight',
        'brown rust': 'brown rust',
        'yellow rust': 'yellow rust',
        'fusarium head blight': 'fusarium head blight',
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

def analyze_dataset(dataset_path):
    """Analyse le dataset et retourne les informations sur les doublons"""
    print("🔍 ANALYSE DU DATASET...")

    subfolders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"   📁 {len(subfolders)} dossiers trouvés")

    folder_data = []
    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        try:
            image_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            normalized = normalize_name(folder)
            folder_data.append({
                'original_name': folder,
                'normalized_name': normalized,
                'image_count': image_count
            })
        except Exception as e:
            print(f"   ⚠️ Erreur avec {folder}: {e}")

    df = pd.DataFrame(folder_data)

    # Grouper par nom normalisé
    grouped = df.groupby('normalized_name')
    duplicate_groups = []
    merge_plan = []

    for normalized_name, group in grouped:
        if len(group) > 1:
            duplicate_groups.append(group)
            # Trier par nombre d'images (descendant)
            sorted_group = group.sort_values('image_count', ascending=False)
            canonical = sorted_group.iloc[0]['original_name']
            duplicates = sorted_group.iloc[1:]['original_name'].tolist()

            merge_plan.append({
                'canonical_folder': canonical,
                'duplicate_folders': duplicates,
                'total_images': sorted_group['image_count'].sum()
            })

    print(f"   🔄 {len(duplicate_groups)} groupes de doublons identifiés")
    return df, merge_plan

def create_backup(dataset_path, backup_path):
    """Crée une sauvegarde du dataset"""
    print("🛡️ CRÉATION DE LA SAUVEGARDE...")

    try:
        os.makedirs(backup_path, exist_ok=True)
        folders_copied = 0
        total_files = 0

        for item in os.listdir(dataset_path):
            src_path = os.path.join(dataset_path, item)
            dst_path = os.path.join(backup_path, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                folders_copied += 1
                total_files += sum(1 for root, dirs, files in os.walk(src_path) for file in files)

        print(f"   ✅ Sauvegarde créée: {folders_copied} dossiers, {total_files} fichiers")
        return True

    except Exception as e:
        print(f"   ❌ Erreur sauvegarde: {e}")
        return False

def merge_duplicates(dataset_path, merge_plan, empty_folders_path="empty_folders", dry_run=False):
    """Fusionne les dossiers dupliqués"""
    print("🔄 FUSION DES DOUBLONS...")

    if dry_run:
        print("   🔍 MODE SIMULATION - Aucun changement réel")

    empty_folders_full_path = os.path.join(dataset_path, empty_folders_path)
    if not dry_run:
        os.makedirs(empty_folders_full_path, exist_ok=True)

    total_moved = 0
    total_merged = 0

    for plan in merge_plan:
        canonical_folder = plan['canonical_folder']
        duplicate_folders = plan['duplicate_folders']

        canonical_path = os.path.join(dataset_path, canonical_folder)
        print(f"\n   📁 Fusion vers: {canonical_folder}")

        for duplicate_folder in duplicate_folders:
            duplicate_path = os.path.join(dataset_path, duplicate_folder)

            if not os.path.exists(duplicate_path):
                print(f"     ⚠️ Dossier manquant: {duplicate_folder}")
                continue

            # Compter les images
            duplicate_images = len([f for f in os.listdir(duplicate_path) if os.path.isfile(os.path.join(duplicate_path, f))])
            print(f"     📸 Déplacement de {duplicate_images} images depuis {duplicate_folder}")

            if not dry_run:
                moved = 0
                for filename in os.listdir(duplicate_path):
                    src_file = os.path.join(duplicate_path, filename)
                    dst_file = os.path.join(canonical_path, filename)

                    if os.path.isfile(src_file):
                        # Gérer les conflits de noms
                        counter = 1
                        base_name, ext = os.path.splitext(filename)
                        original_dst = dst_file
                        while os.path.exists(dst_file):
                            dst_file = os.path.join(canonical_path, f"{base_name}_{counter}{ext}")
                            counter += 1

                        shutil.move(src_file, dst_file)
                        moved += 1

                # Déplacer le dossier vide
                remaining_files = os.listdir(duplicate_path)
                if not remaining_files:
                    empty_dst = os.path.join(empty_folders_full_path, duplicate_folder)
                    shutil.move(duplicate_path, empty_dst)
                    print(f"     🗂️ Dossier vide archivé: {duplicate_folder}")
                else:
                    print(f"     ⚠️ Dossier {duplicate_folder} contient encore {len(remaining_files)} éléments")

                total_moved += moved
                total_merged += 1

    if not dry_run:
        print("
✅ FUSION TERMINÉE!"        print(f"   • Dossiers fusionnés: {total_merged}")
        print(f"   • Images déplacées: {total_moved}")
        print(f"   • Dossiers vides archivés dans: {empty_folders_path}/")
    else:
        print("
🔍 SIMULATION TERMINÉE"        print(f"   • Dossiers à fusionner: {total_merged}")
        print(f"   • Images à déplacer: {total_moved}")

def main():
    parser = argparse.ArgumentParser(description="Fusion des doublons dans le dataset Plantdataset")
    parser.add_argument('--path', required=True, help='Chemin vers le dossier Plantdataset')
    parser.add_argument('--no-backup', action='store_true', help='Ne pas créer de sauvegarde (dangereux!)')
    parser.add_argument('--dry-run', action='store_true', help='Simuler sans effectuer les changements')

    args = parser.parse_args()

    dataset_path = args.path

    if not os.path.exists(dataset_path):
        print(f"❌ Chemin introuvable: {dataset_path}")
        sys.exit(1)

    print("🌱 FUSION DES DOUBLONS - DATASET PLANTDATASET")
    print("=" * 60)
    print(f"Chemin: {dataset_path}")
    print(f"Sauvegarde: {'NON' if args.no_backup else 'OUI'}")
    print(f"Mode: {'SIMULATION' if args.dry_run else 'RÉEL'}")
    print("=" * 60)

    # Analyse
    df, merge_plan = analyze_dataset(dataset_path)

    if not merge_plan:
        print("✅ Aucun doublon trouvé!")
        return

    # Sauvegarde
    if not args.no_backup and not args.dry_run:
        backup_path = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_backup")
        if not create_backup(dataset_path, backup_path):
            print("❌ Échec de la sauvegarde. Arrêt.")
            sys.exit(1)

    # Fusion
    merge_duplicates(dataset_path, merge_plan, dry_run=args.dry_run)

    # Rapport final
    print("\n📊 RAPPORT FINAL:")
    final_folders = len([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"   • Dossiers restants: {final_folders}")

    if not args.dry_run:
        empty_folders_path = os.path.join(dataset_path, "empty_folders")
        if os.path.exists(empty_folders_path):
            empty_count = len([d for d in os.listdir(empty_folders_path) if os.path.isdir(os.path.join(empty_folders_path, d))])
            print(f"   • Dossiers vides archivés: {empty_count}")

    print("\n🎉 Processus terminé!")

if __name__ == "__main__":
    main()