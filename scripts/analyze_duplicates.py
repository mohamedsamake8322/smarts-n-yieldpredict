import os
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

def analyze_duplicates(dataset_path):
    """
    Analyse les doublons dans le dataset et crée un rapport détaillé
    """
    print("🔍 Analyse des doublons dans le dataset...")
    print(f"Chemin: {dataset_path}")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"❌ Chemin introuvable: {dataset_path}")
        return

    # Lister tous les sous-dossiers
    subfolders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Total des sous-dossiers trouvés: {len(subfolders)}")

    # Analyser chaque dossier
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
            print(f"⚠️ Erreur avec {folder}: {e}")

    # Créer un DataFrame
    df = pd.DataFrame(folder_data)

    # Grouper par nom normalisé
    grouped = df.groupby('normalized_name')

    print("
📊 ANALYSE DES DOUBLONS:"    duplicate_groups = []
    total_duplicates = 0
    total_duplicate_images = 0

    for normalized_name, group in grouped:
        if len(group) > 1:
            duplicate_groups.append(group)
            total_duplicates += len(group) - 1  # Nombre de doublons (sans compter l'original)
            total_duplicate_images += group['image_count'].sum()

            print(f"\n🔄 Groupe '{normalized_name}' ({len(group)} dossiers):")
            for _, row in group.iterrows():
                print(f"   • {row['original_name']} ({row['image_count']} images)")

    print("
📈 STATISTIQUES:"    print(f"   • Groupes de doublons: {len(duplicate_groups)}")
    print(f"   • Dossiers dupliqués: {total_duplicates}")
    print(f"   • Images dans les doublons: {total_duplicate_images}")
    print(f"   • Dossiers uniques: {len(subfolders) - total_duplicates}")

    # Sauvegarder les résultats
    output_file = os.path.join(dataset_path, 'duplicate_analysis.csv')
    df.to_csv(output_file, index=False)
    print(f"\n💾 Analyse sauvegardée: {output_file}")

    # Créer un résumé des actions de fusion
    if duplicate_groups:
        merge_summary = []
        for group_df in duplicate_groups:
            # Trier par nombre d'images (descendant)
            sorted_group = group_df.sort_values('image_count', ascending=False)
            canonical = sorted_group.iloc[0]['original_name']
            duplicates = sorted_group.iloc[1:]['original_name'].tolist()

            merge_summary.append({
                'canonical_folder': canonical,
                'duplicate_folders': duplicates,
                'total_images': sorted_group['image_count'].sum(),
                'canonical_images': sorted_group.iloc[0]['image_count']
            })

        merge_df = pd.DataFrame(merge_summary)
        merge_file = os.path.join(dataset_path, 'merge_plan.csv')
        merge_df.to_csv(merge_file, index=False)
        print(f"💾 Plan de fusion sauvegardé: {merge_file}")

        print("
🔧 PLAN DE FUSION RECOMMANDÉ:"        for _, row in merge_df.iterrows():
            print(f"   📁 {row['canonical_folder']} ← fusionner {len(row['duplicate_folders'])} dossiers ({row['total_images']} images total)")

    print("\n✅ Analyse terminée!")

def main():
    # Chemin du dataset (à adapter selon votre environnement)
    dataset_path = r"C:\path\to\your\Plantdataset"  # MODIFIEZ CE CHEMIN

    analyze_duplicates(dataset_path)

if __name__ == "__main__":
    main()