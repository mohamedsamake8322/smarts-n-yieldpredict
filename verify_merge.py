import os
import pandas as pd
from pathlib import Path

def verify_merge_results(dataset_path, original_analysis_file=None):
    """
    Vérifie les résultats de la fusion des doublons
    """
    print("🔍 VÉRIFICATION DES RÉSULTATS DE FUSION")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"❌ Chemin introuvable: {dataset_path}")
        return

    # Analyser l'état actuel
    current_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Dossiers actuels: {len(current_folders)}")

    # Compter les images par dossier
    folder_stats = []
    total_images = 0

    for folder in current_folders:
        folder_path = os.path.join(dataset_path, folder)
        try:
            image_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            folder_stats.append({
                'folder': folder,
                'images': image_count
            })
            total_images += image_count
        except Exception as e:
            print(f"⚠️ Erreur avec {folder}: {e}")

    # Statistiques
    df_stats = pd.DataFrame(folder_stats)
    print("
📊 STATISTIQUES:"    print(f"   • Total d'images: {total_images}")
    print(f"   • Dossiers avec images: {len(df_stats[df_stats['images'] > 0])}")
    print(f"   • Dossiers vides: {len(df_stats[df_stats['images'] == 0])}")

    if len(df_stats) > 0:
        print(f"   • Images moyennes par dossier: {df_stats['images'].mean():.1f}")
        print(f"   • Dossier le plus rempli: {df_stats.loc[df_stats['images'].idxmax(), 'folder']} ({df_stats['images'].max()} images)")
        print(f"   • Dossier le moins rempli: {df_stats.loc[df_stats['images'].idxmin(), 'folder']} ({df_stats['images'].min()} images)")

    # Vérifier les dossiers vides archivés
    empty_folders_path = os.path.join(dataset_path, "empty_folders")
    if os.path.exists(empty_folders_path):
        archived_empty = [d for d in os.listdir(empty_folders_path) if os.path.isdir(os.path.join(empty_folders_path, d))]
        print(f"   • Dossiers vides archivés: {len(archived_empty)}")

        # Vérifier qu'ils sont vraiment vides
        actually_empty = 0
        for folder in archived_empty:
            folder_path = os.path.join(empty_folders_path, folder)
            if len(os.listdir(folder_path)) == 0:
                actually_empty += 1

        if actually_empty == len(archived_empty):
            print("   ✅ Tous les dossiers archivés sont vides")
        else:
            print(f"   ⚠️ {len(archived_empty) - actually_empty} dossiers archivés contiennent encore des fichiers")

    # Comparer avec l'analyse originale si disponible
    if original_analysis_file and os.path.exists(original_analysis_file):
        print("
📈 COMPARAISON AVEC L'ANALYSE ORIGINALE:"        try:
            original_df = pd.read_csv(original_analysis_file)
            original_total = original_df['image_count'].sum()
            print(f"   • Images avant fusion: {original_total}")
            print(f"   • Images après fusion: {total_images}")
            print(f"   • Différence: {total_images - original_total}")

            if total_images == original_total:
                print("   ✅ Nombre d'images conservé")
            else:
                print("   ⚠️ Nombre d'images différent!")

        except Exception as e:
            print(f"   ⚠️ Erreur lecture fichier original: {e}")

    # Sauvegarder les statistiques
    output_file = os.path.join(dataset_path, 'merge_verification.csv')
    df_stats.to_csv(output_file, index=False)
    print(f"\n💾 Statistiques sauvegardées: {output_file}")

    # Générer un rapport de santé
    health_report = generate_health_report(df_stats, dataset_path)
    report_file = os.path.join(dataset_path, 'dataset_health_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(health_report)
    print(f"💾 Rapport de santé: {report_file}")

    print("\n✅ Vérification terminée!")

def generate_health_report(df_stats, dataset_path):
    """Génère un rapport de santé du dataset"""
    report = []
    report.append("RAPPORT DE SANTÉ DU DATASET APRES FUSION")
    report.append("=" * 50)
    report.append("")

    # Statistiques générales
    total_images = df_stats['images'].sum()
    total_folders = len(df_stats)
    empty_folders = len(df_stats[df_stats['images'] == 0])

    report.append("STATISTIQUES GÉNÉRALES:")
    report.append(f"  • Dossiers totaux: {total_folders}")
    report.append(f"  • Images totales: {total_images}")
    report.append(f"  • Dossiers vides: {empty_folders}")
    report.append(f"  • Dossiers avec images: {total_folders - empty_folders}")
    report.append("")

    # Distribution des images
    if total_folders > 0:
        avg_images = df_stats['images'].mean()
        median_images = df_stats['images'].median()
        max_images = df_stats['images'].max()
        min_images = df_stats['images'].min()

        report.append("DISTRIBUTION DES IMAGES:")
        report.append(f"  • Moyenne par dossier: {avg_images:.1f}")
        report.append(f"  • Médiane par dossier: {median_images:.1f}")
        report.append(f"  • Maximum: {max_images}")
        report.append(f"  • Minimum: {min_images}")
        report.append("")

    # Classes les plus représentées
    if len(df_stats[df_stats['images'] > 0]) > 0:
        top_classes = df_stats[df_stats['images'] > 0].nlargest(10, 'images')
        report.append("TOP 10 CLASSES LES PLUS REPRÉSENTÉES:")
        for i, (_, row) in enumerate(top_classes.iterrows(), 1):
            report.append(f"  {i:2d}. {row['folder']:<40} {row['images']:>4d} images")
        report.append("")

    # Classes sous-représentées
    low_classes = df_stats[(df_stats['images'] > 0) & (df_stats['images'] < 10)]
    if len(low_classes) > 0:
        report.append("CLASSES SOUS-REPRÉSENTÉES (< 10 images):")
        for _, row in low_classes.iterrows():
            report.append(f"  • {row['folder']}: {row['images']} images")
        report.append("")

    # Vérifications d'intégrité
    report.append("VÉRIFICATIONS D'INTÉGRITÉ:")
    integrity_issues = []

    # Vérifier les noms de fichiers dupliqués dans le même dossier
    duplicate_files_found = False
    for _, row in df_stats.iterrows():
        if row['images'] > 0:
            folder_path = os.path.join(dataset_path, row['folder'])
            try:
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                if len(files) != len(set(files)):
                    duplicate_files_found = True
                    integrity_issues.append(f"Noms de fichiers dupliqués dans {row['folder']}")
            except:
                integrity_issues.append(f"Erreur accès dossier {row['folder']}")

    if not duplicate_files_found:
        report.append("  ✅ Aucun nom de fichier dupliqué détecté")
    else:
        report.append("  ⚠️ Noms de fichiers dupliqués détectés")

    if integrity_issues:
        report.append("  ⚠️ Problèmes d'intégrité:")
        for issue in integrity_issues:
            report.append(f"     • {issue}")
    else:
        report.append("  ✅ Aucune anomalie d'intégrité détectée")

    report.append("")
    report.append("RECOMMANDATIONS:")
    if empty_folders > 0:
        report.append(f"  • Considérer supprimer ou archiver {empty_folders} dossiers vides")
    if len(low_classes) > 0:
        report.append(f"  • {len(low_classes)} classes ont moins de 10 images - considérer augmentation données")
    if avg_images < 50:
        report.append("  • Dataset semble petit - considérer augmentation données pour meilleur entraînement")

    return "\n".join(report)

def main():
    # Chemin du dataset (à adapter selon votre environnement)
    dataset_path = r"C:\path\to\your\Plantdataset"  # MODIFIEZ CE CHEMIN
    original_analysis = os.path.join(dataset_path, 'duplicate_analysis.csv')

    verify_merge_results(dataset_path, original_analysis)

if __name__ == "__main__":
    main()