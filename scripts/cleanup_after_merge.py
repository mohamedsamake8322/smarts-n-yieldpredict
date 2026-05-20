import os
import shutil
from pathlib import Path

def cleanup_reports(dataset_path, keep_reports=False):
    """
    Nettoie les fichiers de rapport générés par les scripts de fusion
    """
    print("🧹 NETTOYAGE DES RAPPORTS...")

    report_files = [
        'duplicate_analysis.csv',
        'merge_plan.csv',
        'merge_verification.csv',
        'dataset_health_report.txt'
    ]

    cleaned = 0
    for report in report_files:
        report_path = os.path.join(dataset_path, report)
        if os.path.exists(report_path):
            if not keep_reports:
                os.remove(report_path)
                print(f"   🗑️ Supprimé: {report}")
                cleaned += 1
            else:
                print(f"   📄 Conservé: {report}")

    if cleaned > 0:
        print(f"   ✅ {cleaned} rapports nettoyés")
    else:
        print("   ℹ️ Aucun rapport à nettoyer")

def cleanup_backup(dataset_path, confirm_delete=False):
    """
    Nettoie la sauvegarde après validation
    """
    print("\n🛡️ NETTOYAGE DE LA SAUVEGARDE...")

    backup_path = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_backup")

    if not os.path.exists(backup_path):
        print("   ℹ️ Aucune sauvegarde trouvée")
        return False

    print(f"   📁 Sauvegarde trouvée: {backup_path}")

    if not confirm_delete:
        response = input("   Voulez-vous vraiment supprimer la sauvegarde? (y/n): ")
        if response.lower() not in ['y', 'yes', 'oui']:
            print("   ❌ Suppression annulée")
            return False

    try:
        shutil.rmtree(backup_path)
        print("   ✅ Sauvegarde supprimée")
        return True
    except Exception as e:
        print(f"   ❌ Erreur suppression: {e}")
        return False

def cleanup_empty_folders(dataset_path, confirm_delete=False):
    """
    Nettoie le dossier des dossiers vides archivés
    """
    print("\n📂 NETTOYAGE DES DOSSIERS VIDES ARCHIVÉS...")

    empty_folders_path = os.path.join(dataset_path, "empty_folders")

    if not os.path.exists(empty_folders_path):
        print("   ℹ️ Aucun dossier d'archive trouvé")
        return False

    try:
        archived_folders = [d for d in os.listdir(empty_folders_path) if os.path.isdir(os.path.join(empty_folders_path, d))]
        print(f"   📁 {len(archived_folders)} dossiers vides archivés")

        if len(archived_folders) == 0:
            # Supprimer le dossier vide
            os.rmdir(empty_folders_path)
            print("   ✅ Dossier d'archive vide supprimé")
            return True

        if not confirm_delete:
            response = input(f"   Voulez-vous supprimer {len(archived_folders)} dossiers vides archivés? (y/n): ")
            if response.lower() not in ['y', 'yes', 'oui']:
                print("   ❌ Suppression annulée")
                return False

        shutil.rmtree(empty_folders_path)
        print("   ✅ Dossiers vides archivés supprimés")
        return True

    except Exception as e:
        print(f"   ❌ Erreur suppression: {e}")
        return False

def show_disk_usage(dataset_path):
    """
    Affiche l'utilisation disque du dataset
    """
    print("\n💾 UTILISATION DISQUE:")

    try:
        total_size = 0
        total_files = 0

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    total_files += 1
                except:
                    pass

        # Convertir en MB
        size_mb = total_size / (1024 * 1024)

        print(f"   • Taille totale: {size_mb:.2f} MB")
        print(f"   • Nombre de fichiers: {total_files}")

        # Vérifier la sauvegarde
        backup_path = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_backup")
        if os.path.exists(backup_path):
            backup_size = 0
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    try:
                        backup_size += os.path.getsize(os.path.join(root, file))
                    except:
                        pass
            backup_mb = backup_size / (1024 * 1024)
            print(f"   • Taille sauvegarde: {backup_mb:.2f} MB")

    except Exception as e:
        print(f"   ❌ Erreur calcul taille: {e}")

def main():
    print("🧹 UTILITAIRE DE NETTOYAGE POST-FUSION")
    print("=" * 50)

    # Chemin du dataset (à adapter selon votre environnement)
    dataset_path = r"C:\path\to\your\Plantdataset"  # MODIFIEZ CE CHEMIN

    if not os.path.exists(dataset_path):
        print(f"❌ Chemin introuvable: {dataset_path}")
        print("Modifiez la variable dataset_path dans le script")
        return

    print(f"Dataset: {dataset_path}")

    # Afficher l'utilisation disque
    show_disk_usage(dataset_path)

    # Options de nettoyage
    print("\n🔧 OPTIONS DE NETTOYAGE:")

    # 1. Rapports
    keep_reports = input("Conserver les rapports d'analyse? (y/n): ").lower() in ['y', 'yes', 'oui']
    cleanup_reports(dataset_path, keep_reports)

    # 2. Sauvegarde
    if input("Nettoyer la sauvegarde? (y/n): ").lower() in ['y', 'yes', 'oui']:
        cleanup_backup(dataset_path, confirm_delete=True)

    # 3. Dossiers vides
    if input("Nettoyer les dossiers vides archivés? (y/n): ").lower() in ['y', 'yes', 'oui']:
        cleanup_empty_folders(dataset_path, confirm_delete=True)

    print("\n✅ Nettoyage terminé!")

    # Résumé final
    show_disk_usage(dataset_path)

if __name__ == "__main__":
    main()