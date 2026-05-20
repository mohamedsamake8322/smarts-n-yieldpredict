import os
import shutil
from pathlib import Path

def create_backup(dataset_path, backup_path=None):
    """
    Crée une sauvegarde du dataset avant la fusion
    """
    if backup_path is None:
        backup_path = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_backup")

    print("🔄 Création de la sauvegarde...")
    print(f"Source: {dataset_path}")
    print(f"Destination: {backup_path}")

    try:
        # Créer le dossier de sauvegarde
        os.makedirs(backup_path, exist_ok=True)

        # Copier tous les dossiers
        folders_copied = 0
        total_files = 0

        for item in os.listdir(dataset_path):
            src_path = os.path.join(dataset_path, item)
            dst_path = os.path.join(backup_path, item)

            if os.path.isdir(src_path):
                print(f"   📁 Copie: {item}")
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                folders_copied += 1

                # Compter les fichiers
                for root, dirs, files in os.walk(src_path):
                    total_files += len(files)

        print("
✅ Sauvegarde terminée!"        print(f"   • Dossiers sauvegardés: {folders_copied}")
        print(f"   • Fichiers sauvegardés: {total_files}")
        print(f"   • Emplacement: {backup_path}")

        return True

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return False

def verify_backup(original_path, backup_path):
    """
    Vérifie que la sauvegarde est complète
    """
    print("\n🔍 Vérification de la sauvegarde...")

    original_folders = [d for d in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, d))]
    backup_folders = [d for d in os.listdir(backup_path) if os.path.isdir(os.path.join(backup_path, d))]

    print(f"   • Dossiers originaux: {len(original_folders)}")
    print(f"   • Dossiers sauvegardés: {len(backup_folders)}")

    # Vérifier que tous les dossiers sont présents
    missing_folders = set(original_folders) - set(backup_folders)
    if missing_folders:
        print(f"   ⚠️ Dossiers manquants: {missing_folders}")
        return False

    # Compter les fichiers dans chaque dossier
    total_original = 0
    total_backup = 0

    for folder in original_folders:
        orig_folder = os.path.join(original_path, folder)
        back_folder = os.path.join(backup_path, folder)

        orig_files = len([f for f in os.listdir(orig_folder) if os.path.isfile(os.path.join(orig_folder, f))])
        back_files = len([f for f in os.listdir(back_folder) if os.path.isfile(os.path.join(back_folder, f))])

        total_original += orig_files
        total_backup += back_files

        if orig_files != back_files:
            print(f"   ⚠️ Nombre de fichiers différent pour {folder}: {orig_files} vs {back_files}")

    print(f"   • Fichiers originaux: {total_original}")
    print(f"   • Fichiers sauvegardés: {total_backup}")

    if total_original == total_backup:
        print("   ✅ Sauvegarde vérifiée avec succès!")
        return True
    else:
        print("   ❌ Incohérence dans la sauvegarde!")
        return False

def main():
    # Chemin du dataset (à adapter selon votre environnement)
    dataset_path = r"C:\path\to\your\Plantdataset"  # MODIFIEZ CE CHEMIN

    if not os.path.exists(dataset_path):
        print(f"❌ Chemin du dataset introuvable: {dataset_path}")
        print("Veuillez modifier la variable dataset_path dans le script")
        return

    print("🛡️ SYSTÈME DE SAUVEGARDE POUR FUSION DE DATASET")
    print("=" * 60)

    # Créer la sauvegarde
    backup_path = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_backup")
    success = create_backup(dataset_path, backup_path)

    if success:
        # Vérifier la sauvegarde
        verify_backup(dataset_path, backup_path)

        print("
📋 INSTRUCTIONS:"        print("   1. Vérifiez que la sauvegarde est complète"        print("   2. Lancez ensuite le script de fusion: merge_duplicate_folders.py"        print("   3. Si quelque chose se passe mal, restaurez depuis la sauvegarde"        print("
🔄 Pour restaurer: copier le contenu de la sauvegarde vers le dataset original"    else:
        print("\n❌ Échec de la sauvegarde. Arrêt du processus.")

if __name__ == "__main__":
    main()