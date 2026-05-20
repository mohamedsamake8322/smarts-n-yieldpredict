#!/usr/bin/env python3
"""
SCRIPT DE LANCEMENT RAPIDE POUR LA FUSION DES DOUBLONS

Utilise le fichier merge_config.ini pour la configuration.
Lance automatiquement tout le pipeline: analyse → sauvegarde → fusion → vérification.
"""

import os
import sys
import configparser
import argparse

def load_config(config_file):
    """Charge la configuration depuis le fichier INI"""
    if not os.path.exists(config_file):
        print(f"❌ Fichier de configuration introuvable: {config_file}")
        print("Créez d'abord le fichier merge_config.ini")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    return config

def run_pipeline(config, dry_run=False, skip_backup=False):
    """Exécute le pipeline complet de fusion"""

    dataset_path = config.get('PATHS', 'dataset_path', fallback='')
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"❌ Chemin du dataset invalide: {dataset_path}")
        print("Modifiez dataset_path dans merge_config.ini")
        sys.exit(1)

    print("🚀 LANCEMENT DU PIPELINE DE FUSION")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Mode: {'SIMULATION' if dry_run else 'RÉEL'}")
    print(f"Sauvegarde: {'NON' if skip_backup else 'OUI'}")
    print("=" * 60)

    try:
        # Étape 1: Analyse
        print("\n1️⃣ ANALYSE DES DOUBLONS...")
        from analyze_duplicates import analyze_duplicates
        analyze_duplicates(dataset_path)

        # Étape 2: Sauvegarde (si activée)
        if not skip_backup and config.getboolean('BACKUP', 'always_backup', fallback=True):
            print("\n2️⃣ CRÉATION DE LA SAUVEGARDE...")
            from backup_dataset import create_backup, verify_backup
            backup_path = config.get('PATHS', 'backup_path', fallback='')
            if not backup_path:
                backup_path = os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_backup")

            if create_backup(dataset_path, backup_path):
                verify_backup(dataset_path, backup_path)
            else:
                print("❌ Échec de la sauvegarde. Arrêt du pipeline.")
                return False

        # Étape 3: Fusion
        print("\n3️⃣ FUSION DES DOUBLONS...")
        from merge_plant_dataset import analyze_dataset, merge_duplicates
        df, merge_plan = analyze_dataset(dataset_path)

        if not merge_plan:
            print("✅ Aucun doublon trouvé - fusion ignorée")
        else:
            empty_folders_dir = config.get('PATHS', 'empty_folders_dir', fallback='empty_folders')
            merge_duplicates(dataset_path, merge_plan, empty_folders_dir, dry_run=dry_run)

        # Étape 4: Vérification
        print("\n4️⃣ VÉRIFICATION FINALE...")
        from verify_merge import verify_merge_results
        original_analysis = os.path.join(dataset_path, 'duplicate_analysis.csv')
        verify_merge_results(dataset_path, original_analysis)

        print("\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        return True

    except Exception as e:
        print(f"\n❌ ERREUR DANS LE PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_status(config):
    """Affiche l'état actuel du dataset"""
    dataset_path = config.get('PATHS', 'dataset_path', fallback='')

    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset non trouvé")
        return

    print("📊 ÉTAT ACTUEL DU DATASET")
    print("=" * 40)

    folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Dossiers: {len(folders)}")

    total_images = 0
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        try:
            images = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            total_images += images
        except:
            pass

    print(f"Images: {total_images}")

    # Vérifier les fichiers de rapport
    reports = ['duplicate_analysis.csv', 'merge_verification.csv', 'dataset_health_report.txt']
    print("\nRapports disponibles:")
    for report in reports:
        report_path = os.path.join(dataset_path, report)
        if os.path.exists(report_path):
            print(f"  ✅ {report}")
        else:
            print(f"  ❌ {report}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de fusion des doublons")
    parser.add_argument('--config', default='merge_config.ini', help='Fichier de configuration')
    parser.add_argument('--dry-run', action='store_true', help='Mode simulation')
    parser.add_argument('--skip-backup', action='store_true', help='Ignorer la sauvegarde')
    parser.add_argument('--status', action='store_true', help='Afficher l\'état du dataset')

    args = parser.parse_args()

    # Charger la configuration
    config = load_config(args.config)

    if args.status:
        show_status(config)
        return

    # Exécuter le pipeline
    success = run_pipeline(config, dry_run=args.dry_run, skip_backup=args.skip_backup)

    if success:
        print("\n💡 PROCHAINES ÉTAPES:")
        print("   1. Vérifiez les rapports générés")
        print("   2. Validez la qualité du dataset fusionné")
        print("   3. Supprimez la sauvegarde si tout est OK")
    else:
        print("\n💡 EN CAS DE PROBLÈME:")
        print("   1. Vérifiez les messages d'erreur ci-dessus")
        print("   2. Restaurez depuis la sauvegarde si nécessaire")
        print("   3. Contactez le support si besoin")

if __name__ == "__main__":
    main()