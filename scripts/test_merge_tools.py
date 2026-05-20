#!/usr/bin/env python3
"""
SCRIPT DE TEST POUR LES OUTILS DE FUSION DE DOUBLONS

Ce script crée un petit dataset de test avec des doublons simulés
pour valider le fonctionnement des scripts de fusion.
"""

import os
import shutil
from pathlib import Path

def create_test_dataset(base_path):
    """Crée un petit dataset de test avec des doublons"""
    print("🧪 CRÉATION DU DATASET DE TEST...")

    test_path = os.path.join(base_path, "test_plant_dataset")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)

    # Créer des dossiers avec des doublons simulés
    test_folders = {
        # Groupe 1: Bacterial Spot
        "Bacterial Spot": ["image1.jpg", "image2.jpg", "image3.jpg"],
        "Bacterial_Spot": ["image4.jpg", "image5.jpg"],
        "bacterial spot (variant)": ["image6.jpg"],

        # Groupe 2: Leaf Blight
        "Leaf Blight": ["leaf1.jpg", "leaf2.jpg"],
        "leaf_blight": ["leaf3.jpg", "leaf4.jpg"],
        "Leaf Blight Disease": ["leaf5.jpg"],

        # Groupe 3: Healthy (pas de doublon)
        "Healthy": ["healthy1.jpg", "healthy2.jpg", "healthy3.jpg", "healthy4.jpg"],

        # Groupe 4: Empty folder
        "Empty Folder": [],

        # Groupe 5: Another duplicate
        "Powdery Mildew": ["mildew1.jpg"],
        "powdery_mildew": ["mildew2.jpg", "mildew3.jpg"],
    }

    for folder_name, images in test_folders.items():
        folder_path = os.path.join(test_path, folder_name)
        os.makedirs(folder_path)

        # Créer des fichiers vides simulés
        for image in images:
            image_path = os.path.join(folder_path, image)
            with open(image_path, 'w') as f:
                f.write("")  # Fichier vide

    print(f"   ✅ Dataset de test créé: {test_path}")
    print(f"   📁 {len(test_folders)} dossiers créés")

    # Compter les images
    total_images = sum(len(images) for images in test_folders.values())
    print(f"   📸 {total_images} images simulées")

    return test_path

def run_test_pipeline(test_path):
    """Exécute tous les scripts de test"""
    print("\n🧪 EXÉCUTION DU PIPELINE DE TEST...")

    # Test 1: Analyse des doublons
    print("\n1️⃣ TEST: Analyse des doublons")
    try:
        # Simuler l'exécution d'analyse
        from analyze_duplicates import analyze_duplicates
        analyze_duplicates(test_path)
        print("   ✅ Analyse réussie")
    except Exception as e:
        print(f"   ❌ Erreur analyse: {e}")

    # Test 2: Sauvegarde
    print("\n2️⃣ TEST: Sauvegarde")
    try:
        from backup_dataset import create_backup, verify_backup
        backup_path = os.path.join(os.path.dirname(test_path), "test_backup")
        if create_backup(test_path, backup_path):
            if verify_backup(test_path, backup_path):
                print("   ✅ Sauvegarde réussie")
            else:
                print("   ❌ Vérification sauvegarde échouée")
        else:
            print("   ❌ Création sauvegarde échouée")
    except Exception as e:
        print(f"   ❌ Erreur sauvegarde: {e}")

    # Test 3: Fusion (mode simulation)
    print("\n3️⃣ TEST: Fusion (simulation)")
    try:
        from merge_plant_dataset import analyze_dataset, merge_duplicates
        df, merge_plan = analyze_dataset(test_path)
        merge_duplicates(test_path, merge_plan, dry_run=True)
        print("   ✅ Simulation de fusion réussie")
    except Exception as e:
        print(f"   ❌ Erreur fusion: {e}")

    # Test 4: Vérification
    print("\n4️⃣ TEST: Vérification")
    try:
        from verify_merge import verify_merge_results
        verify_merge_results(test_path)
        print("   ✅ Vérification réussie")
    except Exception as e:
        print(f"   ❌ Erreur vérification: {e}")

def cleanup_test(test_path):
    """Nettoie le dataset de test"""
    print("\n🧹 NETTOYAGE DU TEST...")

    try:
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
            print("   ✅ Dataset de test supprimé")

        backup_path = os.path.join(os.path.dirname(test_path), "test_backup")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
            print("   ✅ Sauvegarde de test supprimée")

    except Exception as e:
        print(f"   ❌ Erreur nettoyage: {e}")

def main():
    print("🧪 TESTS DES SCRIPTS DE FUSION DE DOUBLONS")
    print("=" * 60)

    # Chemin de base pour les tests
    base_path = os.getcwd()  # Utilise le répertoire courant

    try:
        # Créer le dataset de test
        test_path = create_test_dataset(base_path)

        # Exécuter les tests
        run_test_pipeline(test_path)

        print("\n🎉 TESTS TERMINÉS!")
        print("Vérifiez les fichiers générés dans le dossier de test.")

    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")

    finally:
        # Demander avant nettoyage
        response = input("\nVoulez-vous supprimer le dataset de test? (y/n): ")
        if response.lower() in ['y', 'yes', 'oui']:
            cleanup_test(test_path)
        else:
            print(f"Dataset de test conservé: {test_path}")

if __name__ == "__main__":
    main()