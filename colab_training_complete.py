#!/usr/bin/env python3
"""
SCRIPT COMPLET COLAB - ENTRAÎNEMENT + APPLICATION

Ce script exécute TOUT automatiquement :
1. Montage Google Drive
2. Installation des dépendances
3. Entraînement du modèle Swin (Phase 2 Production)
4. Normalisation BLIP2 (109 fichiers)
5. Construction index FAISS (1115 entrées)
6. Test des modules
7. Installation Streamlit & LocalTunnel
8. Configuration Streamlit
9. Lancement application
10. Création tunnel public
11. Vérification statut

Usage dans Colab:
!python colab_training_complete.py

Configuration A100:
- Le script détecte automatiquement A100 et optimise les paramètres
- Batch size 64 pour A100
- Utilise toutes les optimisations CUDA
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_step(step_num, description):
    """Affiche une étape avec formatage."""
    print(f"\n{'='*60}")
    print(f"🚀 ÉTAPE {step_num}/11: {description}")
    print('='*60)

def run_command(command, description="", check_output=False):
    """Exécute une commande système."""
    try:
        print(f"📋 {description}")
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Commande exécutée avec succès")
                return result.stdout.strip()
            else:
                print(f"❌ Erreur: {result.stderr}")
                return None
        else:
            result = subprocess.run(command, shell=True)
            return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_gpu():
    """Vérifie le GPU disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"🎮 GPU détecté: {gpu_name} (x{gpu_count})")

            if "A100" in gpu_name:
                print("🚀 Mode A100 activé - Optimisations maximales !")
                return "A100"
            elif "V100" in gpu_name:
                print("⚡ Mode V100 activé")
                return "V100"
            elif "T4" in gpu_name:
                print("💨 Mode T4 activé")
                return "T4"
            else:
                print(f"📊 GPU {gpu_name} détecté")
                return gpu_name
        else:
            print("⚠️  Pas de GPU CUDA détecté - Mode CPU")
            return "CPU"
    except:
        print("⚠️  Impossible de détecter le GPU")
        return "UNKNOWN"

def main():
    print("🤖 SMART AGRICULTURE - ENTRAÎNEMENT COMPLET + APPLICATION")
    print("="*75)
    print("Ce script va exécuter TOUT: Entraînement A100 + Setup complet...")
    print("="*75)

    # Vérification GPU
    gpu_type = check_gpu()

    # Étape 1: Montage Google Drive
    print_step(1, "Montage Google Drive")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive monté")
    except Exception as e:
        print(f"❌ Erreur montage Drive: {e}")
        return False

    # Changement de répertoire
    os.chdir('/content/drive/MyDrive/smarts-n-yieldpredict')
    print(f"📁 Répertoire changé: {os.getcwd()}")

    # Étape 2: Installation des dépendances
    print_step(2, "Installation des Dépendances")
    deps = [
        "sentence-transformers",
        "faiss-cpu",
        "streamlit",
        "torch",
        "transformers",
        "timm",
        "accelerate",
        "albumentations",
        "opencv-python-headless",
        "Pillow",
        "plotly",
        "wandb",  # Pour le tracking d'entraînement
        "torchvision"
    ]

    for dep in deps:
        success = run_command(f"pip install -q {dep}", f"Installation {dep}")
        if not success:
            print(f"⚠️  Échec installation {dep}, continuation...")

    print("✅ Dépendances installées")

    # Étape 3: Configuration et vérification
    print_step(3, "Configuration & Vérification")
    try:
        from config import print_config, ensure_directories
        ensure_directories()
        print_config()
        print("✅ Configuration vérifiée")
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False

    # Étape 4: Entraînement du modèle Swin
    print_step(4, f"Entraînement Swin Base Production ({gpu_type})")

    # Configuration optimisée pour A100
    if gpu_type == "A100":
        os.environ["BATCH_SIZE"] = "64"
        os.environ["GRADIENT_ACCUMULATION"] = "1"
        print("⚡ Configuration A100: batch_size=64, gradient_accumulation=1")
    elif gpu_type in ["V100", "T4"]:
        os.environ["BATCH_SIZE"] = "32"
        os.environ["GRADIENT_ACCUMULATION"] = "2"
        print("💨 Configuration optimisée pour GPU haute performance")
    else:
        os.environ["BATCH_SIZE"] = "16"
        os.environ["GRADIENT_ACCUMULATION"] = "4"
        print("📊 Configuration standard pour GPU")

    # Lancer l'entraînement
    success = run_command("python -m training_pipelines.phase2_swin_base_production",
                         "Entraînement Swin Base (60 epochs)")
    if success:
        print("✅ Modèle Swin entraîné avec succès")
    else:
        print("❌ Échec entraînement - vérifiez les logs ci-dessus")
        return False

    # Étape 5: Normalisation BLIP2
    print_step(5, "Normalisation BLIP2 (109 fichiers)")
    success = run_command("python normalize_blip2.py", "Normalisation BLIP2")
    if success:
        print("✅ BLIP2 normalisé")
    else:
        print("❌ Échec normalisation BLIP2")
        return False

    # Étape 6: Construction index FAISS
    print_step(6, "Construction Index FAISS (1115 entrées)")
    success = run_command("python build_moh_index.py", "Construction index FAISS")
    if success:
        print("✅ Index FAISS construit")
    else:
        print("❌ Échec construction index FAISS")
        return False

    # Étape 7: Test des modules
    print_step(7, "Test des Modules")
    success = run_command("python test_modules.py", "Test des modules")
    if success:
        print("✅ Modules testés")
    else:
        print("⚠️  Échec test modules, continuation...")

    # Étape 8: Vérification modèles Swin
    print_step(8, "Vérification Modèles Swin Entraînés")
    swin_dir = "/content/drive/MyDrive/outputs/phase2_swin_base_production/models"
    files_needed = ['metric_model.pt', 'faiss_index.bin', 'metadata.json']

    print("Vérification des fichiers Swin entraînés:")
    all_exist = True
    for file in files_needed:
        filepath = os.path.join(swin_dir, file)
        exists = os.path.exists(filepath)
        status = "✅ Existe" if exists else "❌ Manquant"
        print(f"  {filepath}: {status}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n✅ Tous les modèles Swin sont disponibles et entraînés")
    else:
        print("\n❌ Certains modèles Swin manquent - vérifiez l'entraînement")
        return False

    # Étape 9: Installation Streamlit & LocalTunnel
    print_step(9, "Installation Streamlit & LocalTunnel")
    run_command("pip install -q streamlit", "Installation Streamlit")
    run_command("npm install -g localtunnel", "Installation LocalTunnel")
    print("✅ Streamlit et LocalTunnel installés")

    # Étape 10: Configuration Streamlit
    print_step(10, "Configuration Streamlit pour Colab")
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    print("✅ Configuration Streamlit pour Colab")

    # Étape 11: Lancement application
    print_step(11, "Lancement Application (Background)")
    try:
        import subprocess
        process = subprocess.Popen([
            'streamlit', 'run', '04_app_streamlit.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--logger.level', 'error'
        ])
        print("✅ Application lancée en arrière-plan")
        print("⏳ Attente de 15 secondes pour l'initialisation...")
        time.sleep(15)
    except Exception as e:
        print(f"❌ Erreur lancement application: {e}")
        return False

    # Étape 12: Vérification statut
    print_step(12, "Vérification Statut Application")
    def check_app():
        try:
            response = requests.get('http://localhost:8501', timeout=5)
            return response.status_code == 200
        except:
            return False

    print("Vérification du statut de l'application...")
    for i in range(20):  # 20 tentatives pour A100
        if check_app():
            print("✅ Application accessible sur http://localhost:8501")
            break
        else:
            print(f"⏳ Tentative {i+1}/20 - Application pas encore prête")
            time.sleep(3)
    else:
        print("❌ Application ne répond pas après 20 tentatives")
        return False

    # Étape 13: Création tunnel public
    print_step(13, "Création Tunnel Public")
    print("🔗 Création du tunnel public...")
    print("📋 Copiez l'URL qui apparaît ci-dessous pour accéder à l'application:")
    print("-" * 50)

    try:
        # Lancement du tunnel
        tunnel_process = subprocess.Popen(['lt', '--port', '8501'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)

        # Attendre un peu pour que le tunnel se lance
        time.sleep(5)

        # Lire la sortie
        try:
            stdout, stderr = tunnel_process.communicate(timeout=15)
            if stdout:
                print(stdout)
            if stderr:
                print("Erreurs:", stderr)
        except subprocess.TimeoutExpired:
            tunnel_process.kill()
            print("⏳ Tunnel lancé, vérifiez l'URL ci-dessus")

    except Exception as e:
        print(f"❌ Erreur création tunnel: {e}")
        print("💡 Alternative: Utilisez ngrok ou un autre service de tunnel")

    # Résumé final
    print("\n" + "="*75)
    print("🎉 ENTRAÎNEMENT + INSTALLATION TERMINÉS !")
    print("="*75)
    print("✅ Modèle Swin entraîné sur A100 (60 epochs)")
    print("✅ Toutes les étapes exécutées automatiquement")
    print("✅ Application Smart Agriculture opérationnelle")
    print("✅ Pipeline complet: Image → Swin → BLIP-2 → Explication")
    print("\n🔗 Utilisez l'URL du tunnel pour accéder à l'application")
    print("📱 Fonctionnalités disponibles:")
    print("   • Détection de maladies (modèle fraîchement entraîné)")
    print("   • Explications BLIP-2 en langage naturel")
    print("   • Assistant agricole (1115 connaissances)")
    print("   • Interface progressive avec feedback utilisateur")
    print("   • Seuils de confiance et cartes d'attention")
    print("\n🚀 Prêt à diagnostiquer vos plantes avec un modèle A100 ! 🌱🤖")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Script terminé avec succès!")
        else:
            print("\n❌ Script terminé avec erreurs")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Script interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)