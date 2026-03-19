#!/usr/bin/env python3
"""
SMART AGRICULTURE - Module d'Application
Script spécialisé pour le lancement de l'application Streamlit
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import SWIN_MODEL_PATH, SWIN_FAISS_INDEX

def check_app_requirements():
    """Vérifier les prérequis pour l'application."""
    checks = {
        "Modèle Swin entraîné": os.path.exists(SWIN_MODEL_PATH),
        "Index FAISS (phase2)": os.path.exists(SWIN_FAISS_INDEX),
        "Application Streamlit": os.path.exists("04_app_streamlit.py"),
        "Données BLIP-2": os.path.exists("data/blip2_normalized"),
        "Configuration": os.path.exists("config.py")
    }

    print("🔍 Vérification des prérequis d'application:")
    all_ok = True
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check}: {status}")
        if not status:
            all_ok = False

    return all_ok

def create_tunnel(port=8501):
    """Créer un tunnel public avec fallback."""
    print("\n🌐 Création du tunnel public...")

    # Essayer LocalTunnel d'abord
    try:
        print("🔄 Tentative LocalTunnel...")
        tunnel_process = subprocess.Popen(
            ['lt', '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(5)

        stdout, stderr = tunnel_process.communicate(timeout=10)
        if stdout and 'https://' in stdout:
            print("✅ LocalTunnel réussi!")
            print(stdout.strip())
            return tunnel_process

    except Exception as e:
        print(f"⚠️ LocalTunnel échoué: {e}")

    # Fallback vers Ngrok (nécessite configuration)
    try:
        print("🔄 Tentative Ngrok...")
        # Note: Nécessite 'pip install pyngrok' et configuration du token
        from pyngrok import ngrok
        # ngrok.set_auth_token("YOUR_TOKEN_HERE")  # À configurer

        public_url = ngrok.connect(port)
        print("✅ Ngrok réussi!")
        print(f"🌐 URL: {public_url}")
        return "ngrok"

    except Exception as e:
        print(f"⚠️ Ngrok échoué: {e}")
        print("💡 Configurez votre token Ngrok ou utilisez un autre service de tunnel")

    print("✅ Application accessible localement sur http://localhost:8501")
    return None

def start_application(port=8501):
    """Démarrer l'application Streamlit."""
    print("\n🚀 Démarrage de l'application Smart Agriculture...")

    # Variables d'environnement
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = str(port)
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    # Lancement de l'application
    process = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', '04_app_streamlit.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0',
        '--logger.level', 'error'
    ])

    # Attente du démarrage
    print("⏳ Attente du démarrage de l'application...")
    for i in range(30):
        try:
            response = requests.get(f'http://localhost:{port}', timeout=2)
            if response.status_code == 200:
                print("✅ Application démarrée avec succès!")
                return process
        except:
            pass

        time.sleep(2)
        print(f"⏳ Tentative {i+1}/30...")

    print("❌ Échec du démarrage de l'application")
    process.terminate()
    return None

def main():
    """Fonction principale de l'application."""
    print("="*80)
    print("🌱 SMART AGRICULTURE - MODULE D'APPLICATION")
    print("Lancement spécialisé de l'application de diagnostic")
    print("="*80)

    # Vérifications préalables
    if not check_app_requirements():
        print("❌ Prérequis non satisfaits. Corrigez les problèmes ci-dessus.")
        sys.exit(1)

    port = 8501

    # Démarrage de l'application
    app_process = start_application(port)
    if not app_process:
        sys.exit(1)

    # Création du tunnel
    tunnel = create_tunnel(port)

    print("\n" + "="*60)
    print("🎉 APPLICATION OPÉRATIONNELLE!")
    print("="*60)
    print("🌐 Accès local: http://localhost:8501")
    if tunnel and tunnel != "ngrok":
        print("🌐 Tunnel actif (vérifiez l'URL ci-dessus)")
    print("\n💡 Pour arrêter: Ctrl+C")
    print("📱 Ouvrez l'URL dans votre navigateur pour diagnostiquer!")

    try:
        # Maintenir l'application active
        app_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'application...")
        app_process.terminate()
        if tunnel and hasattr(tunnel, 'terminate'):
            tunnel.terminate()
        print("✅ Application arrêtée")

if __name__ == "__main__":
    main()