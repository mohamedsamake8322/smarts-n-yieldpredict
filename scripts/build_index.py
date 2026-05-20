#!/usr/bin/env python3
"""
SMART AGRICULTURE - Module d'Indexation FAISS
Script spécialisé pour la construction de l'index FAISS
"""

import os
import sys
from pathlib import Path

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from modules.agricultural_assistant import build_faiss_index

def check_indexing_requirements():
    """Vérifier les prérequis pour l'indexation."""
    checks = {
        "Modèle Swin entraîné": os.path.exists("models/swin_base_patch4_window7_224.pth"),
        "Données Plantwise": os.path.exists("data/disease_info.json"),
        "Scripts d'indexation": os.path.exists("build_moh_index.py"),
        "Configuration": os.path.exists("config.py")
    }

    print("🔍 Vérification des prérequis d'indexation:")
    all_ok = True
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check}: {status}")
        if not status:
            all_ok = False

    return all_ok

def main():
    """Fonction principale d'indexation."""
    print("="*80)
    print("🔍 SMART AGRICULTURE - MODULE D'INDEXATION FAISS")
    print("Construction spécialisée de l'index de recherche vectorielle")
    print("="*80)

    # Vérifications préalables
    if not check_indexing_requirements():
        print("❌ Prérequis non satisfaits. Corrigez les problèmes ci-dessus.")
        sys.exit(1)

    print("\n🚀 Lancement de l'indexation FAISS...")

    try:
        # Construction de l'index FAISS
        print("📊 Construction de l'index FAISS (1115 entrées Plantwise)...")
        build_faiss_index()
        print("✅ Index FAISS construit avec succès!")

        # Normalisation BLIP-2
        print("\n🔄 Normalisation des données BLIP-2...")
        os.system("python normalize_blip2.py")
        print("✅ Données BLIP-2 normalisées!")

        print("\n🎉 Indexation terminée avec succès!")
        print("📁 Fichiers générés:")
        print("   - models/faiss_index.bin")
        print("   - models/metadata.json")
        print("   - data/blip2_normalized/")

    except Exception as e:
        print(f"\n❌ Erreur lors de l'indexation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()