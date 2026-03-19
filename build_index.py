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

from config import (
    BASE_PATH,
    MOH_DIR,
    SWIN_MODEL_PATH,
    SWIN_FAISS_INDEX,
)

try:
    from build_moh_index import build_index as build_faiss_index
except ImportError:
    build_faiss_index = None

def check_indexing_requirements():
    """Vérifier les prérequis pour l'indexation."""
    checks = {
        "Modèle Swin entraîné": os.path.exists(SWIN_MODEL_PATH),
        "Données Plantwise (Moh)": os.path.exists(MOH_DIR),
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
        # Construction de l'index FAISS (MOH / Plantwise)
        if build_faiss_index is None:
            print("❌ build_moh_index.build_index non disponible")
            sys.exit(1)
        print("📊 Construction de l'index FAISS MOH (Plantwise)...")
        build_faiss_index()
        print("✅ Index FAISS construit avec succès!")

        # Normalisation BLIP-2
        print("\n🔄 Normalisation des données BLIP-2...")
        os.system("python normalize_blip2.py")
        print("✅ Données BLIP-2 normalisées!")

        print("\n🎉 Indexation terminée avec succès!")
        print("📁 Fichiers générés:")
        print("   - moh_index.faiss")
        print("   - moh_metadata.json")
        print("   - BLIP2_normalized/ (après normalize_blip2)")

    except Exception as e:
        print(f"\n❌ Erreur lors de l'indexation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()