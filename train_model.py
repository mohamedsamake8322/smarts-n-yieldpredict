#!/usr/bin/env python3
"""
SMART AGRICULTURE - Module d'Entraînement
Script spécialisé pour l'entraînement du modèle Swin Base
"""

import os
import sys
import torch
from pathlib import Path

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from training_pipelines.phase2_swin_base_production import main as train_main

def check_training_requirements():
    """Vérifier les prérequis pour l'entraînement."""
    checks = {
        "GPU disponible": torch.cuda.is_available(),
        "CUDA version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "Mémoire GPU suffisante": False,
        "Données d'entraînement": os.path.exists("dataset_light"),
        "Configuration": os.path.exists("config.py")
    }

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        checks["Mémoire GPU suffisante"] = gpu_mem >= 8  # Au moins 8GB

    print("🔍 Vérification des prérequis d'entraînement:")
    all_ok = True
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check}: {status}")
        if not status:
            all_ok = False

    return all_ok

def setup_training_environment():
    """Configurer l'environnement d'entraînement."""
    print("\n⚙️ Configuration de l'environnement d'entraînement...")

    # Optimisations GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU détecté: {gpu_name}")

        if "A100" in gpu_name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("🚀 Optimisations A100 activées")
        elif "V100" in gpu_name or "T4" in gpu_name:
            torch.backends.cudnn.benchmark = True
            print("⚡ Optimisations GPU haute performance activées")

    # Variables d'environnement
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    print("✅ Environnement configuré")

def main():
    """Fonction principale d'entraînement."""
    print("="*80)
    print("🤖 SMART AGRICULTURE - MODULE D'ENTRAÎNEMENT")
    print("Entraînement spécialisé du modèle Swin Base")
    print("="*80)

    # Vérifications préalables
    if not check_training_requirements():
        print("❌ Prérequis non satisfaits. Corrigez les problèmes ci-dessus.")
        sys.exit(1)

    # Configuration
    setup_training_environment()

    # Lancement de l'entraînement
    print("\n🚀 Lancement de l'entraînement Swin Base Production...")
    print("⏰ Durée estimée: 2-3 heures sur GPU haute performance")

    try:
        train_main()
        print("\n✅ Entraînement terminé avec succès!")
        print("📊 Vérifiez les métriques dans outputs/phase2_swin_base_production/")

    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()