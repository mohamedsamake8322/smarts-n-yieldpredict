"""
PHASE 2 — Modèle production (Swin Base)

Objectifs:
- Modèle robuste et généralisable
- Bon ratio performance / coût de calcul
- Base pour déploiement (web/mobile type Plantix/PlantSnap)

Usage LOCAL:
- python -m training_pipelines.phase2_swin_base_production

Usage COLAB:
- Cellule 1: coller tout metric_training_core.py et exécuter
- Cellule 2: coller ce script et exécuter
"""

import os

try:
    from .metric_training_core import get_default_config, run_training
except ImportError:
    try:
        from metric_training_core import get_default_config, run_training  # type: ignore
    except ImportError:
        get_default_config = globals().get("get_default_config")
        run_training = globals().get("run_training")
        if get_default_config is None or run_training is None:
            raise ImportError(
                "Colab: collez le contenu de metric_training_core.py dans la cellule "
                "précédente et exécutez-la d'abord !"
            )


def main():
    # Swin Base, configuration proche de ton script Colab complet
    config = get_default_config(
        model_name="swin_base_patch4_window7_224",
        embedding_dim=768,
        num_epochs=60,
        experiment_name="phase2_swin_base_production",
    )

    # A100 → tu peux te permettre des batchs plus gros
    config.batch_size = 64

    dataset_override = os.environ.get("DATASET_PATH")
    if dataset_override:
        config.dataset_path = dataset_override

    # Print de sécurité avant de lancer l'entraînement
    print("========== TRAINING CONFIG ==========")
    print(f"Model: {config.model_name}")
    print(f"Dataset path: {config.dataset_path}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Embedding dim: {config.embedding_dim}")
    print("=====================================")

    run_training(config)


if __name__ == "__main__":
    main()

