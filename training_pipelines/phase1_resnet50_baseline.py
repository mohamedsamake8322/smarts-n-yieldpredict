"""
PHASE 1 — Baseline scientifique (ResNet50)

Objectifs:
- Vérifier la qualité du dataset
- Obtenir une métrique de référence (Recall@1, mAP, etc.)
- Servir de comparaison académique pour les modèles Swin

Usage LOCAL:
- python -m training_pipelines.phase1_resnet50_baseline
- python training_pipelines/phase1_resnet50_baseline.py

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
        # Colab: le core a été collé dans la cellule précédente
        get_default_config = globals().get("get_default_config")
        run_training = globals().get("run_training")
        if get_default_config is None or run_training is None:
            raise ImportError(
                "Colab: collez le contenu de metric_training_core.py dans la cellule "
                "précédente et exécutez-la d'abord !"
            )


def main():
    # ResNet50 baseline, 15 epochs

    config = get_default_config(
        model_name="resnet50",
        embedding_dim=512,
        num_epochs=15,
        experiment_name="phase1_resnet50_baseline",
    )
    # Augmentation plus simple pour la baseline scientifique
    config.strong_augmentation = False

    # Optionnel: override rapide via variables d'environnement
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

