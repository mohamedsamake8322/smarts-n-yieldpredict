"""
PHASE 3 — Benchmark recherche (Swin Large)

Objectifs:
- Pousser la performance maximale
- Comparaison académique sérieuse vs ResNet50 & Swin Base
- Décider si le surcoût de Swin Large vaut le gain

Usage LOCAL:
- python -m training_pipelines.phase3_swin_large_research

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
    # Swin Large pour benchmark recherche
    config = get_default_config(
        model_name="swin_large_patch4_window7_224",
        embedding_dim=1024,
        num_epochs=80,
        experiment_name="phase3_swin_large_research",
    )

    # Swin Large est plus lourd → on garde un batch raisonnable
    config.batch_size = 48

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

