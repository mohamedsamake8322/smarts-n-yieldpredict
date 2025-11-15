# Script Plantset - Modèle Vision-Langage Multimodal

Ensemble de scripts pour entraîner un modèle multimodal vision-langage puissant sur 133 classes de maladies de plantes, optimisé pour TPU v4/v5/v6.

## 🚀 Fonctionnalités

- **Architecture moderne** : Inspiré de Florence-2, Qwen-VL, FLAVA, PaLM2-VAdapter
- **Support TPU** : Optimisé pour TPU v4, v5, v6 avec torch_xla
- **Dataset multimodal** : Images + descriptions textuelles enrichies
- **Mixed precision** : Entraînement accéléré avec AMP
- **Scalabilité** : Support multi-TPU et dataset sharding
- **Métriques complètes** : Classification + génération de texte (BLEU/ROUGE)

## 📁 Structure des Scripts

```
script_plantset/
├── __init__.py              # Package initialisation
├── data_cleaner.py          # Nettoyage du dataset
├── text_mapper.py           # Mapping images + textes
├── dataset_loader.py        # DataLoader PyTorch scalable
├── model_builder.py         # Architecture multimodale
├── train.py                 # Entraînement standard
├── evaluate.py              # Évaluation complète
├── infer.py                 # Prédictions
├── scaler_tpu.py           # Support TPU
└── README.md               # Documentation
```

## 🛠️ Installation

```bash
# Installer les dépendances
pip install torch torchvision torch_xla
pip install transformers datasets
pip install scikit-learn matplotlib seaborn
pip install nltk rouge-score
pip install tqdm pillow

# Pour TPU (Google Cloud)
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## 📊 Pipeline Complet

### 1. Nettoyage des Données

```bash
python data_cleaner.py \
    --data-dir /path/to/dataset \
    --min-size 256 \
    --target-size 512 \
    --output-file dataset_clean.jsonl
```

### 2. Mapping Textuel

```bash
python text_mapper.py \
    --diseases-json /path/to/maladies_enrichies.json \
    --dataset-jsonl dataset_clean.jsonl \
    --output-file multimodal_dataset.jsonl \
    --language fr
```

### 3. Entraînement Standard

```bash
python train.py \
    --jsonl-file multimodal_dataset.jsonl \
    --root-dir /path/to/images \
    --num-classes 133 \
    --vision-backbone resnet50 \
    --text-model microsoft/DialoGPT-medium \
    --epochs 100 \
    --batch-size 32 \
    --mixed-precision \
    --checkpoint-dir checkpoints
```

### 4. Entraînement TPU

```bash
python scaler_tpu.py \
    --jsonl-file multimodal_dataset.jsonl \
    --root-dir /path/to/images \
    --num-classes 133 \
    --tpu-cores 8 \
    --epochs 100 \
    --batch-size 32 \
    --mixed-precision
```

### 5. Évaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --jsonl-file multimodal_dataset.jsonl \
    --root-dir /path/to/images \
    --num-classes 133 \
    --output-dir evaluation_results
```

### 6. Prédiction

```bash
python infer.py \
    --checkpoint checkpoints/best_model.pth \
    --num-classes 133 \
    --image /path/to/image.jpg \
    --text "Description optionnelle" \
    --top-k 5 \
    --generate-description
```

## 🏗️ Architecture du Modèle

### Vision Encoder
- **Backbones supportés** : ResNet18/50/101, EfficientNet, ViT
- **Pré-entraînement** : ImageNet ou custom
- **Projection** : Linear + LayerNorm vers espace commun

### Text Encoder
- **Modèles supportés** : BERT, RoBERTa, DialoGPT, LLaMA
- **Tokenisation** : HuggingFace tokenizers
- **Pooling** : Attention-based ou global average

### Fusion Module
- **Cross-Attention** : Attention croisée entre vision et texte
- **Multi-head** : 8 têtes d'attention par défaut
- **Layers** : 2 couches de fusion par défaut

### Têtes de Sortie
- **Classification** : Softmax sur 133 classes
- **Génération** : Embeddings pour descriptions textuelles

## ⚡ Optimisations TPU

### Configuration Recommandée
- **TPU v4** : 8 cœurs, batch_size=32-64
- **TPU v5** : 8 cœurs, batch_size=64-128
- **TPU v6** : 8 cœurs, batch_size=128-256

### Mixed Precision
```python
# Automatique avec torch_xla
use_mixed_precision=True
```

### Dataset Sharding
```python
# Automatique avec DataLoader
num_workers=4  # Réduit pour TPU
pin_memory=False  # Pas nécessaire sur TPU
```

## 📈 Métriques

### Classification
- **Accuracy** : Précision globale
- **F1-Score** : Macro et weighted
- **AUC** : Area Under Curve
- **Confusion Matrix** : Visualisation des erreurs

### Génération de Texte
- **BLEU** : Bilingual Evaluation Understudy
- **ROUGE** : Recall-Oriented Understudy for Gisting Evaluation
- **Perplexity** : Mesure de cohérence

## 🔧 Configuration Avancée

### Modèle Personnalisé
```python
from model_builder import create_multimodal_model

model = create_multimodal_model(
    num_classes=133,
    vision_backbone="efficientnet_b4",
    text_model="microsoft/DialoGPT-large",
    vision_dim=768,
    text_dim=768,
    fusion_dim=512,
    num_attention_heads=12,
    num_attention_layers=4
)
```

### DataLoader Personnalisé
```python
from dataset_loader import create_data_module

data_module = create_data_module(
    jsonl_file="multimodal_dataset.jsonl",
    root_dir="/path/to/images",
    batch_size=64,
    image_size=384,
    text_length=256,
    augment=True
)
```

## 🐛 Dépannage

### Erreurs TPU
```bash
# Vérifier la configuration TPU
python -c "import torch_xla; print(torch_xla.__version__)"

# Tester la connexion TPU
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

### Erreurs de Mémoire
- Réduire `batch_size`
- Utiliser `gradient_checkpointing=True`
- Réduire `image_size` ou `text_length`

### Erreurs de Convergence
- Ajuster `learning_rate` (1e-5 à 1e-3)
- Utiliser `warmup_steps`
- Vérifier les `class_weights`

## 📚 Références

- [Florence-2](https://arxiv.org/abs/2311.00542) - Microsoft
- [Qwen-VL](https://arxiv.org/abs/2308.12966) - Alibaba
- [FLAVA](https://arxiv.org/abs/2112.04482) - Facebook
- [PaLM2-VAdapter](https://arxiv.org/abs/2305.17023) - Google
- [torch_xla](https://github.com/pytorch/xla) - PyTorch TPU

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - Voir LICENSE pour plus de détails.

## 📞 Support

- **Issues** : GitHub Issues
- **Discussions** : GitHub Discussions
- **Email** : support@smartagro.com

---

**Développé par l'équipe Smart Agro** 🌱








