# 🎯 Kaggle DINOv2 Training — Complete Guide

## Overview

Ce script est **entièrement adapté à Kaggle** pour entraîner DINOv2 sur votre dataset déséquilibré de 340 classes de maladies de plantes. Il implémente une **stratégie complète d'équilibrage du dataset**.

---

## 📋 Étapes pour démarrer sur Kaggle

### 1. **Créer un compte Kaggle** (si vous n'en avez pas)
   - Allez sur [kaggle.com](https://www.kaggle.com)
   - Inscrivez-vous gratuitement

### 2. **Préparer votre dataset**

Vous avez 3 options :

#### **Option A : Upload sur Kaggle (Recommandé)**
   1. Allez dans **Datasets** → **Create New Dataset**
   2. Uploadez votre dossier avec la structure :
      ```
      plant-diseases-dataset/
      ├── Alfalfa_Leaf_Mosaic_Virus/
      │   ├── img1.jpg
      │   ├── img2.jpg
      │   └── ...
      ├── Apple_Black_Rot/
      │   └── ...
      └── ... (340 dossiers)
      ```
   3. Attendez que le dataset soit disponible (peut prendre quelques minutes)
   4. Notez le **slug** du dataset (p.ex. `your-username/plant-diseases-dataset`)

### 3. **Télécharger depuis Google Drive** (Pour 80GB+)
   **Avantages** : Pas d'upload, accès direct à votre Drive
   **Limites** : Lent pour 80GB, peut timeout après 9h Kaggle

   1. **Partagez votre dossier Drive** :
      - Faites un clic droit sur votre dossier dataset
      - "Partager" → "Tout le monde avec le lien"
      - Copiez l'ID du dossier depuis l'URL :
        ```
        https://drive.google.com/drive/folders/1ABC...XYZ
        ```
        L'ID est `1ABC...XYZ`

   2. **Dans le script Kaggle** :
      ```python
      DOWNLOAD_FROM_DRIVE = True
      DRIVE_FOLDER_ID = "1ABC...XYZ"  # Votre ID réel
      ```

   3. **Téléchargement progressif** :
      - Le script télécharge par batches de 50 classes
      - Si timeout, relancez le notebook (reprise auto)

   **⚠️ Important** : 80GB peut prendre 2-4h à télécharger. Commencez avec un test !

---

## 💾 Gestion d'un Dataset de 80GB

### Stratégies pour Kaggle

#### **1. Téléchargement Progressif** ✅ (Implémenté)
   - Télécharge par batches de 50 classes
   - Reprise possible si interruption
   - Timeout Kaggle géré automatiquement

#### **2. Sous-dataset Initial** (Fortement recommandé)
   ```python
   # Commencez avec 20-30 classes représentatives
   subset_classes = [
       # Maladies communes
       "Apple_Black_Rot", "Apple_Healthy", "Apple_Scab",
       "Banana_Healthy", "Banana_Sigatoka_Leaf_Spot", 
       "Corn_Common_Rust", "Corn_Healthy",
       "Tomato_Early_Blight", "Tomato_Healthy", "Tomato_Late_Blight",
       # Ajoutez selon vos besoins
   ]
   ```

#### **3. Streaming pendant l'entraînement**
   - Au lieu de tout télécharger, chargez les images à la volée depuis Drive
   - Plus lent mais économise l'espace Kaggle
   - Nécessite modification du Dataset class

#### **4. Compression & Optimisation**
   - Compressez les images (JPEG quality 85% au lieu de 100%)
   - Redimensionnez à 224px pendant le préprocessing
   - Supprimez les classes avec < 50 images (trop bruitées)

### Temps Estimé pour 80GB
- **Téléchargement complet** : 2-4 heures (selon connexion)
- **Sous-dataset (20 classes)** : 5-15 minutes
- **Entraînement** : 30-60 min par epoch (dépend du batch size)

### Alternatives si Kaggle ne suffit pas
Si 80GB est trop pour Kaggle :
1. **Google Colab Pro** : Plus cher mais supporte Drive natif
2. **Local GPU** : Si vous avez un bon PC
3. **Cloud instances** : AWS/GCP avec stockage persistant
4. **Dataset curation** : Nettoyez et réduisez le dataset

#### **Option C : Utiliser un dataset existant**
   - Kaggle a des datasets publics de maladies de plantes
   - Vous pouvez en linker un et adapter le script

### 3. **Créer un Notebook Kaggle**
   1. Allez dans **Code** → **Create New Notebook**
   2. Sélectionnez **Python** comme language
   3. **Important** : Activez le GPU dans les settings (⚙️ → GPU)

### 4. **Copier le script**
   - Copiez/collez le contenu de `kaggle_dinov2_training.py` dans votre notebook
   - Ou uploadez le fichier directement

### 5. **Configurer le chemin du dataset**
   Dans le script, trouvez cette ligne (Cell 3) :
   ```python
   DATA_DIR = Path('/kaggle/input/plant-diseases-dataset')  # Remplacez par votre nom
   ```
   
   Remplacez `plant-diseases-dataset` par le **slug de votre dataset Kaggle** (ou le chemin local si vous utilisez gdown).

### 6. **Lancer l'entraînement**
   - Cliquez sur ▶️ **Run** ou **Ctrl+Shift+Enter**
   - Le script va :
     - 📥 Découvrir les 340 classes
     - 📊 Analyser le déséquilibre (min/max/médiane)
     - ⚖️ Créer des samplers pondérés
     - 🚀 Entraîner le modèle pendant ~20 epochs (ou moins avec early stopping)
     - 💾 Sauvegarder le meilleur modèle

---

## 🎯 Stratégies d'Équilibrage du Dataset

Votre dataset est **très déséquilibré** :
- **Min** : 18 images (Lettuce_Chlorosis_Virus)
- **Max** : 1200 images
- **Médiane** : ~1000 images

Le script implémente **3 stratégies** pour y remédier :

### 1. **Weighted Random Sampler** ✅ (Activé par défaut)
   - Donne plus de poids aux classes peu représentées
   - Chaque batch contient un mélange équilibré de classes
   - Les petites classes (< 300 images) sont **3x sur-échantillonnées**
   - Les grandes classes sont sous-échantillonnées proportionnellement

   **Avantage** : Simple, efficace, réduit le biais du modèle vers les grandes classes

### 2. **Weighted Cross-Entropy Loss**
   - Chaque classe a un poids inverse à sa fréquence
   - Les erreurs sur les petites classes pénalisent plus
   - Classe avec 18 images → poids ~67x plus élevé que classe avec 1200 images

   **Avantage** : Compense pendant le training

### 3. **Data Augmentation Agressive**
   - Les classes avec < 300 images reçoivent plus d'augmentation
   - Transformations : rotation (45°), flip, perspective, color jitter, coarse dropout
   - Cela crée de la variabilité pour les petites classes

   **Avantage** : Augmente la diversité des samples

### 4. **Label Smoothing**
   - Réduit le surapprentissage sur les petites classes
   - Valeur : 0.05 (standard)

---

## 📊 Résultats Attendus

Avec ces stratégies :
- **Accuracy globale** : 75-85% (dépend de la difficulté des classes)
- **Performance sur petites classes** : Meilleure qu'avec un model non-équilibré
- **Temps d'entraînement** : ~1-2 heures sur Kaggle GPU (20 epochs, 16 batch)
- **Utilisation GPU** : ~15-20 GB VRAM (compatible avec A100/T4)

---

## 📁 Fichiers de Sortie

À la fin de l'entraînement, vous obtiendrez dans `/kaggle/working/` :

```
/kaggle/working/
├── models_dinov2_kaggle/
│   ├── final_model.pt          # Modèle final
│   ├── class_mapping.json      # Mapping classe → index
│   └── training_config.json    # Configuration
├── checkpoints_dinov2_kaggle/
│   └── best_model.pt           # Meilleur checkpoint (early stopping)
└── logs_dinov2_kaggle/         # Logs (si TensorBoard activé)
```

**Téléchargement** : Cliquez sur le dossier `working/` dans l'onglet **Output** du notebook

---

## 📦 Scripts Supplémentaires

### `create_subset_dataset.py` — Gestion des 80GB
Script spécialisé pour créer des sous-datasets depuis votre Drive 80GB :

```python
# Configuration
DRIVE_FOLDER_ID = "votre_id_drive"
SUBSET_CLASSES = [
    "Apple_Black_Rot", "Apple_Healthy", "Apple_Scab",
    "Banana_Healthy", "Banana_Sigatoka_Leaf_Spot",
    # 20-30 classes représentatives
]

# Exécution
# Télécharge seulement les classes sélectionnées
# Crée un sous-dataset équilibré (500 images/classe max)
```

**Avantages** :
- Évite de télécharger 80GB d'un coup
- Test rapide du pipeline
- Scaling progressif vers le dataset complet

**Usage** : Copiez le script dans votre notebook Kaggle et exécutez-le avant l'entraînement.

---

## 🔧 Ajustements Possibles

Selon vos besoins, vous pouvez modifier :

### Pour accélérer l'entraînement :
```python
CONFIG['batch_size'] = 8        # Plus petit batch
CONFIG['num_epochs'] = 10       # Moins d'epochs
CONFIG['image_size'] = 192      # Images plus petites
```

### Pour améliorer la qualité (plus coûteux) :
```python
CONFIG['backbone'] = 'dinov2_vitb14'  # Plus gros modèle (87M params)
CONFIG['batch_size'] = 24              # Plus grand batch
CONFIG['num_epochs'] = 40              # Plus d'epochs
CONFIG['image_size'] = 336             # Images plus grandes
```

### Pour plus d'augmentation :
```python
CONFIG['underrep_augmentation_factor'] = 5.0  # Augmenter au lieu de 3.0
```

---

## 🚨 Troubleshooting

### ❌ "Dataset not found at /kaggle/input/..."
- ✅ Vérifiez que le dataset est lié au notebook (Data → Add Data)
- ✅ Ou changez `DATA_DIR` pour pointer au bon chemin

### ❌ "Out of memory"
- ✅ Réduisez `batch_size` (8 au lieu de 16)
- ✅ Réduisez `image_size` (192 au lieu de 224)
- ✅ Utilisez `grad_accum_steps = 2` (gradient accumulation)

### ❌ "GPU not detected"
- ✅ Allez dans **Notebook Settings** (⚙️) → **GPU** → Sélectionnez **GPU**

### ❌ "Epoch too long (> 9 hours)"
- ✅ Les sessions Kaggle sont limitées à 9 heures
- ✅ Réduisez le nombre d'epochs ou la taille du dataset
- ✅ Ou utilisez `CONFIG['batch_size'] = 32` pour des batches plus gros (plus rapide mais moins stable)

---

## 📈 Comparaison : Équilibré vs Non-équilibré

| Métrique | Sans équilibrage | Avec équilibrage |
|----------|-----------------|-----------------|
| Accuracy globale | 60-70% | 75-85% |
| Recall petites classes | 20-40% | 60-80% |
| Recall grandes classes | 85-95% | 75-85% |
| Biais modèle | Fort (faveur grandes classes) | Réduit |

---

## 🎓 Prochaines Étapes

1. **Inférence** : Chargez le modèle et testez sur de nouvelles images
2. **Fine-tuning** : Continuez l'entraînement si accuracy < 80%
3. **Ensemble** : Entraînez plusieurs modèles et moyennez les prédictions
4. **Déploiement** : Convertissez en ONNX/TensorFlow pour production

---

## 🔗 Ressources

- **Kaggle Docs** : https://kaggle.com/docs
- **DINOv2 Paper** : https://arxiv.org/abs/2304.07193
- **Class Imbalance** : https://arxiv.org/abs/1901.05555

---

## 💬 Questions ?

Si vous rencontrez des problèmes :
1. Vérifiez les chemins de données
2. Vérifiez que le GPU est activé
3. Consultez les logs du script
4. Réduisez la complexité (batch size, image size, epochs)

**Bon entraînement ! 🚀**
