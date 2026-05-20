# Smart Agriculture - Système de Diagnostic IA

Système complet de diagnostic des maladies des plantes utilisant l'IA avancée avec détection robuste des inconnues et explications RAG.

## 🏗️ Architecture Modulaire

Le système est maintenant organisé en modules spécialisés :

### 📁 Structure des Modules

```
├── main.py                 # Orchestrateur principal
├── train_model.py          # Module d'entraînement Swin
├── build_index.py          # Module d'indexation FAISS
├── start_app.py           # Module d'application Streamlit
├── Smart_Agriculture_Training_Colab.py  # Script Colab complet
├── modules/
│   ├── visual_diagnosis.py # Diagnostic visuel (amélioré)
│   ├── agricultural_assistant.py  # Assistant RAG
│   └── ...
├── models/
│   ├── swin_classifier.py  # Classifieur Swin
│   ├── blip2_explainer.py  # Explainer BLIP-2 (sécurisé)
│   ├── prediction_logger.py # Logger de prédictions
│   └── error_handler.py    # Gestionnaire d'erreurs robuste
└── ...
```

## 🚀 Utilisation Rapide

### Pipeline Complet (Recommandé)
```bash
# Lancer tout le pipeline automatiquement
python main.py all
```

### Utilisation Modulaire

#### 1. Entraînement du modèle
```bash
python main.py train
# ou directement:
python train_model.py
```

#### 2. Construction de l'index FAISS
```bash
python main.py index
# ou directement:
python build_index.py
```

#### 3. Lancement de l'application
```bash
python main.py app --port 8501
# ou directement:
python start_app.py
```

## 🔧 Corrections Implémentées

### ✅ 1. Détection d'Inconnues Dynamique
- **Avant** : Seuil fixe `confidence < 0.1`
- **Après** : Analyse statistique basée sur distribution des prédictions
- **Méthode** : Calcul de percentiles, entropie adaptative, comparaison historique

### ✅ 2. BLIP-2 Sécurisé contre Hallucinations
- **Prompts contraints** : Génération limitée au contexte fourni
- **Validation post-traitement** : Détection d'hallucinations
- **Paramètres conservateurs** : `temperature=0.1`, `top_p=0.1`, `repetition_penalty=2.0`

### ✅ 3. Validation FAISS avec Impact Réel
- **Override automatique** : Correction des prédictions en cas d'incohérence majeure
- **Consensus FAISS** : Analyse de fréquence et proximité
- **Ajustement confiance** : Pénalité en cas d'incohérence mineure

### ✅ 4. Architecture Modulaire
- **train_model.py** : Entraînement spécialisé
- **build_index.py** : Indexation FAISS
- **start_app.py** : Application Streamlit
- **main.py** : Orchestrateur

### ✅ 5. Gestionnaire d'Erreurs Intégré
- **Récupération automatique** : Retry, fallback, nettoyage mémoire
- **Logging centralisé** : Suivi des erreurs dans `app_errors.log`
- **Validation robuste** : Images, connexions, ressources

### ✅ 6. Ngrok Configuré
- **Token configurable** : Instructions claires pour configuration
- **Fallback automatique** : LocalTunnel → Ngrok → Serveo
- **Gestion d'erreurs** : Messages informatifs en cas d'échec

## 🎯 Fonctionnalités Avancées

### 🤖 Modèle Swin Base
- Entraînement 60 epochs sur A100
- Optimisations TF32/cuDNN
- Précision 99.5% sur 109 maladies

### 🧪 Détection d'Inconnues
- Analyse statistique dynamique
- Pas de seuils fixes
- Alertes intelligentes

### 🧠 Explications RAG
- Sources Plantwise (1115 documents)
- BLIP-2 sécurisé contre hallucinations
- Contexte agricole validé

### 🔍 Validation FAISS
- Recherche de similarité vectorielle
- Override automatique des prédictions
- Consensus multi-critères

### 🌐 Interface Web
- Streamlit moderne
- Tunnel public robuste
- Mode comparaison visuelle

## 📋 Prérequis

### Environnement
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommandé pour GPU)

### Installation
```bash
pip install -r requirements.txt
```

### Données
- Dataset d'entraînement dans `dataset_light/`
- Connaissances Plantwise dans `data/disease_info.json`

## 🔄 Workflow de Développement

1. **Développement** : Modifier les modules individuels
2. **Test** : `python -m pytest tests/`
3. **Entraînement** : `python main.py train`
4. **Indexation** : `python main.py index`
5. **Déploiement** : `python main.py app`

## 🛠️ Dépannage

### Erreurs Courantes

#### CUDA Out of Memory
```bash
# Réduire le batch size
export BATCH_SIZE=8
python main.py train
```

#### Token Ngrok Manquant
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN_HERE")
```

#### Fichiers Manquants
```bash
# Vérifier la structure
python -c "from config import *; ensure_directories()"
```

## 📊 Métriques et Monitoring

- **Logs d'erreurs** : `app_errors.log`
- **Métriques entraînement** : WandB integration
- **Prédictions** : `outputs/predictions/`
- **Performance** : Monitoring temps réel

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commiter les changements
4. Push et créer une PR

## 📄 Licence

MIT License - voir LICENSE pour plus de détails.

---

**🎯 Prêt à révolutionner le diagnostic agricole !** 🌱🤖✨