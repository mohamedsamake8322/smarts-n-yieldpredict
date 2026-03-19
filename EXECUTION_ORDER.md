# 🎯 SCRIPTS D'EXÉCUTION - ORDRE COMPLET

## 📋 Résumé des changements effectués

### ✅ Modifications apportées:

1. **config.py** - Enrichi avec gestion des chemins pour Colab et local
2. **modules/visual_diagnosis.py** - Ajout des top-3 prédictions et info basique/détaillée
3. **modules/agricultural_assistant.py** - Support des chemins Colab
4. **pages/2_Disease_Detection.py** - Nouvelle interface avec workflow progressif
5. **pages/3_Agricultural_Assistant.py** - Nouvelle interface Q&A
6. **Documents guides** - SETUP_AND_EXECUTION_GUIDE.md, quickstart.py


---

## 🚀 EXÉCUTION COMPLÈTE - ÉTAPE PAR ÉTAPE

### **OPTION 1: Exécution locale (Windows/Mac/Linux)**

#### Étape 1: Préparer l'environnement
```bash
# Ouvrir le terminal à la racine du projet
cd c:\smarts-n-yieldpredict.git
# (ou le chemin exact de votre projet)

# Activer l'environnement Python
.\env311\Scripts\Activate.ps1    # Sur Windows
source env311/bin/activate       # Sur Mac/Linux
```

#### Étape 2: Installer les dépendances (si nécessaire)
```bash
pip install -r requirements.txt
```

#### Étape 3: Normaliser les 109 fichiers BLIP2
```bash
python normalize_blip2.py
```
✅ **Résultat attendu:** Crée le dossier `BLIP2_normalized/` avec 109 fichiers JSON normalisés

#### Étape 4: Construire l'index FAISS pour la base Plantwise
```bash
python build_moh_index.py
```
✅ **Résultat attendu:** Crée `moh_index.faiss` et `moh_metadata.json`

#### Étape 5: Tester les modules
```bash
python test_modules.py
```
✅ **Résultat attendu:** Message "Ready for integration into Streamlit app!"

#### Étape 6: Lancer l'application Streamlit
```bash
streamlit run 04_app_streamlit.py
```
✅ **Accès:** Ouvrir http://localhost:8501 dans le navigateur


---

### **OPTION 2: Utiliser le script Quick Start (PLUS FACILE)**

```bash
# Tout faire en une seule commande
python quickstart.py
```

✅ Cela exécutera automatiquement les 4 étapes (normalisation, index, test, Streamlit)


---

### **OPTION 3: Exécution sur Google Colab**

Créer un nouveau notebook Colab et exécuter les cellules dans cet ordre:

#### **Cellule 1: Monter Google Drive et installer les packages**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/smarts-n-yieldpredict')

# Install packages
!pip install -q sentence-transformers faiss-cpu streamlit torch transformers pillow
```

#### **Cellule 2: Vérifier la configuration**
```python
from config import print_config, ensure_directories
ensure_directories()
print_config()
```

#### **Cellule 3: Normaliser les fichiers BLIP2**
```bash
!python normalize_blip2.py 2>&1 | tail -20
```

#### **Cellule 4: Construire l'index FAISS**
```bash
!python build_moh_index.py 2>&1 | tail -10
```

#### **Cellule 5: Tester les modules**
```bash
!python test_modules.py
```

#### **Cellule 6: Lancer l'application Streamlit**
```bash
!streamlit run 04_app_streamlit.py --logger.level=error
```

✅ Cliquer sur le lien public pour accéder à l'application


---

## 📊 Ordre d'exécution complet (Résumé)

```
1️⃣ Terminal/Colab Setup
   ├─ Activer l'environnement
   └─ Installer les dépendances (si nécessaire)

2️⃣ Normalisation des données
   └─ python normalize_blip2.py
      → Crée: BLIP2_normalized/ (109 fichiers normalisés)

3️⃣ Construction de l'index
   └─ python build_moh_index.py
      → Crée: moh_index.faiss + moh_metadata.json

4️⃣ Test des modules
   └─ python test_modules.py
      → Vérifie que tout fonctionne

5️⃣ Lancer l'application
   └─ streamlit run 04_app_streamlit.py
      → Ouvrir http://localhost:8501
```


---

## ✨ Améliorations implémentées

### 1. **Gestion des chemins pour Colab**
   - Variable `BASE_PATH` dans `config.py`
   - Support automatique de Google Drive
   - Chemins portables et testés

### 2. **Workflow progressif dans l'interface**
   - Afficher d'abord les info basiques (nom, agent, symptômes)
   - Demander la confirmation: "Est-ce que cela correspond ?"
   - Afficher les détails complets (management, prévention) si confirmé

### 3. **Top-3 prédictions**
   - Afficher les 3 maladies les plus probables avec scores
   - Format visuel clair avec pourcentages (ex: "Corn smut — 97%")

### 4. **Assistant agricole séparé**
   - Nouvelle page dédiée pour les questions
   - Recherche sémantique sur 1115 entrées Plantwise
   - Interface de navigation claire

### 5. **Documentation complète**
   - `SETUP_AND_EXECUTION_GUIDE.md` - Guide détaillé
   - `quickstart.py` - Automatisation complète
   - Ce fichier - Ordre d'exécution


---

## 🔧 Dépannage rapide

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | Installer les packages: `pip install -r requirements.txt` |
| FAISS index non trouvé | Exécuter: `python build_moh_index.py` |
| Chemins invalides | Vérifier que le projet est dans le bon dossier |
| Streamlit ne démarre pas | Essayer: `python -m streamlit run 04_app_streamlit.py` |
| Colab erreurs de chemin | Vérifier que le projet est à `/MyDrive/smarts-n-yieldpredict/` |


---

## 📁 Fichiers créés/modifiés

| Fichier | Modification | Ligne de commande |
|---------|--------------|------------------|
| `config.py` | ✏️ Enrichi | (Partie de la configuration globale) |
| `modules/visual_diagnosis.py` | ✏️ Top-3 + info basique/détaillée | `python -m modules.visual_diagnosis` |
| `modules/agricultural_assistant.py` | ✏️ Chemins Colab | `python -m modules.agricultural_assistant` |
| `pages/2_Disease_Detection.py` | 🆕 Nouvelle interface | `streamlit run pages/2_Disease_Detection.py` |
| `pages/3_Agricultural_Assistant.py` | 🆕 Nouvelle interface Q&A | `streamlit run pages/3_Agricultural_Assistant.py` |
| `SETUP_AND_EXECUTION_GUIDE.md` | 🆕 Documentation complète | N/A |
| `quickstart.py` | 🆕 Script d'automatisation | `python quickstart.py` |
| `setup_colab.py` | 🆕 Setup Colab | (Exécutable dans Colab) |


---

## ✅ Checklist avant lancement

- [ ] Python 3.8+ installé
- [ ] Dépendances installées: `pip install -r requirements.txt`
- [ ] Dossier `BLIP2/` contient 109 fichiers JSON
- [ ] Dossier `Moh/` contient 1115 fichiers JSON
- [ ] Dossier `modules/` existe avec les fichiers Python
- [ ] Dossier `pages/` existe avec les interfaces

---

## 🎉 Résultat final

Une fois le processus complet exécuté, vous aurez:

✅ **Base Plantwise indexée** - Recherche sémantique rapide
✅ **Fichiers BLIP2 normalisés** - Parsing uniforme
✅ **Interface de diagnostic** - Workflow progressif avec confirmation
✅ **Assistant agricole** - Q&A sur 1115 entrées de connaissances
✅ **Compatibilité Colab** - Exécution directe sur Google Drive

**Accès:** http://localhost:8501 (ou le lien public Colab)

---

**Dernière mise à jour:** 16 mars 2026
