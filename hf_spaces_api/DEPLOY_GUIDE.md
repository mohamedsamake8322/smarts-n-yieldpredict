# 🚀 Déploiement Hugging Face Spaces - Guide Complet

## 📋 Prérequis

1. **Compte Hugging Face**: https://huggingface.co/join
2. **Git installé** sur votre machine
3. **Modèle déjà uploadé**: `mohamedsamake8322/plant-diseaseS-swin-faiss`

---

## 🏗️ Étape 1: Créer le Space

### Via Interface Web

1. **Aller sur**: https://huggingface.co/spaces
2. **Cliquer**: "Create new Space"
3. **Remplir**:
   - **Space name**: `plant-disease-api`
   - **License**: `MIT`
   - **SDK**: `Docker`
   - **Visibility**: `Public`
4. **Créer** le Space

### Via Terminal (recommandé)

```bash
# Créer le Space
huggingface-cli create-space plant-disease-api \
    --sdk docker \
    --license mit \
    --public
```

---

## 📁 Étape 2: Préparer les fichiers

Assurez-vous d'avoir ces fichiers dans `hf_spaces_api/`:

```
hf_spaces_api/
├── app.py              # API FastAPI
├── requirements.txt    # Dépendances
├── Dockerfile         # Configuration Docker
├── README.md          # Documentation
├── test_api.py        # Tests
├── package.json       # Métadonnées HF
├── .gitignore         # Fichiers à ignorer
└── deploy.sh          # Script de déploiement
```

---

## 🔧 Étape 3: Configurer Git

```bash
# Aller dans le dossier API
cd hf_spaces_api

# Initialiser git (si pas déjà fait)
git init

# Ajouter le remote Hugging Face
git remote add origin https://huggingface.co/spaces/mohamedsamake8322/plant-disease-api.git

# Ou si vous utilisez un autre nom:
# git remote add origin https://huggingface.co/spaces/VOTRE_USERNAME/plant-disease-api.git
```

---

## 🚀 Étape 4: Déployer

### Option A: Script automatique

```bash
# Rendre le script exécutable
chmod +x deploy.sh

# Lancer le déploiement
./deploy.sh
```

### Option B: Manuel

```bash
# Ajouter tous les fichiers
git add .

# Commit
git commit -m "Deploy Plant Disease Detection API"

# Push vers HF Spaces
git push origin main
```

---

## 🧪 Étape 5: Tester

### Test local (avant déploiement)

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API localement
uvicorn app:app --reload --host 0.0.0.0 --port 7860

# Tester dans un autre terminal
python test_api.py http://localhost:7860
```

### Test après déploiement

```bash
# Tester l'API déployée
python test_api.py https://mohamedsamake8322-plant-disease-api.hf.space
```

---

## 🔗 Étape 6: Intégrer dans Streamlit

Mettez à jour votre `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
API_URL = "https://mohamedsamake8322-plant-disease-api.hf.space"
```

---

## 📊 URLs importantes

Après déploiement:

- **API Base**: `https://mohamedsamake8322-plant-disease-api.hf.space`
- **Documentation**: `https://mohamedsamake8322-plant-disease-api.hf.space/docs`
- **Health Check**: `https://mohamedsamake8322-plant-disease-api.hf.space/health`

---

## 🐛 Dépannage

### Erreur: "Model not loaded"

- Vérifiez que votre modèle est bien sur HF: `mohamedsamake8322/plant-diseaseS-swin-faiss`
- Vérifiez les logs du Space dans l'onglet "Logs"

### Erreur: "CUDA out of memory"

- Le Space utilise probablement un GPU - vérifiez les limites
- Passez à CPU dans le Dockerfile si nécessaire

### Erreur: "Timeout"

- Augmentez le timeout dans Streamlit: `API_TIMEOUT = 60`
- Vérifiez la connectivité réseau

---

## 📈 Performance

| Métrique | Local | HF Spaces Free | HF Spaces Pro |
|----------|-------|----------------|----------------|
| RAM | 800MB+ | ~400MB | ~400MB |
| CPU/GPU | Variable | CPU Basic | GPU |
| Latence | 50ms | 200-500ms | 100-300ms |
| Coût | 0€ | 0€ | 0.001€/req |

---

## 🎯 Prochaines étapes

1. ✅ **Test local** de l'API
2. ✅ **Déploiement** sur HF Spaces
3. ✅ **Test** de l'API déployée
4. ✅ **Intégration** dans Streamlit
5. 🔄 **Monitoring** et optimisation

---

## 💡 Conseils

- **Monitoring**: Utilisez les logs HF Spaces pour surveiller
- **Scaling**: Upgrade vers Pro si beaucoup de trafic
- **Backup**: Gardez une copie locale de l'API
- **Updates**: Push vers main pour déployer automatiquement

---

## 📞 Support

Si problème:
1. Vérifiez les logs du Space
2. Testez localement d'abord
3. Consultez la doc HF Spaces: https://huggingface.co/docs/hub/spaces