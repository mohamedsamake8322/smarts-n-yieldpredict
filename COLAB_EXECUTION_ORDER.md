# 🚀 GUIDE COMPLET GOOGLE COLAB - Ordre d'Exécution

## ⚡ OPTION ONE-CLICK (Recommandée)

Au lieu de 12 cellules séparées, utilisez simplement:

```python
!python colab_setup_complete.py
```

**✅ Avantages:**
- Tout automatique en 5-10 minutes
- Pas besoin de copier-coller 12 cellules
- Gestion d'erreurs intégrée
- Logs détaillés de progression

**📖 Guide complet:** Voir `COLAB_ONE_CLICK.md`

---

## Version Manuelle (12 Cells)

### Cell 1 - Montage Drive & Installation des Dépendances
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/smarts-n-yieldpredict')

# Installation des dépendances
!pip install -q sentence-transformers faiss-cpu streamlit torch transformers timm accelerate
!pip install -q albumentations opencv-python-headless Pillow plotly
print("✅ Dépendances installées")
```

### Cell 2 - Configuration & Vérification
```python
from config import print_config, ensure_directories
ensure_directories()
print_config()
print("\n✅ Configuration vérifiée")
```

### Cell 3 - Normalisation BLIP2 (109 fichiers)
```python
!python normalize_blip2.py 2>&1 | tail -20
print("✅ BLIP2 normalisé")
```

### Cell 4 - Construction Index FAISS (1115 entrées)
```python
!python build_moh_index.py 2>&1 | tail -10
print("✅ Index FAISS construit")
```

### Cell 5 - Test des Modules
```python
!python test_modules.py
print("✅ Modules testés")
```

### Cell 6 - Vérification Modèles Swin
```python
import os
swin_dir = "/content/drive/MyDrive/outputs/phase2_swin_base_production/models"
files_needed = ['senedisease_macro_f1.pt', 'faiss_index.bin', 'metadata.json']

print("Vérification des fichiers Swin:")
for file in files_needed:
    path = os.path.join(swin_dir, file)
    exists = os.path.exists(path)
    print(f"  {file}: {'✅' if exists else '❌'} {'Existe' if exists else 'Manquant'}")

if all(os.path.exists(os.path.join(swin_dir, f)) for f in files_needed):
    print("\n✅ Tous les modèles Swin sont disponibles")
else:
    print("\n⚠️  Certains modèles Swin manquent - vérifiez votre entraînement")
```

### Cell 7 - Installation Streamlit & LocalTunnel
```python
!pip install -q streamlit
!npm install -g localtunnel
print("✅ Streamlit et LocalTunnel installés")
```

### Cell 8 - Configuration Streamlit pour Colab
```python
import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
print("✅ Configuration Streamlit pour Colab")
```

### Cell 9 - Lancement Application (Background)
```python
import subprocess
import time

# Lancement en arrière-plan
process = subprocess.Popen([
    'streamlit', 'run', '04_app_streamlit.py',
    '--server.port', '8501',
    '--server.address', '0.0.0.0',
    '--logger.level', 'error'
])

print("✅ Application lancée en arrière-plan")
print("Attente de 5 secondes pour l'initialisation...")
time.sleep(5)
```

### Cell 10 - Création Tunnel Public
```python
# Création du tunnel public
print("🔗 Création du tunnel public...")
print("Copiez l'URL qui apparaît ci-dessous:")
!lt --port 8501
```

### Cell 11 - Vérification Statut Application
```python
import requests
import time

def check_app():
    try:
        response = requests.get('http://localhost:8501', timeout=5)
        return response.status_code == 200
    except:
        return False

print("Vérification du statut de l'application...")
for i in range(10):
    if check_app():
        print("✅ Application accessible sur http://localhost:8501")
        break
    else:
        print(f"⏳ Tentative {i+1}/10 - Application pas encore prête")
        time.sleep(3)
else:
    print("❌ Application ne répond pas - vérifiez les logs")
```

### Cell 12 - Commandes Utiles (optionnel)
```python
# Pour arrêter l'application
# process.terminate()

# Pour voir les logs
# !ps aux | grep streamlit

# Pour redémarrer
# !pkill -f streamlit
# !streamlit run 04_app_streamlit.py --server.port 8501 --server.address 0.0.0.0 &

print("🎉 Application prête ! Utilisez l'URL du tunnel pour accéder à l'app")
```

---

## 📋 Résumé de l'Ordre d'Exécution

1. **Montage Drive** + **Installation** (Cell 1)
2. **Configuration** (Cell 2)
3. **Normalisation BLIP2** (Cell 3)
4. **Index FAISS** (Cell 4)
5. **Tests Modules** (Cell 5)
6. **Vérification Swin** (Cell 6)
7. **Installation Streamlit** (Cell 7)
8. **Config Streamlit** (Cell 8)
9. **Lancement App** (Cell 9)
10. **Tunnel Public** (Cell 10)
11. **Vérification** (Cell 11)
12. **Commandes Utiles** (Cell 12)

---

## ⚠️ Points Importants

- **Exécutez les cellules dans l'ordre** - elles dépendent les unes des autres
- **Le modèle Swin** doit déjà être entraîné dans votre Google Drive
- **LocalTunnel** crée une URL publique pour accéder à l'app
- **L'application tourne en arrière-plan** - ne fermez pas le notebook

---

## 🎯 Résultat Final

Après exécution complète, vous aurez:
- ✅ 109 fichiers BLIP2 normalisés
- ✅ Index FAISS avec 1115 entrées
- ✅ Modèles Swin opérationnels
- ✅ Application Streamlit accessible via URL publique
- ✅ Pipeline complet: Image → Swin → BLIP-2 → Explication

**URL d'accès:** L'URL générée par LocalTunnel (Cell 10)