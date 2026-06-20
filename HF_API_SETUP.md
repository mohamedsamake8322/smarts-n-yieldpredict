# Hugging Face API Setup - Zero RAM Streamlit

## ⚠️ Situation actuelle

Votre Streamlit charge le modèle localement → **800MB+ RAM**

## ✅ Objectif : Architecture serveur/client

```
Streamlit (Browser)
    ↓
    HTTP Request avec image
    ↓
Hugging Face Spaces / API
    ↓
    Prédiction (RAM sur serveur)
    ↓
Streamlit reçoit résultat
    ↓
    RAM utilisé: <100MB ✅
```

---

## 📋 Ressources requises

Votre modèle sur HF a BESOIN de:
- ✅ `senedisease_macro_f1.pt` (Swin Base)
- ✅ `metadata.pkl` (classes + image_paths)
- ✅ `faiss_index.bin` (pour les voisins)

## 🚀 Solution 1: Hugging Face Spaces (Gratuit)

### Étape 1: Créer un Spaces FastAPI

1. Allez sur https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Choisissez:
   - **Space SDK**: Docker
   - **License**: openrail
   - **Name**: `plant-disease-api`

### Étape 2: Ajouter le code FastAPI

Créez un dossier avec cette structure:

```
plant-disease-api/
├── Dockerfile
├── app.py (FastAPI)
├── requirements.txt
└── .gitignore
```

**app.py:**
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

app = FastAPI()

# Charger UNE SEULE FOIS au démarrage du serveur
MODEL_REPO = "mohamedsamake8322/plant-diseaseS-swin-faiss"
model, index, metadata, prototypes, prototype_labels, device = None, None, None, None, None, None

def load_model():
    global model, index, metadata, prototypes, prototype_labels, device
    
    # Télécharger depuis HF
    metric_model_path = Path(hf_hub_download(
        repo_id=MODEL_REPO, 
        filename="senedisease_macro_f1.pt"
    ))
    
    # Charger le modèle une seule fois
    model = torch.load(metric_model_path, map_location="cpu")
    model.eval()
    
    # ... (charger index, metadata, prototypes)
    return model, index, metadata, prototypes, prototype_labels

@app.on_event("startup")
async def startup_event():
    global model
    model, index, metadata, prototypes, prototype_labels, device = load_model()
    print("✅ Model loaded on startup")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédiction d'une image
    
    Retourne: {
        "predicted_disease": "Apple Scab",
        "confidence": 0.92,
        "topk_neighbors": [...]
    }
    """
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Inférence
        with torch.no_grad():
            result = infer_on_image(
                model, index, metadata, prototypes, 
                prototype_labels, image, device
            )
        
        return {
            "predicted_disease": result["predicted_disease"],
            "confidence": result["predicted_similarity"],
            "topk_neighbors": result["topk_neighbors"],
            "is_unknown": result["is_unknown"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**requirements.txt:**
```
fastapi==0.104.0
uvicorn==0.24.0
pillow==10.0.0
torch==2.1.0
numpy==1.24.0
faiss-cpu==1.7.4
huggingface-hub==0.17.0
```

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Étape 3: Modifier Streamlit pour appeler l'API

```python
import requests
import streamlit as st

API_URL = "https://votre-username-plant-disease-api.hf.space/predict"

def diagnose_via_api(image_bytes: bytes):
    """Appelle le Spaces API"""
    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# Dans Streamlit
if st.button("Diagnose"):
    result = diagnose_via_api(image_bytes)
    if result:
        st.write(f"**Disease**: {result['predicted_disease']}")
        st.write(f"**Confidence**: {result['confidence']:.2%}")
```

---

## 🚀 Solution 2: Replicate API (Pay-as-you-go)

Alternative: https://replicate.com

Créez un modèle Replicate et invoquez depuis Streamlit:

```python
import replicate

prediction = replicate.run(
    "mohamedsamake8322/plant-disease-swin:latest",
    input={"image": open("path.jpg", "rb")}
)
```

---

## 📊 Comparaison

| Approche | RAM Streamlit | Setup | Coût | Latence |
|----------|--------------|-------|------|---------|
| Local | 800MB+ | 5 min | $0 | 50ms |
| **HF Spaces** | **<100MB** | **20 min** | **$0** | **200ms** |
| Replicate | <100MB | 15 min | $0.01/req | 500ms |

---

## ⚡ Status Actuel

✅ Votre modèle est sur HF: `mohamedsamake8322/plant-diseaseS-swin-faiss`
✅ Streamlit utilise `@st.cache_resource` (charge 1x par session)
⚠️ RAM reste ~400-500MB avec optimization

## Prochaines étapes

1. **Rapide** (~30min): Créer un HF Spaces avec FastAPI (RAM ~0 dans Streamlit)
2. **Alternative**: Utiliser Replicate
3. **Local**: Continuer avec optimization actuelle

Besoin d'aide pour créer le Spaces? 
