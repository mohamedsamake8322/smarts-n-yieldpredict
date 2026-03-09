# Plant Disease Detection API

🚀 **Zero-RAM Streamlit** - API endpoint for plant disease diagnosis using metric learning

## 🌟 Features

- ✅ **FastAPI** backend with automatic OpenAPI docs
- ✅ **Metric Learning** with FAISS for similar image search
- ✅ **Batch processing** support
- ✅ **Health checks** and monitoring
- ✅ **Docker deployment** ready

## � Déploiement sur HF Spaces

### Méthode Automatique (Recommandée)

1. **Définir votre token HF:**
   ```bash
   export HUGGINGFACE_TOKEN="votre_token_ici"
   ```

2. **Lancer le déploiement automatique:**
   ```bash
   # Sur Linux/Mac
   ./deploy_to_spaces.sh

   # Sur Windows
   .\deploy_to_spaces.ps1
   ```

### Méthode Manuelle

1. **Aller sur votre Space HF:** https://huggingface.co/spaces/mohamedsamake8322/sene-disease-api

2. **Uploader ces fichiers:**
   - `app_simple.py` → renommer en `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`

3. **Le Space va redémarrer automatiquement**

## �📋 API Endpoints

### `GET /`
Health check endpoint

### `GET /health`
Detailed health status

### `POST /predict`
Predict disease from single image

**Request:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "predicted_disease": "Apple Scab",
  "predicted_score": 0.92,
  "is_unknown": false,
  "topk_neighbors": [...],
  "proto_ranking": [...]
}
```

### `POST /batch_predict`
Predict diseases from multiple images

**Request:**
- `files`: Multiple image files

## 🚀 Deployment on Hugging Face Spaces

### Step 1: Create New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Choose:
   - **Space SDK**: Docker
   - **License**: MIT
   - **Name**: `plant-disease-api`

### Step 2: Upload Files

Upload these files to your Space:
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`

### Step 3: Configure Space

In your Space settings:
- **Repository**: `mohamedsamake8322/plant-disease-api`
- **Visibility**: Public
- **Hardware**: CPU Basic (free) or GPU if needed

### Step 4: Deploy

The Space will automatically build and deploy using the Dockerfile.

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app:app --reload --host 0.0.0.0 --port 7860

# Test API
curl -X POST "http://localhost:7860/health"
```

## 📊 Integration with Streamlit

Replace your current Streamlit code with:

```python
import requests
import streamlit as st

API_URL = "https://mohamedsamake8322-plant-disease-api.hf.space/predict"

def diagnose_via_api(image_bytes: bytes):
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

# In your Streamlit app
if st.button("Diagnose"):
    result = diagnose_via_api(image_bytes)
    if result:
        st.write(f"**Disease**: {result['predicted_disease']}")
        st.write(f"**Confidence**: {result['predicted_score']:.2%}")
```

## 📈 Performance

- **RAM Usage**: ~400MB (model loaded once)
- **Inference Time**: ~200-500ms per image
- **Concurrent Requests**: 1-2 (limited by HF Spaces free tier)

## 🔒 Security

- Input validation for image files
- Error handling with appropriate HTTP status codes
- No sensitive data exposure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details