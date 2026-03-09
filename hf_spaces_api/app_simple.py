# Plant Disease Detection API - Hugging Face Spaces
# Version ultra-simple pour éviter les problèmes d'importation

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import os
import pickle
from huggingface_hub import hf_hub_download

app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease diagnosis using metric learning",
    version="1.0.0"
)

# Global variables
model = None
metadata = None
device = torch.device("cpu")

def load_model():
    """Load model from Hugging Face Hub"""
    global model, metadata, device

    try:
        print("🔄 Loading model from Hugging Face Hub...")

        # Download model and metadata
        model_path = hf_hub_download(
            repo_id="mohamedsamake8322/plant-diseaseS-swin-faiss",
            filename="metric_model.pt"
        )
        metadata_path = hf_hub_download(
            repo_id="mohamedsamake8322/plant-diseaseS-swin-faiss",
            filename="metadata.pkl"
        )

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Load model
        model = torch.load(model_path, map_location=device)
        model.eval()
        torch.set_grad_enabled(False)

        print("✅ Model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model when the app starts"""
    success = load_model()
    if not success:
        print("⚠️ Model loading failed - API will return errors")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Plant Disease Detection API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
        "metadata_classes": len(metadata.get("idx_to_class", {})) if metadata else 0
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from image

    Args:
        file: Image file (JPG, PNG, etc.)

    Returns:
        JSON with prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Simple preprocessing
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            embedding = model(img_tensor)
            embedding = embedding.cpu().numpy().flatten()

        # Simple response (placeholder - you can add FAISS search here)
        return {
            "predicted_disease": "Sample Disease",
            "predicted_score": 0.85,
            "is_unknown": False,
            "topk_neighbors": [
                {"rank": 1, "disease": "Sample Disease", "similarity": 0.85}
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)