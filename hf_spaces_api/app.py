"""
Plant Disease Detection API - Hugging Face Spaces
FastAPI endpoint for plant disease diagnosis using DINOv2.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch

# Import from local model_core (same directory)
from model_core import (
    load_model_and_metadata,
    infer_on_image,
    MODEL_REPO,
)

try:
    from model_core import LOADED_MODEL_PATH
except Exception:
    LOADED_MODEL_PATH = None

app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease diagnosis using DINOv2",
    version="2.0.0",
)

# Global variables for model and data (loaded once at startup)
model = None
idx_to_class = None
device = None


def load_model_once():
    """Load model and data once at startup"""
    global model, idx_to_class, device

    try:
        print("🔄 Loading DINOv2 model and metadata...")
        model, idx_to_class, device = load_model_and_metadata()
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
    success = load_model_once()
    if not success:
        print("⚠️ Model loading failed - API will return errors")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Plant Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "metadata_classes": len(idx_to_class) if idx_to_class else 0,
    }


@app.get("/version")
async def version():
    """Return model path/version for debugging which checkpoint is loaded."""
    return {
        "model_loaded": model is not None,
        "loaded_model_path": LOADED_MODEL_PATH,
        "model_repo": MODEL_REPO,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from image.

    Retourne un dict compatible avec pages/1_Detection.py (etape "detection
    seule") : diagnostic, confidence, confidence_pct, is_uncertain, top3.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = infer_on_image(model, idx_to_class, image, device, top_k=5)

        pred_disease = result["predicted_disease"]
        confidence = result["predicted_similarity"] or 0.0

        top3 = [
            {
                "rank": r["rank"],
                "display_name": r["disease"],
                "disease": r["disease"],
                "confidence": r["similarity"],
                "confidence_pct": f"{r['similarity']*100:.1f}%",
            }
            for r in result["topk_prototypes"][:3]
        ]

        # Seuil provisoire, pas encore calibre (pas de temperature scaling
        # pour DINOv2). A ajuster une fois qu'on aura mesure la distribution
        # reelle des confiances sur un jeu de validation.
        UNCERTAIN_THRESHOLD = 0.30
        is_uncertain = confidence < UNCERTAIN_THRESHOLD

        return {
            "diagnostic": pred_disease,
            "confidence": confidence,
            "confidence_pct": f"{confidence*100:.1f}%",
            "is_uncertain": is_uncertain,
            "top3": top3,
            # Champs conserves pour compatibilite avec d'anciens appelants
            "predicted_disease": pred_disease,
            "predicted_score": confidence,
            "is_unknown": is_uncertain,
            "topk_neighbors": [],
            "proto_ranking": [
                {"rank": p["rank"], "disease": p["disease"], "similarity": p["similarity"]}
                for p in result["topk_prototypes"]
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)