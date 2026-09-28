# Plant Disease Detection API - Hugging Face Spaces (DINOv2)
# Repond au format attendu par pages/1_Detection.py (diagnostic, confidence,
# confidence_pct, is_uncertain, top3). quality/reliability/crop/category/conseil
# sont volontairement omis pour l'instant : le client a deja un repli "-".

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import os

from model_core import load_model_and_metadata, infer_on_image, DEVICE

app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease diagnosis using DINOv2",
    version="2.0.0",
)

model = None
idx_to_class = None


def load_model():
    global model, idx_to_class
    try:
        print("Loading DINOv2 model from Hugging Face Hub...")
        model, idx_to_class, _ = load_model_and_metadata()
        torch.set_grad_enabled(False)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    if not load_model():
        print("Model loading failed - API will return errors")


@app.get("/")
async def root():
    return {
        "message": "Plant Disease Detection API (DINOv2)",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "metadata_classes": len(idx_to_class) if idx_to_class else 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from image.
    Retourne un dict compatible avec 1_Detection.py (etape "detection seule").
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = infer_on_image(model, idx_to_class, image, DEVICE, top_k=3)

        pred_disease = result["predicted_disease"]
        confidence = result["predicted_similarity"] or 0.0
        # Pas encore de calibration/seuil valide pour DINOv2 -> on ne marque
        # jamais "incertain" pour l'instant, le temps de voir les vrais scores
        # sur des images reelles.
        is_uncertain = False

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

        return {
            "diagnostic": pred_disease,
            "confidence": confidence,
            "confidence_pct": f"{confidence*100:.1f}%",
            "is_uncertain": is_uncertain,
            "top3": top3,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)