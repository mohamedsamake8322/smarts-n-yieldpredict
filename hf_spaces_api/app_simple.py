# Plant Disease Detection API - Hugging Face Spaces (DINOv2)

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
        print("🔄 Loading DINOv2 model from Hugging Face Hub...")
        model, idx_to_class, _ = load_model_and_metadata()
        torch.set_grad_enabled(False)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    if not load_model():
        print("⚠️ Model loading failed - API will return errors")


@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API (DINOv2)", "status": "running", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": len(idx_to_class) if idx_to_class else 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return infer_on_image(model, idx_to_class, image, DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)