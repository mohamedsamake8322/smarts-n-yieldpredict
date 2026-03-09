"""
Plant Disease Detection API - Hugging Face Spaces
FastAPI endpoint for plant disease diagnosis using metric learning
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model_core import (
    load_phase2_model_and_metadata,
    infer_on_image,
    preprocess_image_pil,
    MODELS_PATH_PHASE2,
    MODEL_REPO
)

app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease diagnosis using metric learning",
    version="1.0.0"
)

# Global variables for model and data (loaded once at startup)
model = None
index = None
metadata = None
prototypes = None
prototype_labels = None
device = None

def load_model_once():
    """Load model and data once at startup"""
    global model, index, metadata, prototypes, prototype_labels, device

    try:
        print("🔄 Loading model and metadata...")
        model, index, metadata, prototypes, prototype_labels, device = (
            load_phase2_model_and_metadata(MODELS_PATH_PHASE2)
        )

        # Put model in eval mode for inference
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
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
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
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference
        with torch.no_grad():
            result = infer_on_image(
                model=model,
                index=index,
                metadata=metadata,
                prototypes=prototypes,
                prototype_labels=prototype_labels,
                image=image,
                device=device,
                top_k=5,
                unknown_threshold=0.55
            )

        # Format response
        response = {
            "predicted_disease": result["predicted_disease"],
            "predicted_score": float(result["predicted_similarity"]) if result["predicted_similarity"] else None,
            "is_unknown": result["is_unknown"],
            "topk_neighbors": [
                {
                    "rank": n["rank"],
                    "disease": n["disease"],
                    "similarity": float(n["similarity"]),
                    "image_path": n.get("image_path")
                }
                for n in result["topk_neighbors"]
            ],
            "proto_ranking": [
                {
                    "rank": p["rank"],
                    "disease": p["disease"],
                    "similarity": float(p["similarity"])
                }
                for p in result["topk_prototypes"]
            ]
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict diseases for multiple images

    Args:
        files: List of image files

    Returns:
        JSON with batch prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )

    results = []

    for file in files:
        try:
            # Reuse single prediction logic
            prediction = await predict(file)
            results.append({
                "filename": file.filename,
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)