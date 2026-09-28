"""
FastAPI backend pour le modele de production (DINOv2).

Objectif (etape actuelle - detection seule):
- Recevoir une image (upload multipart)
- Retourner la maladie predite + confiance (softmax) + top-k

La description, les symptomes et le traitement seront ajoutes dans une
etape suivante, une fois la detection validee.

Usage local:
    uvicorn disease_api:app --reload --host 0.0.0.0 --port 8000

Le modele est telecharge automatiquement depuis Hugging Face
(mohamedsamake8322/maladie-plantes-dinov2) au demarrage.
"""

import io
import os
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from model_core import (
    DEVICE,
    load_model_and_metadata,
    infer_on_image,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Chargement du modele DINOv2 (au demarrage)
# ---------------------------------------------------------------------------

model, idx_to_class, DEVICE = load_model_and_metadata()


# ---------------------------------------------------------------------------
# Schemas de reponse FastAPI
# ---------------------------------------------------------------------------


class PrototypeRank(BaseModel):
    rank: int
    disease: str
    similarity: float


class DiagnosisResponse(BaseModel):
    disease: str
    similarity: Optional[float]
    is_unknown: bool
    topk_prototypes: List[PrototypeRank]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Plant Disease Detection API (DINOv2)",
    description="Detection de maladies de plantes par classification DINOv2. "
    "Etape 1 : detection seule. Description/symptomes/traitement a venir.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, summary="Interface web de debug")
async def index():
    html = """
    <html>
      <head>
        <title>Plant Disease Diagnostic (DINOv2)</title>
        <style>
          body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 900px; margin: 2rem auto; }
          h1 { color: #1b6e3b; }
          .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; margin-top: 1.5rem; }
          .unknown { color: #b45309; font-weight: 600; }
          table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
          th, td { border: 1px solid #eee; padding: 0.35rem 0.5rem; text-align: left; font-size: 0.9rem; }
          th { background: #f9fafb; }
          label { display: block; margin-top: 0.8rem; }
          input[type="file"] { margin-top: 0.3rem; }
          input[type="number"] { padding: 0.25rem 0.35rem; }
          button { margin-top: 1rem; padding: 0.4rem 0.9rem; border-radius: 999px; border: none; background: #15803d; color: white; cursor: pointer; font-weight: 500; }
          button:hover { background: #166534; }
        </style>
      </head>
      <body>
        <h1>Plant Disease Diagnostic - DINOv2 (etape 1: detection)</h1>
        <p>Upload une image pour obtenir la maladie predite et le top-k.</p>

        <form action="/web/diagnose" method="post" enctype="multipart/form-data">
          <label>Image:
            <input name="file" type="file" accept="image/*" required>
          </label>
          <label>Top K:
            <input name="top_k" type="number" value="5" min="1" max="10">
          </label>
          <button type="submit">Diagnose</button>
        </form>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "num_classes": len(idx_to_class) if idx_to_class else 0,
        "model": "dinov2",
    }


@app.post("/diagnose", response_model=DiagnosisResponse, summary="Diagnose plant disease")
async def diagnose_endpoint(
    file: UploadFile = File(...),
    top_k: int = TOP_K_DEFAULT,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image")

    result = infer_on_image(model, idx_to_class, image, DEVICE, top_k=top_k)

    return DiagnosisResponse(
        disease=result["predicted_disease"] or "UNKNOWN DISEASE",
        similarity=result["predicted_similarity"],
        is_unknown=result["is_unknown"],
        topk_prototypes=[
            PrototypeRank(rank=r["rank"], disease=r["disease"], similarity=r["similarity"])
            for r in result["topk_prototypes"]
        ],
    )


@app.post("/web/diagnose", response_class=HTMLResponse, summary="Diagnose via interface web")
async def diagnose_web(
    file: UploadFile = File(...),
    top_k: int = Form(TOP_K_DEFAULT),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image")

    result = infer_on_image(model, idx_to_class, image, DEVICE, top_k=top_k)

    disease_name = result["predicted_disease"] or "UNKNOWN DISEASE"
    similarity = result["predicted_similarity"]

    def fmt(x):
        return f"{x:.2%}" if x is not None else "N/A"

    rows_proto = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['disease']}</td><td>{fmt(r['similarity'])}</td></tr>"
        for r in result["topk_prototypes"]
    )

    html = f"""
    <html>
      <head>
        <title>Diagnostic result</title>
        <meta charset="utf-8" />
      </head>
      <body>
        <a href="/">&larr; Nouvelle image</a>
        <div class="card">
          <h2>Diagnostic</h2>
          <p><strong>Maladie:</strong> {disease_name}</p>
          <p><strong>Confiance:</strong> {fmt(similarity)}</p>
        </div>

        <div class="card">
          <h3>Top-{top_k} predictions</h3>
          <table>
            <tr><th>Rank</th><th>Maladie</th><th>Confiance</th></tr>
            {rows_proto}
          </table>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "disease_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )