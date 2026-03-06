"""
FastAPI backend pour le modèle de production (Phase 2 - Swin Base).

Objectif:
- Recevoir une image (upload multipart)
- Retourner:
    - maladie prédite (prototype-based)
    - score de similarité
    - indicateur UNKNOWN si sous un seuil
    - top‑k voisins FAISS (optionnel)

Usage local:
    uvicorn disease_api:app --reload --host 0.0.0.0 --port 8000

Les artefacts du modèle doivent être présents sous:
    ./outputs/phase2_swin_base_production/models/
"""

import io
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from model_core import (
    MODELS_PATH_PHASE2,
    DEVICE,
    load_phase2_model_and_metadata,
    infer_on_image,
)

# ---------------------------------------------------------------------------
# Chemins et constantes
# ---------------------------------------------------------------------------

UNKNOWN_THRESHOLD_DEFAULT = float(os.getenv("UNKNOWN_THRESHOLD", "0.55"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Chargement du modèle et des artefacts (au démarrage)
# ---------------------------------------------------------------------------

model, index, metadata, prototypes, prototype_labels, DEVICE = load_phase2_model_and_metadata(
    MODELS_PATH_PHASE2
)


# ---------------------------------------------------------------------------
# Schémas de réponse FastAPI
# ---------------------------------------------------------------------------


class Neighbor(BaseModel):
    rank: int
    disease: str
    similarity: float
    image_path: Optional[str] = None


class PrototypeRank(BaseModel):
    rank: int
    disease: str
    similarity: float


class DiagnosisResponse(BaseModel):
    disease: str
    similarity: Optional[float]
    is_unknown: bool
    topk_prototypes: List[PrototypeRank]
    topk_neighbors: List[Neighbor]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Plant Disease Metric API (Phase 2 - Swin Base)",
    description="API centrale pour le diagnostic par similarite (metric learning) "
    "et interface web de debug.",
    version="1.0.0",
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
    # Interface web minimale: upload + affichage texte des resultats
    html = """
    <html>
      <head>
        <title>Plant Disease Diagnostic (Phase 2)</title>
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
          input[type="number"], input[type="text"] { padding: 0.25rem 0.35rem; }
          button { margin-top: 1rem; padding: 0.4rem 0.9rem; border-radius: 999px; border: none; background: #15803d; color: white; cursor: pointer; font-weight: 500; }
          button:hover { background: #166534; }
        </style>
      </head>
      <body>
        <h1>🌿 Plant Disease Diagnostic — Phase 2 (Swin Base)</h1>
        <p>Upload une image pour obtenir le diagnostic (nom de maladie + similarite) et les voisins les plus proches.</p>

        <form action="/web/diagnose" method="post" enctype="multipart/form-data">
          <label>Image:
            <input name="file" type="file" accept="image/*" required>
          </label>
          <label>Top K voisins:
            <input name="top_k" type="number" value="5" min="1" max="10">
          </label>
          <label>Seuil unknown (prototype similarity):
            <input name="unknown_threshold" type="text" value="0.55">
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
        "has_faiss": faiss is not None and index is not None,
        "num_images": len(metadata.get("image_paths", [])),
        "num_classes": metadata.get("num_classes"),
    }


@app.post("/diagnose", response_model=DiagnosisResponse, summary="Diagnose plant disease")
async def diagnose_endpoint(
    file: UploadFile = File(...),
    top_k: int = TOP_K_DEFAULT,
    unknown_threshold: float = UNKNOWN_THRESHOLD_DEFAULT,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image")

    result = infer_on_image(
        image=image, top_k=top_k, unknown_threshold=unknown_threshold
    )

    disease_name = result["predicted_disease"]
    return DiagnosisResponse(
        disease=disease_name if disease_name is not None else "UNKNOWN DISEASE",
        similarity=result["predicted_similarity"],
        is_unknown=result["is_unknown"],
        topk_prototypes=[
            PrototypeRank(
                rank=r["rank"], disease=r["disease"], similarity=r["similarity"]
            )
            for r in result["topk_prototypes"]
        ],
        topk_neighbors=[
            Neighbor(
                rank=n["rank"],
                disease=n["disease"],
                similarity=n["similarity"],
                image_path=n.get("image_path"),
            )
            for n in result["topk_neighbors"]
        ],
    )


@app.post("/web/diagnose", response_class=HTMLResponse, summary="Diagnose via interface web")
async def diagnose_web(
    file: UploadFile = File(...),
    top_k: int = Form(TOP_K_DEFAULT),
    unknown_threshold: float = Form(UNKNOWN_THRESHOLD_DEFAULT),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image")

    result = infer_on_image(
        image=image, top_k=top_k, unknown_threshold=unknown_threshold
    )

    disease_name = result["predicted_disease"] or "UNKNOWN DISEASE"
    is_unknown = result["is_unknown"]
    similarity = result["predicted_similarity"]

    def fmt(x):
        return f"{x:.2%}" if x is not None else "N/A"

    rows_proto = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['disease']}</td><td>{fmt(r['similarity'])}</td></tr>"
        for r in result["topk_prototypes"]
    )
    rows_neighbors = "".join(
        f"<tr><td>{n['rank']}</td><td>{n['disease']}</td><td>{fmt(n['similarity'])}</td><td>{n.get('image_path','')}</td></tr>"
        for n in result["topk_neighbors"]
    )

    unknown_html = (
        '<p class="unknown">⚠️ Marque comme UNKNOWN DISEASE (similarite sous le seuil).</p>'
        if is_unknown
        else ""
    )

    html = f"""
    <html>
      <head>
        <title>Diagnostic result</title>
        <meta charset="utf-8" />
      </head>
      <body>
        <a href="/">← Nouvelle image</a>
        <div class="card">
          <h2>Diagnostic</h2>
          <p><strong>Maladie:</strong> {disease_name}</p>
          <p><strong>Similarite prototypale:</strong> {fmt(similarity)}</p>
          {unknown_html}
        </div>

        <div class="card">
          <h3>Top-{top_k} prototypes</h3>
          <table>
            <tr><th>Rank</th><th>Maladie</th><th>Similarite</th></tr>
            {rows_proto}
          </table>
        </div>

        <div class="card">
          <h3>Top-{top_k} voisins FAISS</h3>
          <table>
            <tr><th>Rank</th><th>Maladie</th><th>Similarite</th><th>Chemin image (local)</th></tr>
            {rows_neighbors}
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

