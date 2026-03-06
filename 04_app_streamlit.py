"""
STREAMLIT APP - Interactive Diagnosis Interface
Utilisation: streamlit run 04_app_streamlit.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import pickle
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Workaround minimal pour éviter les warnings autour de torch.classes.__path__
try:
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except Exception:
    pass

try:
    import streamlit_authenticator as stauth  # type: ignore
except Exception:
    stauth = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from model_core import (
    MODELS_PATH_PHASE2,
    DATASET_ROOT_LOCAL,
    DEVICE,
    load_phase2_model_and_metadata,
    infer_on_image,
)

from utils.gradcam import generate_gradcam

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="🌾 Plant Disease Diagnostic System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
<style>
.main {
    background-color: #f8f9fa;
}
.block-container {
    padding-top: 2rem;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
}
img {
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

"""
LOAD MODEL & INDEX (cached)
On pointe vers les artefacts de la phase 2 (Swin Base production),
et on passe par model_core pour garantir l'unicité de la logique IA.
"""

MODELS_PATH = MODELS_PATH_PHASE2


@st.cache_resource
def load_model_and_index():
    try:
        model, index, metadata, prototypes, prototype_labels, device = (
            load_phase2_model_and_metadata(MODELS_PATH)
        )
    except Exception as e:
        st.error(f"❌ Error loading model/metadata: {e}")
        st.stop()
    return model, index, metadata, prototypes, prototype_labels, device

def preprocess_image(image, size=224):
    """Prétraitement (conserve la signature Streamlit mais délègue à model_core)."""
    from model_core import preprocess_image_pil

    return preprocess_image_pil(image, size=size)

def _map_image_path_to_local(raw_path: str) -> str:
    """Raccourci vers model_core.map_image_path_to_local."""
    from model_core import map_image_path_to_local

    return map_image_path_to_local(raw_path)


def diagnose(
    model,
    index,
    metadata,
    prototypes,
    prototype_labels,
    image,
    device,
    k=5,
    unknown_threshold: float = 0.55,
):
    """Appelle model_core.infer_on_image puis adapte le resultat au format Streamlit."""

    result = infer_on_image(
        model=model,
        index=index,
        metadata=metadata,
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        image=image,
        device=device,
        top_k=k,
        unknown_threshold=unknown_threshold,
    )

    # Adaptation pour l'UI Streamlit
    results = []
    for n in result["topk_neighbors"]:
        results.append(
            {
                "rank": n["rank"],
                "disease": n["disease"],
                "confidence": n["similarity"],
                "path": n.get("image_path"),
            }
        )

    diagnosis = {
        "predicted_label": result["predicted_label"],
        "predicted_disease": result["predicted_disease"],
        "predicted_score": result["predicted_similarity"],
        "is_unknown": result["is_unknown"],
        "proto_ranking": result["topk_prototypes"],
    }

    return results, diagnosis

# ============================================================================
# UI
# ============================================================================
st.title("🌾 Plant Disease Diagnostic System")
st.markdown("""
This AI system diagnoses plant diseases from images by finding visually similar 
examples from a training dataset. *Not a classification system - a diagnostic assistant.*
""")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of similar images (K)", 1, 10, 5)
    show_ref_images = st.checkbox("Show reference images", value=True)
    unknown_threshold = st.slider(
        "Unknown threshold (prototype similarity)",
        min_value=0.3,
        max_value=0.9,
        value=0.55,
        step=0.01,
    )

# Load models
try:
    model, index, metadata, prototypes, prototype_labels, device = load_model_and_index()
    st.success("✅ Models loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ============================================================================
# MAIN INTERFACE
# ============================================================================
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📸 Input Image")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload plant image", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Image")
        
        # Run diagnosis
        if st.button("🔍 Diagnose"):
            progress_bar = st.progress(0)

            with st.spinner("Analyzing..."):
                progress_bar.progress(30)
                results, diagnosis = diagnose(
                    model,
                    index,
                    metadata,
                    prototypes,
                    prototype_labels,
                    image,
                    device,
                    k=k,
                    unknown_threshold=unknown_threshold,
                )
                progress_bar.progress(100)

                # Grad-CAM heatmap (si possible)
                try:
                    # On réutilise le prétraitement pour obtenir le tensor d'entrée
                    img_tensor = preprocess_image(image, size=metadata.get("image_size", 224)).to(device)
                    # Cible par défaut: on tente d'utiliser un backbone du modèle, sinon le modèle complet
                    target_layer = getattr(model, "backbone", None) or model
                    cam = generate_gradcam(model, img_tensor, target_layer)

                    # Mise à l'échelle et superposition
                    cam_resized = cv2.resize(cam, image.size)
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    base = image.convert("RGB")
                    heatmap_img = Image.fromarray(heatmap).convert("RGB")
                    overlay = Image.blend(base, heatmap_img, alpha=0.4)

                    st.session_state.gradcam_overlay = overlay
                except Exception:
                    st.session_state.gradcam_overlay = None
            
            # Save results for display
            st.session_state.results = results
            st.session_state.diagnosis = diagnosis
            st.session_state.uploaded_image = image

with col2:
    st.header("📊 Diagnosis Results")
    
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        diagnosis = st.session_state.get("diagnosis", {})

        pred_disease = diagnosis.get("predicted_disease")
        pred_score = diagnosis.get("predicted_score")
        is_unknown = diagnosis.get("is_unknown", False)

        st.subheader("🦠 Probable Disease")

        if is_unknown:
            st.error("Unknown Disease – Please consult an expert.")
        else:
            st.markdown(f"## 🌿 {pred_disease}")

        # Informations textuelles sur la maladie (JSON externe)
        try:
            with open("data/disease_info.json", "r", encoding="utf-8") as f:
                disease_info = json.load(f)
        except Exception:
            disease_info = {}

        if (not is_unknown) and pred_disease and pred_disease in disease_info:
            info = disease_info[pred_disease]
            st.markdown("### 📖 Disease Information")
            st.write(info.get("description", ""))
            if info.get("symptoms"):
                st.markdown("**Symptoms:**")
                st.write(info["symptoms"])
            if info.get("treatment"):
                st.markdown("**Treatment:**")
                st.write(info["treatment"])
        
        # Heatmap Grad-CAM si disponible
        if st.session_state.get("gradcam_overlay") is not None:
            st.markdown("### 🔥 Grad-CAM Heatmap")
            st.image(
                st.session_state["gradcam_overlay"],
                use_column_width=True,
                caption="Regions most influential for the model (Grad-CAM).",
            )
    else:
        st.info("👆 Upload an image and click 'Diagnose' to start")

# ============================================================================
# VISUAL CONFIRMATION (Plantix Style)
# ============================================================================
if "results" in st.session_state and st.session_state.results:
    st.divider()
    st.subheader("🔎 Visual Confirmation")

    diagnosis = st.session_state.get("diagnosis", {})
    pred_class = diagnosis.get("predicted_disease")

    if pred_class and pred_class != "UNKNOWN DISEASE":
        confirmation_images = [
            r for r in st.session_state.results if r["disease"] == pred_class
        ][:4]
    else:
        confirmation_images = st.session_state.results[:4]

    if confirmation_images:
        cols = st.columns(len(confirmation_images))

        for col, result in zip(cols, confirmation_images):
            with col:
                try:
                    ref_image = Image.open(result["path"])
                    st.image(ref_image, use_column_width=True)
                except Exception:
                    st.warning("Image not available")

    if st.button("❓ Not matching?"):
        st.info("Try another image or consult an agronomist.")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()

st.markdown("""
### 📝 How it works:

1. **Upload** an image of a plant disease
2. **AI analyzes** the visual patterns
3. **Finds similar** images from training dataset
4. **Shows diagnostic** options for validation

### ⚠️ Important:

- This is a **diagnostic assistant**, not a classifier
- Always validate with domain experts
- Based on visual similarity to **26,203 training examples**
- Can process new disease classes **without retraining**

### 🚀 Features:

- ✅ Fast inference (~50ms per image)
- ✅ Transparent: shows reference images
- ✅ Scalable: add classes without retraining
- ✅ Confidence scores based on similarity
""")
