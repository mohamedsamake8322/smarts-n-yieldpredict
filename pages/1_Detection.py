"""
Disease detection page – Plantix mode:
- Identifies ONLY the disease
- Displays the name + some similar images
"""

# WORKAROUND: Prevent Streamlit from inspecting torch.classes (compatibility issue)
import os
import sys
from pathlib import Path
import io
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply PyTorch/Streamlit workaround
try:
    from utils.pytorch_fix import apply_pytorch_fix
    apply_pytorch_fix()
except ImportError:
    # Minimal fallback: simply neutralize __path__ if present
    try:
        import torch

        if hasattr(torch, "classes"):
            torch.classes.__path__ = []
    except (ImportError, AttributeError):
        pass

import streamlit as st
from PIL import Image

from model_core import (
    MODELS_PATH_PHASE2,
    load_phase2_model_and_metadata,
    infer_on_image,
)
from utils.blip2_explainer import generate_explanation_for_image, load_disease_info
from utils.i18n import language_selector, get_lang, t, LANGUAGE_OPTIONS
import numpy as np


# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Detection - Agro-Scan",
    page_icon="📸",
    layout="wide",
)

from utils.styles import load_custom_css

load_custom_css()

# Session state initialization
def init_session_state():
    defaults = {
        "uploaded_image": None,
        "image_bytes": None,
        "uploaded_image_path": None,
        "lang": "en",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar language selector
language_selector()

# Folder for light images for visual confirmation
DATASET_LIGHT_ROOT = Path("dataset_light")


def _map_to_dataset_light(original_path: str) -> str:
    """
    Tries to remap an image path from the full dataset to dataset_light.
    Keeps the same relative structure from 'dataset_final' if possible.
    """
    try:
        p = Path(original_path)
        parts = p.parts
        if "dataset_final" in parts:
            idx = parts.index("dataset_final")
            rel = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path(".")
            candidate = DATASET_LIGHT_ROOT / rel
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass
    # Fallback: keep the original path if nothing found in dataset_light
    return original_path


def _get_light_images_for_disease(disease_name: str, max_images: int = 4) -> list[str]:
    """
    Retrieves up to max_images images in dataset_light for a given disease.
    Tries several folder name variants for robustness
    (spaces vs underscores).
    """
    candidates = [
        disease_name,
        disease_name.replace(" ", "_"),
        disease_name.replace(" ", ""),
    ]

    for name in candidates:
        disease_dir = DATASET_LIGHT_ROOT / name
        if disease_dir.exists() and disease_dir.is_dir():
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                imgs.extend(sorted(str(p) for p in disease_dir.glob(ext)))
            if imgs:
                return imgs[:max_images]

    return []


def _is_plant_like(image: Image.Image, green_threshold: float = 0.18) -> bool:
    """
    Simple heuristic to filter images that are obviously not crop leaves
    (based on green channel dominance).
    This is not a classifier, just a UX safeguard.
    """
    try:
        arr = np.array(image.convert("RGB")).astype("float32") / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        green_dominance = (g > r) & (g > b)
        ratio_green = float(green_dominance.mean())
        return ratio_green >= green_threshold
    except Exception:
        return True  # in case of doubt, let the AI decide


# --------------------------
# Simple internationalization
# --------------------------

SUPPORTED_LANGS = {code: label for label, code in LANGUAGE_OPTIONS}

# Titre
st.title(t("page_title"))
st.markdown(t("page_subtitle"))

@st.cache_resource
def load_model_and_index():
    """Unique loading of the metric learning model and FAISS index."""
    model, index, metadata, prototypes, prototype_labels, device = (
        load_phase2_model_and_metadata(MODELS_PATH_PHASE2)
    )
    return model, index, metadata, prototypes, prototype_labels, device


def _is_confident_unknown(result, faiss_threshold: float = 0.55, min_agreement: int = 3):
    """
    Double logic pro for 'unknown disease':
    - threshold on prototype similarity (already handled in infer_on_image)
    - coherence of FAISS neighbors (at least min_agreement of the same class)
    """
    is_unknown_flag = result.get("is_unknown", False)
    neighbors = result.get("topk_neighbors", [])

    # If there are no neighbors (FAISS missing, empty index, etc.),
    # we rely only on the internal logic of model_core.
    if not neighbors:
        return bool(is_unknown_flag)

    # Check if the top neighbors share the same class
    top = neighbors[:min_agreement]
    diseases = [n["disease"] for n in top]
    main = diseases[0]
    same_count = sum(1 for d in diseases if d == main)

    # We mark as unknown only if:
    # - the internal model already considers it unknown
    #   AND
    # - the neighbors do not agree among themselves.
    return bool(is_unknown_flag) and same_count < min_agreement


def diagnose_image(
    image: Image.Image,
    uploaded_image_path: str | None = None,
    k: int = 5,
    unknown_threshold: float = 0.55,
):
    """Common inference pipeline for this page + Plantix logic."""
    (
        model,
        index,
        metadata,
        prototypes,
        prototype_labels,
        device,
    ) = load_model_and_index()

    # Resize for mobile / performance
    image_resized = image.resize((224, 224))

    result = infer_on_image(
        model=model,
        index=index,
        metadata=metadata,
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        image=image_resized,
        device=device,
        top_k=k,
        unknown_threshold=unknown_threshold,
    )

    # More robust "unknown" overlay
    is_unknown = _is_confident_unknown(result, faiss_threshold=unknown_threshold)

    neighbors = result["topk_neighbors"]

    # Avoid showing the uploaded image in neighbors (if same path)
    filtered_neighbors = []
    for n in neighbors:
        path = n.get("image_path")
        if uploaded_image_path and path and Path(path) == Path(uploaded_image_path):
            continue
        filtered_neighbors.append(n)

    diagnosis = {
        "predicted_disease": result["predicted_disease"],
        "is_unknown": is_unknown,
        "neighbors": filtered_neighbors,
    }
    return diagnosis

# Sidebar
st.sidebar.title("📸 Detection")

# Language selector
current_lang = get_lang()
lang_options = [f"{code.upper()} - {label}" for code, label in SUPPORTED_LANGS.items()]
default_index = list(SUPPORTED_LANGS.keys()).index(current_lang)
selected = st.sidebar.selectbox("🌐 Langue / Language", lang_options, index=default_index)
for code, label in SUPPORTED_LANGS.items():
    if selected.startswith(code.upper()):
        st.session_state["lang"] = code
        break

st.sidebar.markdown("### Instructions")
st.sidebar.markdown(t("sidebar_instructions"))

# Capture/upload area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Take a photo")
    camera_input = st.camera_input("Capture an image", label_visibility="collapsed")
    
    if camera_input:
        try:
            # Get the image bytes
            image_bytes = camera_input.getvalue()
            if not isinstance(image_bytes, bytes):
                image_bytes = bytes(image_bytes)
            
            # Check that bytes are not empty
            if len(image_bytes) == 0:
                st.error("❌ The captured image is empty. Please try again.")
            else:
                # Open the image to display
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.session_state.uploaded_image = image
                st.session_state.image_bytes = image_bytes
                st.session_state.uploaded_image_path = None
        except Exception as e:
            st.error(f"❌ Error processing the image: {str(e)}")

with col2:
    st.subheader("📁 Upload an image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            # Read the file bytes
            uploaded_file.seek(0)  # Ensure we're at the beginning
            image_bytes = uploaded_file.read()
            
            if not isinstance(image_bytes, bytes):
                image_bytes = bytes(image_bytes)
            
            # Check that bytes are not empty
            if len(image_bytes) == 0:
                st.error("❌ The image file is empty. Please select another file.")
            else:
                # Open the image to display
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.session_state.uploaded_image = image
                st.session_state.image_bytes = image_bytes
                st.session_state.uploaded_image_path = uploaded_file.name
                
                # Reset the pointer for later use
                uploaded_file.seek(0)
        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

# Display the selected image
if 'uploaded_image' in st.session_state:
    st.markdown("---")
    st.subheader(t("image_to_analyze"))
    
    st.markdown("#### Analyzed image")
    # Use image_bytes instead of PIL object to avoid format issues
    if 'image_bytes' in st.session_state and st.session_state.image_bytes:
        st.image(st.session_state.image_bytes, width=None)
    elif st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, width=None)
    
    # Detection button
    if st.button(t("analyze_button"), type="primary", use_container_width=True):
        with st.spinner("🔬 Analyzing image..."):
            try:
                # Open the image
                image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")

                # Safeguard: obviously non-plant image
                if not _is_plant_like(image):
                    st.error(t("not_leaf_message"))
                    st.session_state.show_results = False
                else:
                    # Launch the metric learning pipeline
                    diagnosis = diagnose_image(
                        image,
                        uploaded_image_path=st.session_state.get("uploaded_image_path"),
                        k=5,
                        unknown_threshold=0.55,
                    )

                    st.session_state.detection_result = diagnosis
                    st.session_state.image_for_explanation = image
                    st.session_state.show_results = True
                
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.session_state.show_results = False

# Display results – Plantix mode (name + similar images)
if st.session_state.get("show_results", False) and "detection_result" in st.session_state:
    diagnosis = st.session_state.detection_result

    st.markdown("---")
    st.success(t("analysis_done"))

    pred_disease = diagnosis.get("predicted_disease")
    is_unknown = diagnosis.get("is_unknown", False)
    neighbors = diagnosis.get("neighbors", [])

    st.subheader(t("probable_disease"))
    if is_unknown or not pred_disease or pred_disease == "UNKNOWN DISEASE":
        st.error(t("unknown_disease"))
    else:
        st.markdown(f"## 🌿 {pred_disease}")

    # Load the base knowledge from JSON (deterministic source of truth)
    if pred_disease and pred_disease != "UNKNOWN DISEASE":
        disease_data = load_disease_info(
            pred_disease,
            allow_fuzzy=False,
            language_code=get_lang(),
        )
    else:
        disease_data = {
            "disease": pred_disease or "Unknown Disease",
            "symptoms": [],
            "cause": "",
            "management": [],
        }

    # Confidence
    confidence = diagnosis.get("predicted_similarity")
    if confidence is not None:
        st.markdown(f"**Confidence:** {confidence*100:.0f}%")

    # Show symptoms / cause / management directly from JSON (authoritative)
    st.markdown(t("symptoms_heading"))
    if disease_data.get("symptoms"):
        for item in disease_data["symptoms"]:
            st.markdown(f"- {item}")
    else:
        st.markdown(t("no_data"))

    st.markdown(t("cause_heading"))
    if disease_data.get("cause"):
        st.markdown(disease_data["cause"])
    else:
        st.markdown(t("no_data"))

    st.markdown(t("management_heading"))
    if disease_data.get("management"):
        for item in disease_data["management"]:
            st.markdown(f"- {item}")
    else:
        st.markdown(t("no_data"))

    source_file = disease_data.get("_source_file")
    if source_file:
        st.caption(f"🔗 Source JSON: {source_file}")

    # Visual confirmation: 3–4 images from dataset_light for the predicted class
    if not is_unknown and pred_disease:
        st.markdown(t("visual_confirmation"))

        light_images = _get_light_images_for_disease(pred_disease, max_images=4)

        if light_images:
            cols = st.columns(min(3, len(light_images)))
            for idx, img_path in enumerate(light_images):
                col = cols[idx % len(cols)]
                with col:
                    try:
                        ref_img = Image.open(img_path).convert("RGB")
                        st.image(ref_img, width=None)
                    except Exception:
                        st.warning("Image not available")

        # Optional: generate an AI explanation for the predicted disease
        if "image_for_explanation" in st.session_state:
            if st.checkbox("Generate AI explanation", value=False):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = generate_explanation_for_image(
                            st.session_state.image_for_explanation,
                            pred_disease,
                            language_code=get_lang(),
                        )
                        st.markdown("### 🤖 AI Explanation")
                        st.write(explanation)
                    except Exception as e:
                        st.error(f"⚠️ Failed to generate explanation: {e}")

    # Button for new detection
    if st.button("🔄 New detection", use_container_width=True):
        for key in [
            "uploaded_image",
            "image_bytes",
            "image_for_explanation",
            "detection_result",
            "show_results",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

