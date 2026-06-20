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
from typing import Dict, List, Any

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
import datetime
import json
import asyncio

from utils.blip2_explainer import generate_explanation_for_image, load_disease_info
from utils.i18n import language_selector, get_lang, t, LANGUAGE_OPTIONS
from utils.helpers import get_user_id
from services.database_service import DatabaseService
import numpy as np
import requests

API_BASE_URL = os.getenv("API_URL", "https://mohamedsamake8322-sene-disease-api.hf.space")
API_TIMEOUT = 30


def get_api_url() -> str:
    return os.getenv("API_URL") or API_BASE_URL


def call_hf_api(image_bytes: bytes) -> Dict[str, Any]:
    api_url = get_api_url()
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{api_url}/predict", files=files, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"❌ HF API error: {exc}")
        return {}


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


def save_analyzed_image(image: Image.Image, disease_name: str, confidence: float = None, disease_data: dict = None):
    """
    Saves the analyzed image to a folder named after the detected disease.
    Also saves metadata to local SQLite database.
    
    Creates folder structure: prediction_logs/<disease_name>/
    Returns the path where the image was saved.
    """
    try:
        # Create base prediction logs directory
        base_dir = Path("prediction_logs")
        base_dir.mkdir(exist_ok=True)
        
        # Sanitize disease name for folder creation
        safe_disease_name = disease_name.replace(" ", "_").replace("/", "_")
        disease_dir = base_dir / safe_disease_name
        disease_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}_analysis.jpg"
        image_path = disease_dir / image_filename
        
        # Save the image
        image.save(str(image_path), quality=95)
        
        # Save metadata as JSON
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "disease": disease_name,
            "confidence": float(confidence) if confidence else None,
            "image_file": image_filename
        }
        metadata_filename = f"{timestamp}_metadata.json"
        metadata_path = disease_dir / metadata_filename
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Also save to SQLite database
        try:
            db_service = DatabaseService()
            asyncio.run(db_service.save_analysis(
                user_id=get_user_id(),
                disease_name=disease_name,
                confidence=confidence,
                image_path_log=str(image_path),
                disease_data=disease_data
            ))
        except Exception as db_err:
            st.warning(f"⚠️ Could not save to database: {str(db_err)}")
        
        return str(image_path)
    
    except Exception as e:
        st.warning(f"⚠️ Could not save analyzed image: {str(e)}")
        return None


# --------------------------
# Simple internationalization
# --------------------------

SUPPORTED_LANGS = {code: label for label, code in LANGUAGE_OPTIONS}

# Titre
st.title(t("page_title"))
st.markdown(t("page_subtitle"))

def diagnose_image(
    image: Image.Image,
    uploaded_image_path: str | None = None,
    k: int = 5,
    unknown_threshold: float = 0.55,
):
    """
    Call the Hugging Face API for disease prediction.
    Compatible avec le nouveau format de réponse SmartAgri v2.0 :
      diagnostic, confidence, confidence_pct, is_uncertain,
      quality, quality_score, reliability, crop, scientific_name,
      category, top3, conseil
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()

    result = call_hf_api(image_bytes)
    if not result:
        return {
            "predicted_disease":    "UNKNOWN DISEASE",
            "predicted_similarity": None,
            "is_unknown":           True,
            "neighbors":            [],
            "confidence_pct":       "—",
            "quality":              "—",
            "reliability":          "—",
            "crop":                 "—",
            "scientific_name":      "—",
            "category":             "—",
            "conseil":              "",
            "top3":                 [],
        }

    # Nouveau format : top3 remplace topk_neighbors
    top3 = result.get("top3", [])
    neighbors = [
        {
            "rank":       t.get("rank", i + 1),
            "disease":    t.get("display_name", t.get("disease", "Unknown")),
            "similarity": t.get("confidence", 0.0),
            "image_path": None,
        }
        for i, t in enumerate(top3)
    ]

    # Nom de la maladie : nouveau champ "diagnostic"
    pred_disease = (
        result.get("diagnostic")
        or result.get("predicted_disease")
        or "UNKNOWN DISEASE"
    )
    # Nettoie les emojis/préfixes du diagnostic
    if pred_disease.startswith("🌿 "):
        pred_disease = pred_disease[2:].strip()
    if pred_disease.startswith("⚠️"):
        pred_disease = pred_disease.replace("⚠️", "").strip()

    # Confiance : nouveau champ "confidence"
    confidence = (
        result.get("confidence")
        or result.get("predicted_score")
    )

    # Incertitude : nouveau champ "is_uncertain"
    is_unknown = result.get("is_uncertain", result.get("is_unknown", True))

    return {
        "predicted_disease":    pred_disease,
        "predicted_similarity": confidence,
        "is_unknown":           is_unknown,
        "neighbors":            neighbors,
        "confidence_pct":       result.get("confidence_pct", f"{confidence*100:.1f}%" if confidence else "—"),
        "quality":              result.get("quality", "—"),
        "reliability":          result.get("reliability", "—"),
        "crop":                 result.get("crop", "—"),
        "scientific_name":      result.get("scientific_name", "—"),
        "category":             result.get("category", "—"),
        "conseil":              result.get("conseil", ""),
        "top3":                 top3,
    }


def _is_confident_unknown(result, faiss_threshold: float = 0.55, min_agreement: int = 3):
    """
    Double logic pro for 'unknown disease':
    - threshold on prototype similarity
    - coherence of HF neighbor responses
    """
    is_unknown_flag = result.get("is_unknown", False)
    neighbors = result.get("neighbors", [])

    if not neighbors:
        return bool(is_unknown_flag)

    top = neighbors[:min_agreement]
    diseases = [n["disease"] for n in top if n.get("disease")]
    if not diseases:
        return bool(is_unknown_flag)
    main = diseases[0]
    same_count = sum(1 for d in diseases if d == main)

    return bool(is_unknown_flag) and same_count < min_agreement

# Sidebar
st.sidebar.title("📸 Detection")

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

st.sidebar.markdown(" Instructions")
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
    
    st.markdown(" Analyzed image")
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
                    
                    # Save the analyzed image to prediction_logs folder
                    pred_disease = diagnosis.get("predicted_disease", "Unknown")
                    confidence   = diagnosis.get("predicted_similarity") or diagnosis.get("confidence")
                    if pred_disease and pred_disease != "UNKNOWN DISEASE":
                        save_analyzed_image(image, pred_disease, confidence, diagnosis)
                
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.session_state.show_results = False

# Display results – Plantix mode (name + similar images)
if st.session_state.get("show_results", False) and "detection_result" in st.session_state:
    diagnosis = st.session_state.detection_result

    st.markdown("---")
    st.success(t("analysis_done"))

    pred_disease   = diagnosis.get("predicted_disease")
    is_unknown     = diagnosis.get("is_unknown", False)
    neighbors      = diagnosis.get("neighbors", [])
    confidence     = diagnosis.get("predicted_similarity")
    confidence_pct = diagnosis.get("confidence_pct", f"{confidence*100:.1f}%" if confidence else "—")
    quality        = diagnosis.get("quality", "—")
    reliability    = diagnosis.get("reliability", "—")
    crop           = diagnosis.get("crop", "—")
    sci_name       = diagnosis.get("scientific_name", "—")
    category       = diagnosis.get("category", "—")
    conseil        = diagnosis.get("conseil", "")

    st.subheader(t("probable_disease"))

    if is_unknown or not pred_disease or pred_disease == "UNKNOWN DISEASE":
        st.warning("⚠️ Résultat incertain — " + t("unknown_disease"))
        # Affiche quand même le top-3 pour aider l'utilisateur
        top3 = diagnosis.get("top3", [])
        if top3:
            st.markdown("**Meilleures hypothèses :**")
            for item in top3:
                st.markdown(f"- {item.get('display_name', item.get('disease', '?'))} — {item.get('confidence_pct', '')}")
    else:
        st.markdown(f"## 🌿 {pred_disease}")

    # ── Métriques Niveau 5 ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Confiance", confidence_pct)
    m2.metric("🖼️ Qualité",   quality)
    m3.metric("🎯 Fiabilité", reliability)
    m4.metric("📂 Catégorie", category)

    # ── Détails scientifiques ──
    if sci_name and sci_name != "—":
        st.caption(f"🔬 *{sci_name}* — 🌾 {crop}")

    # ── Conseil agronomique ──
    if conseil:
        st.info(conseil)

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

    # ====== STEP 1: Show basic information ======
    if disease_data.get("description"):
        st.markdown("### 📖 Description")
        st.write(disease_data["description"])

    # Scientific name & pathogen type
    col1, col2 = st.columns(2)
    with col1:
        if disease_data.get("scientific_name"):
            st.markdown(f"**Scientific Name:** {disease_data['scientific_name']}")
    with col2:
        if disease_data.get("pathogen_type"):
            st.markdown(f"**Pathogen Type:** {disease_data['pathogen_type']}")

    # Hosts
    if disease_data.get("hosts"):
        st.markdown("### 🌾 Hosts Affected")
        for host in disease_data["hosts"]:
            st.markdown(f"- {host}")

    # Susceptibility
    if disease_data.get("susceptibility"):
        st.markdown("### 🛡️ Susceptibility")
        suscept = disease_data["susceptibility"]
        
        if suscept.get("highly_susceptible"):
            st.markdown("**Highly Susceptible:**")
            for item in suscept["highly_susceptible"]:
                st.markdown(f"- {item}")
        
        if suscept.get("moderately_susceptible"):
            st.markdown("**Moderately Susceptible:**")
            for item in suscept["moderately_susceptible"]:
                st.markdown(f"- {item}")
        
        if suscept.get("more_tolerant"):
            st.markdown("**More Tolerant:**")
            for item in suscept["more_tolerant"]:
                st.markdown(f"- {item}")

    # ====== BUTTON to expand for more details ======
    if st.button("📋 View detailed treatment & management", type="secondary", use_container_width=True):
        st.session_state.show_detailed_info = True

    # ====== STEP 2: Show detailed information (after button click) ======
    if st.session_state.get("show_detailed_info", False):
        st.markdown("---")
        st.markdown("### 📋 Detailed Information")

        # Symptoms and damage
        if disease_data.get("symptoms"):
            st.markdown("**Symptoms and Damage:**")
            for item in disease_data["symptoms"]:
                st.markdown(f"- {item}")

        # Disease cycle and spread
        if disease_data.get("disease_cycle_and_spread"):
            st.markdown("**Disease Cycle and Spread:**")
            for item in disease_data["disease_cycle_and_spread"]:
                st.markdown(f"- {item}")

        # Favorable conditions
        if disease_data.get("favorable_conditions"):
            st.markdown("**Favorable Conditions:**")
            for item in disease_data["favorable_conditions"]:
                st.markdown(f"- {item}")

        # Pathogen characteristics
        if disease_data.get("pathogen_characteristics"):
            st.markdown("**Pathogen Characteristics:**")
            for item in disease_data["pathogen_characteristics"]:
                st.markdown(f"- {item}")

        # Monitoring
        if disease_data.get("monitoring"):
            st.markdown("**Monitoring:**")
            for item in disease_data["monitoring"]:
                st.markdown(f"- {item}")

        # Management
        if disease_data.get("management"):
            st.markdown("**Management and Control:**")
            for item in disease_data["management"]:
                st.markdown(f"- {item}")

        # Prevention (if available)
        if disease_data.get("prevention"):
            st.markdown("**Prevention:**")
            for item in disease_data["prevention"]:
                st.markdown(f"- {item}")

        source_file = disease_data.get("_source_file")
        if source_file:
            st.caption(f"🔗 Source: {source_file}")

    # Visual confirmation: 3–4 images from dataset_light for the predicted class
    if not is_unknown and pred_disease:
        st.markdown("---")
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
                        st.markdown(" 🤖 AI Explanation")
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

