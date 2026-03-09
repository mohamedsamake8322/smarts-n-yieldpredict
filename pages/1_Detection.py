"""
Disease detection page – Plantix mode:
- Identifies ONLY the disease
- Displays the name + some similar images
"""

# WORKAROUND: Prevent Streamlit from inspecting torch.classes (compatibility issue)
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

SUPPORTED_LANGS = {
    "fr": "Français",
    "en": "English",
    "tr": "Türkçe",
    "sw": "Kiswahili",
    "ha": "Hausa",
}

TRANSLATIONS = {
    "page_title": {
        "fr": "📸 Détection Intelligente des Plantes",
        "en": "📸 Smart Plant Disease Detection",
        "tr": "📸 Akıllı Bitki Hastalığı Tespiti",
        "sw": "📸 Utambuzi Mahiri wa Magonjwa ya Mimea",
        "ha": "📸 Ganewar Cutar Shuka Mai Hikima",
    },
    "page_subtitle": {
        "fr": "Téléversez une image pour identifier la **maladie probable** et comparer avec quelques images similaires.",
        "en": "Upload an image to identify the **most probable disease** and compare with similar examples.",
        "tr": "En olası hastalığı belirlemek ve benzer örneklerle karşılaştırmak için bir görüntü yükleyin.",
        "sw": "Pakia picha ili kutambua **ugonjwa unaowezekana zaidi** na kuulinganisha na mifano inayofanana.",
        "ha": "Loda hoto don gano **mummanan cutar da za ta fi yiwuwa** kuma kwatanta ta da hotuna makamanta.",
    },
    "sidebar_instructions": {
        "fr": "1. **Prenez une photo** avec votre caméra\n2. **Ou téléversez** une image depuis votre galerie\n3. Attendez l'**analyse par l'IA**\n4. Consultez les **résultats et recommandations**",
        "en": "1. **Take a photo** with your camera\n2. **Or upload** an image from your gallery\n3. Wait for the **AI analysis**\n4. Check the **results and recommendations**",
        "tr": "1. Kameranızla **fotoğraf çekin**\n2. Ya da galerinizden bir **görüntü yükleyin**\n3. **Yapay zekâ analizini** bekleyin\n4. **Sonuçları ve önerileri** inceleyin",
        "sw": "1. **Piga picha** kwa kutumia kamera\n2. Au **pakia** picha kutoka kwenye galeri\n3. Subiri **uchambuzi wa AI**\n4. Angalia **matokeo na mapendekezo**",
        "ha": "1. **Dauki hoto** da kamara\n2. Ko **loda** hoto daga gallery\n3. Jira **binciken AI**\n4. Duba **sakamako da shawarwari**",
    },
    "image_to_analyze": {
        "fr": "🖼️ Image à analyser",
        "en": "🖼️ Image to analyze",
        "tr": "🖼️ Analiz edilecek görüntü",
        "sw": "🖼️ Picha ya kuchambua",
        "ha": "🖼️ Hoton da za a bincika",
    },
    "analyze_button": {
        "fr": "🔍 Lancer la détection",
        "en": "🔍 Run detection",
        "tr": "🔍 Tespiti başlat",
        "sw": "🔍 Anzisha utambuzi",
        "ha": "🔍 Fara gano cuta",
    },
    "analysis_done": {
        "fr": "✅ Analyse terminée !",
        "en": "✅ Analysis completed!",
        "tr": "✅ Analiz tamamlandı!",
        "sw": "✅ Uchambuzi umekamilika!",
        "ha": "✅ Bincike ya kammala!",
    },
    "probable_disease": {
        "fr": "🦠 Maladie probable",
        "en": "🦠 Probable disease",
        "tr": "🦠 Muhtemel hastalık",
        "sw": "🦠 Ugonjwa unaowezekana",
        "ha": "🦠 Cutar da ake zargi",
    },
    "unknown_disease": {
        "fr": "Maladie inconnue – veuillez essayer une autre image ou consulter un expert.",
        "en": "Unknown disease – please try another image or consult an expert.",
        "tr": "Bilinmeyen hastalık – lütfen başka bir görüntü deneyin veya bir uzmana danışın.",
        "sw": "Ugonjwa haujatambuliwa – tafadhali jaribu picha nyingine au wasiliana na mtaalam.",
        "ha": "An kasa gane cutar – don Allah a gwada wani hoto ko a tuntuɓi ƙwararre.",
    },
    "visual_confirmation": {
        "fr": "### 🔎 Confirmation visuelle",
        "en": "### 🔎 Visual confirmation",
        "tr": "### 🔎 Görsel doğrulama",
        "sw": "### 🔎 Uthibitisho wa kuona",
        "ha": "### 🔎 Tabbatarwa ta gani",
    },
    "not_leaf_message": {
        "fr": "Cette image ne ressemble pas à une feuille de culture. Veuillez prendre une photo plus proche de la feuille ou choisir une autre image.",
        "en": "This image does not look like a crop leaf. Please take a closer photo of the leaf or choose another image.",
        "tr": "Bu görüntü bir ürün yaprağına benzemiyor. Lütfen yaprağa daha yakın bir fotoğraf çekin veya başka bir görüntü seçin.",
        "sw": "Picha hii haionekani kama jani la zao. Tafadhali piga picha karibu zaidi ya jani au chagua picha nyingine.",
        "ha": "Wannan hoto ba ya kama da ganyen amfanin gona. Don Allah dauki hoto kusa da ganyen ko ka zabi wani hoto.",
    },
}


def get_lang() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "fr"
    return st.session_state["lang"]


def t(key: str) -> str:
    lang = get_lang()
    return TRANSLATIONS.get(key, {}).get(
        lang, TRANSLATIONS.get(key, {}).get("fr", key)
    )


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
        st.image(st.session_state.image_bytes, use_column_width=True)
    elif st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, use_column_width=True)
    
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
                        st.image(ref_img, use_column_width=True)
                    except Exception:
                        st.warning("Image not available")

    # Button for new detection
    if st.button("🔄 New detection", use_container_width=True):
        for key in ["uploaded_image", "image_bytes", "detection_result", "show_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

