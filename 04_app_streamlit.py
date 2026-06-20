"""
STREAMLIT APP - Interactive Diagnosis Interface
Utilisation: streamlit run 04_app_streamlit.py

✅ ZERO RAM OPTIMIZATION: Uses Hugging Face Spaces API
RAM usage: <50MB (model runs on HF servers)
Modèle : Swin-Base + ArcFace | 516 classes | SmartAgriDataset v2.1
"""

import streamlit as st
import numpy as np
import json
import requests
import io
from pathlib import Path
from PIL import Image
import plotly.express as px
from typing import Dict, List, Tuple, Any
import os
from dotenv import load_dotenv
from datetime import datetime

from utils.i18n import get_lang, language_selector, t
from utils.blip2_explainer import load_disease_info as load_disease_info_blip2

try:
    import pandas as pd
except ImportError:
    pd = None

load_dotenv()

# ============================================================================
# API CONFIG — Nouveau modèle Swin-Base sur HuggingFace Spaces
# ============================================================================
API_BASE_URL = "https://mohamedsamake8322-sene-disease-api.hf.space"
API_TIMEOUT  = 60  # secondes (cold start HF peut prendre 30-60s)

def get_api_url() -> str:
    custom = os.environ.get("API_URL") or st.secrets.get("API_URL", None)
    return custom if custom else API_BASE_URL

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="🌾 AI Plant Disease Diagnostic Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults = {
        "results":        None,
        "diagnosis":      None,
        "uploaded_image": None,
        "history":        [],
        "expert_mode":    False,
        "multi_image_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.block-container { padding-top: 2rem; }
.stMetric { background-color: white; padding: 15px; border-radius: 10px; }
img { border-radius: 12px; }
.reliability-high   { color: #2d6a4f; font-weight: bold; }
.reliability-medium { color: #f4a261; font-weight: bold; }
.reliability-low    { color: #e63946; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    try:
        r = requests.get(f"{get_api_url()}/health", timeout=10)
        data = r.json()
        return r.status_code == 200 and data.get("model_loaded", False)
    except Exception:
        return False


def call_hf_api(image_bytes: bytes) -> Dict[str, Any] | None:
    """Appelle le nouveau endpoint /predict et retourne le JSON brut."""
    try:
        response = requests.post(
            f"{get_api_url()}/predict",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Délai dépassé — le serveur met trop de temps (cold start). Réessayez dans 30s.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur API : {e}")
        return None


def diagnose_via_api(image_bytes: bytes) -> Tuple[List[Dict], Dict]:
    """
    Appelle l'API et adapte la réponse au format interne de l'app.

    Nouveau format API :
      diagnostic, confidence, confidence_pct, is_uncertain,
      quality, quality_score, reliability, crop, scientific_name,
      category, top3 (list), conseil

    Format interne Streamlit (résultats + diagnostic) :
      results  → liste de dicts {rank, disease, confidence, scientific_name}
      diagnosis → dict {predicted_disease, predicted_score, is_unknown, ...}
    """
    raw = call_hf_api(image_bytes)
    if raw is None:
        return [], {}

    try:
        # ── Résultats top-3 ──
        results = [
            {
                "rank":            t["rank"],
                "disease":         t["display_name"],
                "scientific_name": t.get("scientific_name", ""),
                "confidence":      t["confidence"],
                "confidence_pct":  t["confidence_pct"],
            }
            for t in raw.get("top3", [])
        ]

        # ── Diagnostic principal ──
        diagnosis = {
            "predicted_disease":   raw.get("diagnostic", "INCONNU"),
            "predicted_score":     raw.get("confidence", 0.0),
            "confidence_pct":      raw.get("confidence_pct", "0%"),
            "is_unknown":          raw.get("is_uncertain", True),
            "quality":             raw.get("quality", "—"),
            "quality_score":       raw.get("quality_score", 0.0),
            "reliability":         raw.get("reliability", "—"),
            "crop":                raw.get("crop", "—"),
            "scientific_name":     raw.get("scientific_name", "—"),
            "category":            raw.get("category", "—"),
            "conseil":             raw.get("conseil", ""),
        }

        return results, diagnosis

    except Exception as e:
        st.error(f"Erreur parsing réponse API : {e}")
        return [], {}


def get_confirmation_images(disease_name: str, top3: List[Dict]) -> List[Dict]:
    """
    Retourne des images de confirmation pour la section visuelle.
    Nouveau modèle : pas d'images locales → utilise HuggingFace dataset public
    ou affiche un placeholder avec le nom scientifique.
    """
    # On retourne les top3 enrichis pour l'affichage
    return [r for r in top3 if r["disease"] == disease_name][:4] or top3[:4]


@st.cache_data(ttl=300)
def load_disease_info_json():
    try:
        with open("data/disease_info.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    language_selector(container="sidebar")
    st.markdown("---")
    st.header("⚙️ Settings")
    show_ref_images  = st.checkbox("Show reference images", value=True)
    expert_mode      = st.toggle("Expert Mode",            value=st.session_state.expert_mode)
    multi_image_mode = st.toggle("Multi-Image Comparison", value=st.session_state.multi_image_mode)
    st.session_state.expert_mode      = expert_mode
    st.session_state.multi_image_mode = multi_image_mode

    if st.session_state.history:
        st.subheader("📚 Recent Diagnoses")
        for i, diag in enumerate(st.session_state.history[-5:]):
            st.write(f"{i+1}. {diag['disease']} ({diag['score']:.2f})")

# ============================================================================
# HEADER
# ============================================================================
st.title(t("app_title"))
st.markdown(t("app_description"))

# Statut API
api_healthy = check_api_health()
if api_healthy:
    st.success("✅ API connectée — Swin-Base (516 classes) opérationnel")
else:
    st.warning("⚠️ API non disponible — vérifiez votre connexion ou attendez le cold start (~30s)")

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📸 Input Image")

    if st.session_state.multi_image_mode:
        uploaded_files = st.file_uploader(
            "Upload plant images", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )
        if uploaded_files:
            images = [Image.open(f).convert("RGB") for f in uploaded_files]
            st.image(images, width=None, caption=[f"Image {i+1}" for i in range(len(images))])
            selected = st.selectbox("Select image for diagnosis", range(len(images)),
                                    format_func=lambda x: f"Image {x+1}")
            image = images[selected] if images else None
        else:
            image = None
    else:
        uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=None, caption="Uploaded Image")
        else:
            image = None

    if image and st.button("🔍 Diagnose"):
        progress = st.progress(0)
        with st.spinner("Analyzing via API..."):
            progress.progress(20)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG", quality=95)
            image_bytes = img_buffer.getvalue()
            progress.progress(50)
            results, diagnosis = diagnose_via_api(image_bytes)
            progress.progress(100)

        if results and diagnosis:
            st.session_state.results        = results
            st.session_state.diagnosis      = diagnosis
            st.session_state.uploaded_image = image
            st.session_state.history.append({
                "disease": diagnosis.get("predicted_disease", "Unknown"),
                "score":   diagnosis.get("predicted_score", 0.0),
            })
        else:
            st.error("❌ Diagnosis failed — please try again")

# ============================================================================
# RÉSULTATS
# ============================================================================
with col2:
    st.header("📊 Diagnosis Results")

    if st.session_state.results and st.session_state.diagnosis:
        results   = st.session_state.results
        diagnosis = st.session_state.diagnosis

        pred_disease = diagnosis.get("predicted_disease", "INCONNU")
        pred_score   = diagnosis.get("predicted_score", 0.0)
        is_unknown   = diagnosis.get("is_unknown", False)

        st.subheader(t("probable_disease"))

        # ── Diagnostic principal ──
        if is_unknown:
            st.error("⚠️ " + t("unknown_disease"))
        else:
            st.markdown(f"## 🌿 {pred_disease}")

        # ── Métriques Niveau 5 ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confiance",      diagnosis.get("confidence_pct", "—"))
        m2.metric("Qualité image",  diagnosis.get("quality", "—"))
        m3.metric("Fiabilité",      diagnosis.get("reliability", "—"))
        m4.metric("Catégorie",      diagnosis.get("category", "—"))

        # ── Détails scientifiques ──
        with st.expander("🔬 Détails scientifiques", expanded=False):
            st.write(f"**Culture** : {diagnosis.get('crop', '—')}")
            st.write(f"**Nom scientifique** : {diagnosis.get('scientific_name', '—')}")
            st.write(f"**Catégorie** : {diagnosis.get('category', '—')}")

        # ── Conseil agronomique ──
        if diagnosis.get("conseil"):
            st.info(diagnosis["conseil"])

        # ── Informations sur la maladie ──
        if not is_unknown and pred_disease:
            info = load_disease_info_blip2(pred_disease, language_code=get_lang())
            if info:
                st.markdown("### 📖 Disease Information")
                st.write(info.get("description", ""))
                if info.get("symptoms"):
                    st.markdown("**Symptoms:**")
                    st.write(info["symptoms"])
                if info.get("management"):
                    st.markdown("**Management:**")
                    st.write(info["management"])
                if info.get("prevention"):
                    st.markdown("**Prevention:**")
                    st.write(info["prevention"])

        # ── Top-3 graphique ──
        if results:
            st.markdown("### 📈 Top-3 Predictions")
            diseases = [r["disease"] for r in results]
            scores   = [r["confidence"] for r in results]
            fig = px.bar(
                x=diseases, y=scores,
                labels={"x": "Disease", "y": "Confidence (calibrated)"},
                title="Top-3 Predicted Diseases",
                color=scores,
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Mode expert ──
        if st.session_state.expert_mode:
            st.markdown("### 🔬 Expert Details")
            st.json({
                "top3":           results,
                "quality_score":  diagnosis.get("quality_score"),
                "confidence_raw": diagnosis.get("predicted_score"),
            })

        # ── Feedback ──
        st.markdown("### 💬 Help Improve the System")
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            user_feedback = st.radio(
                "Was this diagnosis correct?",
                ["Select...", "Yes, correct", "No, incorrect", "Unsure"],
                key="feedback_radio"
            )
        with fb_col2:
            correct_disease = (
                st.text_input("What was the actual disease?", key="correct_disease")
                if user_feedback == "No, incorrect" else None
            )

        additional_notes = st.text_area(
            "Additional notes (optional)",
            placeholder="Any observations about symptoms, treatment, etc.",
            key="additional_notes"
        )

        if st.button("📤 Submit Feedback", key="submit_feedback"):
            if user_feedback != "Select...":
                st.success("✅ Thank you! Your feedback helps improve the system.")
            else:
                st.warning("Please select your feedback before submitting.")

    else:
        st.info("👆 Upload an image and click 'Diagnose' to start")

# ============================================================================
# CONFIRMATION VISUELLE — Nouveau système sans images locales
# ============================================================================
if st.session_state.results and st.session_state.diagnosis:
    st.divider()
    st.subheader("🔎 Visual Confirmation")

    diagnosis    = st.session_state.diagnosis
    results      = st.session_state.results
    pred_disease = diagnosis.get("predicted_disease", "")
    is_unknown   = diagnosis.get("is_unknown", False)

    if not is_unknown and pred_disease:
        st.markdown(
            f"Le modèle a détecté **{pred_disease}** "
            f"({diagnosis.get('scientific_name', '')}) avec une confiance de "
            f"**{diagnosis.get('confidence_pct', '')}**."
        )
        st.markdown("Comparez les 3 meilleures prédictions ci-dessous :")

        conf_cols = st.columns(len(results[:3]))
        for col, r in zip(conf_cols, results[:3]):
            with col:
                sci = r.get("scientific_name", "")
                st.markdown(f"**{r['rank']}. {r['disease']}**")
                if sci:
                    st.caption(f"*{sci}*")
                st.metric("Confiance", r["confidence_pct"])
                # Placeholder image avec nom de la maladie
                st.image(
                    f"https://via.placeholder.com/200x150/2d6a4f/ffffff?"
                    f"text={r['disease'].replace(' ', '+')[:20]}",
                    width=200,
                )

        st.info(
            "ℹ️ Les images de référence locales ne sont plus disponibles avec le nouveau modèle. "
            "Une prochaine version intégrera des images depuis le dataset SmartAgri v2.1."
        )

    if st.button("❓ Not matching?"):
        st.info("Try another image or consult an agronomist.")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
### 📝 How it works:
1. **Upload** an image of a plant disease
2. **AI analyzes** via Swin Transformer Base (516 classes, 356k training images)
3. **Returns diagnosis** with calibrated confidence score
4. **Shows top-3** predictions with scientific names

### ⚠️ Important:
- This is a **diagnostic assistant**, not a final diagnosis²
- Always validate with domain experts
- Confidence scores are **calibrated** (Temperature Scaling T=3.1)

### 🚀 Features:
- ✅ **Zero RAM**: Model runs on Hugging Face servers
- ✅ **516 classes** : maladies, ravageurs, insectes, acariens, nématodes
- ✅ **Noms scientifiques** universels
- ✅ Qualité image + Fiabilité globale
- ✅ Conseil agronomique automatique
- ✅ Expert mode for detailed analysis

### 🔧 Technical:
- **Architecture**: Streamlit → HTTP API → HF Spaces (FastAPI + Gradio)
- **Model**: Swin Transformer Base 384px + ArcFace
- **Calibration**: Temperature Scaling (T=3.1)
- **API Docs**: https://mohamedsamake8322-sene-disease-api.hf.space/docs
""")
