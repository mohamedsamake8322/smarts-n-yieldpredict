"""
DETECT DISEASE - Main AI Diagnosis Page
Utilisation: via streamlit navigation
"""

import streamlit as st
import torch
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
import plotly.express as px
from io import BytesIO

from model_core import (
    MODELS_PATH_PHASE2,
    DATASET_ROOT_LOCAL,
    DEVICE,
    load_phase2_model_and_metadata,
    infer_on_image,
)

from utils.gradcam import generate_gradcam
from utils.storage import save_diagnosis, save_feedback, save_wrong_image_for_review
from utils.blip2_explainer import load_disease_info
from utils.i18n import get_lang, language_selector, t

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="🔍 Detect Disease",
    page_icon="🔍",
    layout="wide",
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    defaults = {
        "results": None,
        "diagnosis": None,
        "uploaded_image": None,
        "gradcam_overlay": None,
        "current_image_name": None,
        "detection_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
<style>
.main {
    background-color: #f8f9fa;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
}
img {
    border-radius: 12px;
}
.diagnosis-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

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

@st.cache_data
def load_disease_database():
    return load_disease_info()

def preprocess_image(image, size=224):
    """Image preprocessing"""
    from model_core import preprocess_image_pil
    return preprocess_image_pil(image, size=size)

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
    """Run diagnosis and format results"""

    with torch.no_grad():
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

    # Format for UI
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
# MAIN UI
# ============================================================================
st.title(t("app_title"))
st.markdown(t("app_description"))

st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header(t("sidebar_settings"))
    language_selector(container="sidebar")
    st.markdown("---")
    k = st.slider("Number of similar images (K)", 1, 10, 5)
    unknown_threshold = st.slider(
        "Unknown threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.55,
        step=0.01,
    )
    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
    show_ref_images = st.toggle("Show reference images", value=True)

# Load models
try:
    model, index, metadata, prototypes, prototype_labels, device = load_model_and_index()
    st.sidebar.success("✅ Models loaded")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ============================================================================
# UPLOAD & DIAGNOSIS INTERFACE
# ============================================================================
col1, col2 = st.columns([1.2, 1.5])

with col1:
    st.subheader("📸 Upload Image")
    
    uploaded_file = st.file_uploader("Choose a plant image", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=None, caption="Uploaded Image")
        
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        
        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)

            with st.spinner("Analyzing plant disease..."):
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
                progress_bar.progress(70)

                # Grad-CAM
                if show_gradcam:
                    try:
                        img_tensor = preprocess_image(image, size=metadata.get("image_size", 224)).to(device)
                        target_layer = getattr(model, "backbone", None) or model
                        cam = generate_gradcam(model, img_tensor, target_layer)
                        cam_resized = cv2.resize(cam, image.size)
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        base = image.convert("RGB")
                        heatmap_img = Image.fromarray(heatmap).convert("RGB")
                        overlay = Image.blend(base, heatmap_img, alpha=0.4)
                        st.session_state.gradcam_overlay = overlay
                    except:
                        st.session_state.gradcam_overlay = None

                progress_bar.progress(100)

            # Save diagnosis
            st.session_state.results = results
            st.session_state.diagnosis = diagnosis
            st.session_state.uploaded_image = image
            st.session_state.current_image_name = uploaded_file.name
            st.session_state.detection_done = True

with col2:
    st.subheader("📊 Diagnosis Results")
    
    if 'diagnosis' in st.session_state and st.session_state.diagnosis:
        diagnosis = st.session_state.diagnosis
        results = st.session_state.results

        pred_disease = diagnosis.get("predicted_disease")
        pred_score = diagnosis.get("predicted_score")
        is_unknown = diagnosis.get("is_unknown", False)

        # Main diagnosis display
        if is_unknown:
            st.error("⚠️ **Unknown Disease**\nPlease consult an expert or try another image.")
        else:
            st.markdown(f"""
            <div class="diagnosis-box">
            <h2 style="margin: 0; font-size: 2rem;">{pred_disease}</h2>
            <h3 style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Confidence: {pred_score:.1%}
            </h3>
            </div>
            """, unsafe_allow_html=True)

# Disease information via BLIP2_i18n / BLIP2 fallback
    if not is_unknown and pred_disease:
        info = load_disease_info(pred_disease, language_code=get_lang())
        with st.expander("📖 Disease Information", expanded=True):
            st.markdown(f"**Description:** {info.get('description', '')}")
            if info.get("symptoms"):
                st.markdown("**Symptoms:**")
                st.write(info["symptoms"])
            if info.get("management"):
                st.markdown("**Management:**")
                st.write(info["management"])
            if info.get("prevention"):
                st.markdown("**Prevention:**")
                st.write(info["prevention"])
            if info.get("_source_file"):
                st.caption(f"🔗 Source JSON: {info.get('_source_file')}")

        # Metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Top Match", results[0]["disease"] if results else "N/A", f"{results[0]['confidence']:.1%}" if results else "")
        with col_m2:
            st.metric("Confidence", f"{pred_score:.1%}")

        # Heatmap
        if show_gradcam and st.session_state.get("gradcam_overlay"):
            with st.expander("🔥 Grad-CAM Analysis"):
                st.image(st.session_state["gradcam_overlay"], width=None,
                        caption="Heat regions: brighter = more important for diagnosis")

        # Similarity chart
        if results:
            with st.expander("📈 Similarity Scores"):
                diseases = [r["disease"] for r in results]
                scores = [r["confidence"] for r in results]
                fig = px.bar(x=diseases, y=scores, labels={"x": "Disease", "y": "Similarity"},
                           title="Top Similar Diseases")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Upload an image and click 'Analyze Image' to start")

# ============================================================================
# VISUAL CONFIRMATION & FEEDBACK
# ============================================================================
if st.session_state.detection_done and 'diagnosis' in st.session_state:
    st.divider()
    st.subheader("📸 Visual Confirmation")

    diagnosis = st.session_state.get("diagnosis", {})
    results = st.session_state.get("results", [])
    pred_disease = diagnosis.get("predicted_disease")

    # Show reference images
    if show_ref_images and results:
        confirmation_images = [r for r in results if r["disease"] == pred_disease][:4]
        
        if confirmation_images:
            st.markdown("Similar cases from our database:")
            cols = st.columns(len(confirmation_images))
            
            for col, result in zip(cols, confirmation_images):
                with col:
                    if result["path"] and Path(result["path"]).exists():
                        try:
                            ref_image = Image.open(result["path"])
                            st.image(ref_image, width=None)
                            st.caption(f"{result['disease']}\n{result['confidence']:.1%}")
                        except:
                            st.warning("Image not available")

    st.divider()
    st.subheader("✅ Feedback")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        if st.button("✔️ Correct Diagnosis", use_container_width=True, type="primary"):
            try:
                save_diagnosis(
                    st.session_state.current_image_name,
                    diagnosis.get("predicted_disease"),
                    diagnosis.get("predicted_score")
                )
                save_feedback(
                    diagnosis.get("predicted_disease"),
                    "correct",
                    diagnosis.get("predicted_score")
                )
                st.success("✅ Thank you! Your feedback helps improve the system.")
                st.balloons()
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

    with col_f2:
        if st.button("❌ Incorrect Diagnosis", use_container_width=True):
            try:
                save_feedback(
                    diagnosis.get("predicted_disease"),
                    "incorrect",
                    diagnosis.get("predicted_score")
                )
                if st.session_state.uploaded_image:
                    img_bytes = BytesIO()
                    st.session_state.uploaded_image.save(img_bytes, format="JPEG")
                    save_wrong_image_for_review(
                        img_bytes.getvalue(),
                        diagnosis.get("predicted_disease"),
                        diagnosis.get("predicted_score")
                    )
                st.warning("⚠️ This image has been saved for review by our team to improve the model.")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_f3:
        if st.button("🤔 Unsure", use_container_width=True):
            try:
                save_feedback(
                    diagnosis.get("predicted_disease"),
                    "unsure",
                    diagnosis.get("predicted_score")
                )
                st.info("ℹ️ Your uncertainty helps us identify edge cases.")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

st.markdown("""
### 💡 Tips for Best Results:
- **Clear lighting** - Good sunlight or indoor lighting
- **Close-up** - Focus on the affected area
- **Multiple angles** - Try different angles if unsure
- **No filter** - Avoid using photo filters or effects
- **Single leaf** - Best results with one clear leaf in frame
""")
