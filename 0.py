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
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="🌾 Plant Disease Diagnostic System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL & INDEX (cached)
# ============================================================================
# On pointe vers les artefacts de la phase 2 (Swin Base production)
MODELS_PATH = Path("./outputs/phase2_swin_base_production/models")
DATASET_ROOT_LOCAL = Path("./dataset_final")

@st.cache_resource
def load_model_and_index():
    """Load metric model (phase 2), FAISS index, prototypes et metadata."""

    # Fichiers obligatoires
    metric_model_path = MODELS_PATH / "metric_model.pt"
    metadata_path = MODELS_PATH / "metadata.pkl"

    for f in [metric_model_path, metadata_path]:
        if not f.exists():
            st.error(f"❌ Missing file: {f}")
            st.info("📥 Assure-toi d'avoir copié les artefacts de la phase 2 dans outputs/phase2_swin_base_production/models")
            st.stop()

    # Charge metadata (embeddings_shape, image_paths, labels, prototypes, etc.)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Détermine le chemin de l'index FAISS à partir de la metadata
    faiss_index_path = metadata.get("faiss_index_path") or str(MODELS_PATH / "faiss_index.bin")
    faiss_index_file = Path(faiss_index_path)

    # Définition du modèle identique à celui utilisé pendant l'entraînement
    from training_pipelines.metric_training_core import DiagnosticModel  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(metric_model_path, map_location=device)

    cfg = checkpoint.get("config", {})
    model_name = cfg.get("model_name", "swin_base_patch4_window7_224")
    embedding_dim = cfg.get("embedding_dim", metadata.get("embedding_dim", 768))
    image_size = cfg.get("image_size", 224)

    model = DiagnosticModel(model_name=model_name, embedding_dim=embedding_dim, image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Charge FAISS index si possible
    index = None
    if faiss is not None and faiss_index_file.exists():
        try:
            index = faiss.read_index(str(faiss_index_file))
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger l'index FAISS ({e}). Seul le diagnostic par prototypes sera disponible.")
    else:
        st.info("ℹ️ FAISS non disponible ou index absent. Diagnostic basé uniquement sur prototypes.")

    # Préparation des prototypes (numpy float32)
    prototypes = None
    prototype_labels = None
    if "prototypes" in metadata and "prototype_labels" in metadata:
        prototypes = np.asarray(metadata["prototypes"], dtype="float32")
        prototype_labels = np.asarray(metadata["prototype_labels"], dtype=int)

    return model, index, metadata, prototypes, prototype_labels, device

def preprocess_image(image, size=224):
    """Preprocess PIL image for model inference"""
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((size, size))
    
    # Convert to numpy
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def _map_image_path_to_local(raw_path: str) -> str:
    """
    Mappe un chemin absolu Colab (/content/drive/MyDrive/dataset_final/...)
    vers le dataset local ./dataset_final/... si necessaire.
    """
    p = Path(raw_path)
    if p.exists():
        return str(p)

    # Essaie de retrouver la partie relative a partir de 'dataset_final'
    parts = p.parts
    if "dataset_final" in parts:
        idx = parts.index("dataset_final")
        rel = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path(".")
        candidate = DATASET_ROOT_LOCAL / rel
        if candidate.exists():
            return str(candidate)

    # Fallback: renvoie le chemin brut (l'ouverture pourra echouer mais sera capturee)
    return raw_path


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
    """Run inference et retourne:
    - top_k voisins (FAISS si dispo)
    - ranking par prototypes (pour diagnostic + seuil unknown)
    """

    image_paths = metadata["image_paths"]
    labels = metadata["labels"]
    idx_to_class = metadata["idx_to_class"]

    def class_name_for(label):
        # idx_to_class peut avoir des cles int ou str selon la serialization
        if label in idx_to_class:
            return idx_to_class[label]
        if str(label) in idx_to_class:
            return idx_to_class[str(label)]
        return f"class_{label}"

    # Preprocess
    img_tensor = preprocess_image(image).to(device)

    # Get embedding
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().astype("float32")  # (1, D)

    # Normalisation L2
    emb_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)

    # --- Diagnostic par prototypes ---
    proto_ranking = []
    predicted_label = None
    predicted_score = None

    if prototypes is not None and prototype_labels is not None:
        # (C, D) x (D, 1) -> (C,)
        sims = prototypes @ emb_norm.T  # (C, 1)
        sims = sims.squeeze(axis=1)

        order = np.argsort(sims)[::-1]
        for rank, ci in enumerate(order[: max(k, 5)]):
            class_id = int(prototype_labels[ci])
            proto_ranking.append(
                {
                    "rank": rank + 1,
                    "label": class_id,
                    "disease": class_name_for(class_id),
                    "similarity": float(sims[ci]),
                }
            )

        if proto_ranking:
            predicted_label = proto_ranking[0]["label"]
            predicted_score = proto_ranking[0]["similarity"]

    # Unknown detection
    is_unknown = (
        predicted_score is not None and predicted_score < float(unknown_threshold)
    )

    # --- Recherche des voisins via FAISS (pour les images de reference) ---
    results = []
    if index is not None:
        distances, indices = index.search(emb_norm.astype("float32"), k=k)

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            pred_label = labels[idx]
            pred_class = class_name_for(pred_label)
            raw_path = image_paths[idx]
            mapped_path = _map_image_path_to_local(raw_path)

            results.append(
                {
                    "rank": rank + 1,
                    "disease": pred_class,
                    "confidence": float(dist),
                    "path": mapped_path,
                }
            )

    diagnosis = {
        "predicted_label": predicted_label,
        "predicted_disease": "UNKNOWN DISEASE"
        if is_unknown
        else (class_name_for(predicted_label) if predicted_label is not None else None),
        "predicted_score": predicted_score,
        "is_unknown": is_unknown,
        "proto_ranking": proto_ranking,
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
        if st.button("🔍 Diagnose", use_container_width=True):
            with st.spinner("Analyzing..."):
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
            
            # Save results for display
            st.session_state.results = results
            st.session_state.diagnosis = diagnosis
            st.session_state.uploaded_image = image

with col2:
    st.header("📊 Diagnosis Results")
    
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        diagnosis = st.session_state.get("diagnosis", {})
        
        # Top diagnosis (prototypes + seuil unknown)
        st.subheader("🎯 Most Probable Disease (Prototype-based)")

        pred_disease = diagnosis.get("predicted_disease")
        pred_score = diagnosis.get("predicted_score")
        is_unknown = diagnosis.get("is_unknown", False)
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Disease", pred_disease if pred_disease is not None else "N/A")
        with metric_col2:
            if pred_score is not None:
                st.metric("Similarity", f"{pred_score:.2%}")
            else:
                st.metric("Similarity", "N/A")

        if is_unknown:
            st.warning(
                "⚠️ Similarity below threshold: case marked **UNKNOWN DISEASE**. "
                "Please check with an expert or add as a new class."
            )
        
        # All results table
        st.subheader("🔍 Top Similar Images (FAISS)")
        
        results_data = []
        for r in results:
            results_data.append({
                'Rank': r['rank'],
                'Disease': r['disease'],
                'Confidence': f"{r['confidence']:.2%}",
            })
        
        st.dataframe(results_data, use_container_width=True)
        
        # Class distribution
        st.subheader("📈 Class Distribution")
        class_counts = {}
        for r in results:
            cls = r['disease']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(list(class_counts.keys()), list(class_counts.values()), color='steelblue')
        ax.set_xlabel('Number of Matches')
        ax.set_title('Disease Match Distribution')
        st.pyplot(fig)
    else:
        st.info("👆 Upload an image and click 'Diagnose' to start")

# ============================================================================
# REFERENCE IMAGES
# ============================================================================
if 'results' in st.session_state and st.session_state.results:
    
    st.header("📚 Reference Images from Training Dataset")
    
    results = st.session_state.results
    
    # Display top K reference images
    cols = st.columns(min(k, 5))
    
    for i, (col, result) in enumerate(zip(cols, results)):
        with col:
            try:
                ref_image = Image.open(result['path'])
                st.image(ref_image, use_column_width=True)
                st.caption(f"{result['disease']}\n({result['confidence']:.2%})")
            except Exception as e:
                st.warning(f"Could not load image {result['path']}")

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
