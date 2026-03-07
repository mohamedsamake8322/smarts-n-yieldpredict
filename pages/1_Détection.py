"""
Page de détection des maladies – mode Plantix :
- Identifie UNIQUEMENT la maladie
- Affiche le nom + quelques images similaires
"""

# WORKAROUND: Empêcher Streamlit d'inspecter torch.classes (problème de compatibilité)
import sys
from pathlib import Path
import io
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# Appliquer le workaround PyTorch/Streamlit
try:
    from utils.pytorch_fix import apply_pytorch_fix
    apply_pytorch_fix()
except ImportError:
    # Fallback minimal : neutraliser simplement __path__ si présent
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


# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Détection - Agro-Scan",
    page_icon="📸",
    layout="wide",
)

from utils.styles import load_custom_css

load_custom_css()

# Dossier des images légères pour la confirmation visuelle
DATASET_LIGHT_ROOT = Path("dataset_light")


def _map_to_dataset_light(original_path: str) -> str:
    """
    Essaie de remapper un chemin d'image du dataset complet vers dataset_light.
    On conserve la même structure relative à partir de 'dataset_final' si possible.
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
    # Fallback: on garde le chemin original si on ne trouve rien dans dataset_light
    return original_path

# Titre
st.title("📸 Détection Intelligente des Plantes")
st.markdown(
    "Téléversez une image pour identifier la **maladie probable** et comparer avec quelques images similaires."
)


@st.cache_resource
def load_model_and_index():
    """Chargement unique du modèle metric learning et de l'index FAISS."""
    model, index, metadata, prototypes, prototype_labels, device = (
        load_phase2_model_and_metadata(MODELS_PATH_PHASE2)
    )
    return model, index, metadata, prototypes, prototype_labels, device


def _is_confident_unknown(result, faiss_threshold: float = 0.55, min_agreement: int = 3):
    """
    Double logique pro pour 'maladie inconnue':
    - seuil sur la similarité prototype (déjà géré dans infer_on_image)
    - cohérence des voisins FAISS (au moins min_agreement de la même classe)
    """
    is_unknown_flag = result.get("is_unknown", False)
    neighbors = result.get("topk_neighbors", [])

    # S'il n'y a pas de voisins (FAISS absent, index vide, etc.),
    # on se fie uniquement à la logique interne de model_core.
    if not neighbors:
        return bool(is_unknown_flag)

    # Vérifier si les top voisins partagent la même classe
    top = neighbors[:min_agreement]
    diseases = [n["disease"] for n in top]
    main = diseases[0]
    same_count = sum(1 for d in diseases if d == main)

    # On ne marque comme inconnu que si:
    # - le modèle interne le considère déjà comme inconnu
    #   ET
    # - les voisins ne sont pas d'accord entre eux.
    return bool(is_unknown_flag) and same_count < min_agreement


def diagnose_image(
    image: Image.Image,
    uploaded_image_path: str | None = None,
    k: int = 5,
    unknown_threshold: float = 0.55,
):
    """Pipeline d'inférence commun pour cette page + logique Plantix."""
    (
        model,
        index,
        metadata,
        prototypes,
        prototype_labels,
        device,
    ) = load_model_and_index()

    # Redimensionner pour mobile / perfs
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

    # Surcouche "unknown" plus robuste
    is_unknown = _is_confident_unknown(result, faiss_threshold=unknown_threshold)

    neighbors = result["topk_neighbors"]

    # Éviter de montrer l'image uploadée dans les voisins (si même chemin)
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
st.sidebar.title("📸 Détection")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. **Prenez une photo** avec votre caméra
2. **Ou téléversez** une image depuis votre galerie
3. Attendez l'**analyse par l'IA**
4. Consultez les **résultats et recommandations**
""")

# Zone de capture/téléversement
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Prendre une photo")
    camera_input = st.camera_input("Capturez une image", label_visibility="collapsed")
    
    if camera_input:
        try:
            # Obtenir les bytes de l'image
            image_bytes = camera_input.getvalue()
            if not isinstance(image_bytes, bytes):
                image_bytes = bytes(image_bytes)
            
            # Vérifier que les bytes ne sont pas vides
            if len(image_bytes) == 0:
                st.error("❌ L'image capturée est vide. Veuillez réessayer.")
            else:
                # Ouvrir l'image pour l'afficher
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.session_state.uploaded_image = image
                st.session_state.image_bytes = image_bytes
                st.session_state.uploaded_image_path = None
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image: {str(e)}")

with col2:
    st.subheader("📁 Téléverser une image")
    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            # Lire les bytes du fichier
            uploaded_file.seek(0)  # S'assurer qu'on est au début
            image_bytes = uploaded_file.read()
            
            if not isinstance(image_bytes, bytes):
                image_bytes = bytes(image_bytes)
            
            # Vérifier que les bytes ne sont pas vides
            if len(image_bytes) == 0:
                st.error("❌ Le fichier image est vide. Veuillez sélectionner un autre fichier.")
            else:
                # Ouvrir l'image pour l'afficher
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.session_state.uploaded_image = image
                st.session_state.image_bytes = image_bytes
                st.session_state.uploaded_image_path = uploaded_file.name
                
                # Réinitialiser le pointeur pour une utilisation ultérieure
                uploaded_file.seek(0)
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")

# Afficher l'image sélectionnée
if 'uploaded_image' in st.session_state:
    st.markdown("---")
    st.subheader("🖼️ Image à analyser")
    
    st.markdown("#### Image analysée")
    st.image(st.session_state.uploaded_image, use_column_width=True)
    
    # Bouton de détection
    if st.button("🔍 Lancer la détection", type="primary", use_container_width=True):
        with st.spinner("🔬 Analyse de l'image en cours..."):
            try:
                # Ouvrir l'image et lancer le pipeline metric learning
                image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")
                diagnosis = diagnose_image(
                    image,
                    uploaded_image_path=st.session_state.get("uploaded_image_path"),
                    k=5,
                    unknown_threshold=0.55,
                )

                st.session_state.detection_result = diagnosis
                st.session_state.show_results = True
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection: {str(e)}")
                st.session_state.show_results = False

# Afficher les résultats – mode Plantix (nom + images similaires)
if st.session_state.get("show_results", False) and "detection_result" in st.session_state:
    diagnosis = st.session_state.detection_result

    st.markdown("---")
    st.success("✅ Analyse terminée !")

    pred_disease = diagnosis.get("predicted_disease")
    is_unknown = diagnosis.get("is_unknown", False)
    neighbors = diagnosis.get("neighbors", [])

    st.subheader("🦠 Maladie probable")
    if is_unknown or not pred_disease or pred_disease == "UNKNOWN DISEASE":
        st.error("Maladie inconnue – veuillez essayer une autre image ou consulter un expert.")
    else:
        st.markdown(f"## 🌿 {pred_disease}")

    # Visual confirmation : 3–4 images similaires de la même classe
    if neighbors:
        st.markdown("### 🔎 Confirmation visuelle")

        if pred_disease and pred_disease != "UNKNOWN DISEASE":
            confirmation = [n for n in neighbors if n["disease"] == pred_disease][:4]
        else:
            confirmation = neighbors[:4]

        if confirmation:
            cols = st.columns(3)
            for idx, n in enumerate(confirmation):
                col = cols[idx % 3]
                with col:
                    try:
                        source_path = n["image_path"]
                        light_path = _map_to_dataset_light(source_path)
                        ref_img = Image.open(light_path).convert("RGB")
                        st.image(ref_img, use_column_width=True)
                    except Exception:
                        st.warning("Image non disponible")

    # Bouton pour nouvelle détection
    if st.button("🔄 Nouvelle détection", use_container_width=True):
        for key in ["uploaded_image", "image_bytes", "detection_result", "show_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

