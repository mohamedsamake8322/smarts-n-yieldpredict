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

# Initialisation de l'état de session
def init_session_state():
    defaults = {
        "uploaded_image": None,
        "image_bytes": None,
        "uploaded_image_path": None,
        "lang": "fr",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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


def _get_light_images_for_disease(disease_name: str, max_images: int = 4) -> list[str]:
    """
    Récupère jusqu'à max_images images dans dataset_light pour une maladie donnée.
    On essaie plusieurs variantes de nom de dossier pour être robuste
    (espaces vs underscores).
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
    Heuristique simple pour filtrer les images qui ne sont manifestement
    pas des feuilles de cultures (basée sur la dominance du canal vert).
    Ce n'est pas un classifieur, juste un garde‑fou UX.
    """
    try:
        arr = np.array(image.convert("RGB")).astype("float32") / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        green_dominance = (g > r) & (g > b)
        ratio_green = float(green_dominance.mean())
        return ratio_green >= green_threshold
    except Exception:
        return True  # en cas de doute, on laisse l'IA décider


# --------------------------
# Internationalisation simple
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

# Sélecteur de langue
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
    st.subheader(t("image_to_analyze"))
    
    st.markdown("#### Image analysée")
    # Utiliser image_bytes au lieu de l'objet PIL pour éviter les problèmes de format
    if 'image_bytes' in st.session_state and st.session_state.image_bytes:
        st.image(st.session_state.image_bytes, use_column_width=True)
    elif st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, use_column_width=True)
    
    # Bouton de détection
    if st.button(t("analyze_button"), type="primary", use_container_width=True):
        with st.spinner("🔬 Analyse de l'image en cours..."):
            try:
                # Ouvrir l'image
                image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")

                # Garde‑fou: image manifestement non végétale
                if not _is_plant_like(image):
                    st.error(t("not_leaf_message"))
                    st.session_state.show_results = False
                else:
                    # Lancer le pipeline metric learning
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
    st.success(t("analysis_done"))

    pred_disease = diagnosis.get("predicted_disease")
    is_unknown = diagnosis.get("is_unknown", False)
    neighbors = diagnosis.get("neighbors", [])

    st.subheader(t("probable_disease"))
    if is_unknown or not pred_disease or pred_disease == "UNKNOWN DISEASE":
        st.error(t("unknown_disease"))
    else:
        st.markdown(f"## 🌿 {pred_disease}")

    # Visual confirmation : 3–4 images de dataset_light pour la classe prédite
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
                        st.warning("Image non disponible")

    # Bouton pour nouvelle détection
    if st.button("🔄 Nouvelle détection", use_container_width=True):
        for key in ["uploaded_image", "image_bytes", "detection_result", "show_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

