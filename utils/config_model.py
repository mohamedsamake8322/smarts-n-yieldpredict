import os
import urllib.request
import json
from tensorflow import keras
import streamlit as st

# 📦 Liens Hugging Face
MODEL_URL = "https://huggingface.co/mohamedsamake8322/smartagro-efficientnet-resnet/resolve/main/best_model.keras"
MODEL_PATH = "model/best_model.keras"

LABELS_URL = "https://huggingface.co/mohamedsamake8322/smartagro-efficientnet-resnet/resolve/main/labels.json"
LABELS_PATH = "model/labels.json"


def download_if_missing(url: str, dest_path: str, description: str):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with st.spinner(f"📥 Téléchargement {description}..."):
            try:
                urllib.request.urlretrieve(url, dest_path)
                st.success(f"✅ {description} téléchargé avec succès.")
            except Exception as e:
                st.error(f"❌ Échec du téléchargement de {description} : {e}")
                st.stop()


@st.cache_resource
def load_model():
    """Conservé pour compatibilité: charge l'ancien modèle Keras si nécessaire."""
    download_if_missing(MODEL_URL, MODEL_PATH, "du modèle")
    return keras.models.load_model(MODEL_PATH, compile=False)


def load_labels(output_dir: str = "out") -> dict:
    """Charge en priorité les labels du nouveau modèle (out/idx2label.json), sinon fallback labels.json distant."""
    idx2label_path = os.path.join(output_dir, "idx2label.json")
    if os.path.exists(idx2label_path):
        try:
            with open(idx2label_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # assurer des clés str pour compat
            return {str(k): v for k, v in raw.items()}
        except Exception as e:
            st.warning(f"⚠️ Échec lecture {idx2label_path}: {e}. Bascule vers labels.json distant.")

    download_if_missing(LABELS_URL, LABELS_PATH, "des étiquettes (labels.json)")
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Impossible de charger labels.json : {e}")
        st.stop()
        return {}
