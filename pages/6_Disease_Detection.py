import streamlit as st
import os
import sys
import json
from PIL import Image, ImageEnhance
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvmodels

st.set_page_config(page_title="Diagnostic Agricole Pro", page_icon="🩺", layout="wide")
sys.path.append(os.path.abspath("."))

@st.cache_data
def load_descriptions_fallback():
    try:
        with open("data/all_diseases_translated.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

@st.cache_data
def load_label_mappings(output_dir: str = "out"):
    idx2label_path = os.path.join(output_dir, "idx2label.json")
    label2idx_path = os.path.join(output_dir, "label2idx.json")
    if not (os.path.exists(idx2label_path) and os.path.exists(label2idx_path)):
        return {}, {}
    with open(idx2label_path, "r", encoding="utf-8") as f:
        idx2label_raw = json.load(f)
    # clés indices en int
    idx2label = {int(k): v for k, v in idx2label_raw.items()}
    with open(label2idx_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    return idx2label, label2idx

class ImageClassifier(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", num_classes: int = 2, proj_dim: int = 512, pretrained: bool = False):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            try:
                weights = tvmodels.ResNet18_Weights.DEFAULT if pretrained else None
                backbone = tvmodels.resnet18(weights=weights)
            except Exception:
                backbone = tvmodels.resnet18(pretrained=pretrained)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            try:
                weights = tvmodels.ResNet50_Weights.DEFAULT if pretrained else None
                backbone = tvmodels.resnet50(weights=weights)
            except Exception:
                backbone = tvmodels.resnet50(pretrained=pretrained)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")

        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, images):
        feats = self.backbone(images)
        emb = self.proj(feats)
        logits = self.classifier(emb)
        return logits, emb

@st.cache_resource
def get_model_and_mappings(output_dir: str = "out", backbone: str = "resnet18", proj_dim: int = 512):
    idx2label, label2idx = load_label_mappings(output_dir)
    if not idx2label:
        st.warning("Aucun mapping de labels trouvé dans 'out/'. Entraînez d'abord le modèle.")
        return None, idx2label, label2idx
    num_classes = len(idx2label)
    model = ImageClassifier(backbone_name=backbone, num_classes=num_classes, proj_dim=proj_dim, pretrained=False)
    ckpt_path = os.path.join(output_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        st.warning("Aucun fichier de modèle trouvé dans 'out/best.pth'. Entraînez d'abord le modèle.")
        return None, idx2label, label2idx
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])  # aligné avec train_and_infer.py
        model.eval()
    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        return None, idx2label, label2idx
    return model, idx2label, label2idx

# Charger le modèle de manière paresseuse pour éviter les conflits Streamlit/PyTorch
model = None
idx2label = {}
label2idx = {}

def get_model():
    global model, idx2label, label2idx
    if model is None:
        model, idx2label, label2idx = get_model_and_mappings()
    return model, idx2label, label2idx

DISEASE_ICONS = {
    "Healthy": "✅",
    "Aphids on Vegetables": "🐛🥦",
    "Armyworms on Vegetables": "🐛🍃",
    "Blister Beetle": "🪲🔥",
    "Beet Leafhopper": "🪲🌿",
    "Colorado Potato Beetle": "🥔🪲",
    "Western Striped and Spotted Cucumber Beetle": "🥒🪲",
    "Spotted Cucumber Beetle": "🥒🐞",
    "Cutworms on Vegetables": "🐛✂️",
    "False Chinch Bug": "🐜❌",
    "Flea Beetles": "🪲🔬",
    "Tomato and Tobacco Hornworms": "🍅🐛",
    "Thrips on Vegetables": "🦟🥦",
    "Potato Leafhopper": "🥔🌿",
    "Two-Spotted Spider Mite": "🕷️🌱",
    "Corn Earworm / Tomato Fruitworm": "🌽🍅🐛",
    "Tomato Russet Mite": "🍅🕷️",
    "Whiteflies (Family: Aleyrodidae)": "🦟🌿",
    "Alfalfa Mosaic Virus": "🦠🌱",
    "Bacterial Canker": "🦠⚠️",
    "Bacterial Speck": "🦠🍅",
    "Beet Curly Top Virus": "🌀🦠",
    "Big Bud": "🌿💥",
    "Blossom End Rot": "🍅⚫",
    "Damping-Off": "🌱🚫",
    "Early Blight": "🍅🟠",
    "Fusarium Crown/Root Rot": "🌿🦠",
    "Fusarium Wilt": "🌾⚠️",
    "Late Blight": "🍅🔥",
    "Root-Knot Nematodes": "🌱🐛",
    "Phytophthora Root, Stem, and Crown Rots": "🌿🦠",
    "Powdery Mildew on Vegetables": "🍃🌫️",
    "Tobacco Mosaic Virus & Tomato Mosaic Virus": "🍅🌿🦠",
    "Tomato Spotted Wilt Virus": "🍅🔴",
    "Verticillium Wilt": "🌾🔴",
    "Cercospora Leaf Spot (Frogeye)": "🌿⚪",
    "Choanephora Blight (Wet Rot)": "🌿💧",
    "Gray Leaf Spot": "🌿🔘",
    "Phomopsis Blight": "🌿🔥",
}

@st.cache_data
def load_enriched_mapping(path: str = r"C:\Users\moham\Music\plantdataset\maladies_enrichies.json"):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

maladies_enrichies = load_enriched_mapping()
disease_descriptions = load_descriptions_fallback()

def estimate_progression(conf):
    if conf > 90: return "🔴 Critique"
    elif conf > 75: return "🟠 Avancé"
    elif conf > 50: return "🟡 Début"
    else: return "🟢 Faible impact"

def predict_disease(image_pil, confidence_threshold=0.2, topk=3):
    model, idx2label, label2idx = get_model()
    if model is None or not idx2label:
        raise RuntimeError("Modèle non chargé. Entraînez d'abord le modèle dans 'out/'.")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    topk = int(min(topk, len(probs)))
    top_idx = probs.argsort()[::-1][:topk]
    results = []
    for i in top_idx:
        cls_name = idx2label.get(int(i), str(i))
        p = float(probs[i]) * 100.0
        results.append({"disease": cls_name, "confidence": p})

    preds = []
    for r in results:
        if r["confidence"] < confidence_threshold * 100:
            continue
        info = maladies_enrichies.get(r["disease"], {}) if isinstance(maladies_enrichies, dict) else {}
        desc = ""
        sympt = ""
        mgmt = ""
        if isinstance(info, dict):
            desc = info.get("description", {})
            if isinstance(desc, dict):
                desc = desc.get("fr", next(iter(desc.values()), ""))
            sympt = info.get("symptômes", "")
            if isinstance(sympt, dict):
                sympt = sympt.get("fr", "")
            mgmt = info.get("traitement", "")
            if isinstance(mgmt, dict):
                mgmt = mgmt.get("fr", "")
        if not desc:
            # fallback aux anciennes descriptions si correspondance par nom
            fallback = next((d for d in disease_descriptions if d.get("name", "").strip().lower() == r["disease"].strip().lower()), {})
            desc = fallback.get("symptoms", "")
            mgmt = fallback.get("management", "")

        preds.append({
            "name": f"{DISEASE_ICONS.get(r['disease'], '🦠')} {r['disease']}",
            "confidence": r["confidence"],
            "progression_stage": estimate_progression(r["confidence"]),
            "symptoms": sympt or (desc if desc else "❌ Non disponibles"),
            "recommendations": mgmt or "❌ Aucune recommandation",
        })
    if not preds:
        # montrer quand même topk si seuil trop strict
        for r in results:
            info = maladies_enrichies.get(r["disease"], {}) if isinstance(maladies_enrichies, dict) else {}
            desc = ""
            sympt = ""
            mgmt = ""
            if isinstance(info, dict):
                desc = info.get("description", {})
                if isinstance(desc, dict):
                    desc = desc.get("fr", next(iter(desc.values()), ""))
                sympt = info.get("symptômes", "")
                if isinstance(sympt, dict):
                    sympt = sympt.get("fr", "")
                mgmt = info.get("traitement", "")
                if isinstance(mgmt, dict):
                    mgmt = mgmt.get("fr", "")
            if not desc:
                fallback = next((d for d in disease_descriptions if d.get("name", "").strip().lower() == r["disease"].strip().lower()), {})
                desc = fallback.get("symptoms", "")
                mgmt = fallback.get("management", "")
            preds.append({
                "name": f"{DISEASE_ICONS.get(r['disease'], '🦠')} {r['disease']}",
                "confidence": r["confidence"],
                "progression_stage": estimate_progression(r["confidence"]),
                "symptoms": sympt or (desc if desc else "❌ Non disponibles"),
                "recommendations": mgmt or "❌ Aucune recommandation",
            })
    return preds

def render_diagnostic_card(result):
    color = {
        "🔴 Critique": "red",
        "🟠 Avancé": "orange",
        "🟡 Début": "gold",
        "🟢 Faible impact": "green"
    }.get(result["progression_stage"], "gray")

    with st.container():
        st.markdown("---")
        st.markdown(f"### {result['name']}")
        st.markdown(f"<span style='color:{color}; font-weight:bold;'>Gravité : {result['progression_stage']}</span>", unsafe_allow_html=True)
        st.markdown(f"📊 **Confiance IA :** {result['confidence']:.1f}%")
        st.markdown(f"🧬 **Symptômes :** {result['symptoms']}")
        st.markdown(f"💊 **Recommandations :** {result['recommendations']}")

st.title("🌿 Disease Detector Pro")

tab_upload, tab_camera = st.tabs(["📤 Upload", "📷 Caméra"])

with tab_upload:
    uploaded = st.file_uploader("Téléverser une image de la plante 🌱", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

with tab_camera:
    camera_image = st.camera_input("Prendre une photo avec la caméra")

image_source = None
if uploaded is not None:
    image_source = uploaded
elif camera_image is not None:
    image_source = camera_image

if image_source:
    try:
        image = Image.open(image_source).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="🌱 Image originale", use_container_width=True)

        enhance = st.checkbox("🔬 Améliorer le contraste ?", value=True)
        if enhance:
            image = ImageEnhance.Contrast(image).enhance(1.2)

        threshold = st.slider("🎚️ Seuil de confiance IA (%)", min_value=1, max_value=100, value=60, step=1) / 100
        force_top3 = st.checkbox("📋 Afficher les 3 meilleures prédictions (même avec faible confiance)", value=False)
        with st.spinner("🧠 Diagnostic en cours... 🔎🧪✨"):
            # petite animation d'emojis en placeholder
            ph = st.empty()
            ph.markdown("🔎")
            predictions = predict_disease(image, confidence_threshold=threshold if not force_top3 else 0.0, topk=3)
            ph.empty()
            st.write("🧠 Résultats bruts :", predictions)
        if predictions:
            st.success("✅ Analyse terminée")
            for result in predictions[:3]:
                render_diagnostic_card(result)
        else:
            st.info("ℹ️ Aucun symptôme détecté avec une confiance suffisante.")
    except Exception as e:
        st.error(f"❌ Erreur lors de l’analyse : {e}")
else:
    st.info("📷 Téléversez une image ou utilisez la caméra pour commencer.")
