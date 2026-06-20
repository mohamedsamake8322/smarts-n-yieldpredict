"""
SmartAgri — Test Local du Modèle
=================================
Lance : python test_local.py
Ouvre : http://localhost:7860
"""

import json
import math
import os
from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageStat
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────
# Chemins par défaut
# ─────────────────────────────────────────────
DEFAULT_MODEL   = r"C:\Users\moham\Pictures\test detection\senedisease_macro_f1.pt"
DEFAULT_MAPPING = r"C:\Users\moham\Pictures\test detection\class_mapping.json"

IMG_SIZE     = 384
TEMPERATURE  = 3.1
CONFIDENCE_THR = 0.55

# ─────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s; self.m = m
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        x = x.float()
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.float()))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if label is None:
            return self.s * cosine
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh   = torch.zeros_like(cosine)
        oh.scatter_(1, label.view(-1, 1).long(), 1.0)
        return ((oh * phi) + ((1.0 - oh) * cosine)) * self.s


class SwinArcFaceModel(nn.Module):
    def __init__(self, num_classes, num_crops, num_dtypes, num_health,
                 dropout=0.3, s=30.0, m=0.35, img_size=384):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=False, num_classes=0, global_pool="avg",
            img_size=img_size, strict_img_size=False,
        )
        feat_dim = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim), nn.GELU(),
        )
        self.arc         = ArcMarginProduct(feat_dim, num_classes, s=s, m=m)
        self.crop_head   = nn.Linear(feat_dim, num_crops)
        self.dtype_head  = nn.Linear(feat_dim, num_dtypes)
        self.health_head = nn.Linear(feat_dim, num_health)

    def forward(self, x):
        feat = self.backbone(x).float()
        emb  = F.normalize(self.neck(feat), p=2, dim=1)
        return self.arc(emb, label=None)


# ─────────────────────────────────────────────
# État global
# ─────────────────────────────────────────────
STATE = {
    "model":   None,
    "mapping": None,
    "device":  "cpu",
    "status":  "❌ Aucun modèle chargé",
}

TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.05)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Chargement modèle
# ─────────────────────────────────────────────
def load_model(model_path: str, mapping_path: str) -> str:
    """Charge le modèle et le mapping. Retourne un message de statut."""
    global STATE

    model_path   = model_path.strip().strip('"')
    mapping_path = mapping_path.strip().strip('"')

    if not Path(model_path).exists():
        msg = f"❌ Modèle introuvable : {model_path}"
        STATE["status"] = msg
        return msg

    if not Path(mapping_path).exists():
        msg = f"❌ class_mapping.json introuvable : {mapping_path}"
        STATE["status"] = msg
        return msg

    try:
        # Checkpoint
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        sd   = ckpt["model_state_dict"]

        num_classes = int(sd["arc.weight"].shape[0])
        num_crops   = int(sd["crop_head.weight"].shape[0])
        num_dtypes  = int(sd["dtype_head.weight"].shape[0])
        num_health  = int(sd["health_head.weight"].shape[0])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = SwinArcFaceModel(num_classes, num_crops, num_dtypes, num_health).to(device)

        # EMA
        ema_shadow = ckpt.get("ema_shadow", {})
        merged     = dict(sd)
        applied    = 0
        for name, tensor in ema_shadow.items():
            if name in merged and merged[name].shape == tensor.shape:
                merged[name] = tensor
                applied += 1
        model.load_state_dict(merged, strict=False)
        model.eval()

        # Mapping
        with open(mapping_path, encoding="utf-8") as f:
            raw_mapping = json.load(f)

        label_idx_to_info = {}
        for idx_str, cid in raw_mapping["label_idx_to_class_id"].items():
            info = raw_mapping["class_id_to_info"].get(str(cid), {})
            label_idx_to_info[int(idx_str)] = info

        STATE["model"]   = model
        STATE["mapping"] = label_idx_to_info
        STATE["device"]  = device

        msg = (
            f"✅ Modèle chargé avec succès !\n"
            f"   Classes     : {num_classes}\n"
            f"   Crops       : {num_crops}\n"
            f"   Disease types : {num_dtypes}\n"
            f"   Health states : {num_health}\n"
            f"   EMA appliquée : {applied} poids\n"
            f"   Device      : {device.upper()}\n"
            f"   Mapping     : {len(label_idx_to_info)} entrées"
        )
        STATE["status"] = msg
        return msg

    except Exception as e:
        msg = f"❌ Erreur chargement : {e}"
        STATE["status"] = msg
        return msg


def load_default() -> str:
    """Charge les fichiers depuis les chemins par défaut."""
    return load_model(DEFAULT_MODEL, DEFAULT_MAPPING)


# ─────────────────────────────────────────────
# Qualité image
# ─────────────────────────────────────────────
def assess_quality(img: Image.Image) -> tuple[str, float]:
    try:
        import cv2
        gray  = np.array(img.convert("L"))
        stat  = ImageStat.Stat(img)
        lap   = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp = min(lap / 500.0, 1.0)
        bri   = stat.mean[0] / 255.0
        bri_s = 1.0 - abs(bri - 0.5) * 2.0
        cont  = min(stat.stddev[0] / 80.0, 1.0)
        score = 0.5 * sharp + 0.25 * bri_s + 0.25 * cont
    except Exception:
        score = 0.6
    label = "🟢 Bonne" if score >= 0.70 else ("🟡 Moyenne" if score >= 0.45 else "🔴 Faible")
    return label, float(score)


# ─────────────────────────────────────────────
# Prédiction
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(image: Image.Image) -> tuple:
    """
    Retourne (diagnostic, confiance, qualité, fiabilité,
              culture, sci_name, catégorie, top3_text, conseil)
    """
    if STATE["model"] is None:
        msg = "⚠️ Aucun modèle chargé — utilisez le panneau de chargement ci-dessus"
        return (msg,) + ("—",) * 8

    if image is None:
        return ("❌ Aucune image",) + ("—",) * 8

    # Qualité
    quality_label, quality_score = assess_quality(image)

    # Inférence
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(STATE["device"])
    logits = STATE["model"](tensor)
    probs  = F.softmax(logits / TEMPERATURE, dim=1)[0].cpu().numpy()
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(int(i), float(probs[i])) for i in top3_idx]

    # Infos top-1
    top1_idx, top1_conf = top3[0]
    info     = STATE["mapping"].get(top1_idx, {}) if STATE["mapping"] else {}
    disp     = info.get("display_name") or info.get("scientific_name") or f"Classe {top1_idx}"
    crop     = info.get("crop", "—")
    sci      = info.get("scientific_name") or "—"
    category = info.get("category") or "—"

    # Incertitude
    gap       = top3[0][1] - top3[1][1]
    uncertain = top1_conf < CONFIDENCE_THR or gap < 0.15

    if uncertain:
        diagnostic = "⚠️ Résultat incertain — prenez une autre photo"
        conf_str   = f"{top1_conf*100:.1f}% (faible)"
    else:
        diagnostic = f"🌿 {disp}"
        conf_str   = f"{top1_conf*100:.1f}%"

    # Fiabilité
    rel_score = 0.5 * top1_conf + 0.3 * quality_score + 0.2 * min(gap / 0.15, 1.0)
    reliability = "🟢 Élevée" if rel_score >= 0.75 else ("🟡 Modérée" if rel_score >= 0.50 else "🔴 Faible")

    # Top-3
    top3_lines = []
    for rank, (idx, prob) in enumerate(top3, 1):
        inf  = STATE["mapping"].get(idx, {}) if STATE["mapping"] else {}
        name = inf.get("display_name") or f"Classe {idx}"
        bar  = "█" * int(prob * 20)
        top3_lines.append(f"{rank}. {name}\n   {bar} {prob*100:.1f}%")
    top3_text = "\n\n".join(top3_lines)

    # Conseil
    cat_map = {
        "Fungal":    "🍄 Maladie fongique — consultez un agronome pour un traitement fongicide.",
        "Bacterial": "🦠 Maladie bactérienne — évitez l'humidité et consultez un spécialiste.",
        "Viral":     "🧬 Maladie virale — contrôlez les insectes vecteurs.",
        "Pest":      "🐛 Ravageur — identifiez l'insecte et appliquez un traitement ciblé.",
    }
    if uncertain:
        conseil = "📸 Photographiez la feuille de près, en bonne lumière, sans flou."
    elif "Healthy" in str(info.get("class_name", "")):
        conseil = "✅ Plante saine — aucun traitement nécessaire."
    else:
        conseil = cat_map.get(category, "ℹ️ Consultez un agronome pour confirmation.")

    return (
        diagnostic,
        conf_str,
        quality_label,
        reliability,
        f"🌾 {crop}",
        f"🔬 {sci}",
        f"📂 {category}",
        top3_text,
        conseil,
    )


# ─────────────────────────────────────────────
# Interface Gradio
# ─────────────────────────────────────────────
with gr.Blocks(title="🌿 SmartAgri — Test Local") as demo:

    gr.Markdown("# 🌿 SmartAgri — Test Local du Modèle")
    gr.Markdown("Charge le modèle `senedisease_macro_f1.pt` et teste des prédictions.")

    # ── Section chargement ──
    with gr.Group():
        gr.Markdown("## 1️⃣ Chargement du modèle")

        with gr.Row():
            model_path_input = gr.Textbox(
                value=DEFAULT_MODEL,
                label="📁 Chemin du modèle (.pt)",
                placeholder=r"C:\...\senedisease_macro_f1.pt",
                scale=3,
            )
            mapping_path_input = gr.Textbox(
                value=DEFAULT_MAPPING,
                label="📁 Chemin class_mapping.json",
                placeholder=r"C:\...\class_mapping.json",
                scale=3,
            )

        with gr.Row():
            btn_default = gr.Button("⚡ Charger depuis les chemins par défaut", variant="primary")
            btn_custom  = gr.Button("📂 Charger depuis les chemins ci-dessus",  variant="secondary")

        status_box = gr.Textbox(
            label="Statut du chargement",
            value="❌ Aucun modèle chargé",
            interactive=False,
            lines=7,
        )

    gr.Markdown("---")

    # ── Section prédiction ──
    with gr.Group():
        gr.Markdown("## 2️⃣ Test de prédiction")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📷 Image à tester")
                btn_predict = gr.Button("🔍 Analyser", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Résultat")
                out_diag = gr.Textbox(label="🩺 Diagnostic",        interactive=False)
                out_conf = gr.Textbox(label="📊 Confiance calibrée", interactive=False)

                with gr.Row():
                    out_qual = gr.Textbox(label="🖼️ Qualité image",   interactive=False)
                    out_rel  = gr.Textbox(label="🎯 Fiabilité",        interactive=False)

                with gr.Accordion("🔬 Détails", open=False):
                    out_crop = gr.Textbox(label="Culture",            interactive=False)
                    out_sci  = gr.Textbox(label="Nom scientifique",   interactive=False)
                    out_cat  = gr.Textbox(label="Catégorie",          interactive=False)

                with gr.Accordion("📊 Top-3", open=True):
                    out_top3 = gr.Textbox(label="", interactive=False, lines=9)

                out_conseil = gr.Textbox(label="💡 Conseil", interactive=False, lines=2)

    # ── Événements ──
    outputs = [out_diag, out_conf, out_qual, out_rel, out_crop, out_sci, out_cat, out_top3, out_conseil]

    btn_default.click(fn=load_default, outputs=[status_box])
    btn_custom.click(fn=load_model,    inputs=[model_path_input, mapping_path_input], outputs=[status_box])
    btn_predict.click(fn=predict,      inputs=[image_input], outputs=outputs)
    image_input.change(fn=predict,     inputs=[image_input], outputs=outputs)

    gr.Markdown("""
    ---
    **Modèle** : Swin-Base 384px + ArcFace | **Calibration** : Temperature Scaling T=3.1
    """)

if __name__ == "__main__":
    # Tente de charger automatiquement au démarrage
    print("SmartAgri — Test Local")
    print(f"Modèle   : {DEFAULT_MODEL}")
    print(f"Mapping  : {DEFAULT_MAPPING}")

    if Path(DEFAULT_MODEL).exists() and Path(DEFAULT_MAPPING).exists():
        print(load_default())
    else:
        print("⚠️ Fichiers par défaut non trouvés — chargez manuellement dans l'interface")

    demo.launch(server_port=7860, share=False, inbrowser=True)
