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
import os
import base64
import mimetypes

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
from utils.blip2_explainer import generate_explanation_for_image, load_disease_info
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

# Dossier dataset_light pour la confirmation visuelle (léger, compatible Streamlit Cloud)
DATASET_LIGHT_ROOT = Path(os.environ.get("DATASET_LIGHT_ROOT", "dataset_light"))


def _get_images_for_disease(disease_name: str, max_images: int = 4) -> list[str]:
    """
    Récupère jusqu'à max_images images depuis dataset_light pour une maladie.
    Recherche robuste:
    - dataset_light/<classe>/
    - dataset_light/train/<classe>/
    - dataset_light/val/<classe>/
    - dataset_light/test/<classe>/
    """
    candidates = [
        disease_name,
        disease_name.replace(" ", "_"),
        disease_name.replace(" ", ""),
    ]

    for name in candidates:
        imgs = []
        for base in [
            DATASET_LIGHT_ROOT / name,
            DATASET_LIGHT_ROOT / "train" / name,
            DATASET_LIGHT_ROOT / "val" / name,
            DATASET_LIGHT_ROOT / "test" / name,
            DATASET_LIGHT_ROOT / "train_new" / name,
        ]:
            if base.exists() and base.is_dir():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                    imgs.extend(sorted(str(p) for p in base.glob(ext)))
        if imgs:
            return imgs[:max_images]

    return []


def _img_to_data_uri(img_path: str) -> str | None:
    """Convertit une image locale en data-URI (base64) pour afficher une galerie horizontale."""
    try:
        mime, _ = mimetypes.guess_type(img_path)
        if mime is None:
            ext = Path(img_path).suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def render_horizontal_gallery(img_paths: list[str], height_px: int = 180) -> None:
    """Affiche 4 images en ligne horizontale avec scroll (UX pro)."""
    img_paths = [p for p in img_paths if p]
    if not img_paths:
        return

    items_html = ""
    for p in img_paths:
        uri = _img_to_data_uri(p)
        if not uri:
            continue
        items_html += (
            "<div style='flex:0 0 auto; margin-right:12px; text-align:center;'>"
            f"<img src='{uri}' style='height:{height_px}px; width:auto; object-fit:cover; "
            "border-radius:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);'/>"
            "</div>"
        )

    if not items_html:
        return

    st.markdown(
        f"""
        <div style="
          display:flex;
          flex-wrap:nowrap;
          overflow-x:auto;
          padding-bottom:6px;
          margin-top:6px;
        ">
          {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    "ar": "العربية",
    "zh": "中文",
    "ff": "Pulaar",
    "bm": "Bambara",
    "wo": "Wolof",
}

TRANSLATIONS = {
    "page_title": {
        "fr": "📸 Détection Intelligente des Plantes",
        "en": "📸 Smart Plant Disease Detection",
        "tr": "📸 Akıllı Bitki Hastalığı Tespiti",
        "sw": "📸 Utambuzi Mahiri wa Magonjwa ya Mimea",
        "ha": "📸 Ganewar Cutar Shuka Mai Hikima",
        "ar": "📸 الكشف الذكي عن أمراض النباتات",
        "zh": "📸 智能植物病害检测",
        "ff": "📸 Taramol wiɓɓude e cutaali ko ɓernde",
        "bm": "📸 La déteksyon ni cuta bɛn",
        "wo": "📸 Diisaan bi am solo biñ goor",
    },
    "page_subtitle": {
        "fr": "Téléversez une image pour identifier la **maladie probable** et comparer avec quelques images similaires.",
        "en": "Upload an image to identify the **most probable disease** and compare with similar examples.",
        "tr": "En olası hastalığı belirlemek ve benzer örneklerle karşılaştırmak için bir görüntü yükleyin.",
        "sw": "Pakia picha ili kutambua **ugonjwa unaowezekana zaidi** na kuulinganisha na mifano inayofanana.",
        "ha": "Loda hoto don gano **mummanan cutar da za ta fi yiwuwa** kuma kwatanta ta da hotuna makamanta.",
        "ar": "قم برفع صورة لتحديد **أكثر مرض محتمل** ومقارنتها بأمثلة مشابهة.",
        "zh": "上传一张图片，识别**最可能的病害**，并与相似示例进行对比。",
        "ff": "Wiiɓu e sawriŋ ɓernde e ladde **ko a amu ndammal** tee ɓe soodataa ko ñaari ɗo.",
        "bm": "Bérè raa e image-gi la a bɛ so gɛstine **cuta bi a ma sɔrɔ** ka tɛɛn ɲɛnman ɲin bɛfanin.",
        "wo": "Dugal bi mu ngiisaal si **xam-xam gi am na wax** walla, ak yeneen jëfandikoo ci moom dɔ.",
    },
    "sidebar_instructions": {
        "fr": "1. **Prenez une photo** avec votre caméra\n2. **Ou téléversez** une image depuis votre galerie\n3. Attendez l'**analyse par l'IA**\n4. Consultez les **résultats et recommandations**",
        "en": "1. **Take a photo** with your camera\n2. **Or upload** an image from your gallery\n3. Wait for the **AI analysis**\n4. Check the **results and recommendations**",
        "tr": "1. Kameranızla **fotoğraf çekin**\n2. Ya da galerinizden bir **görüntü yükleyin**\n3. **Yapay zekâ analizini** bekleyin\n4. **Sonuçları ve önerileri** inceleyin",
        "sw": "1. **Piga picha** kwa kutumia kamera\n2. Au **pakia** picha kutoka kwenye galeri\n3. Subiri **uchambuzi wa AI**\n4. Angalia **matokeo na mapendekezo**",
        "ha": "1. **Dauki hoto** da kamara\n2. Ko **loda** hoto daga gallery\n3. Jira **binciken AI**\n4. Duba **sakamako da shawarwari**",
        "ar": "1. **التقط صورة** بكاميرتك\n2. **أو ارفع** صورة من المعرض\n3. انتظر **تحليل الذكاء الاصطناعي**\n4. راجع **النتائج والتوصيات**",
        "zh": "1. **用相机拍照**\n2. **或从相册上传**图片\n3. 等待**AI分析**\n4. 查看**结果和建议**",
        "ff": "1. **Daaɗi foto** e kamara-gi\n2. Ko **wiiɓu** e sawriŋ e gelar-gi\n3. Jooɗii **AI analysis**\n4. Wondi **rezultaaji** ko **ndeynaabe**",
        "bm": "1. **Tira foto** ci kamera\n2. Walla **bérè** image ci galerie\n3. Tara **AI analysis** san\n4. Nani **rezilta** ka **tuntum**",
        "wo": "1. **Tir dalal** ci kamera\n2. Bi nekk, **sàppal** image ci galari\n3. Ñaata **AI analysis** bi\n4. Doŋŋ **rezilta** bi ak **toppante**",
    },
    "image_to_analyze": {
        "fr": "🖼️ Image à analyser",
        "en": "🖼️ Image to analyze",
        "tr": "🖼️ Analiz edilecek görüntü",
        "sw": "🖼️ Picha ya kuchambua",
        "ha": "🖼️ Hoton da za a bincika",
        "ar": "🖼️ الصورة المراد تحليلها",
        "zh": "🖼️ 待分析图片",
        "ff": "🖼️ Sawriŋ e ɓeyda",
        "bm": "🖼️ Image bi bɛ jɛfandikoo",
        "wo": "🖼️ Image bu am na jëfandikoo",
    },
    "analyze_button": {
        "fr": "🔍 Lancer la détection",
        "en": "🔍 Run detection",
        "tr": "🔍 Tespiti başlat",
        "sw": "🔍 Anzisha utambuzi",
        "ha": "🔍 Fara gano cuta",
        "ar": "🔍 شغّل الكشف",
        "zh": "🔍 开始检测",
        "ff": "🔍 Wani e diisaan",
        "bm": "🔍 De biyɛ",
        "wo": "🔍 Gën a diisaan",
    },
    "analysis_done": {
        "fr": "✅ Analyse terminée !",
        "en": "✅ Analysis completed!",
        "tr": "✅ Analiz tamamlandı!",
        "sw": "✅ Uchambuzi umekamilika!",
        "ha": "✅ Bincike ya kammala!",
        "ar": "✅ تم اكتمال التحليل!",
        "zh": "✅ 分析完成！",
        "ff": "✅ AI analysis wà yàgg",
        "bm": "✅ Analysis bi kɔrɔ",
        "wo": "✅ Analysis bi am na jëfandikoo",
    },
    "probable_disease": {
        "fr": "🦠 Maladie probable",
        "en": "🦠 Probable disease",
        "tr": "🦠 Muhtemel hastalık",
        "sw": "🦠 Ugonjwa unaowezekana",
        "ha": "🦠 Cutar da ake zargi",
        "ar": "🦠 مرض محتمل",
        "zh": "🦠 可能的病害",
        "ff": "🦠 Cuta ɓernde ko amatu",
        "bm": "🦠 Cuta bi a ka kɔnɔ",
        "wo": "🦠 Cuta bu moom",
    },
    "unknown_disease": {
        "fr": "Maladie inconnue – veuillez essayer une autre image ou consulter un expert.",
        "en": "Unknown disease – please try another image or consult an expert.",
        "tr": "Bilinmeyen hastalık – lütfen başka bir görüntü deneyin veya bir uzmana danışın.",
        "sw": "Ugonjwa haujatambuliwa – tafadhali jaribu picha nyingine au wasiliana na mtaalam.",
        "ha": "An kasa gane cutar – don Allah a gwada wani hoto ko a tuntuɓi ƙwararre.",
        "ar": "مرض غير معروف - يرجى تجربة صورة أخرى أو استشارة مختص.",
        "zh": "未知病害 - 请尝试另一张图片或咨询专家。",
        "ff": "Cuta e wondi yiɗi - waɗii ɓernde fow ndi ko yiɗi ndewli.",
        "bm": "Cuta bi si bɛ xam - bésin image bi ka mɔgɔ ka bɛ jɛfandikoo, baara tɔ.",
        "wo": "Cuta bu baaxuma - liggéey bi wone, walla tan nañu ci sàmm.",
    },
    "visual_confirmation": {
        "fr": "### 🔎 Confirmation visuelle",
        "en": "### 🔎 Visual confirmation",
        "tr": "### 🔎 Görsel doğrulama",
        "sw": "### 🔎 Uthibitisho wa kuona",
        "ha": "### 🔎 Tabbatarwa ta gani",
        "ar": "### 🔎 تأكيد بصري",
        "zh": "### 🔎 视觉确认",
        "ff": "### 🔎 Onɗii e am",
        "bm": "### 🔎 Kɔrɔ bi",
        "wo": "### 🔎 Am solo bi",
    },
    "not_leaf_message": {
        "fr": "Cette image ne ressemble pas à une feuille de culture. Veuillez prendre une photo plus proche de la feuille ou choisir une autre image.",
        "en": "This image does not look like a crop leaf. Please take a closer photo of the leaf or choose another image.",
        "tr": "Bu görüntü bir ürün yaprağına benzemiyor. Lütfen yaprağa daha yakın bir fotoğraf çekin veya başka bir görüntü seçin.",
        "sw": "Picha hii haionekani kama jani la zao. Tafadhali piga picha karibu zaidi ya jani au chagua picha nyingine.",
        "ha": "Wannan hoto ba ya kama da ganyen amfanin gona. Don Allah dauki hoto kusa da ganyen ko ka zabi wani hoto.",
        "ar": "هذه الصورة لا تبدو كأنها ورقة محصول. يرجى التقاط صورة أقرب للورقة أو اختيار صورة أخرى.",
        "zh": "这张图片不像是作物的叶子。请拍摄更近的叶片照片或选择另一张图片。",
        "ff": "Sawriŋ-gi heɓa o ɓeydi e gaddi goɗi. Jeyndii ɓernde ko fii ɓaɗɗi ɗum tawa.",
        "bm": "Image-gi bɛmɛna mɔgɔ ka ɲɔgɔn jani bi bɛgi. Tara foto bi ɲɔgɔn bɔ, walla jɛ image bi mɔgɔ.",
        "wo": "Image bu neexuma bu amul jëmm ci wàllu bi. Doxal dalal bu ɓegg, walla doŋŋ biine.",
    },
    # UI strings (Plantix-like sections)
    "sidebar_title": {
        "fr": "📸 Détection",
        "en": "📸 Detection",
        "tr": "📸 Tespit",
        "sw": "📸 Utambuzi",
        "ha": "📸 Gano Cuta",
        "ar": "📸 الكشف",
        "zh": "📸 检测",
        "ff": "📸 Ñaari",
        "bm": "📸 Déteksyon",
        "wo": "📸 Diisaan",
    },
    "instructions_header": {
        "fr": "### Instructions",
        "en": "### Instructions",
        "tr": "### Talimatlar",
        "sw": "### Maelekezo",
        "ha": "### Umarnai",
        "ar": "### التعليمات",
        "zh": "### 使用说明",
        "ff": "### Heɓi ko",
        "bm": "### Ntɔgɔ",
        "wo": "### Tur",
    },
    "take_photo_subheader": {
        "fr": "📷 Prenez une photo",
        "en": "📷 Take a photo",
        "tr": "📷 Fotoğraf çekin",
        "sw": "📷 Piga picha",
        "ha": "📷 Dauki hoto",
        "ar": "📷 التقط صورة",
        "zh": "📷 拍照",
        "ff": "📷 Daaɗi foto",
        "bm": "📷 Tira foto",
        "wo": "📷 Tir dalal",
    },
    "upload_image_subheader": {
        "fr": "📁 Téléversez une image",
        "en": "📁 Upload an image",
        "tr": "📁 Görüntü yükleyin",
        "sw": "📁 Pakia picha",
        "ha": "📁 Loda hoto",
        "ar": "📁 ارفع صورة",
        "zh": "📁 上传图片",
        "ff": "📁 Wiiɓu sawriŋ",
        "bm": "📁 Bérè image",
        "wo": "📁 Sàppal image",
    },
    "captured_image_empty_error": {
        "fr": "❌ L'image capturée est vide. Veuillez réessayer.",
        "en": "❌ The captured image is empty. Please try again.",
        "tr": "❌ Yakalanan görüntü boş. Lütfen tekrar deneyin.",
        "sw": "❌ Picha iliyopatikana haina kitu. Tafadhali jaribu tena.",
        "ha": "❌ An daskararren hoto babu komai. Don Allah a sake gwadawa.",
        "ar": "❌ الصورة الملتقطة فارغة. يرجى المحاولة مرة أخرى.",
        "zh": "❌ 已捕获的图片为空。请重试。",
        "ff": "❌ Sawriŋ-gi ba na jeyndii. Sani i ɓernde goɗɗi.",
        "bm": "❌ Image bi bɛ capturé-gi ma na komana. Kelen tɛ.",
        "wo": "❌ Dalal bu gis na wàllu bu fau. Jëfandikoo biine.",
    },
    "uploaded_image_empty_error": {
        "fr": "❌ Le fichier image est vide. Veuillez sélectionner un autre fichier.",
        "en": "❌ The image file is empty. Please select another file.",
        "tr": "❌ Görsel dosyası boş. Lütfen başka bir dosya seçin.",
        "sw": "❌ Faili ya picha haina kitu. Tafadhali chagua faili nyingine.",
        "ha": "❌ Fayil ɗin hoto babu komai. Don Allah ka zaɓi wani fayil.",
        "ar": "❌ ملف الصورة فارغ. يرجى اختيار ملف آخر.",
        "zh": "❌ 图片文件为空。请选择其他文件。",
        "ff": "❌ File bi am na sawriŋ ba na komana. Seɗii file wiiɗi.",
        "bm": "❌ File image-gi mɔgɔ ma na komana. Bɔ file biine.",
        "wo": "❌ Fàayil image bi fau. Dogal biine.",
    },
    "analyzed_image_header": {
        "fr": "#### Image analysée",
        "en": "#### Analyzed image",
        "tr": "#### Analiz edilen görsel",
        "sw": "#### Picha iliyochambuliwa",
        "ha": "#### Hoton da aka bincika",
        "ar": "#### الصورة بعد التحليل",
        "zh": "#### 已分析图片",
        "ff": "#### Sawriŋ bi tawa",
        "bm": "#### Image bi bɛ kɔrɔ",
        "wo": "#### Image bu gis",
    },
    "analyzing_image_spinner": {
        "fr": "🔬 Analyse en cours...",
        "en": "🔬 Analyzing image...",
        "tr": "🔬 Görsel analiz ediliyor...",
        "sw": "🔬 Inachambua picha...",
        "ha": "🔬 Yana binciken hoto...",
        "ar": "🔬 جارٍ تحليل الصورة...",
        "zh": "🔬 正在分析图片...",
        "ff": "🔬 ɓeyda sawriŋ...",
        "bm": "🔬 Analysis bi tɛ...",
        "wo": "🔬 Analysing image...",
    },
    "symptoms_header": {
        "fr": "### Symptômes",
        "en": "### Symptoms",
        "tr": "### Belirtiler",
        "sw": "### Dalili",
        "ha": "### Alamomi",
        "ar": "### الأعراض",
        "zh": "### 症状",
        "ff": "### Iɓi ɗi",
        "bm": "### Tilé",
        "wo": "### Daalil",
    },
    "no_structured_symptoms_data": {
        "fr": "_Aucune donnée structurée sur les symptômes._",
        "en": "_No structured symptoms data available._",
        "tr": "_Yapılandırılmış belirti verisi yok._",
        "sw": "_Hakuna data ya dalili iliyoandaliwa._",
        "ha": "_Babu ingantaccen bayanin alamomi._",
        "ar": "_لا تتوفر بيانات أعراض منظمة._",
        "zh": "_暂无结构化症状数据。_",
        "ff": "_Ba data e dalili ko ɓeyɗaa._",
        "bm": "_Kɔndɔ data tilé ka bɛ jɛn._",
        "wo": "_Man na data dalili bu tàngal._",
    },
    "description_header": {
        "fr": "### Description",
        "en": "### Description",
        "tr": "### Açıklama",
        "sw": "### Maelezo",
        "ha": "### Bayani",
        "ar": "### الوصف",
        "zh": "### 描述",
        "ff": "### Fulɓe",
        "bm": "### Yɔrɔ",
        "wo": "### Njax",
    },
    "hosts_header": {
        "fr": "### Hôtes",
        "en": "### Hosts",
        "tr": "### Konukçular",
        "sw": "### Wenyeji",
        "ha": "### Masu karɓa",
        "ar": "### العوائل",
        "zh": "### 宿主",
        "ff": "### Suɓaaje",
        "bm": "### Tè",
        "wo": "### Jëfandikoo",
    },
    "susceptibility_header": {
        "fr": "### Susceptibilité",
        "en": "### Susceptibility",
        "tr": "### Duyarlılık",
        "sw": "### Uwezekano",
        "ha": "### Yiwuwar kamuwa",
        "ar": "### القابلية للإصابة",
        "zh": "### 易感性",
        "ff": "### Ko amatu ndiyoo",
        "bm": "### Bɛ foyi",
        "wo": "### Njàngale bi",
    },
    "scientific_name_label": {
        "fr": "Nom scientifique",
        "en": "Scientific name",
        "tr": "Bilimsel ad",
        "sw": "Jina la kisayansi",
        "ha": "Sunan kimiyya",
        "ar": "الاسم العلمي",
        "zh": "学名",
        "ff": "Jina joɗaa",
        "bm": "Jina bi syen",
        "wo": "Sunan sayansi",
    },
    "pathogen_type_label": {
        "fr": "Type de pathogène",
        "en": "Pathogen type",
        "tr": "Patojen türü",
        "sw": "Aina ya pathojeni",
        "ha": "Nau'in kwayar cuta",
        "ar": "نوع الممرض",
        "zh": "病原体类型",
        "ff": "Niiɗi e patojɛn",
        "bm": "Kaw anw sɔrɔ",
        "wo": "Noor bi ëpp a",
    },
    "confidence_label": {
        "fr": "Confiance",
        "en": "Confidence",
        "tr": "Güven",
        "sw": "Uhakika",
        "ha": "Tabbacin",
        "ar": "الثقة",
        "zh": "置信度",
        "ff": "Uyari",
        "bm": "Kɔndɔkɔ",
        "wo": "Wàllu",
    },
    "visual_confirmation_caption": {
        "fr": "Confirmation visuelle",
        "en": "Visual confirmation",
        "tr": "Görsel doğrulama",
        "sw": "Uthibitisho wa kuona",
        "ha": "Tantancewa ta gani",
        "ar": "تأكيد بصري",
        "zh": "视觉确认",
        "ff": "Onɗii e am",
        "bm": "Kɔrɔ bi",
        "wo": "Am solo bi",
    },
    "highly_susceptible_label": {
        "fr": "Très sensible",
        "en": "Highly susceptible",
        "tr": "Highly susceptible",
        "sw": "Highly susceptible",
        "ha": "Highly susceptible",
        "ar": "عالي القابلية للإصابة",
        "zh": "高度易感",
        "ff": "Highly susceptible",
        "bm": "Highly susceptible",
        "wo": "Highly susceptible",
    },
    "moderately_susceptible_label": {
        "fr": "Sensibilité modérée",
        "en": "Moderately susceptible",
        "tr": "Moderately susceptible",
        "sw": "Moderately susceptible",
        "ha": "Moderately susceptible",
        "ar": "متوسط القابلية للإصابة",
        "zh": "中等易感",
        "ff": "Moderately susceptible",
        "bm": "Moderately susceptible",
        "wo": "Moderately susceptible",
    },
    "more_tolerant_label": {
        "fr": "Plus tolérant",
        "en": "More tolerant",
        "tr": "More tolerant",
        "sw": "More tolerant",
        "ha": "More tolerant",
        "ar": "أكثر تحملا",
        "zh": "更耐受",
        "ff": "More tolerant",
        "bm": "More tolerant",
        "wo": "More tolerant",
    },
    "what_caused_it": {
        "fr": "Qu'est-ce qui l'a causé ?",
        "en": "What caused it?",
        "tr": "Buna ne sebep oldu?",
        "sw": "Ni nini kilisababisha?",
        "ha": "Me ya kawo shi?",
        "ar": "ما سبب ذلك؟",
        "zh": "是什么导致的？",
        "ff": "E ɓiɗɗi ɗum ɓo?",
        "bm": "A ka kelen bɛ? ",
        "wo": "Bi neex na dumm ci moom?",
    },
    "confirm_treatment_btn": {
        "fr": "✅ Confirmer & voir le traitement",
        "en": "✅ Confirm & see treatment",
        "tr": "✅ Onayla ve tedaviyi gör",
        "sw": "✅ Thibitisha & uone matibabu",
        "ha": "✅ Tabbatar & duba magani",
        "ar": "✅ أكد وشاهد العلاج",
        "zh": "✅ 确认并查看治疗",
        "ff": "✅ Yigga & ndeeɓi ɓannu",
        "bm": "✅ Tɔgɔ ka kelen ɲɛ",
        "wo": "✅ Xàccal ak donee",
    },
    "disease_cycle_and_spread_header": {
        "fr": "### Cycle de la maladie et propagation",
        "en": "### Disease cycle and spread",
        "tr": "### Hastalık döngüsü ve yayılma",
        "sw": "### Mzunguko wa ugonjwa na kuenea",
        "ha": "### Zagaye cuta da yaduwa",
        "ar": "### دورة المرض والانتشار",
        "zh": "### 病程与传播",
        "ff": "### Mbaawɗi cuta e ɓeydunde",
        "bm": "### Cuta bi ka kɔrɔw",
        "wo": "### Ware bi ak defar",
    },
    "favorable_conditions_header": {
        "fr": "### Conditions favorables",
        "en": "### Favorable conditions",
        "tr": "### Uygun koşullar",
        "sw": "### Masharti mazuri",
        "ha": "### Yanayi masu kyau",
        "ar": "### الظروف الملائمة",
        "zh": "### 适宜条件",
        "ff": "### Ko am ɓurndude",
        "bm": "### Yanayi sugu",
        "wo": "### Xam-xam bu am solo",
    },
    "pathogen_characteristics_header": {
        "fr": "### Caractéristiques du pathogène",
        "en": "### Pathogen characteristics",
        "tr": "### Patojen özellikleri",
        "sw": "### Sifa za pathojeni",
        "ha": "### Halayen cutar",
        "ar": "### خصائص العامل الممرض",
        "zh": "### 病原体特征",
        "ff": "### Ɓeydii e cuta",
        "bm": "### Halé bèyɛ",
        "wo": "### Yëngu bi",
    },
    "monitoring_header": {
        "fr": "### Suivi",
        "en": "### Monitoring",
        "tr": "### İzleme",
        "sw": "### Ufuatiliaji",
        "ha": "### Kula",
        "ar": "### المتابعة",
        "zh": "### 监测",
        "ff": "### Jeyndii",
        "bm": "### Kɔnɔ",
        "wo": "### Wàkk",
    },
    "management_treatment_header": {
        "fr": "### Gestion / Traitement",
        "en": "### Management / Treatment",
        "tr": "### Yönetim / Tedavi",
        "sw": "### Usimamizi / Matibabu",
        "ha": "### Kulawa / Magani",
        "ar": "### الإدارة / العلاج",
        "zh": "### 管理 / 治疗",
        "ff": "### Ɓeyndirde / Dawɗii",
        "bm": "### Kulaw / Têgɛ",
        "wo": "### Tàqqal / Daaw",
    },
    "no_management_guidance": {
        "fr": "_Aucune recommandation de gestion disponible._",
        "en": "_No management guidance available._",
        "tr": "_Yönetim önerisi yok._",
        "sw": "_Hakuna mwongozo wa usimamizi._",
        "ha": "_Babu shawarwarin kulawa._",
        "ar": "_لا توجد إرشادات إدارة متاحة._",
        "zh": "_暂无管理指导。_",
        "ff": "_Ko bindi ɓeyndirde ma na._",
        "bm": "_Kelen bi bɛ? _",
        "wo": "_Jàngale bi man na._",
    },
    "prevention_header": {
        "fr": "### Prévention",
        "en": "### Prevention",
        "tr": "### Önleme",
        "sw": "### Kinga",
        "ha": "### Rigakafi",
        "ar": "### الوقاية",
        "zh": "### 预防",
        "ff": "### Taaɓal",
        "bm": "### N’o",
        "wo": "### Sukur",
    },
    "optional_ai_explanation_expander": {
        "fr": "Optionnel : Explication IA (BLIP-2)",
        "en": "Optional: AI explanation (BLIP-2)",
        "tr": "Opsiyonel: Yapay Zeka açıklaması (BLIP-2)",
        "sw": "Hiari: ufafanuzi wa AI (BLIP-2)",
        "ha": "Na zaɓi: bayani na AI (BLIP-2)",
        "ar": "اختياري: شرح بالذكاء الاصطناعي (BLIP-2)",
        "zh": "可选：AI 解释（BLIP-2）",
        "ff": "Na ɓuri: AI explanation (BLIP-2)",
        "bm": "Ɲa ɓuri: AI explanation (BLIP-2)",
        "wo": "Ndax li benn: AI explanation (BLIP-2)",
    },
    "generate_ai_explanation_checkbox": {
        "fr": "Générer une explication IA",
        "en": "Generate AI explanation",
        "tr": "Yapay zekâ açıklaması oluştur",
        "sw": "Tengeneza ufafanuzi wa AI",
        "ha": "Ƙirƙiri bayanin AI",
        "ar": "إنشاء شرح بالذكاء الاصطناعي",
        "zh": "生成 AI 解释",
        "ff": "Tee AI explanation",
        "bm": "Tii AI explanation",
        "wo": "Wutal AI explanation",
    },
    "ai_explanation_header": {
        "fr": "### 🤖 Explication IA",
        "en": "### 🤖 AI Explanation",
        "tr": "### 🤖 Yapay Zeka Açıklaması",
        "sw": "### 🤖 Ufafanuzi wa AI",
        "ha": "### 🤖 Bayanin AI",
        "ar": "### 🤖 شرح بالذكاء الاصطناعي",
        "zh": "### 🤖 AI 解释",
        "ff": "### 🤖 AI explanation",
        "bm": "### 🤖 AI explanation",
        "wo": "### 🤖 AI explanation",
    },
    "new_detection_btn": {
        "fr": "🔄 Nouvelle détection",
        "en": "🔄 New detection",
        "tr": "🔄 Yeni tespit",
        "sw": "🔄 Ugunduzi mpya",
        "ha": "🔄 Sabon gano cuta",
        "ar": "🔄 كشف جديد",
        "zh": "🔄 新检测",
        "ff": "🔄 Diisaan e wiiɗi",
        "bm": "🔄 Déteksyon biine",
        "wo": "🔄 Diisaan biine",
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
st.sidebar.title(t("sidebar_title"))

# Language selector
current_lang = get_lang()
lang_options = [f"{code.upper()} - {label}" for code, label in SUPPORTED_LANGS.items()]
default_index = list(SUPPORTED_LANGS.keys()).index(current_lang)
selected = st.sidebar.selectbox("🌐 Langue / Language", lang_options, index=default_index)
for code, label in SUPPORTED_LANGS.items():
    if selected.startswith(code.upper()):
        st.session_state["lang"] = code
        break

st.sidebar.markdown(t("instructions_header"))
st.sidebar.markdown(t("sidebar_instructions"))

# Capture/upload area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(t("take_photo_subheader"))
    camera_input = st.camera_input("Capture an image", label_visibility="collapsed")
    
    if camera_input:
        try:
            # Get the image bytes
            image_bytes = camera_input.getvalue()
            if not isinstance(image_bytes, bytes):
                image_bytes = bytes(image_bytes)
            
            # Check that bytes are not empty
            if len(image_bytes) == 0:
                st.error(t("captured_image_empty_error"))
            else:
                # Open the image to display
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.session_state.uploaded_image = image
                st.session_state.image_bytes = image_bytes
                st.session_state.uploaded_image_path = None
        except Exception as e:
            st.error(f"❌ Error processing the image: {str(e)}")

with col2:
    st.subheader(t("upload_image_subheader"))
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
                st.error(t("uploaded_image_empty_error"))
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
    
    st.markdown(t("analyzed_image_header"))
    # Use image_bytes instead of PIL object to avoid format issues
    if 'image_bytes' in st.session_state and st.session_state.image_bytes:
        st.image(st.session_state.image_bytes, use_column_width=True)
    elif st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, use_column_width=True)
    
    # Detection button
    if st.button(t("analyze_button"), type="primary", use_container_width=True):
        with st.spinner(t("analyzing_image_spinner")):
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

    # Load the base knowledge from JSON (deterministic source of truth)
    # Source: BLIP2_normalized/ (109 JSON). Aucune source n'est affichée.
    disease_data = load_disease_info(
        pred_disease,
        allow_fuzzy=False,
        language_code=get_lang(),
    )

    # Plantix UX: infos scientifiques en premier (avant Symptoms)
    scientific_name = disease_data.get("scientific_name") or ""
    pathogen_type = disease_data.get("pathogen_type") or ""
    description = disease_data.get("description") or ""
    hosts = disease_data.get("hosts") or []
    susceptibility = disease_data.get("susceptibility") or {}

    if scientific_name:
        st.markdown(f"**{t('scientific_name_label')}:** {scientific_name}")
    if pathogen_type:
        st.markdown(f"**{t('pathogen_type_label')}:** {pathogen_type}")
        # Pro UX: images de confirmation directement après pathogen type, avant Description.
        light_images = _get_images_for_disease(pred_disease, max_images=4)
        if light_images:
            st.caption(t("visual_confirmation_caption"))
            render_horizontal_gallery(light_images, height_px=180)
    if description:
        st.markdown(t("description_header"))
        st.write(description)
    if hosts:
        st.markdown(t("hosts_header"))
        for h in hosts:
            st.markdown(f"- {h}")
    if isinstance(susceptibility, dict) and susceptibility:
        st.markdown(t("susceptibility_header"))
        key_order = [
            ("highly_susceptible", t("highly_susceptible_label")),
            ("moderately_susceptible", t("moderately_susceptible_label")),
            ("more_tolerant", t("more_tolerant_label")),
        ]
        rendered_any = False
        for k, label in key_order:
            items = susceptibility.get(k) or []
            if items:
                st.markdown(f"**{label}:**")
                for item in items:
                    st.markdown(f"- {item}")
                rendered_any = True
        if not rendered_any:
            # fallback: show whatever keys exist
            for k, items in susceptibility.items():
                if items:
                    # If it's one of our known keys, show a translated label.
                    fallback_label = {
                        "highly_susceptible": t("highly_susceptible_label"),
                        "moderately_susceptible": t("moderately_susceptible_label"),
                        "more_tolerant": t("more_tolerant_label"),
                    }.get(k, k)
                    st.markdown(f"**{fallback_label}:**")
                    for item in items:
                        st.markdown(f"- {item}")

    # Confidence
    confidence = diagnosis.get("predicted_similarity")
    if confidence is not None:
        st.markdown(f"**{t('confidence_label')}:** {confidence*100:.0f}%")

    # -----------------------------
    # Plantix UX: preview then confirm
    # -----------------------------
    if "confirmed_disease" not in st.session_state:
        st.session_state.confirmed_disease = None

    # Preview: name + symptoms only
    st.markdown(t("symptoms_header"))
    if disease_data.get("symptoms"):
        for item in disease_data["symptoms"]:
            st.markdown(f"- {item}")
    else:
        st.markdown(t("no_structured_symptoms_data"))

    # (Optionnel) Cause courte
    if disease_data.get("cause"):
        with st.expander(t("what_caused_it")):
            st.write(disease_data["cause"])

    # Confirm button: reveals treatment/prevention
    if (not is_unknown) and pred_disease and st.session_state.confirmed_disease != pred_disease:
        if st.button(
            t("confirm_treatment_btn"),
            use_container_width=True,
            key="confirm_treatment_btn",
        ):
            st.session_state.confirmed_disease = pred_disease

    if st.session_state.confirmed_disease == pred_disease:
        # After confirmation: show full Plantix-like card content.
        # On évite volontairement d'afficher "sources" (pour jurys/recherche).

        def _render_bullets(title: str, items: list[str]):
            if items:
                st.markdown(f"### {title}")
                for item in items:
                    st.markdown(f"- {item}")

        _render_bullets(
            t("disease_cycle_and_spread_header").replace("### ", ""),
            disease_data.get("disease_cycle_and_spread") or [],
        )
        _render_bullets(
            t("favorable_conditions_header").replace("### ", ""),
            disease_data.get("favorable_conditions") or [],
        )
        _render_bullets(
            t("pathogen_characteristics_header").replace("### ", ""),
            disease_data.get("pathogen_characteristics") or [],
        )
        _render_bullets(t("monitoring_header").replace("### ", ""), disease_data.get("monitoring") or [])

        st.markdown(t("management_treatment_header"))
        mgmt = disease_data.get("management") or []
        if mgmt:
            for item in mgmt:
                st.markdown(f"- {item}")
        else:
            st.markdown(t("no_management_guidance"))

        _render_bullets(t("prevention_header").replace("### ", ""), disease_data.get("prevention") or [])

        hosts = disease_data.get("hosts") or []
        _render_bullets(t("hosts_header").replace("### ", ""), hosts)

    # Optionnel: explication IA (désactivée par défaut, coûteuse).
    # Les images de confirmation sont déjà affichées juste après pathogen type.
    if "image_for_explanation" in st.session_state:
        with st.expander(t("optional_ai_explanation_expander"), expanded=False):
            if st.checkbox(t("generate_ai_explanation_checkbox"), value=False):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = generate_explanation_for_image(
                            st.session_state.image_for_explanation,
                            pred_disease,
                            library_dir=Path("BLIP2_normalized"),
                            language_code=get_lang(),
                        )
                        st.markdown(t("ai_explanation_header"))
                        st.write(explanation)
                    except Exception as e:
                        st.error(f"⚠️ Failed to generate explanation: {e}")

    # Button for new detection
    if st.button(t("new_detection_btn"), use_container_width=True):
        for key in [
            "uploaded_image",
            "image_bytes",
            "image_for_explanation",
            "detection_result",
            "show_results",
            "confirmed_disease",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

