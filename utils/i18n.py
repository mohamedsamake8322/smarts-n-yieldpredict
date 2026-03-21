import os
import streamlit as st

LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("Français", "fr"),
    ("العربية", "ar"),
    ("中文", "zh"),
    ("Русский", "ru"),
    ("Español", "es"),
    ("Deutsch", "de"),
    ("Türkçe", "tr"),
    ("Swahili", "sw"),
    ("አማርኛ", "am"),
    ("Igbo", "ig"),
    ("Hausa", "ha"),
]

LANGUAGE_CODE_MAP = {label: code for label, code in LANGUAGE_OPTIONS}
LANGUAGE_LABEL_MAP = {code: label for label, code in LANGUAGE_OPTIONS}

TRANSLATIONS = {
    "app_title": {
        "en": "🌾 AI Plant Disease Diagnostic Assistant",
        "fr": "🌾 Assistant de diagnostic des maladies des plantes",
    },
    "app_description": {
        "en": "This AI system diagnoses plant diseases from images by finding visually similar examples from a training dataset.",
        "fr": "Ce système d'IA diagnostique les maladies des plantes à partir d'images en trouvant des exemples visuellement similaires.",
    },
    "sidebar_settings": {
        "en": "⚙️ Settings",
        "fr": "⚙️ Paramètres",
    },
    "select_language": {
        "en": "🌍 Language / Langue",
        "fr": "🌍 Langue / Language",
    },
    "language_active": {
        "en": "Active language",
        "fr": "Langue active",
    },
    "unknown_disease": {
        "en": "Unknown disease – please try another image or consult an expert.",
        "fr": "Maladie inconnue – veuillez essayer une autre image ou consulter un expert.",
    },
    "probable_disease": {
        "en": "🦠 Probable disease",
        "fr": "🦠 Maladie probable",
    },
    "analysis_done": {
        "en": "✅ Analysis completed!",
        "fr": "✅ Analyse terminée !",
    },
    "api_connection_success": {
        "en": "✅ API connection successful - Zero RAM mode active!",
        "fr": "✅ Connexion API réussie - mode Zero RAM activé !",
    },
    "api_connection_failed": {
        "en": "⚠️ API not available - Using fallback mode",
        "fr": "⚠️ API non disponible - mode de secours activé",
    },
    "probable_disease": {
        "en": "🦠 Probable Disease",
        "fr": "🦠 Maladie probable",
    },
    "unknown_disease": {
        "en": "Unknown disease – Please consult an expert.",
        "fr": "Maladie inconnue – Veuillez consulter un expert.",
    },
    "visual_confirmation": {
        "en": "### 🔎 Visual confirmation",
        "fr": "### 🔎 Confirmation visuelle",
    },
    "symptoms_heading": {
        "en": "### Symptoms",
        "fr": "### Symptômes",
    },
    "cause_heading": {
        "en": "### Cause",
        "fr": "### Cause",
    },
    "management_heading": {
        "en": "### Management",
        "fr": "### Gestion",
    },
    "no_data": {
        "en": "_No data available._",
        "fr": "_Aucune donnée disponible._",
    },
    "page_title": {
        "en": "🌾 Plant Disease Detection",
        "fr": "🌾 Détection des maladies des plantes",
    },
    "page_subtitle": {
        "en": "Upload an image of a plant leaf to diagnose potential diseases",
        "fr": "Téléchargez une image d'une feuille de plante pour diagnostiquer les maladies potentielles",
    },
    "sidebar_instructions": {
        "en": "### 📋 Instructions\n1. Upload or capture a clear image of a plant leaf\n2. Click 'Analyze' to get diagnosis\n3. Review results and disease information",
        "fr": "### 📋 Instructions\n1. Téléchargez ou capturez une image claire d'une feuille de plante\n2. Cliquez sur 'Analyser' pour obtenir le diagnostic\n3. Examinez les résultats et les informations sur la maladie",
    },
    "image_to_analyze": {
        "en": "### 📷 Image to Analyze",
        "fr": "### 📷 Image à analyser",
    },
    "analyze_button": {
        "en": "🔍 Analyze Image",
        "fr": "🔍 Analyser l'image",
    },
    "not_leaf_message": {
        "en": "⚠️ The uploaded image does not appear to contain a plant leaf. Please upload a clear image of a plant leaf for accurate diagnosis.",
        "fr": "⚠️ L'image téléchargée ne semble pas contenir de feuille de plante. Veuillez télécharger une image claire d'une feuille de plante pour un diagnostic précis.",
    },
}


def get_lang() -> str:
    if "lang" not in st.session_state or not st.session_state.get("lang"):
        st.session_state["lang"] = os.environ.get("DEFAULT_LANG", "en")
    return st.session_state["lang"]


def language_selector(container="sidebar") -> None:
    current = LANGUAGE_LABEL_MAP.get(get_lang(), "English")
    options = [label for label, _ in LANGUAGE_OPTIONS]
    current_index = options.index(current) if current in options else 0

    if container == "sidebar":
        label = st.sidebar.selectbox("🌍 Choisir la langue", options, index=current_index)
        st.sidebar.caption(f"Langue active : **{label}** ({LANGUAGE_CODE_MAP[label]})")
    else:
        label = st.selectbox("🌍 Choisir la langue", options, index=current_index)
        st.caption(f"Langue active : **{label}** ({LANGUAGE_CODE_MAP[label]})")

    st.session_state["lang"] = LANGUAGE_CODE_MAP.get(label, "en")


def t(key: str) -> str:
    lang = get_lang()
    if key not in TRANSLATIONS:
        return key
    return TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get("en", key))
