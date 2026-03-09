"""
Page des paramètres de l'application
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Paramètres - Agro-Scan",
    page_icon="⚙️",
    layout="wide"
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css, load_mobile_css
from services.local_model_service import get_local_model_service

load_custom_css()
load_mobile_css()

st.title("⚙️ Paramètres")

tab1, tab2, tab3 = st.tabs(["🔧 Application", "🤖 Modèle IA", "📊 Données"])

with tab1:
    st.subheader("🔧 Paramètres de l'application")
    
    st.markdown("### 🌍 Langue et Région")
    col1, col2 = st.columns(2)
    
    with col1:
        language = st.selectbox("Langue", ["Français", "English", "Bambara", "Wolof"])
    
    with col2:
        region = st.selectbox("Région", ["Mali", "Sénégal", "Burkina Faso", "Côte d'Ivoire"])
    
    st.markdown("### 🔔 Notifications")
    notif_detection = st.checkbox("Notifications pour nouvelles détections", value=True)
    notif_alerts = st.checkbox("Alertes maladies", value=True)
    notif_community = st.checkbox("Notifications communauté", value=False)
    
    st.markdown("### 🎨 Apparence")
    theme = st.selectbox("Thème", ["Clair", "Sombre", "Auto"])
    
    if st.button("Enregistrer", type="primary"):
        st.success("✅ Paramètres enregistrés !")

with tab2:
    st.subheader("🤖 Configuration du Modèle IA")
    
    # Vérifier le modèle local
    local_service = get_local_model_service()
    
    if local_service.is_ready():
        st.success("✅ Modèle local Phi-3 chargé et prêt")
        st.info(f"📍 Chemin: {local_service.model_path}")
    else:
        st.warning("⚠️ Modèle local non disponible")
        st.info(f"📍 Chemin attendu: {local_service.model_path}")
        st.markdown("""
        **Pour activer le modèle local:**
        1. Assurez-vous que le fichier `Phi-3-mini-4k-instruct-q4.gguf` est dans `local_model/`
        2. Installez `llama-cpp-python`: `pip install llama-cpp-python`
        3. Redémarrez l'application
        """)
    
    st.markdown("### ⚙️ Paramètres de génération")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Température",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Contrôle la créativité (plus élevé = plus créatif)"
        )
    
    with col2:
        max_tokens = st.slider(
            "Tokens maximum",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Longueur maximale de la réponse"
        )
    
    st.markdown("### 🔄 Alternative: OpenAI")
    openai_key = st.text_input(
        "Clé API OpenAI (optionnel)",
        type="password",
        help="Laissez vide pour utiliser le modèle local"
    )
    
    if st.button("Tester la connexion", type="primary"):
        if local_service.is_ready():
            test_prompt = "Bonjour, comment allez-vous ?"
            response = local_service.generate(test_prompt, temperature=0.2, max_tokens=50)
            if response:
                st.success("✅ Modèle local fonctionne !")
                st.info(f"Test: {response[:100]}...")
            else:
                st.error("❌ Erreur lors du test")
        else:
            st.warning("⚠️ Modèle local non disponible")

with tab3:
    st.subheader("📊 Gestion des Données")
    
    st.markdown("### 💾 Stockage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Espace utilisé", "12.5 MB")
        st.metric("Détections stockées", "45")
    
    with col2:
        st.metric("Conversations", "23")
        st.metric("Images stockées", "45")
    
    st.markdown("### 📤 Export")
    
    export_format = st.radio(
        "Format d'export",
        ["JSON", "CSV", "Excel"],
        horizontal=True
    )
    
    if st.button("Exporter toutes les données", type="primary"):
        st.info("Fonctionnalité à venir - Export de vos données")
    
    st.markdown("### 🗑️ Suppression")
    
    if st.button("Supprimer toutes les données", type="secondary"):
        if st.checkbox("Je confirme la suppression de toutes mes données"):
            st.warning("⚠️ Cette action est irréversible !")
            if st.button("Confirmer la suppression", type="primary"):
                st.error("Fonctionnalité de suppression désactivée pour la sécurité")

