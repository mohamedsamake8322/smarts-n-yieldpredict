"""
Page de profil utilisateur
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Profil - Agro-Scan",
    page_icon="👤",
    layout="wide"
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css, load_mobile_css
from utils.helpers import get_user_id
from services.database_service import DatabaseService
from utils.service_adapters import SyncDatabaseService

load_custom_css()
load_mobile_css()

st.title("👤 Mon Profil")

# Initialiser le service
@st.cache_resource
def get_database_service():
    async_service = DatabaseService()
    return SyncDatabaseService(async_service)

database_service = get_database_service()

# Informations utilisateur
tab1, tab2, tab3 = st.tabs(["📊 Statistiques", "⚙️ Paramètres", "📱 À propos"])

with tab1:
    st.subheader("📊 Mes Statistiques")
    
    try:
        stats = database_service.get_user_stats(get_user_id())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Détections totales", stats.get('total_detections', 0))
        
        with col2:
            st.metric("Conversations", stats.get('total_chats', 0))
        
        with col3:
            st.metric("Maladies détectées", len(stats.get('top_diseases', {})))
        
        with col4:
            st.metric("Cultures suivies", len(stats.get('top_plants', {})))
        
        # Graphiques
        if stats.get('top_plants'):
            st.markdown("### 🌱 Cultures les plus détectées")
            for plant, count in list(stats['top_plants'].items())[:5]:
                st.progress(count / max(stats['top_plants'].values()) if stats['top_plants'] else 0)
                st.write(f"{plant}: {count} détection(s)")
        
        if stats.get('top_diseases'):
            st.markdown("### 🦠 Maladies les plus fréquentes")
            for disease, count in list(stats['top_diseases'].items())[:5]:
                st.write(f"**{disease}**: {count} occurrence(s)")
    
    except Exception as e:
        st.info("Aucune statistique disponible pour le moment")

with tab2:
    st.subheader("⚙️ Paramètres")
    
    st.markdown("### 🔔 Notifications")
    notif_email = st.checkbox("Recevoir des notifications par email", value=False)
    notif_push = st.checkbox("Recevoir des notifications push", value=True)
    
    st.markdown("### 🌍 Langue")
    language = st.selectbox("Langue de l'interface", ["Français", "English", "Bambara"])
    
    st.markdown("### 📏 Unités")
    unit_system = st.radio("Système d'unités", ["Métrique", "Impérial"], horizontal=True)
    
    if st.button("Enregistrer les paramètres", type="primary"):
        st.success("✅ Paramètres enregistrés !")

with tab3:
    st.subheader("📱 À propos")
    
    st.markdown("""
    **Agro-Scan - Version 1.1.0**
    
    Application intelligente de détection des plantes et maladies agricoles.
    
    **Fonctionnalités:**
    - 🔬 Détection intelligente par IA
    - 💬 Assistant conversationnel
    - 📊 Historique et statistiques
    - 🧮 Calculateurs agricoles
    - 📚 Bibliothèque de ressources
    
    **Développé pour les producteurs agricoles**
    
    Pour toute question ou suggestion, contactez-nous.
    """)
    
    st.markdown("---")
    st.markdown("**ID Utilisateur:** " + get_user_id())
    
    if st.button("Exporter mes données", type="primary"):
        st.info("Fonctionnalité à venir - Export de vos données au format JSON/CSV")

