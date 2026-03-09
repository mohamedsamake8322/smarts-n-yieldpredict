"""
Page de l'historique des détections et conversations
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Historique - Agro-Scan",
    page_icon="📊",
    layout="wide"
)

from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_user_id, format_date, get_severity_color, get_severity_label
from utils.styles import load_custom_css
from services.database_service import DatabaseService
from utils.service_adapters import SyncDatabaseService

load_custom_css()

# Titre
st.title("📊 Historique")
st.markdown("Consultez vos détections et conversations précédentes")

# Initialiser le service
@st.cache_resource
def get_database_service():
    async_service = DatabaseService()
    return SyncDatabaseService(async_service)

database_service = get_database_service()

# Sidebar
st.sidebar.title("📊 Historique")
st.sidebar.markdown("### Filtres")
limit = st.sidebar.slider("Nombre d'éléments", 5, 50, 20)

# Onglets
tab1, tab2 = st.tabs(["📸 Détections", "💬 Conversations"])

with tab1:
    st.subheader("📸 Détections précédentes")
    
    try:
        detections = database_service.get_user_detections(get_user_id(), limit)
        
        if not detections:
            st.info("Aucune détection enregistrée. Effectuez votre première détection !")
        else:
            st.success(f"✅ {len(detections)} détection(s) trouvée(s)")
            
            # Affichage des détections
            for i, detection in enumerate(detections):
                with st.expander(
                    f"{get_severity_color(detection.get('severity', 'low'))} "
                    f"{detection.get('plant_name', 'Plante inconnue')} - "
                    f"{format_date(detection.get('created_at', ''))}",
                    expanded=(i == 0)
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Plante:** {detection.get('plant_name', 'N/A')}")
                        st.write(f"**Nom scientifique:** {detection.get('plant_scientific_name', 'N/A')}")
                        st.write(f"**Gravité:** {get_severity_label(detection.get('severity', 'low'))}")
                        st.write(f"**Confiance:** {(detection.get('confidence', 0) * 100):.1f}%")
                        st.write(f"**Date:** {format_date(detection.get('created_at', ''))}")
                        
                        # Maladies
                        if detection.get('diseases'):
                            st.write("**Maladies détectées:**")
                            for disease in detection['diseases']:
                                if isinstance(disease, dict):
                                    st.write(f"- {disease.get('name', 'N/A')} ({disease.get('severity', 'N/A')})")
                    
                    with col2:
                        # Image si disponible
                        if detection.get('image_path'):
                            try:
                                st.image(detection['image_path'], use_container_width=True)
                            except:
                                st.info("Image non disponible")
                        
                        # Bouton de suppression
                        if st.button("🗑️ Supprimer", key=f"del_{detection.get('id', i)}"):
                            try:
                                success = database_service.delete_detection(
                                    detection.get('id'),
                                    get_user_id()
                                )
                                if success:
                                    st.success("✅ Détection supprimée")
                                    st.rerun()
                                else:
                                    st.error("❌ Erreur lors de la suppression")
                            except Exception as e:
                                st.error(f"❌ Erreur: {str(e)}")
                    
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des détections: {str(e)}")
        st.info("Aucune détection disponible")

with tab2:
    st.subheader("💬 Conversations précédentes")
    
    try:
        chats = database_service.get_user_chat_history(get_user_id(), limit)
        
        if not chats:
            st.info("Aucune conversation enregistrée. Commencez à discuter avec l'assistant !")
        else:
            st.success(f"✅ {len(chats)} conversation(s) trouvée(s)")
            
            # Affichage des conversations
            for i, chat in enumerate(chats):
                with st.expander(
                    f"💬 {chat.get('message', 'Message')[:50]}... - {format_date(chat.get('created_at', ''))}",
                    expanded=(i == 0)
                ):
                    st.write(f"**Vous:** {chat.get('message', 'N/A')}")
                    st.write(f"**Assistant:** {chat.get('response', 'N/A')}")
                    st.write(f"**Date:** {format_date(chat.get('created_at', ''))}")
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des conversations: {str(e)}")
        st.info("Aucune conversation disponible")

