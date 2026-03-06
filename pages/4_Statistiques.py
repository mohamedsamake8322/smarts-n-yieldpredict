"""
Page des statistiques et analyses
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Statistiques - Agro-Scan",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_user_id
from utils.styles import load_custom_css
from services.database_service import DatabaseService
from utils.service_adapters import SyncDatabaseService

load_custom_css()

# Titre
st.title("📈 Statistiques et Analyses")
st.markdown("Visualisez vos données et identifiez les tendances")

# Initialiser le service
@st.cache_resource
def get_database_service():
    async_service = DatabaseService()
    return SyncDatabaseService(async_service)

database_service = get_database_service()

# Sidebar
st.sidebar.title("📈 Statistiques")
st.sidebar.markdown("### Options")
show_charts = st.sidebar.checkbox("Afficher les graphiques", value=True)

try:
    # Récupérer les statistiques
    stats = database_service.get_user_stats(get_user_id())
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Détections totales",
            stats.get('total_detections', 0)
        )
    
    with col2:
        st.metric(
            "Conversations",
            stats.get('total_chats', 0)
        )
    
    with col3:
        top_diseases = stats.get('top_diseases', {})
        st.metric(
            "Maladies détectées",
            len(top_diseases)
        )
    
    with col4:
        top_plants = stats.get('top_plants', {})
        st.metric(
            "Plantes différentes",
            len(top_plants)
        )
    
    if show_charts:
        st.markdown("---")
        
        # Graphique des plantes les plus détectées
        if top_plants:
            st.subheader("🌱 Plantes les plus détectées")
            df_plants = pd.DataFrame([
                {'Plante': k, 'Nombre': v} 
                for k, v in top_plants.items()
            ])
            
            fig = px.bar(
                df_plants,
                x='Plante',
                y='Nombre',
                title="Nombre de détections par plante",
                color='Nombre',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Graphique des maladies les plus fréquentes
        if top_diseases:
            st.subheader("🦠 Maladies les plus fréquentes")
            df_diseases = pd.DataFrame([
                {'Maladie': k, 'Occurrences': v} 
                for k, v in top_diseases.items()
            ])
            
            fig = px.pie(
                df_diseases,
                values='Occurrences',
                names='Maladie',
                title="Répartition des maladies détectées"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.markdown("---")
    st.subheader("📋 Récapitulatif")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if top_plants:
            st.write("**Plantes détectées:**")
            df = pd.DataFrame([
                {'Plante': k, 'Détections': v} 
                for k, v in top_plants.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        if top_diseases:
            st.write("**Maladies détectées:**")
            df = pd.DataFrame([
                {'Maladie': k, 'Occurrences': v} 
                for k, v in top_diseases.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Message si pas de données
    if stats.get('total_detections', 0) == 0:
        st.info("📊 Aucune statistique disponible. Effectuez des détections pour voir vos statistiques !")
        
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des statistiques: {str(e)}")
    st.info("Aucune statistique disponible")

