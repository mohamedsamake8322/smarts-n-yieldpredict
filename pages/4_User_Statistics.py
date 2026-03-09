"""
User statistics and analysis page
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit call
st.set_page_config(
    page_title="User Statistics - Agro-Scan",
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

# Title
st.title("📈 User Statistics and Analysis")
st.markdown("View your data and spot trends")

# Initialize service
@st.cache_resource
def get_database_service():
    async_service = DatabaseService()
    return SyncDatabaseService(async_service)

database_service = get_database_service()

# Sidebar
st.sidebar.title("📈 Statistics")
st.sidebar.markdown("### Options")
show_charts = st.sidebar.checkbox("Show charts", value=True)

try:
    # Retrieve statistics
    stats = database_service.get_user_stats(get_user_id())
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Detections",
            stats.get('total_detections', 0)
        )
    
    with col2:
        st.metric(
            "Chats",
            stats.get('total_chats', 0)
        )
    
    with col3:
        top_diseases = stats.get('top_diseases', {})
        st.metric(
            "Diseases Detected",
            len(top_diseases)
        )
    
    with col4:
        top_plants = stats.get('top_plants', {})
        st.metric(
            "Distinct Plants",
            len(top_plants)
        )
    
    if show_charts:
        st.markdown("---")
        
        # Chart of most detected plants
        if top_plants:
            st.subheader("🌱 Most Detected Plants")
            df_plants = pd.DataFrame([
                {'Plant': k, 'Count': v} 
                for k, v in top_plants.items()
            ])
            
            fig = px.bar(
                df_plants,
                x='Plant',
                y='Count',
                title="Detections per Plant",
                color='Count',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Chart of most common diseases
        if top_diseases:
            st.subheader("🦠 Most Common Diseases")
            df_diseases = pd.DataFrame([
                {'Disease': k, 'Occurrences': v} 
                for k, v in top_diseases.items()
            ])
            
            fig = px.pie(
                df_diseases,
                values='Occurrences',
                names='Disease',
                title="Distribution of Detected Diseases"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown("---")
    st.subheader("📋 Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if top_plants:
            st.write("**Detected Plants:**")
            df = pd.DataFrame([
                {'Plant': k, 'Detections': v} 
                for k, v in top_plants.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        if top_diseases:
            st.write("**Detected Diseases:**")
            df = pd.DataFrame([
                {'Disease': k, 'Occurrences': v} 
                for k, v in top_diseases.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Message if no data
    if stats.get('total_detections', 0) == 0:
        st.info("📊 No statistics available yet. Perform detections to see your stats!")
        
except Exception as e:
    st.error(f"❌ Error loading statistics: {str(e)}")
    st.info("No statistics available")

