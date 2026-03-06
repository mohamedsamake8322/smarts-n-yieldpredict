"""
Page de démarrage / Splash Screen
Affiche le logo et charge l'application
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Agro-Scan",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css

load_custom_css()

# CSS pour le splash screen
splash_css = """
<style>
    .splash-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 9999;
    }
    
    .logo-circle {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: pulse 2s ease-in-out infinite;
        margin-bottom: 30px;
    }
    
    .logo-plant {
        font-size: 80px;
        color: white;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }
    
    .app-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        max-width: 300px;
    }
    
    .loading-spinner {
        border: 3px solid rgba(255,255,255,0.3);
        border-top: 3px solid white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-top: 30px;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @media (max-width: 768px) {
        .logo-circle {
            width: 120px;
            height: 120px;
        }
        
        .logo-plant {
            font-size: 60px;
        }
        
        .app-title {
            font-size: 2rem;
        }
    }
</style>
"""

st.markdown(splash_css, unsafe_allow_html=True)

# Contenu du splash screen
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="splash-container">
        <div class="logo-circle">
            <div class="logo-plant">🌱</div>
        </div>
        <div class="app-title">Agro-Scan</div>
        <div class="app-subtitle">Détection intelligente des plantes et maladies</div>
        <div class="loading-spinner"></div>
    </div>
    """, unsafe_allow_html=True)

# Attendre un peu puis rediriger
if 'splash_shown' not in st.session_state:
    time.sleep(2)
    st.session_state.splash_shown = True
    st.switch_page("app.py")

