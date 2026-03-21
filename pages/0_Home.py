"""
Home page with splash screen
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Smart Disease Detection - Home",
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

# CSS pour l'écran de démarrage
splash_css = """
<style>
    .splash-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .logo-circle {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: #20B2AA;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }
    }
    
    .logo-plant {
        font-size: 80px;
        color: white;
        animation: grow 2s ease-in-out infinite;
    }
    
    @keyframes grow {
        0%, 100% {
            transform: scale(1) rotate(0deg);
        }
        50% {
            transform: scale(1.1) rotate(5deg);
        }
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
        text-align: center;
        animation: fadeIn 1s ease-in;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .loading-bar {
        width: 200px;
        height: 4px;
        background: rgba(255,255,255,0.3);
        border-radius: 2px;
        overflow: hidden;
        margin-top: 2rem;
    }
    
    .loading-progress {
        height: 100%;
        background: white;
        border-radius: 2px;
        animation: loading 2s ease-in-out;
    }
    
    @keyframes loading {
        from {
            width: 0%;
        }
        to {
            width: 100%;
        }
    }
    
    .version-badge {
        position: absolute;
        bottom: 20px;
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
    }
    
    /* Masquer les éléments Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(
    """
    <style>
    /* Hide root page nav entry when on Home page */
    [data-testid="stSidebarNav"] > div:first-child { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hide root page nav entry for clean flow
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] > div:first-child { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(splash_css, unsafe_allow_html=True)

# Contenu de l'écran de démarrage
st.markdown("""
<div class="splash-container">
    <div class="logo-circle">
        <div class="logo-plant">🌱</div>
    </div>
    <h1 class="app-title">AI-Powered Crop Disease Detection</h1>
    <p class="app-subtitle">Detect plant diseases instantly using advanced AI and image analysis.</p>
    <div class="loading-bar">
        <div class="loading-progress"></div>
    </div>
    <div class="version-badge">Version 1.1.0</div>
</div>
""", unsafe_allow_html=True)

# Call to action for user to go to Detection
st.markdown("""
<div style='text-align:center; margin-top: 25px;'>
    <a href='#' id='start_diagnosis' style='display:inline-flex;
       align-items:center; justify-content:center;
       background-color:#28a745; color:white; padding:14px 28px;
       border-radius:8px; font-size:18px; text-decoration:none;'>
       ▶️ Start Diagnosis
    </a>
</div>
""", unsafe_allow_html=True)

# JavaScript redirection (works in browser click)
st.markdown(
    """
    <script>
    const btn = document.getElementById('start_diagnosis');
    if (btn) {
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            window.location.href = '/?page=Detection';
        });
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# For Streamlit, also render a native button
if st.button("Start Diagnosis"):
    st.experimental_set_query_params(page="Detection")
    st.experimental_rerun()

# Auto-redirect after 3 seconds via browser (safe in Streamlit Cloud)
st.markdown(
    """
    <script>
    setTimeout(function() {
        window.location.href = '/?page=Detection';
    }, 3000);
    </script>
    """,
    unsafe_allow_html=True,
)

st.info("You will be redirected to Detection in 3 seconds; or click Start Diagnosis.")

