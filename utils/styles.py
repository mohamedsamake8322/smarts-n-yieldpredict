"""
Styles CSS personnalisés pour Streamlit
Responsive design pour mobile, tablette et ordinateur
"""

import streamlit as st

def load_mobile_css():
    """Charge le CSS mobile-first pour l'application"""
    
    mobile_css = """
    <style>
    /* Design mobile-first inspiré de Plantix */
    .main {
        padding: 0.5rem;
    }
    
    /* Header */
    .stApp > header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
    }
    
    /* Cards avec ombres et bordures arrondies */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Boutons circulaires pour cultures */
    .crop-button {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 2px solid;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .crop-button:hover {
        transform: scale(1.1);
    }
    
    /* Workflow steps */
    .workflow-step {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.5);
        border-radius: 10px;
        margin: 0.5rem;
    }
    
    /* Badge "Nouveau" */
    .new-badge {
        background: #9c27b0;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
        position: absolute;
        top: -5px;
        right: -5px;
    }
    
    /* Navigation inférieure */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e0e0e0;
        padding: 0.5rem;
        display: flex;
        justify-content: space-around;
        z-index: 1000;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main {
            padding: 0.25rem;
        }
        
        h1, h2, h3 {
            font-size: 1.2rem !important;
        }
        
        .stButton > button {
            font-size: 0.9rem;
            padding: 0.5rem;
        }
    }
    
    /* Masquer les éléments Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    
    st.markdown(mobile_css, unsafe_allow_html=True)

def load_custom_css():
    """Charge le CSS personnalisé pour l'application"""
    
    custom_css = """
    <style>
    /* Styles généraux */
    .main {
        padding: 1rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        
        h2 {
            font-size: 1.2rem !important;
        }
        
        .stButton>button {
            width: 100%;
            font-size: 0.9rem;
            padding: 0.5rem;
        }
        
        .stMetric {
            padding: 0.5rem;
        }
    }
    
    /* Styles pour les cartes */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Styles pour les boutons */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Styles pour les métriques */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Amélioration de la sidebar */
    .css-1d391kg {
        padding-top: 3rem;
    }
    
    /* Styles pour les images */
    img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
    }
    
    /* Styles pour les alertes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Animation pour les chargements */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #2e7d32;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    /* Styles pour les onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    /* Amélioration de la lisibilité sur mobile */
    @media (max-width: 480px) {
        .stMarkdown {
            font-size: 0.9rem;
        }
        
        .stTextInput>div>div>input {
            font-size: 16px; /* Évite le zoom sur iOS */
        }
    }
    
    /* Styles pour les tableaux */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer */
    footer {
        visibility: hidden;
    }
    
    /* Header */
    header {
        visibility: hidden;
    }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def load_mobile_css():
    """Charge le CSS spécifique pour mobile"""
    
    mobile_css = """
    <style>
    /* Optimisations mobile */
    @media (max-width: 768px) {
        /* Réduction des espacements */
        .main .block-container {
            padding: 0.5rem;
        }
        
        /* Boutons plus grands pour le tactile */
        .stButton>button {
            min-height: 44px;
            font-size: 1rem;
        }
        
        /* Cartes compactes */
        .card {
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
        
        /* Texte lisible */
        p, li {
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Colonnes empilées sur mobile */
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    /* Amélioration du scroll sur mobile */
    .main {
        -webkit-overflow-scrolling: touch;
    }
    
    /* Touch targets optimisés */
    button, a, [role="button"] {
        min-height: 44px;
        min-width: 44px;
    }
    </style>
    """
    
    st.markdown(mobile_css, unsafe_allow_html=True)



