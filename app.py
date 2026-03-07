"""
Application principale Sènè Disease Detection avec Streamlit
Interface inspirée de Plantix mais avec des fonctionnalités uniques
"""

# WORKAROUND: Empêcher Streamlit d'inspecter torch.classes (problème de compatibilité)
import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

# Appliquer le workaround PyTorch/Streamlit
try:
    from utils.pytorch_fix import apply_pytorch_fix
    apply_pytorch_fix()
except ImportError:
    # Si le module n'existe pas encore, appliquer le fix directement
    try:
        import torch
        if hasattr(torch, 'classes') and not hasattr(torch.classes, '__path__'):
            class MockPath:
                def __iter__(self):
                    return iter([])
                def __contains__(self, item):
                    return False
            torch.classes.__path__ = MockPath()
    except (ImportError, AttributeError):
        pass

import streamlit as st
from datetime import datetime
import json

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Sènè Disease Detection - Détection Intelligente",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Sènè Disease Detection\nApplication intelligente de détection des maladies des cultures"
    }
)

# Import des utilitaires
try:
    from utils.styles import load_custom_css, load_mobile_css
    from utils.helpers import get_user_id, initialize_session_state, format_date
    from utils.calculators import FertilizerCalculator, PesticideCalculator, FarmingCalculator
except ImportError:
    st.warning("⚠️ Certains utilitaires sont manquants.")
    def load_custom_css(): pass
    def load_mobile_css(): pass
    def get_user_id(): return "user_123"
    def initialize_session_state(): pass
    def format_date(d): return str(d)

# Charger les styles
load_custom_css()
load_mobile_css()

# Initialiser l'état de la session
initialize_session_state()

# Import des services
from services.detection_service import DetectionService
from services.chatbot_service import ChatbotService
from services.database_service import DatabaseService
from utils.service_adapters import SyncDetectionService, SyncChatbotService, SyncDatabaseService

# Initialiser les services
@st.cache_resource
def get_services():
    """Initialise et cache les services"""
    return {
        'detection': SyncDetectionService(DetectionService()),
        'chatbot': SyncChatbotService(ChatbotService()),
        'database': SyncDatabaseService(DatabaseService())
    }

services = get_services()

# ================= HEADER =================
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("### 🌱 Sènè Disease Detection")

with header_col2:
    if st.button("⚙️", help="Paramètres"):
        st.switch_page("pages/5_Paramètres.py")

# ================= SÉLECTION DE CULTURES =================
st.markdown("### 🌾 Mes Cultures")

# Cultures disponibles
crops_data = [
    {"name": "Tomate", "icon": "🍅", "color": "#e53935"},
    {"name": "Maïs", "icon": "🌽", "color": "#fbc02d"},
    {"name": "Riz", "icon": "🌾", "color": "#388e3c"},
    {"name": "Manioc", "icon": "🥔", "color": "#f57c00"},
    {"name": "Banane", "icon": "🍌", "color": "#fdd835"},
    {"name": "Cacao", "icon": "🍫", "color": "#5d4037"},
    {"name": "Café", "icon": "☕", "color": "#6d4c41"},
    {"name": "Arachide", "icon": "🥜", "color": "#ffa726"},
]

# Récupérer les cultures sélectionnées
if 'selected_crops' not in st.session_state:
    st.session_state.selected_crops = ["Tomate", "Maïs", "Riz"]

# Afficher les cultures en scroll horizontal
crop_cols = st.columns(len(crops_data) + 1)

for i, crop in enumerate(crops_data):
    with crop_cols[i]:
        is_selected = crop["name"] in st.session_state.selected_crops
        button_style = f"""
        <style>
        .crop-btn-{i} {{
            background: {'linear-gradient(135deg, ' + crop['color'] + ' 0%, #ffffff 100%)' if is_selected else '#f5f5f5'};
            border: 2px solid {crop['color'] if is_selected else '#ddd'};
            border-radius: 50%;
            width: 80px;
            height: 80px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s;
        }}
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        if st.button(f"{crop['icon']}\n{crop['name']}", key=f"crop_{i}", use_container_width=True):
            if crop["name"] in st.session_state.selected_crops:
                st.session_state.selected_crops.remove(crop["name"])
            else:
                st.session_state.selected_crops.append(crop["name"])
            st.rerun()

# Bouton pour ajouter une culture
with crop_cols[-1]:
    if st.button("➕\nAjouter", use_container_width=True, help="Ajouter une nouvelle culture"):
        st.info("Fonctionnalité à venir")

# ================= CARTES D'INFORMATION =================
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("#### 📅 Météo")
    weather_card = st.container()
    with weather_card:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white;'>
            <div style='font-size: 1.2rem; margin-bottom: 10px;'>{datetime.now().strftime("%d %b")}</div>
            <div style='font-size: 2rem; font-weight: bold;'>25°C</div>
            <div style='font-size: 3rem;'>☀️</div>
        </div>
        """, unsafe_allow_html=True)

with info_col2:
    st.markdown("#### 💧 Conditions de Pulvérisation")
    spray_card = st.container()
    with spray_card:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 15px; color: white;'>
            <div style='font-size: 0.9rem; margin-bottom: 10px;'>Conditions de pulvérisation</div>
            <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 5px;'>Modérées</div>
            <div style='font-size: 0.9rem;'>Jusqu'à 11h</div>
            <div style='font-size: 1.5rem; margin-top: 10px;'>⚠️</div>
        </div>
        """, unsafe_allow_html=True)

# ================= WORKFLOW DE DÉTECTION =================
st.markdown("### 🔍 Détection Intelligente")

workflow_card = st.container()
with workflow_card:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 30px; border-radius: 20px; margin: 20px 0;'>
    """, unsafe_allow_html=True)
    
    # Workflow en 3 étapes
    step_col1, arrow1, step_col2, arrow2, step_col3 = st.columns([2, 0.5, 2, 0.5, 2])
    
    with step_col1:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 10px;'>📸</div>
            <div style='font-weight: bold; font-size: 1.1rem;'>Prendre une photo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow1:
        st.markdown("<div style='font-size: 2rem; text-align: center;'>➡️</div>", unsafe_allow_html=True)
    
    with step_col2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 10px;'>🔬</div>
            <div style='font-weight: bold; font-size: 1.1rem;'>Voir le diagnostic</div>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow2:
        st.markdown("<div style='font-size: 2rem; text-align: center;'>➡️</div>", unsafe_allow_html=True)
    
    with step_col3:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 10px;'>💊</div>
            <div style='font-weight: bold; font-size: 1.1rem;'>Obtenir le traitement</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Bouton principal
    if st.button("📸 Prendre une photo", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Détection.py")

# ================= ACTIVITÉ RÉCENTE =================
st.markdown("### 📋 Activité Récente")

try:
    recent_detections = services['database'].get_user_detections(get_user_id(), 3)
    
    if recent_detections:
        for det in recent_detections[:2]:
            det_col1, det_col2 = st.columns([3, 1])
            
            with det_col1:
                st.markdown(f"""
                <div style='background: white; padding: 15px; border-radius: 10px; 
                            border-left: 4px solid #4caf50; margin-bottom: 10px;'>
                    <div style='font-size: 0.9rem; color: #666;'>{format_date(det.get('created_at', ''))}</div>
                    <div style='font-size: 1.2rem; font-weight: bold; margin-top: 5px;'>
                        {det.get('plant_name', 'Plante inconnue')}
                    </div>
                    <div style='font-size: 0.9rem; color: #666; margin-top: 5px;'>
                        {', '.join([d.get('name', '') for d in det.get('diseases', [])[:2]]) if det.get('diseases') else 'Aucune maladie détectée'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with det_col2:
                st.markdown("""
                <div style='display: flex; align-items: center; height: 100%;'>
                    <div style='background: #4caf50; color: white; padding: 8px 15px; 
                                border-radius: 20px; font-size: 0.8rem; font-weight: bold;'>
                        Terminé
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("👁️", key=f"view_{det.get('id', '')}", help="Voir les détails"):
                    st.switch_page("pages/3_Historique.py")
    else:
        st.info("Aucune activité récente. Effectuez votre première détection !")
        
except Exception as e:
    st.info("Aucune activité récente disponible.")

# ================= CHAT DIRECT AVEC LE MODÈLE =================
st.markdown("### 💬 Discuter avec l'Assistant IA")

# Initialiser l'historique de chat dans la session
if 'home_chat_messages' not in st.session_state:
    st.session_state.home_chat_messages = []

# Afficher l'historique de chat
chat_container = st.container()
with chat_container:
    if st.session_state.home_chat_messages:
        for msg in st.session_state.home_chat_messages[-5:]:  # Afficher les 5 derniers messages
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(msg['content'])
    else:
        st.info("👋 Bonjour ! Je suis votre assistant agricole. Posez-moi vos questions !")

# Zone de saisie pour le chat
user_question = st.chat_input("Posez votre question sur l'agriculture...")

if user_question:
    # Ajouter le message de l'utilisateur
    st.session_state.home_chat_messages.append({
        'role': 'user',
        'content': user_question,
        'timestamp': datetime.now().isoformat()
    })
    
    # Générer la réponse
    with st.spinner("🤔 Réflexion en cours..."):
        try:
            # Récupérer l'historique utilisateur
            user_history = None
            try:
                user_history = services['database'].get_user_chat_history(get_user_id(), 10)
            except:
                pass
            
            # Générer la réponse
            response = services['chatbot'].generate_response(
                message=user_question,
                context=None,
                user_history=user_history
            )
            
            # Ajouter la réponse à l'historique
            st.session_state.home_chat_messages.append({
                'role': 'assistant',
                'content': response.response,
                'timestamp': response.timestamp
            })
            
            # Sauvegarder dans la base de données
            try:
                services['database'].save_chat_message(
                    user_id=get_user_id(),
                    message=user_question,
                    response=response.response,
                    context=None
                )
            except Exception as e:
                pass  # Ignorer les erreurs de sauvegarde
            
        except Exception as e:
            st.session_state.home_chat_messages.append({
                'role': 'assistant',
                'content': "Je rencontre une difficulté technique. Pouvez-vous reformuler votre question ?",
                'timestamp': datetime.now().isoformat()
            })
    
    st.rerun()

# Bouton pour effacer le chat
if st.session_state.home_chat_messages:
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.home_chat_messages = []
        st.rerun()

# Bouton pour ouvrir la page complète du chat
if st.button("💬 Ouvrir le chat complet", use_container_width=True):
    st.switch_page("pages/2_Assistant.py")

st.markdown("---")

# ================= SECTION TOOLS =================
st.markdown("### 🛠️ Outils")

tools_col1, tools_col2, tools_col3 = st.columns(3)

with tools_col1:
    if st.button("🧮\n\nCalculateur d'Engrais", use_container_width=True, help="Calculer les besoins en engrais"):
        st.switch_page("pages/6_Calculateurs.py#fertilizer")

with tools_col2:
    st.markdown("""
    <div style='position: relative;'>
        <span style='background: #9c27b0; color: white; padding: 2px 8px; 
                     border-radius: 10px; font-size: 0.7rem; position: absolute; 
                     top: -5px; right: -5px; z-index: 1;'>Nouveau</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("💉\n\nCalculateur de Pesticides", use_container_width=True, help="Calculer les doses de pesticides"):
        st.switch_page("pages/6_Calculateurs.py#pesticide")

with tools_col3:
    st.markdown("""
    <div style='position: relative;'>
        <span style='background: #9c27b0; color: white; padding: 2px 8px; 
                     border-radius: 10px; font-size: 0.7rem; position: absolute; 
                     top: -5px; right: -5px; z-index: 1;'>Nouveau</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📊\n\nCalculateur Agricole", use_container_width=True, help="Calculs agricoles avancés"):
        st.switch_page("pages/6_Calculateurs.py#farming")

# ================= SECTION LIBRARY =================
st.markdown("### 📚 Bibliothèque")

library_col1, library_col2 = st.columns([2, 1])

with library_col1:
    if st.button("🌿 Conseils de Culture", use_container_width=True, help="Conseils et guides de culture"):
        st.switch_page("pages/7_Bibliothèque.py#tips")

library_row2_col1, library_row2_col2 = st.columns(2)

with library_row2_col1:
    if st.button("🦠 Maladies & Ravageurs", use_container_width=True, help="Base de données des maladies"):
        st.switch_page("pages/7_Bibliothèque.py#diseases")

with library_row2_col2:
    if st.button("⚠️ Alertes Maladies", use_container_width=True, help="Alertes et notifications"):
        st.switch_page("pages/7_Bibliothèque.py#alerts")

# ================= NAVIGATION INFÉRIEURE =================
st.markdown("---")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🌾 Mes Cultures", use_container_width=True, type="primary"):
        st.rerun()

with nav_col2:
    if st.button("💬 Communauté", use_container_width=True):
        st.switch_page("pages/8_Communauté.py")

with nav_col3:
    if st.button("👤 Profil", use_container_width=True):
        st.switch_page("pages/9_Profil.py")

# Cacher les éléments Streamlit par défaut
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
