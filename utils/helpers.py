"""
Fonctions utilitaires pour l'application
"""

import streamlit as st
import uuid
from datetime import datetime

def get_user_id():
    """Récupère ou crée un ID utilisateur unique"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    return st.session_state.user_id

def initialize_session_state():
    """Initialise l'état de la session"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.user_id = get_user_id()
        st.session_state.detections = []
        st.session_state.chat_history = []
        st.session_state.current_detection = None

def format_date(date_string):
    """Formate une date pour l'affichage"""
    try:
        if isinstance(date_string, str):
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            return dt.strftime("%d/%m/%Y %H:%M")
        return str(date_string)
    except:
        return str(date_string)

def get_severity_color(severity):
    """Retourne la couleur selon le niveau de gravité"""
    colors = {
        'low': '🟢',
        'moderate': '🟡',
        'severe': '🟠',
        'critical': '🔴'
    }
    return colors.get(severity.lower(), '⚪')

def get_severity_label(severity):
    """Retourne le label selon le niveau de gravité"""
    labels = {
        'low': 'Faible',
        'moderate': 'Modéré',
        'severe': 'Sévère',
        'critical': 'Critique'
    }
    return labels.get(severity.lower(), severity)










