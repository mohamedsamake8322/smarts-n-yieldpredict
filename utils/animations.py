"""
Utilitaires d'animation pour Streamlit
"""

import streamlit as st
import time
import streamlit.components.v1 as components

def typewriting_effect(placeholder, text, speed=0.03):
    """
    Effet de machine à écrire pour le texte
    """
    try:
        displayed_text = ""
        for char in text:
            displayed_text += char
            placeholder.markdown(displayed_text)
            time.sleep(speed)
    except:
        # En cas d'erreur, afficher directement
        placeholder.markdown(text)

def pulsing_title(components_module):
    """
    Effet de pulsation pour le titre (optionnel)
    """
    try:
        pulsing_css = """
        <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        h1 {
            animation: pulse 3s ease-in-out infinite;
        }
        </style>
        """
        components_module.html(pulsing_css, height=0)
    except:
        pass










