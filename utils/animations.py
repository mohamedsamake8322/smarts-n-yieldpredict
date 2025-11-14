import time
import streamlit as st

def typewriting_effect(placeholder, text, delay=0.04):
    """ Affiche le texte progressivement pour un effet machine à écrire """
    for i in range(len(text)):
        placeholder.markdown(text[:i+1])
        time.sleep(delay)
def pulsing_title(components):
    """ Ajoute un effet de pulsation au titre """
    css_code = """
    <style>
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.03); }
      100% { transform: scale(1); }
    }
    h1 {
      animation: pulse 2s infinite;
    }
    </style>
    """
    components.html(css_code, height=0)
