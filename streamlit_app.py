"""
STREAMLIT APP - Main Entry Point
Redirects to Home page for a professional experience
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Smart Disease Detection",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Immediately redirect to Home page
st.switch_page("pages/0_Home.py")