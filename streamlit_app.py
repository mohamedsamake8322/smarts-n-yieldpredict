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

# Hide the root page entry from sidebar to keep only the main menu pages
st.markdown(
    """
    <style>
    /* Hide first nav item (root app script) in Streamlit multipage sidebar */
    [data-testid="stSidebarNav"] > div:first-child { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Immediately redirect to Home page (pages/0_Home.py) for UX
st.switch_page("Home")