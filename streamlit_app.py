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

# Attempt a safe redirect to Home page.
try:
    st.switch_page("Home")
except Exception as e:
    st.warning("Unable to auto-switch to Home; fallback to manual navigation.")
    st.write("If you don't see the Home page, please click below:")
    if st.button("Go to Home"):
        st.experimental_set_query_params(page="Home")
        st.experimental_rerun()
    st.write("Workaround: set Streamlit main script to pages/0_Home.py in Streamlit Cloud settings.")