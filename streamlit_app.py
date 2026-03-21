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

# Gateway page for Streamlit Cloud entrypoint.
# `st.switch_page` in root can throw StreamlitAPIException in server context.
# We avoid that by displaying a safe home button.

st.write("## Smart Disease Detection")
st.write("Welcome! Click below to open Home.")

if st.button("Go to Home"):
    st.experimental_set_query_params(page="Home")
    st.experimental_rerun()

st.write("---")
st.write("If you are on Streamlit Cloud, set **Main file** to `pages/0_Home.py` for best behavior.")
st.write("The root script `streamlit_app.py` remains a hidden entrypoint for launch logic.")