"""
Utility functions for the application
"""

import streamlit as st
import uuid
from datetime import datetime

def get_user_id():
    """Retrieve or generate a unique user ID"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    return st.session_state.user_id

def initialize_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.user_id = get_user_id()
        st.session_state.detections = []
        st.session_state.chat_history = []
        st.session_state.current_detection = None

def format_date(date_string):
    """Format a date string for display"""
    try:
        if isinstance(date_string, str):
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            return dt.strftime("%d/%m/%Y %H:%M")
        return str(date_string)
    except:
        return str(date_string)

def get_severity_color(severity):
    """Return a color icon based on severity level"""
    colors = {
        'low': '🟢',
        'moderate': '🟡',
        'severe': '🟠',
        'critical': '🔴'
    }
    return colors.get(severity.lower(), '⚪')

def get_severity_label(severity):
    """Return a severity label based on level"""
    labels = {
        'low': 'Low',
        'moderate': 'Moderate',
        'severe': 'Severe',
        'critical': 'Critical'
    }
    return labels.get(severity.lower(), severity)










