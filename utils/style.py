PALETTE = {
    "background": "#F4F9F9",
    "primary": "#4CAF50",
    "secondary": "#FFC107"
}

def apply_style():
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {PALETTE["background"]};
        }}
        .stButton button {{
            background-color: {PALETTE["primary"]};
            color: white;
        }}
        </style>
        """, unsafe_allow_html=True
    )
