import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from datetime import datetime, timedelta
import requests
import streamlit.components.v1 as components
import folium


# ✅ Configuration de la page
st.set_page_config(page_title="African Agricultural Network", page_icon="🌍", layout="wide")

st.title("🌍 African Agricultural Social Network")
st.markdown("### A platform for African farmers and experts to share insights and innovations")

# ✅ Sidebar - Profil utilisateur
st.sidebar.title("🌱 My Farming Profile")

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        "name": "Awa Diouf",
        "type": "Farmer",
        "specialty": "Agroecology & Polyculture",
        "region": "Senegal - Casamance",
        "experience": "12 years",
        "followers": 315,
        "following": 210,
        "posts": 78
    }

profile = st.session_state.user_profile

st.sidebar.markdown(f"**{profile['name']}**")
st.sidebar.markdown(f"{profile['type']} - {profile['specialty']}")
st.sidebar.markdown(f"📍 {profile['region']}")
st.sidebar.markdown(f"🎯 {profile['experience']} of experience")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.metric("Posts", profile['posts'])
with col2:
    st.metric("Followers", profile['followers'])
with col3:
    st.metric("Following", profile['following'])

# ✅ Tabs - Sections principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Community Feed",
    "African Farming Groups",
    "Q&A Experts",
    "Events & Workshops",
    "Local Markets"
])

# ✅ 🌍 **Farming Groups & Discussion Forums**
with tab2:
    st.subheader("🌿 Farming Communities in Africa")

    col1, col2 = st.columns([2, 1])

    african_groups = [
        {"name": "Irrigation & Water Management", "members": 2145, "category": "Climate & Sustainability"},
        {"name": "Polyculture & Organic Farming", "members": 3243, "category": "Traditional Techniques"},
        {"name": "Microfinance & Cooperatives", "members": 1983, "category": "Economics"},
        {"name": "Food Security & Post-Harvest Management", "members": 2456, "category": "Agriculture & Trade"}
    ]

    with col1:
        for group in african_groups:
            with st.container():
                col_info, col_stats, col_action = st.columns([2, 1, 1])
                with col_info:
                    st.markdown(f"**{group['name']}** 🌱")
                    st.markdown(f"*{group['category']}*")
                with col_stats:
                    st.metric("Members", group['members'])
                with col_action:
                    if st.button("📖 Join Group", key=f"join_{group['name']}"):
                        st.success(f"You've joined {group['name']}!")
                st.markdown("---")

    with col2:
        st.markdown("**🔍 Discover More Groups**")
        search_term = st.text_input("Search for a group")
        group_categories = st.multiselect(
            "Categories",
            ["Agroecology", "Water Management", "Livestock", "Cooperatives", "Organic Farming", "Market Access"],
            default=["Agroecology"]
        )

# ✅ 📊 **Tracking African Crops**
with tab3:
    st.subheader("📊 Crop Monitoring in Africa")

    st.sidebar.subheader("📈 Key African Crops")
    crop_data = {
        "🌾 Millet": {"region": "Sahel & West Africa", "yield": "1.2 T/ha"},
        "🌿 Cassava": {"region": "Central & West Africa", "yield": "12 T/ha"},
        "🍫 Cocoa": {"region": "Ivory Coast, Ghana", "yield": "0.8 T/ha"},
        "🌰 Groundnuts": {"region": "Senegal, Nigeria", "yield": "2.5 T/ha"}
    }

    for crop, details in crop_data.items():
        st.markdown(f"🔹 **{crop}** - {details['region']} - Yield: {details['yield']}")

# ✅ 🌍 **Carte interactive des cultures africaines**
m = folium.Map(location=[7, 20], zoom_start=4)

cultures = {
    "Mil": [13.5, -2.1],
    "Manioc": [6.5, 3.3],
    "Cacao": [5.3, -4.0],
    "Arachide": [14.7, -16.5]
}

for crop, coord in cultures.items():
    folium.Marker(location=coord, popup=crop, icon=folium.Icon(color="green")).add_to(m)

st.subheader("🌍 Map of Major African Crops")
st_folium(m, width=700)

# ✅ 🔥 **Trending Topics**
st.sidebar.subheader("🔥 Agricultural Trends in Africa")
african_trends = [
    {"tag": "#SahelDrought", "posts": 134},
    {"tag": "#MilPrice2025", "posts": 98},
    {"tag": "#AgroecologyAfrica", "posts": 67},
    {"tag": "#ManiocMarket", "posts": 45},
    {"tag": "#AgritechInnovations", "posts": 34}
]

for trend in african_trends:
    col_tag, col_count = st.sidebar.columns([2, 1])
    with col_tag:
        st.markdown(f"**{trend['tag']}**")
    with col_count:
        st.markdown(f"{trend['posts']} posts")

# ✅ 🛒 **Agricultural Marketplace**
st.subheader("🛒 African Agricultural Marketplace")

market_items = [
    {"name": "Mil Organic Seeds", "price": "5,000 CFA/kg", "location": "Senegal"},
    {"name": "Natural Fertilizer for Cassava", "price": "8,000 CFA/bag", "location": "Côte d'Ivoire"},
    {"name": "Solar Irrigation Pump", "price": "120,000 CFA", "location": "Mali"}
]

for item in market_items:
    st.markdown(f"**{item['name']}** - {item['price']} ({item['location']})")
    if st.button("🛍️ Buy", key=item["name"]):
        st.success(f"Transaction initiated for {item['name']}")
