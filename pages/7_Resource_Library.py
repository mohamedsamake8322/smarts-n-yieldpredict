"""
Agricultural resources library page
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Resource Library - Agro-Scan",
    page_icon="📚",
    layout="wide"
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css, load_mobile_css

load_custom_css()
load_mobile_css()

st.title("📚 Agricultural Resource Library")

# Tabs
tab1, tab2, tab3 = st.tabs(["🌿 Cultivation Tips", "🦠 Diseases & Pests", "⚠️ Alerts"])

with tab1:
    st.subheader("🌿 Cultivation Tips")
    
    crop_selected = st.selectbox(
        "Select a crop",
        ["Tomato", "Corn", "Rice", "Cassava", "Banana", "Cocoa", "Coffee", "Peanut"]
    )
    
    if crop_selected == "Tomato":
        st.markdown("""
        ### 🍅 Tomato Cultivation Guide
        
        **Planting:**
        - Season: February-March or September-October
        - Spacing: 50-60 cm between plants, 80-100 cm between rows
        - Depth: 5-7 cm
        
        **Soil:**
        - Well-drained soil rich in organic matter
        - Optimal pH: 6.0-6.8
        - Avoid overly acidic soils
        
        **Watering:**
        - Frequency: 2-3 times per week
        - Amount: 2-3 L per plant
        - Avoid wetting leaves
        
        **Fertilization:**
        - NPK 15-15-15 fertilizer: 200-300 kg/ha
        - Compost: 10-15 tonnes/ha
        - Basal dressing before planting
        
        **Harvesting:**
        - 60-90 days after planting
        - Harvest early morning
        - Store at room temperature
        """)
    
    elif crop_selected == "Corn":
        st.markdown("""
        ### 🌽 Corn Cultivation Guide
        
        **Planting:**
        - Season: April-May (rainy season)
        - Spacing: 25-30 cm between plants, 75-80 cm between rows
        - Density: 40,000-50,000 plants/ha
        
        **Soil:**
        - Deep, well-drained soil
        - Optimal pH: 5.5-7.0
        - Avoid compacted soils
        
        **Watering:**
        - Needs: 500-800 mm during cycle
        - Critical during flowering
        - Irrigate if rainfall is insufficient
        
        **Fertilization:**
        - Nitrogen: 120-150 kg/ha
        - Phosphorus: 40-60 kg/ha
        - Potassium: 80-100 kg/ha
        
        **Harvesting:**
        - 90-120 days after planting
        - Harvest at physiological maturity
        - Dry quickly after harvest
        """)
    
    # Ajouter d'autres cultures...

with tab2:
    st.subheader("🦠 Diseases & Pests Database")
    
    search_term = st.text_input("🔍 Search for a disease or pest")
    
    diseases_db = {
        "Late Blight": {
            "description": "Fungal disease caused by Phytophthora infestans",
            "symptoms": "Brown spots on leaves, white downy growth on underside",
            "affected_crops": ["Tomato", "Potato"],
            "treatment": "Bordeaux mixture, systemic fungicides",
            "prevention": "Avoid excessive moisture, space plants apart"
        },
        "Powdery Mildew": {
            "description": "Fungal disease caused by various fungi",
            "symptoms": "White powder on leaves and stems",
            "affected_crops": ["Cucurbits", "Grapevine", "Roses"],
            "treatment": "Wettable sulfur, anti-mildew fungicides",
            "prevention": "Good air circulation, preventive treatment"
        },
        "Rust": {
            "description": "Fungal disease caused by Puccinia spp.",
            "symptoms": "Pustules orange/brunes sur les feuilles",
            "affected_crops": ["Céréales", "Haricot"],
            "treatment": "Fongicides systémiques",
            "prevention": "Variétés résistantes, rotation des cultures"
        }
    }
    
    if search_term:
        filtered = {k: v for k, v in diseases_db.items() if search_term.lower() in k.lower()}
    else:
        filtered = diseases_db
    
    for disease, info in filtered.items():
        with st.expander(f"🦠 {disease}", expanded=False):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Symptômes:** {info['symptoms']}")
            st.write(f"**Cultures affectées:** {', '.join(info['affected_crops'])}")
            st.write(f"**Traitement:** {info['treatment']}")
            st.write(f"**Prévention:** {info['prevention']}")

with tab3:
    st.subheader("⚠️ Alertes Maladies")
    
    st.info("""
    **Système d'alerte en temps réel**
    
    Recevez des notifications sur les risques de maladies dans votre région
    basées sur les conditions météorologiques et les rapports de terrain.
    """)
    
    # Zone de sélection
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox(
            "Région",
            ["Bamako", "Sikasso", "Kayes", "Mopti", "Ségou", "Koulikoro", "Gao", "Tombouctou"]
        )
    
    with col2:
        crop_alert = st.selectbox(
            "Culture",
            ["Toutes", "Tomate", "Maïs", "Riz", "Manioc"]
        )
    
    # Alertes simulées
    alerts = [
        {
            "date": "2024-01-15",
            "severity": "Modérée",
            "disease": "Mildiou",
            "crop": "Tomate",
            "message": "Conditions favorables au mildiou détectées. Traitement préventif recommandé."
        },
        {
            "date": "2024-01-14",
            "severity": "Faible",
            "disease": "Oïdium",
            "crop": "Cucurbitacées",
            "message": "Risque faible d'oïdium. Surveiller les conditions d'humidité."
        }
    ]
    
    for alert in alerts:
        severity_color = {
            "Faible": "🟢",
            "Modérée": "🟡",
            "Élevée": "🟠",
            "Critique": "🔴"
        }
        
        st.markdown(f"""
        <div style='background: white; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #f57c00; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between;'>
                <div>
                    <strong>{severity_color.get(alert['severity'], '⚪')} {alert['severity']}</strong> - {alert['disease']}
                </div>
                <div style='color: #666; font-size: 0.9rem;'>{alert['date']}</div>
            </div>
            <div style='margin-top: 10px;'>{alert['message']}</div>
            <div style='margin-top: 5px; color: #666; font-size: 0.9rem;'>
                Culture: {alert['crop']}
            </div>
        </div>
        """, unsafe_allow_html=True)
