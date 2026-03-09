"""
Page de la bibliothèque de ressources agricoles
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Bibliothèque - Agro-Scan",
    page_icon="📚",
    layout="wide"
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css, load_mobile_css

load_custom_css()
load_mobile_css()

st.title("📚 Bibliothèque Agricole")

# Onglets
tab1, tab2, tab3 = st.tabs(["🌿 Conseils de Culture", "🦠 Maladies & Ravageurs", "⚠️ Alertes"])

with tab1:
    st.subheader("🌿 Conseils de Culture")
    
    crop_selected = st.selectbox(
        "Sélectionnez une culture",
        ["Tomate", "Maïs", "Riz", "Manioc", "Banane", "Cacao", "Café", "Arachide"]
    )
    
    if crop_selected == "Tomate":
        st.markdown("""
        ### 🍅 Guide de culture de la Tomate
        
        **Plantation:**
        - Période: Février-Mars ou Septembre-Octobre
        - Espacement: 50-60 cm entre plants, 80-100 cm entre rangs
        - Profondeur: 5-7 cm
        
        **Sol:**
        - Sol bien drainé, riche en matière organique
        - pH optimal: 6.0-6.8
        - Éviter les sols trop acides
        
        **Arrosage:**
        - Fréquence: 2-3 fois par semaine
        - Quantité: 2-3 L par plant
        - Éviter l'arrosage sur les feuilles
        
        **Fertilisation:**
        - Engrais NPK 15-15-15: 200-300 kg/ha
        - Compost: 10-15 tonnes/ha
        - Fumure de fond avant plantation
        
        **Récolte:**
        - 60-90 jours après plantation
        - Récolter tôt le matin
        - Conserver à température ambiante
        """)
    
    elif crop_selected == "Maïs":
        st.markdown("""
        ### 🌽 Guide de culture du Maïs
        
        **Plantation:**
        - Période: Avril-Mai (saison des pluies)
        - Espacement: 25-30 cm entre plants, 75-80 cm entre rangs
        - Densité: 40,000-50,000 plants/ha
        
        **Sol:**
        - Sol profond, bien drainé
        - pH optimal: 5.5-7.0
        - Éviter les sols compacts
        
        **Arrosage:**
        - Besoins: 500-800 mm pendant le cycle
        - Critique pendant la floraison
        - Irrigation si pluies insuffisantes
        
        **Fertilisation:**
        - Azote: 120-150 kg/ha
        - Phosphore: 40-60 kg/ha
        - Potassium: 80-100 kg/ha
        
        **Récolte:**
        - 90-120 jours après plantation
        - Récolter à maturité physiologique
        - Sécher rapidement après récolte
        """)
    
    # Ajouter d'autres cultures...

with tab2:
    st.subheader("🦠 Base de données des Maladies & Ravageurs")
    
    search_term = st.text_input("🔍 Rechercher une maladie ou un ravageur")
    
    diseases_db = {
        "Mildiou": {
            "description": "Maladie fongique causée par Phytophthora infestans",
            "symptoms": "Taches brunes sur les feuilles, duvet blanc au revers",
            "affected_crops": ["Tomate", "Pomme de terre"],
            "treatment": "Bouillie bordelaise, fongicides systémiques",
            "prevention": "Éviter l'humidité excessive, espacement des plants"
        },
        "Oïdium": {
            "description": "Maladie fongique causée par différents champignons",
            "symptoms": "Poudre blanche sur les feuilles et tiges",
            "affected_crops": ["Cucurbitacées", "Vigne", "Rosiers"],
            "treatment": "Soufre mouillable, fongicides anti-oïdium",
            "prevention": "Bonne circulation d'air, traitement préventif"
        },
        "Rouille": {
            "description": "Maladie fongique causée par Puccinia spp.",
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
