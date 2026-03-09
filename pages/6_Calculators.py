"""
Agricultural calculators page
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Calculators - Agro-Scan",
    page_icon="🧮",
    layout="wide"
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css, load_mobile_css
from utils.calculators import FertilizerCalculator, PesticideCalculator, FarmingCalculator

load_custom_css()
load_mobile_css()

st.title("🧮 Pesticide Calculator")

st.subheader("Pesticide Calculator")
st.markdown("Calculate pesticide doses for your treatments")

col1, col2 = st.columns(2)

with col1:
    area_hectares = st.number_input(
        "Superficie (hectares)",
        min_value=0.1,
        max_value=1000.0,
        value=1.0,
        step=0.1,
        key="pest_area"
    )
    
    pesticide_type = st.selectbox(
        "Type de pesticide",
        ["Liquide", "Poudre"]
    )
    
    recommended_rate = st.number_input(
        "Dose recommandée (L/ha ou kg/ha)",
        min_value=0.1,
        max_value=50.0,
        value=2.0,
        step=0.1
    )

with col2:
    concentration = st.number_input(
        "Concentration (%)",
        min_value=1,
        max_value=100,
        value=50,
        step=1
    )
    
    price_per_unit = st.number_input(
        "Prix par unité (FCFA)",
        min_value=0,
        value=5000,
        step=100
    )

if st.button("Calculer", type="primary", key="calc_pest"):
    calc = PesticideCalculator()
    result = calc.calculate_dose(
        area_hectares,
        pesticide_type.lower(),
        concentration,
        recommended_rate
    )
    
    cost = calc.calculate_cost(result["pesticide_amount"], price_per_unit)
    
    st.success("✅ Calcul terminé !")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            f"Quantité de pesticide ({result['unit']})",
            f"{result['pesticide_amount']} {result['unit']}"
        )
        st.metric("Quantité d'eau", f"{result['water_amount']} L")
    
    with col2:
        st.metric("Coût du traitement", f"{cost} FCFA")
        st.info(f"💡 Concentration: {concentration}%")
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="spacing_row"
            )
        
        with col2:
            width_m = st.number_input(
                "Largeur (mètres)",
                min_value=1,
                value=100,
                step=1,
                key="width"
            )
            
            spacing_plant = st.number_input(
                "Espacement entre plants (m)",
                min_value=0.1,
                value=0.5,
                step=0.1,
                key="spacing_plant"
            )
        
        if st.button("Calculer la densité", type="primary"):
            calc = FarmingCalculator()
            result = calc.calculate_planting_density(
                length_m,
                width_m,
                spacing_row,
                spacing_plant
            )
            
            st.success("✅ Calcul terminé !")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Superficie", f"{result['area_hectares']} ha")
            
            with col2:
                st.metric("Plantes totales", f"{result['total_plants']:,}")
            
            with col3:
                st.metric("Plantes/ha", f"{result['plants_per_hectare']:,}")
