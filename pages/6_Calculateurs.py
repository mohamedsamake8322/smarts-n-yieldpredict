"""
Page des calculateurs agricoles
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Calculateurs - Agro-Scan",
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

st.title("🧮 Calculateurs Agricoles")

# Onglets pour les différents calculateurs
tab1, tab2, tab3 = st.tabs(["🧮 Engrais", "💉 Pesticides", "📊 Agricole"])

with tab1:
    st.subheader("Calculateur d'Engrais")
    st.markdown("Calculez les besoins en engrais NPK pour vos cultures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        crop_type = st.selectbox(
            "Type de culture",
            ["Tomate", "Maïs", "Riz", "Manioc", "Banane", "Cacao", "Café", "Arachide"]
        )
        
        area_hectares = st.number_input(
            "Superficie (hectares)",
            min_value=0.1,
            max_value=1000.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        st.info("""
        **Informations:**
        - Les recommandations sont basées sur les besoins moyens
        - Ajustez selon votre analyse de sol
        - Consultez un agronome pour des recommandations précises
        """)
    
    if st.button("Calculer", type="primary"):
        calc = FertilizerCalculator()
        npk_needs = calc.calculate_npk(area_hectares, crop_type)
        
        st.success("✅ Calcul terminé !")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Azote (N)", f"{npk_needs['N']} kg")
        
        with col2:
            st.metric("Phosphore (P)", f"{npk_needs['P']} kg")
        
        with col3:
            st.metric("Potassium (K)", f"{npk_needs['K']} kg")
        
        with col4:
            st.metric("Total", f"{npk_needs['total']} kg")
        
        # Calcul avec engrais spécifique
        st.markdown("### 💡 Quantité d'engrais nécessaire")
        
        fertilizer_type = st.selectbox(
            "Type d'engrais",
            ["NPK 15-15-15", "NPK 20-10-10", "NPK 12-12-17", "Urée (46% N)"]
        )
        
        npk_ratios = {
            "NPK 15-15-15": {"N": 15, "P": 15, "K": 15},
            "NPK 20-10-10": {"N": 20, "P": 10, "K": 10},
            "NPK 12-12-17": {"N": 12, "P": 12, "K": 17},
            "Urée (46% N)": {"N": 46, "P": 0, "K": 0}
        }
        
        if st.button("Calculer la quantité"):
            ratio = npk_ratios[fertilizer_type]
            amount = calc.calculate_fertilizer_amount(npk_needs, fertilizer_type, ratio)
            st.success(f"📦 Quantité nécessaire: **{amount} kg** de {fertilizer_type}")

with tab2:
    st.subheader("Calculateur de Pesticides")
    st.markdown("Calculez les doses de pesticides pour vos traitements")
    
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

with tab3:
    st.subheader("Calculateur Agricole Général")
    st.markdown("Calculs avancés pour votre exploitation")
    
    calc_type = st.radio(
        "Type de calcul",
        ["Rendement", "Irrigation", "Densité de plantation"],
        horizontal=True
    )
    
    if calc_type == "Rendement":
        st.markdown("### 📈 Calcul de rendement estimé")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area_hectares = st.number_input(
                "Superficie (hectares)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="yield_area"
            )
            
            plants_per_hectare = st.number_input(
                "Plantes par hectare",
                min_value=1,
                value=10000,
                step=100
            )
        
        with col2:
            fruits_per_plant = st.number_input(
                "Fruits par plante",
                min_value=1,
                value=10,
                step=1
            )
            
            avg_fruit_weight = st.number_input(
                "Poids moyen d'un fruit (kg)",
                min_value=0.01,
                value=0.2,
                step=0.01
            )
        
        if st.button("Calculer le rendement", type="primary"):
            calc = FarmingCalculator()
            result = calc.calculate_yield(
                area_hectares,
                plants_per_hectare,
                fruits_per_plant,
                avg_fruit_weight
            )
            
            st.success("✅ Calcul terminé !")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Plantes totales", f"{result['total_plants']:,}")
            
            with col2:
                st.metric("Rendement estimé", f"{result['estimated_yield_tons']} tonnes")
            
            with col3:
                st.metric("Rendement (kg)", f"{result['estimated_yield_kg']:,} kg")
    
    elif calc_type == "Irrigation":
        st.markdown("### 💧 Calcul des besoins en irrigation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area_hectares = st.number_input(
                "Superficie (hectares)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="irr_area"
            )
            
            crop_water_needs = st.number_input(
                "Besoins en eau de la culture (mm)",
                min_value=1,
                value=500,
                step=10
            )
        
        with col2:
            irrigation_efficiency = st.slider(
                "Efficacité d'irrigation (%)",
                min_value=50,
                max_value=100,
                value=80,
                step=5
            ) / 100
        
        if st.button("Calculer l'irrigation", type="primary"):
            calc = FarmingCalculator()
            result = calc.calculate_irrigation(
                area_hectares,
                crop_water_needs,
                irrigation_efficiency
            )
            
            st.success("✅ Calcul terminé !")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Besoins en eau", f"{result['water_needs_m3']} m³")
            
            with col2:
                st.metric("Eau d'irrigation nécessaire", f"{result['irrigation_needs_m3']} m³")
    
    elif calc_type == "Densité de plantation":
        st.markdown("### 🌱 Calcul de la densité de plantation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            length_m = st.number_input(
                "Longueur (mètres)",
                min_value=1,
                value=100,
                step=1,
                key="length"
            )
            
            spacing_row = st.number_input(
                "Espacement entre rangs (m)",
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
