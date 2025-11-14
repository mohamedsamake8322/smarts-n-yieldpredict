import streamlit as st
from smart_agro_tools.input_recommender.recommender import suggest_inputs
import matplotlib.pyplot as plt

def show_npk_tool():
    st.subheader("🔬 Recommandation d'intrants NPK")
    crop = st.selectbox("Culture ciblée", ["Bananas", "Maize", "Mil", "Artichokes"])
    yield_target = st.number_input("Rendement cible (kg/ha)", value=2400)
    ndvi_profile = [0.44, 0.47, 0.50, 0.45, 0.42]
    soil = {"GWETPROF": 0.41, "GWETROOT": 0.38, "GWETTOP": 0.35}
    climate = {"WD10M": 120, "WS10M_RANGE": 4.2}

    if st.button("📈 Générer intrants"):
        npk = suggest_inputs(ndvi_profile, soil, climate, crop, yield_target)
        st.success("Recommandation générée ✅")
        st.write(npk)
        st.bar_chart(npk)
