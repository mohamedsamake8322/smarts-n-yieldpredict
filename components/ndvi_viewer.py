import streamlit as st
from smart_agro_tools.ndvi_engine.extractor import extract_valid_ndvi
import matplotlib.pyplot as plt

def show_ndvi():
    st.subheader("🛰️ Visualisation NDVI")
    lat = st.number_input("Latitude", value=19.66)
    lon = st.number_input("Longitude", value=4.3)
    year = st.slider("Année", min_value=2000, max_value=2025, value=2021)

    if st.button("🔍 Extraire NDVI"):
        result = extract_valid_ndvi(lat, lon, year)
        ndvi = result["ndvi"]
        source = result["source"]
        mean_ndvi = ndvi.mean(dim=["x", "y"]).values

        st.success(f"Source satellite : {source}")
        st.line_chart(ndvi.mean(dim=["x", "y"]))
        st.write(f"**NDVI moyen**: {round(float(mean_ndvi), 3)}")
