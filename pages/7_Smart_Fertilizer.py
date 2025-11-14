# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import os
from io import BytesIO

# 🔌 Custom modules
from utils.fertilizer_api import predict_fertilizer
from utils.pdf_generator import build_pdf, build_excel
from utils.logger import log_prediction
from utils import AdvancedAI

# -----------------------------
# Paths
# -----------------------------
BASE = os.path.dirname(__file__)
DATA_FOLDER = r"C:\plateforme-agricole-complete-v2\Crop-Fertilizer-Analysis"

# -----------------------------
# Extract crops from CSV
# -----------------------------
def get_available_cultures(folder_path: str) -> list:
    cultures = set()
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            for col in df.columns:
                if col.lower().strip() in ["label", "crop type", "culture"]:
                    cultures.update(df[col].dropna().unique())
    return sorted(list(cultures))

available_cultures = get_available_cultures(DATA_FOLDER)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌾 SmartFactLaser – Fertilizer Recommendation")

cols = st.columns(2)
with cols[0]:
    culture = st.selectbox("🌿 Crop Type", available_cultures)
    surface = st.number_input("Area (ha)", min_value=0.1, value=1.0)
with cols[1]:
    soil_type = st.selectbox("🧱 Soil Type", ["Sandy", "Loamy", "Clayey", "Black", "Red"])
    moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=40)

st.markdown("### 🌤 Environmental Parameters")
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=28)
humidity = st.number_input("Air Humidity (%)", min_value=0, max_value=100, value=60)
nitrogen = st.number_input("Nitrogen (N)", min_value=0, value=20)
phosphorous = st.number_input("Phosphorous (P)", min_value=0, value=30)
potassium = st.number_input("Potassium (K)", min_value=0, value=10)

if st.button("🔍 Predict Fertilizer & Generate Plan"):
    user_inputs = {
        "Temperature": temperature,
        "Humidity": humidity,
        "Moisture": moisture,
        "Soil Type": soil_type,
        "Crop Type": culture,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorous,
        "Potassium": potassium
    }

    try:
        fertilizer_name = predict_fertilizer(user_inputs)
        log_prediction(user_inputs, fertilizer_name)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    st.success(f"🧪 Recommended Fertilizer: {fertilizer_name}")

    # PDF
    pdf_bytes = build_pdf(culture, surface, fertilizer_name)
    st.download_button("📄 Download PDF Plan", pdf_bytes, file_name=f"{culture}_fertilization_plan.pdf", mime="application/pdf")

    # Excel
    excel_bytes = build_excel(culture, surface, fertilizer_name)
    st.download_button("📥 Download Excel", excel_bytes, file_name=f"{culture}_fertilization_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Explanation
    st.markdown("### ℹ️ Quick Explanation")
    st.write("- The fertilizer is recommended based on agro-climatic conditions and nutritional needs.")
    st.write("- The model is trained on real fertilization data.")
    st.success("✅ Plan generated — download the PDF or Excel below.")
