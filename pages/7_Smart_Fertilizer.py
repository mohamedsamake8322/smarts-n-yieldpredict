"""
Smart Fertilizer Recommendation Interface

This Streamlit application provides an intelligent fertilizer recommendation system
with enhanced UX, visualizations, and professional design.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from io import BytesIO

# 🔌 Custom modules
from utils.fertilizer_api import predict_fertilizer
from utils.pdf_generator import build_pdf, build_excel
from utils.logger import log_prediction
from utils import AdvancedAI

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="SmartFactLaser – Fertilizer Recommendation",
    page_icon="🌾",
    layout="wide"
)

BASE = os.path.dirname(__file__)
DATA_FOLDER = r"C:\smarts-n-yieldpredict.git\Crop-Fertilizer-Analysis"


def get_available_cultures(folder_path: str) -> list:
    """
    Extracts available crop types from CSV files in the specified folder.
    
    Args:
        folder_path: Path to folder containing CSV files with crop data
        
    Returns:
        Sorted list of unique crop names
    """
    cultures = set()
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(folder_path, file))
                    for col in df.columns:
                        if col.lower().strip() in ["label", "crop type", "culture"]:
                            cultures.update(df[col].dropna().unique())
                except Exception:
                    continue
    return sorted(list(cultures)) if cultures else ["Rice", "Wheat", "Corn", "Soybean"]


# -----------------------------
# Sidebar: Configuration
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    available_cultures = get_available_cultures(DATA_FOLDER)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(
        "This system uses machine learning to recommend the optimal fertilizer "
        "based on your crop type, soil conditions, and environmental parameters."
    )


# -----------------------------
# Main UI
# -----------------------------
st.title("🌾 SmartFactLaser – Fertilizer Recommendation")
st.markdown("### AI-Powered Fertilizer Recommendation System")

# -----------------------------
# Input Section with Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🌿 Crop & Soil", "🌤 Environment", "🔬 Nutrients"])

with tab1:
    st.markdown("#### Crop and Soil Information")
    col1, col2 = st.columns(2)
    
    with col1:
        culture = st.selectbox(
            "🌿 Crop Type",
            available_cultures,
            help="Select the type of crop you are growing"
        )
        surface = st.number_input(
            "Area (ha)",
            min_value=0.1,
            value=1.0,
            step=0.1,
            help="Total cultivation area in hectares"
        )
    
    with col2:
        soil_type = st.selectbox(
            "🧱 Soil Type",
            ["Sandy", "Loamy", "Clayey", "Black", "Red"],
            help="Type of soil in your field"
        )
        moisture = st.number_input(
            "💧 Soil Moisture (%)",
            min_value=0,
            max_value=100,
            value=40,
            help="Current soil moisture percentage"
        )

with tab2:
    st.markdown("#### Environmental Conditions")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-10,
            max_value=50,
            value=28,
            step=0.1,
            help="Average temperature in Celsius"
        )
    
    with col2:
        humidity = st.number_input(
            "💨 Air Humidity (%)",
            min_value=0,
            max_value=100,
            value=60,
            help="Relative air humidity percentage"
        )

with tab3:
    st.markdown("#### Soil Nutrient Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nitrogen = st.number_input(
            "🔵 Nitrogen (N)",
            min_value=0,
            value=20,
            help="Available nitrogen level in soil"
        )
    
    with col2:
        phosphorous = st.number_input(
            "🟢 Phosphorous (P)",
            min_value=0,
            value=30,
            help="Available phosphorous level in soil"
        )
    
    with col3:
        potassium = st.number_input(
            "🟠 Potassium (K)",
            min_value=0,
            value=10,
            help="Available potassium level in soil"
        )

# -----------------------------
# Visualization Section
# -----------------------------
with st.expander("📊 Visualize Nutrient Balance", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        # NPK Bar Chart
        fig_npk = go.Figure()
        nutrients = ['Nitrogen (N)', 'Phosphorous (P)', 'Potassium (K)']
        values = [nitrogen, phosphorous, potassium]
        colors = ['#3498db', '#2ecc71', '#e67e22']
        
        fig_npk.add_trace(go.Bar(
            x=nutrients,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
            name='Nutrient Levels'
        ))
        
        fig_npk.update_layout(
            title="Soil Nutrient Levels (NPK)",
            xaxis_title="Nutrients",
            yaxis_title="Level",
            height=350
        )
        st.plotly_chart(fig_npk, use_container_width=True)
    
    with col2:
        # Environmental Conditions
        fig_env = go.Figure()
        
        fig_env.add_trace(go.Bar(
            x=['Temperature', 'Humidity', 'Moisture'],
            y=[temperature, humidity, moisture],
            marker_color=['#e74c3c', '#3498db', '#16a085'],
            text=[f"{temperature}°C", f"{humidity}%", f"{moisture}%"],
            textposition='auto',
            name='Environmental'
        ))
        
        fig_env.update_layout(
            title="Environmental Conditions",
            xaxis_title="Parameters",
            yaxis_title="Value",
            height=350
        )
        st.plotly_chart(fig_env, use_container_width=True)

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("🔍 Predict Fertilizer & Generate Plan", type="primary", use_container_width=True):
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
            with st.spinner("🤖 Analyzing conditions and generating recommendation..."):
                fertilizer_name = predict_fertilizer(user_inputs)
                log_prediction(user_inputs, fertilizer_name)
            
            # Display result with metric
            st.success("✅ Recommendation Generated Successfully!")
            st.markdown("### 🧪 Recommended Fertilizer")
            st.metric(
                label="Fertilizer Type",
                value=fertilizer_name,
                delta="AI Recommendation"
            )
            
            # Store in session state for downloads
            st.session_state.fertilizer_name = fertilizer_name
            st.session_state.user_inputs = user_inputs
            st.session_state.culture = culture
            st.session_state.surface = surface
            
        except FileNotFoundError as e:
            st.error(f"❌ Model file not found. Please ensure model files are in the correct location.")
            st.exception(e)
        except ValueError as e:
            st.error(f"❌ Invalid input: {str(e)}")
        except TypeError as e:
            st.error(f"❌ Invalid input type: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)

# -----------------------------
# Results Section
# -----------------------------
if 'fertilizer_name' in st.session_state:
    st.markdown("---")
    
    # Downloads Section
    with st.expander("📥 Download Reports", expanded=True):
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            try:
                pdf_bytes = build_pdf(
                    st.session_state.culture,
                    st.session_state.surface,
                    st.session_state.fertilizer_name
                )
                st.download_button(
                    "📄 Download PDF Plan",
                    pdf_bytes,
                    file_name=f"{st.session_state.culture}_fertilization_plan.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
        
        with col_d2:
            try:
                excel_bytes = build_excel(
                    st.session_state.culture,
                    st.session_state.surface,
                    st.session_state.fertilizer_name
                )
                st.download_button(
                    "📊 Download Excel Plan",
                    excel_bytes,
                    file_name=f"{st.session_state.culture}_fertilization_plan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating Excel: {e}")
    
    # Explanation Section
    with st.expander("ℹ️ Why This Recommendation?", expanded=False):
        st.markdown("### 🔬 Recommendation Explanation")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("#### 📊 Model Criteria")
            st.write("- **Crop Requirements**: Based on optimal nutrient needs for your crop type")
            st.write("- **Soil Conditions**: Adjusted for your soil type and moisture levels")
            st.write("- **Environmental Factors**: Optimized for current temperature and humidity")
            st.write("- **Nutrient Balance**: Corrects imbalances in NPK levels")
        
        with col_exp2:
            st.markdown("#### 🧠 AI Model Information")
            st.write("- **Model Type**: XGBoost Classifier")
            st.write("- **Training Data**: Real-world fertilization data")
            st.write("- **Accuracy**: Optimized for agricultural conditions")
            st.write("- **Updates**: Model continuously improved with new data")
        
        st.markdown("---")
        st.info(
            "💡 **Tip**: This recommendation is based on ML analysis. "
            "For best results, combine with local agronomic expertise and soil testing."
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌾 SmartFactLaser – AI-Powered Fertilizer Recommendation System"
    "</div>",
    unsafe_allow_html=True
)
