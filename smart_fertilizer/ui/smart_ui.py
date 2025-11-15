import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
from typing import Dict, List, Optional

# ✅ Imports internes du projet - corrigés
from modules.smart_fertilizer.core.smart_fertilizer_engine import SmartFertilizerEngine
from modules.smart_fertilizer.core.regional_context import RegionalContext
from modules.smart_fertilizer.regions.region_selector import RegionSelector
from modules.smart_fertilizer.ui.crop_selector import CropSelector
from modules.smart_fertilizer.weather.weather_client import WeatherClient
from modules.smart_fertilizer.weather.iot_simulator import IoTSensorSimulator
from modules.smart_fertilizer.exports.pdf_generator import FertilizerReportGenerator
from modules.smart_fertilizer.exports.export_utils import ExportUtilities
from modules.smart_fertilizer.api.models import SoilAnalysis, CropSelection
from modules.smart_fertilizer.ui.translations import get_translator

class SmartFertilizerUI:
    """
    Main Streamlit UI for Smart Fertilizer Application
    """

    def __init__(self):
        self.fertilizer_engine = SmartFertilizerEngine()
        self.regional_context = RegionalContext()
        self.region_selector = RegionSelector()
        self.crop_selector = CropSelector(self.fertilizer_engine)
        self.weather_client = WeatherClient()
        self.iot_simulator = IoTSensorSimulator()
        self.pdf_generator = FertilizerReportGenerator()
        self.export_utils = ExportUtilities()
        self.translator = get_translator()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""

        if 'current_language' not in st.session_state:
            st.session_state.current_language = 'en'

        if 'soil_analysis_data' not in st.session_state:
            st.session_state.soil_analysis_data = {}

        if 'crop_selection_data' not in st.session_state:
            st.session_state.crop_selection_data = {}

        if 'selected_region' not in st.session_state:
            st.session_state.selected_region = None

        if 'recommendation_generated' not in st.session_state:
            st.session_state.recommendation_generated = False

        if 'current_recommendation' not in st.session_state:
            st.session_state.current_recommendation = None

        if 'show_advanced_options' not in st.session_state:
            st.session_state.show_advanced_options = False

    def render_main_interface(self):
        """Render the main application interface"""

        # Header and navigation
        self._render_header()

        # Sidebar for navigation and settings
        self._render_sidebar()

        # Main content area
        main_tab = st.session_state.get('main_tab', 'fertilizer_recommendation')

        if main_tab == 'fertilizer_recommendation':
            self._render_fertilizer_recommendation_tab()
        elif main_tab == 'weather_monitoring':
            self._render_weather_monitoring_tab()
        elif main_tab == 'iot_dashboard':
            self._render_iot_dashboard_tab()
        elif main_tab == 'data_analysis':
            self._render_data_analysis_tab()
        elif main_tab == 'help_support':
            self._render_help_support_tab()

    def _render_header(self):
        """Render application header"""

        st.set_page_config(
            page_title="Smart Fertilizer - African Agriculture",
            page_icon="🌾",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Main title with language support
        title = self.translator.get_text('app_title', st.session_state.current_language)
        st.title(f"🌾 {title}")

        # Subtitle
        subtitle = self.translator.get_text('app_subtitle', st.session_state.current_language)
        st.markdown(f"*{subtitle}*")

        # Quick stats row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Supported Regions",
                value=len(self.regional_context.get_available_regions())
            )

        with col2:
            st.metric(
                label="Available Crops",
                value=len(self.fertilizer_engine.get_available_crops())
            )

        with col3:
            if st.session_state.current_recommendation:
                roi = st.session_state.current_recommendation.roi_percentage
                st.metric(label="Expected ROI", value=f"{roi:.1f}%")
            else:
                st.metric(label="Expected ROI", value="--")

        with col4:
            if st.session_state.current_recommendation:
                cost = st.session_state.current_recommendation.cost_analysis.cost_per_hectare
                currency = st.session_state.current_recommendation.cost_analysis.currency
                st.metric(label="Cost/Hectare", value=f"{currency} {cost:.2f}")
            else:
                st.metric(label="Cost/Hectare", value="--")

        st.divider()

    def _render_sidebar(self):
        """Render sidebar navigation and settings"""

        with st.sidebar:
            st.header("🚀 Navigation")

            # Main navigation
            main_tabs = {
                'fertilizer_recommendation': '🎯 Fertilizer Recommendation',
                'weather_monitoring': '🌤️ Weather Monitoring',
                'iot_dashboard': '📊 IoT Dashboard',
                'data_analysis': '📈 Data Analysis',
                'help_support': '❓ Help & Support'
            }

            for tab_key, tab_label in main_tabs.items():
                if st.button(tab_label, key=f"nav_{tab_key}", use_container_width=True):
                    st.session_state.main_tab = tab_key
                    st.rerun()

            st.divider()

            # Language selection
            st.subheader("🌍 Language / Langue")
            languages = {
                'en': 'English',
                'fr': 'Français',
                'sw': 'Kiswahili',
                'ha': 'Hausa',
                'am': 'Amharic'
            }

            selected_language = st.selectbox(
                "Select Language:",
                options=list(languages.keys()),
                format_func=lambda x: languages[x],
                index=list(languages.keys()).index(st.session_state.current_language)
            )

            if selected_language != st.session_state.current_language:
                st.session_state.current_language = selected_language
                st.rerun()

            st.divider()

            # Quick settings
            st.subheader("⚙️ Settings")

            st.session_state.show_advanced_options = st.checkbox(
                "Show Advanced Options",
                value=st.session_state.show_advanced_options
            )

            # Export options
            if st.session_state.current_recommendation:
                st.subheader("📄 Export Report")

                export_format = st.selectbox(
                    "Format:",
                    options=['pdf', 'xlsx', 'csv', 'json'],
                    format_func=lambda x: x.upper()
                )

                if st.button("Download Report", use_container_width=True):
                    self._handle_export(export_format)

    def _render_fertilizer_recommendation_tab(self):
        """Render main fertilizer recommendation interface"""

        st.header("🎯 Smart Fertilizer Recommendation System")

        # Progress indicator
        progress_steps = ["Region & Crop", "Soil Analysis", "Farm Details", "Generate Recommendation"]
        current_step = self._get_current_step()

        progress_cols = st.columns(len(progress_steps))
        for i, (col, step) in enumerate(zip(progress_cols, progress_steps)):
            with col:
                if i <= current_step:
                    st.success(f"✅ {step}")
                elif i == current_step + 1:
                    st.info(f"🔄 {step}")
                else:
                    st.empty()

        st.divider()

        # Step 1: Region and Crop Selection
        with st.expander("🌍 Step 1: Region & Crop Selection", expanded=current_step == 0):
            self._render_region_crop_selection()

        # Step 2: Soil Analysis Input
        if st.session_state.selected_region:
            with st.expander("🧪 Step 2: Soil Analysis", expanded=current_step == 1):
                self._render_soil_analysis_input()

        # Step 3: Farm Details
        if st.session_state.soil_analysis_data and st.session_state.crop_selection_data:
            with st.expander("🚜 Step 3: Farm Details", expanded=current_step == 2):
                self._render_farm_details_input()

        # Step 4: Generate Recommendation
        if self._all_data_collected():
            with st.expander("🎯 Step 4: Generate Recommendation", expanded=current_step == 3):
                self._render_recommendation_generation()

        # Display results
        if st.session_state.recommendation_generated and st.session_state.current_recommendation:
            st.divider()
            self._render_recommendation_results()

    def _render_region_crop_selection(self):
        """Render enhanced region and crop selection interface"""

        st.markdown("### 🌍 Étape 1: Sélection de la Région et des Cultures")

        # Region selection (enhanced)
        region_data = self.region_selector.render_region_selector()
        if region_data:
            st.session_state.selected_region = region_data

        # Enhanced crop selection
        if st.session_state.selected_region:
            st.divider()
            crop_data = self.crop_selector.render_crop_selection(st.session_state.selected_region)
            if crop_data:
                st.session_state.crop_selection_data = crop_data

                # Display selection summary
                st.markdown("### ✅ Résumé de la Sélection")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.info(f"**Région:** {st.session_state.selected_region['name']}")
                    st.info(f"**Climat:** {st.session_state.selected_region.get('climate_type', 'N/A')}")

                with col2:
                    st.success(f"**Culture:** {crop_data['crop_name']}")
                    st.success(f"**Variété:** {crop_data['variety']}")

                with col3:
                    st.warning(f"**Durée:** {crop_data['growth_duration']} jours")
                    yield_range = crop_data.get('yield_potential', [0, 0, 0])
                    st.warning(f"**Rendement:** {yield_range[0]}-{yield_range[2]} t/ha")

                # Additional crop insights
                with st.expander("📊 Informations Détaillées sur la Culture Sélectionnée"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🌱 Besoins Nutritionnels**")
                        nutrients = crop_data.get('nutritional_requirements', {})
                        if nutrients:
                            st.write(f"• Azote (N): {nutrients.get('N', 0)} kg/ha")
                            st.write(f"• Phosphore (P): {nutrients.get('P', 0)} kg/ha")
                            st.write(f"• Potassium (K): {nutrients.get('K', 0)} kg/ha")

                        st.markdown("**💧 Besoins en Eau**")
                        water_req = crop_data.get('water_requirement', 0)
                        st.write(f"• {water_req} mm par saison")

                    with col2:
                        st.markdown("**💰 Informations Économiques**")
                        price = crop_data.get('market_price', 0)
                        st.write(f"• Prix de marché: ${price}/tonne")

                        processing = crop_data.get('processing_options', [])
                        if processing:
                            st.markdown("**🏭 Options de Transformation**")
                            for option in processing[:3]:
                                st.write(f"• {option}")

                        diseases = crop_data.get('diseases', [])
                        if diseases:
                            st.markdown("**⚠️ Principales Maladies**")
                            for disease in diseases[:3]:
                                st.write(f"• {disease}")

                # Weather and timing recommendations
                with st.expander("🌤️ Recommandations Climatiques et de Calendrier"):
                    self._render_timing_recommendations(region_data, crop_data)

        else:
            st.info("Veuillez d'abord sélectionner une région pour accéder à la sélection des cultures.")

    def _render_timing_recommendations(self, region_data: Dict, crop_data: Dict):
        """Render timing and calendar recommendations"""

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📅 Calendrier de Plantation Optimal**")

            climate_type = region_data.get('climate_type', '').lower()
            growth_duration = crop_data.get('growth_duration', 120)

            # Simple seasonal recommendations based on climate
            if 'tropical' in climate_type:
                if growth_duration <= 90:
                    st.success("🌧️ **Saison des pluies recommandée**")
                    st.write("• Plantation: Mai - Juillet")
                    st.write("• Récolte: Août - Octobre")
                else:
                    st.info("🌦️ **Début de saison des pluies**")
                    st.write("• Plantation: Avril - Mai")
                    st.write("• Récolte: Août - Septembre")

            elif 'arid' in climate_type or 'semi' in climate_type:
                st.warning("💧 **Irrigation requise**")
                st.write("• Plantation: Novembre - Janvier")
                st.write("• Récolte: Mars - Mai")
                st.write("• Irrigation critique: Décembre - Février")

            else:
                st.info("🌤️ **Saison optimale**")
                st.write("• Plantation: Mars - Mai")
                st.write("• Récolte: Juillet - Septembre")

        with col2:
            st.markdown("**⚠️ Considérations Climatiques**")

            water_requirement = crop_data.get('water_requirement', 500)
            rainfall_pattern = region_data.get('rainfall_pattern', 'moderate')

            if water_requirement > 800:
                st.error("🚨 **Forte demande en eau**")
                st.write("• Irrigation essentielle")
                st.write("• Surveillance quotidienne")
            elif water_requirement < 400:
                st.success("✅ **Résistant à la sécheresse**")
                st.write("• Adapté aux zones arides")
                st.write("• Irrigation minimale")
            else:
                st.info("⚖️ **Besoins modérés**")
                st.write("• Irrigation complémentaire")
                st.write("• Dépend des précipitations")

            # Risk factors
            diseases = crop_data.get('diseases', [])
            if diseases:
                st.markdown("**🛡️ Surveillance des Maladies**")
                for disease in diseases[:2]:
                    st.write(f"• {disease}")

            st.markdown("**📊 Facteurs de Réussite**")
            st.write(f"• Température optimale: 20-30°C")
            st.write(f"• Humidité relative: 60-80%")
            st.write(f"• pH du sol: 6.0-7.5")

    def _render_soil_analysis_input(self):
        """Render soil analysis input interface"""

        st.subheader("🧪 Soil Test Results")

        # Option to use IoT data or manual input
        data_source = st.radio(
            "Data Source:",
            options=['manual_input', 'iot_sensors', 'upload_file'],
            format_func=lambda x: {
                'manual_input': '✏️ Manual Input',
                'iot_sensors': '📡 IoT Sensors',
                'upload_file': '📁 Upload File'
            }[x],
            horizontal=True
        )

        if data_source == 'manual_input':
            self._render_manual_soil_input()
        elif data_source == 'iot_sensors':
            self._render_iot_soil_input()
        elif data_source == 'upload_file':
            self._render_file_upload_soil_input()

    def _render_manual_soil_input(self):
        """Render manual soil analysis input"""

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Basic Properties**")

            ph = st.number_input(
                "Soil pH:",
                min_value=3.0,
                max_value=10.0,
                value=st.session_state.soil_analysis_data.get('ph', 6.5),
                step=0.1,
                format="%.1f"
            )

            organic_matter = st.number_input(
                "Organic Matter (%):",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.soil_analysis_data.get('organic_matter', 3.0),
                step=0.1,
                format="%.1f"
            )

            texture = st.selectbox(
                "Soil Texture:",
                options=['sandy', 'loamy', 'clay', 'silt'],
                index=['sandy', 'loamy', 'clay', 'silt'].index(
                    st.session_state.soil_analysis_data.get('texture', 'loamy')
                )
            )

            cec = st.number_input(
                "CEC (cmol/kg):",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.soil_analysis_data.get('cec', 15.0),
                step=0.5,
                format="%.1f"
            )

        with col2:
            st.markdown("**Primary Nutrients (ppm)**")

            nitrogen = st.number_input(
                "Available Nitrogen:",
                min_value=0.0,
                max_value=1000.0,
                value=st.session_state.soil_analysis_data.get('nitrogen', 250.0),
                step=10.0,
                format="%.1f"
            )

            phosphorus = st.number_input(
                "Available Phosphorus:",
                min_value=0.0,
                max_value=200.0,
                value=st.session_state.soil_analysis_data.get('phosphorus', 25.0),
                step=1.0,
                format="%.1f"
            )

            potassium = st.number_input(
                "Available Potassium:",
                min_value=0.0,
                max_value=1000.0,
                value=st.session_state.soil_analysis_data.get('potassium', 200.0),
                step=10.0,
                format="%.1f"
            )

        with col3:
            st.markdown("**Secondary & Micronutrients (ppm)**")

            if st.session_state.show_advanced_options:
                calcium = st.number_input(
                    "Calcium:",
                    min_value=0.0,
                    max_value=5000.0,
                    value=st.session_state.soil_analysis_data.get('calcium', 800.0),
                    step=50.0
                )

                magnesium = st.number_input(
                    "Magnesium:",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state.soil_analysis_data.get('magnesium', 150.0),
                    step=10.0
                )

                sulfur = st.number_input(
                    "Sulfur:",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.soil_analysis_data.get('sulfur', 15.0),
                    step=1.0
                )

                ec = st.number_input(
                    "EC (dS/m):",
                    min_value=0.0,
                    max_value=10.0,
                    value=st.session_state.soil_analysis_data.get('ec', 1.2),
                    step=0.1
                )

                zinc = st.number_input(
                    "Zinc:",
                    min_value=0.0,
                    max_value=50.0,
                    value=st.session_state.soil_analysis_data.get('zinc', 2.5),
                    step=0.1
                )

                iron = st.number_input(
                    "Iron:",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.soil_analysis_data.get('iron', 8.0),
                    step=0.5
                )
            else:
                st.info("Enable 'Advanced Options' in sidebar to input additional nutrients")
                calcium = st.session_state.soil_analysis_data.get('calcium', 800.0)
                magnesium = st.session_state.soil_analysis_data.get('magnesium', 150.0)
                sulfur = st.session_state.soil_analysis_data.get('sulfur', 15.0)
                ec = st.session_state.soil_analysis_data.get('ec', 1.2)
                zinc = st.session_state.soil_analysis_data.get('zinc', 2.5)
                iron = st.session_state.soil_analysis_data.get('iron', 8.0)

        # Store soil analysis data
        st.session_state.soil_analysis_data = {
            'ph': ph,
            'organic_matter': organic_matter,
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'cec': cec,
            'texture': texture,
            'ec': ec,
            'calcium': calcium,
            'magnesium': magnesium,
            'sulfur': sulfur,
            'zinc': zinc,
            'iron': iron
        }

        # Show soil analysis interpretation
        if st.button("Analyze Soil Data"):
            self._display_soil_interpretation()

    def _render_iot_soil_input(self):
        """Render IoT sensor data input"""

        st.info("📡 Reading data from IoT sensors...")

        # Get IoT sensor data
        sensor_data = self.iot_simulator.get_all_current_readings()
        agricultural_summary = self.iot_simulator.get_agricultural_summary()

        if sensor_data and sensor_data.get('sensors'):
            sensors = sensor_data['sensors']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Current Sensor Readings**")

                # Display key sensor readings
                if 'soil_ph' in sensors:
                    ph_reading = sensors['soil_ph']
                    st.metric("Soil pH", f"{ph_reading['value']:.1f}",
                             help=f"Last updated: {ph_reading.get('timestamp', 'Unknown')}")

                if 'soil_moisture' in sensors:
                    moisture_reading = sensors['soil_moisture']
                    st.metric("Soil Moisture", f"{moisture_reading['value']:.1f}%",
                             help=f"Battery: {moisture_reading.get('battery_level', 'Unknown')}%")

                if 'soil_temperature' in sensors:
                    temp_reading = sensors['soil_temperature']
                    st.metric("Soil Temperature", f"{temp_reading['value']:.1f}°C")

                if 'soil_ec' in sensors:
                    ec_reading = sensors['soil_ec']
                    st.metric("Electrical Conductivity", f"{ec_reading['value']:.1f} dS/m")

            with col2:
                st.markdown("**Agricultural Conditions**")

                soil_conditions = agricultural_summary.get('soil_conditions', {})
                environmental = agricultural_summary.get('environmental_conditions', {})
                indices = agricultural_summary.get('agricultural_indices', {})

                st.write(f"**Irrigation Need:** {soil_conditions.get('irrigation_need', 'Unknown').title()}")
                st.write(f"**Growing Conditions:** {indices.get('growing_conditions', 'Unknown').title()}")
                st.write(f"**Fertilizer Application Suitability:** {indices.get('fertilizer_application_suitability', 'Unknown').title()}")

                # Show recommendations
                recommendations = agricultural_summary.get('recommendations', [])
                if recommendations:
                    st.markdown("**IoT Recommendations:**")
                    for rec in recommendations[:3]:
                        st.write(f"• {rec}")

            # Use IoT data for soil analysis
            if st.button("Use IoT Sensor Data"):
                # Convert IoT data to soil analysis format
                st.session_state.soil_analysis_data = {
                    'ph': sensors.get('soil_ph', {}).get('value', 6.5),
                    'organic_matter': 3.0,  # Not typically measured by IoT
                    'nitrogen': 250.0,  # Estimated
                    'phosphorus': 25.0,  # Estimated
                    'potassium': 200.0,  # Estimated
                    'cec': 15.0,  # Estimated
                    'texture': 'loamy',  # Default
                    'ec': sensors.get('soil_ec', {}).get('value', 1.2),
                    'calcium': 800.0,  # Estimated
                    'magnesium': 150.0,  # Estimated
                    'sulfur': 15.0,  # Estimated
                    'zinc': 2.5,  # Estimated
                    'iron': 8.0  # Estimated
                }

                st.success("✅ IoT sensor data imported successfully!")
                st.info("⚠️ Note: Some values are estimated. Consider laboratory testing for complete analysis.")
        else:
            st.warning("⚠️ No IoT sensor data available. Please check sensor connectivity.")

    def _render_file_upload_soil_input(self):
        """Render file upload interface for soil data"""

        st.markdown("**Upload Soil Test Report**")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload a soil test report in CSV, Excel, or JSON format"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    df = pd.DataFrame([data])

                st.write("**Uploaded Data Preview:**")
                st.dataframe(df.head())

                # Map columns to soil parameters
                if st.button("Process Uploaded Data"):
                    # Simple mapping - in production, this would be more sophisticated
                    soil_data = {}

                    for col in df.columns:
                        col_lower = col.lower()
                        if 'ph' in col_lower:
                            soil_data['ph'] = float(df[col].iloc[0])
                        elif 'organic' in col_lower or 'om' in col_lower:
                            soil_data['organic_matter'] = float(df[col].iloc[0])
                        elif 'nitrogen' in col_lower or 'n' in col_lower:
                            soil_data['nitrogen'] = float(df[col].iloc[0])
                        elif 'phosphorus' in col_lower or 'p' in col_lower:
                            soil_data['phosphorus'] = float(df[col].iloc[0])
                        elif 'potassium' in col_lower or 'k' in col_lower:
                            soil_data['potassium'] = float(df[col].iloc[0])
                        elif 'cec' in col_lower:
                            soil_data['cec'] = float(df[col].iloc[0])

                    # Fill in defaults for missing values
                    defaults = {
                        'ph': 6.5, 'organic_matter': 3.0, 'nitrogen': 250.0,
                        'phosphorus': 25.0, 'potassium': 200.0, 'cec': 15.0,
                        'texture': 'loamy', 'ec': 1.2, 'calcium': 800.0,
                        'magnesium': 150.0, 'sulfur': 15.0, 'zinc': 2.5, 'iron': 8.0
                    }

                    for key, default_value in defaults.items():
                        if key not in soil_data:
                            soil_data[key] = default_value

                    st.session_state.soil_analysis_data = soil_data
                    st.success("✅ Soil data imported successfully!")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your file contains soil test parameters with appropriate column names.")

    def _render_farm_details_input(self):
        """Render farm details input interface"""

        st.subheader("🚜 Farm Information")

        col1, col2 = st.columns(2)

        with col1:
            area_hectares = st.number_input(
                "Farm Area (hectares):",
                min_value=0.1,
                max_value=10000.0,
                value=1.0,
                step=0.1,
                format="%.1f"
            )

            target_yield = st.number_input(
                "Target Yield (tons/ha):",
                min_value=0.5,
                max_value=20.0,
                value=5.0,
                step=0.1,
                format="%.1f"
            )

        with col2:
            # Location for weather data
            st.markdown("**Farm Location** (Optional)")

            latitude = st.number_input(
                "Latitude:",
                min_value=-90.0,
                max_value=90.0,
                value=0.0,
                step=0.1,
                format="%.6f"
            )

            longitude = st.number_input(
                "Longitude:",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=0.1,
                format="%.6f"
            )

        # Store farm details
        st.session_state.farm_details = {
            'area_hectares': area_hectares,
            'target_yield': target_yield,
            'latitude': latitude,
            'longitude': longitude
        }

        # Show weather data if location provided
        if latitude != 0.0 or longitude != 0.0:
            with st.expander("🌤️ Current Weather Conditions"):
                weather_data = self.weather_client.get_current_weather(latitude, longitude)
                if weather_data:
                    weather_col1, weather_col2, weather_col3 = st.columns(3)

                    with weather_col1:
                        st.metric("Temperature", f"{weather_data['temperature']:.1f}°C")
                        st.metric("Humidity", f"{weather_data['humidity']:.0f}%")

                    with weather_col2:
                        st.metric("Wind Speed", f"{weather_data['wind_speed']:.1f} m/s")
                        st.metric("Precipitation", f"{weather_data['precipitation']:.1f} mm")

                    with weather_col3:
                        st.metric("Pressure", f"{weather_data['pressure']:.0f} hPa")
                        st.write(f"**Condition:** {weather_data['description'].title()}")

    def _render_recommendation_generation(self):
        """Render recommendation generation interface"""

        st.subheader("🎯 Generate Fertilizer Recommendation")

        # Summary of inputs
        with st.expander("📋 Input Summary", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Region & Crop:**")
                st.write(f"• Region: {st.session_state.selected_region['name']}")
                st.write(f"• Crop: {st.session_state.crop_selection_data['crop_type'].title()}")
                st.write(f"• Variety: {st.session_state.crop_selection_data['variety']}")

                st.write("**Farm Details:**")
                st.write(f"• Area: {st.session_state.farm_details['area_hectares']} ha")
                st.write(f"• Target Yield: {st.session_state.farm_details['target_yield']} tons/ha")

            with col2:
                st.write("**Key Soil Parameters:**")
                soil_data = st.session_state.soil_analysis_data
                st.write(f"• pH: {soil_data['ph']:.1f}")
                st.write(f"• Organic Matter: {soil_data['organic_matter']:.1f}%")
                st.write(f"• Available N: {soil_data['nitrogen']:.0f} ppm")
                st.write(f"• Available P: {soil_data['phosphorus']:.0f} ppm")
                st.write(f"• Available K: {soil_data['potassium']:.0f} ppm")

        # Generation options
        col1, col2 = st.columns(2)

        with col1:
            currency = st.selectbox(
                "Currency for Cost Analysis:",
                options=['USD', 'NGN', 'KES', 'GHS', 'ZAR', 'ETB', 'XOF'],
                index=0
            )

            include_weather = st.checkbox(
                "Include Weather Considerations",
                value=True,
                help="Include current weather and forecast in recommendations"
            )

        with col2:
            optimization_goal = st.selectbox(
                "Optimization Goal:",
                options=['cost_effective', 'maximum_yield', 'balanced'],
                format_func=lambda x: {
                    'cost_effective': 'Cost Effective',
                    'maximum_yield': 'Maximum Yield',
                    'balanced': 'Balanced Approach'
                }[x],
                index=2
            )

            report_language = st.selectbox(
                "Report Language:",
                options=['en', 'fr', 'sw'],
                format_func=lambda x: {
                    'en': 'English',
                    'fr': 'Français',
                    'sw': 'Kiswahili'
                }[x]
            )

        # Generate recommendation button
        if st.button("🚀 Generate Fertilizer Recommendation", type="primary", use_container_width=True):
            with st.spinner("Generating intelligent fertilizer recommendation..."):
                try:
                    # Create soil analysis object
                    soil_analysis = SoilAnalysis(**st.session_state.soil_analysis_data)

                    # Create crop selection object
                    crop_selection = CropSelection(**st.session_state.crop_selection_data)

                    # Get region data
                    region_data = st.session_state.selected_region

                    # Generate recommendation
                    recommendation = self.fertilizer_engine.generate_recommendation(
                        soil_analysis=soil_analysis,
                        crop_selection=crop_selection,
                        region_data=region_data,
                        area_hectares=st.session_state.farm_details['area_hectares'],
                        target_yield=st.session_state.farm_details['target_yield'],
                        currency=currency
                    )

                    # Store recommendation
                    st.session_state.current_recommendation = recommendation
                    st.session_state.recommendation_generated = True

                    st.success("✅ Fertilizer recommendation generated successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")
                    st.info("Please check your inputs and try again.")

    def _render_recommendation_results(self):
        """Render recommendation results"""

        recommendation = st.session_state.current_recommendation

        st.header("📊 Fertilizer Recommendation Results")

        # Key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric(
                "Expected Yield",
                f"{recommendation.expected_yield:.1f} tons/ha",
                delta=f"+{recommendation.expected_yield - recommendation.target_yield:.1f}"
            )

        with metric_col2:
            st.metric(
                "Total Cost",
                f"{recommendation.cost_analysis.currency} {recommendation.cost_analysis.total_cost:.2f}"
            )

        with metric_col3:
            st.metric(
                "Cost per Hectare",
                f"{recommendation.cost_analysis.currency} {recommendation.cost_analysis.cost_per_hectare:.2f}"
            )

        with metric_col4:
            st.metric(
                "ROI",
                f"{recommendation.roi_percentage:.1f}%",
                delta=f"{recommendation.roi_percentage:.1f}%"
            )

        # Tabbed results display
        result_tabs = st.tabs([
            "📋 Summary",
            "🧪 Nutrient Analysis",
            "💰 Cost Breakdown",
            "📅 Application Schedule",
            "🌍 Regional Insights",
            "⚠️ Risk Assessment"
        ])

        with result_tabs[0]:  # Summary
            self._render_summary_tab(recommendation)

        with result_tabs[1]:  # Nutrient Analysis
            self._render_nutrient_analysis_tab(recommendation)

        with result_tabs[2]:  # Cost Breakdown
            self._render_cost_breakdown_tab(recommendation)

        with result_tabs[3]:  # Application Schedule
            self._render_application_schedule_tab(recommendation)

        with result_tabs[4]:  # Regional Insights
            self._render_regional_insights_tab(recommendation)

        with result_tabs[5]:  # Risk Assessment
            self._render_risk_assessment_tab(recommendation)

    def _render_summary_tab(self, recommendation):
        """Render recommendation summary tab"""

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Recommendation Overview")

            nutrient_balance = recommendation.nutrient_balance

            st.write(f"**Total Nitrogen Required:** {nutrient_balance.total_n:.1f} kg/ha")
            st.write(f"**Total Phosphorus Required:** {nutrient_balance.total_p:.1f} kg/ha")
            st.write(f"**Total Potassium Required:** {nutrient_balance.total_k:.1f} kg/ha")

            # Recommended fertilizers
            st.markdown("**Recommended Fertilizers:**")
            for fert in recommendation.recommended_fertilizers:
                st.write(f"• {fert.name} ({fert.n_content}-{fert.p_content}-{fert.k_content})")

        with col2:
            st.subheader("📈 Expected Outcomes")

            # Create yield comparison chart
            yield_data = {
                'Scenario': ['Without Fertilizer', 'With Recommendation'],
                'Yield (tons/ha)': [recommendation.target_yield * 0.7, recommendation.expected_yield]
            }

            fig = px.bar(
                yield_data,
                x='Scenario',
                y='Yield (tons/ha)',
                title='Yield Comparison',
                color='Scenario',
                color_discrete_sequence=['#ff7f7f', '#2E8B57']
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_nutrient_analysis_tab(self, recommendation):
        """Render nutrient analysis tab"""

        st.subheader("🧪 Nutrient Balance Analysis")

        nutrient_balance = recommendation.nutrient_balance

        # Primary nutrients chart
        nutrients_data = {
            'Nutrient': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
            'Required (kg/ha)': [
                nutrient_balance.total_n,
                nutrient_balance.total_p,
                nutrient_balance.total_k
            ]
        }

        fig = px.bar(
            nutrients_data,
            x='Nutrient',
            y='Required (kg/ha)',
            title='Primary Nutrient Requirements',
            color='Nutrient',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )

        st.plotly_chart(fig, use_container_width=True)

        # Secondary and micronutrients
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Secondary Nutrients (kg/ha)**")
            secondary = nutrient_balance.secondary_nutrients
            for nutrient, amount in secondary.items():
                if amount > 0:
                    st.write(f"• {nutrient.upper()}: {amount:.1f}")

        with col2:
            st.markdown("**Micronutrients (kg/ha)**")
            micro = nutrient_balance.micronutrients
            for nutrient, amount in micro.items():
                if amount > 0:
                    st.write(f"• {nutrient.upper()}: {amount:.1f}")

    def _render_cost_breakdown_tab(self, recommendation):
        """Render cost breakdown tab"""

        st.subheader("💰 Cost Analysis")

        cost_analysis = recommendation.cost_analysis

        col1, col2 = st.columns(2)

        with col1:
            # Cost breakdown pie chart
            breakdown = cost_analysis.fertilizer_breakdown
            if breakdown:
                fig = px.pie(
                    values=list(breakdown.values()),
                    names=list(breakdown.keys()),
                    title="Cost Breakdown by Fertilizer Type"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cost summary table
            cost_data = [
                ["Total Program Cost", f"{cost_analysis.currency} {cost_analysis.total_cost:.2f}"],
                ["Cost per Hectare", f"{cost_analysis.currency} {cost_analysis.cost_per_hectare:.2f}"],
                ["Expected Revenue", f"{cost_analysis.currency} {recommendation.expected_yield * 250:.2f}"],
                ["Net Profit", f"{cost_analysis.currency} {(recommendation.expected_yield * 250) - cost_analysis.total_cost:.2f}"],
                ["ROI", f"{recommendation.roi_percentage:.1f}%"]
            ]

            cost_df = pd.DataFrame(cost_data, columns=["Metric", "Value"])
            st.table(cost_df)

        # Detailed breakdown
        st.markdown("**Detailed Cost Breakdown:**")
        if cost_analysis.fertilizer_breakdown:
            breakdown_data = []
            for fert_type, cost in cost_analysis.fertilizer_breakdown.items():
                breakdown_data.append({
                    'Fertilizer Type': fert_type.replace('_', ' ').title(),
                    'Cost': f"{cost_analysis.currency} {cost:.2f}",
                    'Percentage': f"{(cost/cost_analysis.total_cost)*100:.1f}%"
                })

            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)

    def _render_application_schedule_tab(self, recommendation):
        """Render application schedule tab"""

        st.subheader("📅 Fertilizer Application Schedule")

        application_schedule = recommendation.application_schedule

        if application_schedule:
            # Schedule table
            schedule_data = []
            for app in application_schedule:
                schedule_data.append({
                    'Growth Stage': app.stage.replace('_', ' ').title(),
                    'Days After Planting': app.days_after_planting,
                    'Fertilizer': app.fertilizer_type,
                    'Rate (kg/ha)': f"{app.amount_kg_per_ha:.1f}",
                    'Method': app.application_method.replace('_', ' ').title(),
                    'Notes': app.notes
                })

            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(schedule_df, use_container_width=True)

            # Timeline visualization
            fig = go.Figure()

            for i, app in enumerate(application_schedule):
                fig.add_trace(go.Scatter(
                    x=[app.days_after_planting],
                    y=[app.amount_kg_per_ha],
                    mode='markers+text',
                    text=[app.stage.replace('_', ' ').title()],
                    textposition="top center",
                    marker=dict(size=12, color=f'rgba({50+i*50}, {100+i*30}, {150+i*20}, 0.8)'),
                    name=app.fertilizer_type
                ))

            fig.update_layout(
                title="Application Timeline",
                xaxis_title="Days After Planting",
                yaxis_title="Application Rate (kg/ha)",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No application schedule generated.")

    def _render_regional_insights_tab(self, recommendation):
        """Render regional insights tab"""

        st.subheader("🌍 Regional Considerations")

        # Climate considerations
        climate_considerations = recommendation.climate_considerations
        if climate_considerations:
            st.markdown("**Climate Considerations:**")
            for consideration in climate_considerations:
                st.info(f"🌤️ {consideration}")

        # Regional information
        region_data = st.session_state.selected_region

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Regional Information:**")
            st.write(f"• **Region:** {region_data['name']}")
            st.write(f"• **Climate Type:** {region_data.get('climate_type', 'N/A').replace('_', ' ').title()}")
            st.write(f"• **Rainfall Pattern:** {region_data.get('rainfall_pattern', 'N/A').replace('_', ' ').title()}")
            st.write(f"• **Currency:** {region_data.get('currency', 'USD')}")

            subsidies = "Available" if region_data.get('fertilizer_subsidies', False) else "Not Available"
            st.write(f"• **Fertilizer Subsidies:** {subsidies}")

        with col2:
            st.markdown("**Major Crops in Region:**")
            major_crops = region_data.get('major_crops', [])
            for crop in major_crops:
                is_selected = crop == recommendation.crop_selection.crop_type.lower()
                icon = "✅" if is_selected else "•"
                st.write(f"{icon} {crop.replace('_', ' ').title()}")

        # Alternative options
        alternative_options = recommendation.alternative_options
        if alternative_options:
            st.markdown("**Alternative Options:**")
            for option in alternative_options:
                st.success(f"💡 {option}")

    def _render_risk_assessment_tab(self, recommendation):
        """Render risk assessment tab"""

        st.subheader("⚠️ Risk Assessment & Management")

        risk_factors = recommendation.risk_factors

        if risk_factors:
            st.markdown("**Identified Risk Factors:**")
            for risk in risk_factors:
                st.warning(f"⚠️ {risk}")
        else:
            st.success("✅ No significant risk factors identified.")

        # Weather-based risks if location provided
        farm_details = st.session_state.get('farm_details', {})
        if farm_details.get('latitude', 0) != 0 or farm_details.get('longitude', 0) != 0:
            st.markdown("**Weather-Based Risk Assessment:**")

            agricultural_indices = self.weather_client.get_agricultural_indices(
                farm_details['latitude'],
                farm_details['longitude']
            )

            col1, col2 = st.columns(2)

            with col1:
                drought_risk = agricultural_indices.get('drought_risk', {})
                risk_level = drought_risk.get('risk_level', 'unknown')

                if risk_level == 'high':
                    st.error(f"🌵 **Drought Risk:** {risk_level.title()}")
                elif risk_level == 'medium':
                    st.warning(f"🌦️ **Drought Risk:** {risk_level.title()}")
                else:
                    st.success(f"💧 **Drought Risk:** {risk_level.title()}")

                st.write(drought_risk.get('recommendation', ''))

            with col2:
                leaching_risk = agricultural_indices.get('leaching_risk', {})
                risk_level = leaching_risk.get('risk_level', 'unknown')

                if risk_level == 'high':
                    st.error(f"🌊 **Leaching Risk:** {risk_level.title()}")
                elif risk_level == 'medium':
                    st.warning(f"💧 **Leaching Risk:** {risk_level.title()}")
                else:
                    st.success(f"🛡️ **Leaching Risk:** {risk_level.title()}")

                st.write(leaching_risk.get('recommendation', ''))

        # Risk mitigation strategies
        st.markdown("**Risk Mitigation Strategies:**")

        mitigation_strategies = [
            "Regular soil testing to monitor nutrient levels",
            "Split fertilizer applications to reduce loss",
            "Monitor weather conditions before application",
            "Consider organic matter amendments for soil health",
            "Implement integrated pest management practices",
            "Maintain proper fertilizer storage conditions"
        ]

        for strategy in mitigation_strategies:
            st.info(f"🛡️ {strategy}")

    def _render_weather_monitoring_tab(self):
        """Render weather monitoring interface"""

        st.header("🌤️ Weather Monitoring & Forecast")

        # Location input
        col1, col2 = st.columns(2)

        with col1:
            lat = st.number_input("Latitude:", value=0.0, step=0.1, format="%.6f")
        with col2:
            lon = st.number_input("Longitude:", value=0.0, step=0.1, format="%.6f")

        if lat != 0.0 or lon != 0.0:
            # Current weather
            current_weather = self.weather_client.get_current_weather(lat, lon)

            if current_weather:
                st.subheader("🌡️ Current Conditions")

                weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

                with weather_col1:
                    st.metric("Temperature", f"{current_weather['temperature']:.1f}°C")
                    st.metric("Humidity", f"{current_weather['humidity']:.0f}%")

                with weather_col2:
                    st.metric("Wind Speed", f"{current_weather['wind_speed']:.1f} m/s")
                    st.metric("Pressure", f"{current_weather['pressure']:.0f} hPa")

                with weather_col3:
                    st.metric("Precipitation", f"{current_weather['precipitation']:.1f} mm")
                    st.metric("UV Index", f"{current_weather['uv_index']:.0f}")

                with weather_col4:
                    st.write(f"**Condition:** {current_weather['description'].title()}")
                    st.write(f"**Visibility:** {current_weather['visibility']/1000:.1f} km")

            # Agricultural indices
            agricultural_indices = self.weather_client.get_agricultural_indices(lat, lon)

            st.subheader("🌾 Agricultural Weather Indices")

            indices_col1, indices_col2 = st.columns(2)

            with indices_col1:
                # Growing degree days
                gdd = agricultural_indices.get('growing_degree_days', {})
                st.metric("Growing Degree Days", f"{gdd.get('total_accumulated', 0):.1f}")

                # Evapotranspiration
                et = agricultural_indices.get('evapotranspiration', {})
                st.metric("ET (14-day)", f"{et.get('total_et_mm', 0):.1f} mm")

            with indices_col2:
                # Application suitability
                app_suit = agricultural_indices.get('fertilizer_application_suitability', {})
                suitability = app_suit.get('overall_suitability', 0)

                if suitability >= 4:
                    st.success(f"✅ Application Suitability: {suitability:.1f}/5")
                elif suitability >= 3:
                    st.warning(f"⚠️ Application Suitability: {suitability:.1f}/5")
                else:
                    st.error(f"❌ Application Suitability: {suitability:.1f}/5")

                best_day = app_suit.get('best_application_day', 1)
                st.info(f"Best application day: Day {best_day}")

            # Forecast
            forecast = self.weather_client.get_weather_forecast(lat, lon, 7)

            if forecast and forecast.get('daily_forecast'):
                st.subheader("📅 7-Day Forecast")

                forecast_data = []
                for day in forecast['daily_forecast']:
                    forecast_data.append({
                        'Date': day['date'].strftime('%Y-%m-%d'),
                        'Min Temp (°C)': day['temp_min'],
                        'Max Temp (°C)': day['temp_max'],
                        'Humidity (%)': day['humidity'],
                        'Precipitation (mm)': day['precipitation'],
                        'Condition': day['description'].title()
                    })

                forecast_df = pd.DataFrame(forecast_data)
                st.dataframe(forecast_df, use_container_width=True)

                # Temperature trend chart
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Max Temp (°C)'],
                    mode='lines+markers',
                    name='Max Temperature',
                    line=dict(color='red')
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Min Temp (°C)'],
                    mode='lines+markers',
                    name='Min Temperature',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    title="Temperature Trend",
                    xaxis_title="Date",
                    yaxis_title="Temperature (°C)"
                )

                st.plotly_chart(fig, use_container_width=True)

    def _render_iot_dashboard_tab(self):
        """Render IoT sensor dashboard"""

        st.header("📊 IoT Sensor Dashboard")

        # Get sensor data
        sensor_data = self.iot_simulator.get_all_current_readings()
        agricultural_summary = self.iot_simulator.get_agricultural_summary()

        if sensor_data and sensor_data.get('sensors'):
            # Real-time metrics
            st.subheader("📡 Real-time Sensor Readings")

            sensors = sensor_data['sensors']

            sensor_col1, sensor_col2, sensor_col3, sensor_col4 = st.columns(4)

            with sensor_col1:
                if 'soil_moisture' in sensors:
                    moisture = sensors['soil_moisture']
                    st.metric(
                        "Soil Moisture",
                        f"{moisture['value']:.1f}%",
                        help=f"Battery: {moisture.get('battery_level', 'Unknown')}%"
                    )

                if 'soil_ph' in sensors:
                    ph = sensors['soil_ph']
                    st.metric("Soil pH", f"{ph['value']:.1f}")

            with sensor_col2:
                if 'soil_temperature' in sensors:
                    temp = sensors['soil_temperature']
                    st.metric("Soil Temperature", f"{temp['value']:.1f}°C")

                if 'ambient_temperature' in sensors:
                    air_temp = sensors['ambient_temperature']
                    st.metric("Air Temperature", f"{air_temp['value']:.1f}°C")

            with sensor_col3:
                if 'humidity' in sensors:
                    humidity = sensors['humidity']
                    st.metric("Air Humidity", f"{humidity['value']:.1f}%")

                if 'wind_speed' in sensors:
                    wind = sensors['wind_speed']
                    st.metric("Wind Speed", f"{wind['value']:.1f} m/s")

            with sensor_col4:
                if 'soil_ec' in sensors:
                    ec = sensors['soil_ec']
                    st.metric("Soil EC", f"{ec['value']:.1f} dS/m")

                if 'atmospheric_pressure' in sensors:
                    pressure = sensors['atmospheric_pressure']
                    st.metric("Pressure", f"{pressure['value']:.0f} hPa")

            # Agricultural summary
            st.subheader("🌾 Agricultural Conditions Summary")

            soil_conditions = agricultural_summary.get('soil_conditions', {})
            environmental = agricultural_summary.get('environmental_conditions', {})
            indices = agricultural_summary.get('agricultural_indices', {})

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                irrigation_need = soil_conditions.get('irrigation_need', 'unknown')
                if irrigation_need == 'high':
                    st.error(f"💧 **Irrigation Need:** {irrigation_need.title()}")
                elif irrigation_need == 'medium':
                    st.warning(f"💧 **Irrigation Need:** {irrigation_need.title()}")
                else:
                    st.success(f"💧 **Irrigation Need:** {irrigation_need.title()}")

            with summary_col2:
                growing_conditions = indices.get('growing_conditions', 'unknown')
                if growing_conditions == 'optimal':
                    st.success(f"🌱 **Growing Conditions:** {growing_conditions.title()}")
                elif growing_conditions == 'good':
                    st.info(f"🌱 **Growing Conditions:** {growing_conditions.title()}")
                else:
                    st.warning(f"🌱 **Growing Conditions:** {growing_conditions.title()}")

            with summary_col3:
                fert_suitability = indices.get('fertilizer_application_suitability', 'unknown')
                if fert_suitability == 'suitable':
                    st.success(f"🚜 **Fertilizer Application:** {fert_suitability.title()}")
                elif fert_suitability == 'moderate':
                    st.warning(f"🚜 **Fertilizer Application:** {fert_suitability.title()}")
                else:
                    st.error(f"🚜 **Fertilizer Application:** {fert_suitability.title()}")

            # Sensor history charts
            st.subheader("📈 Sensor Trends (Last 24 Hours)")

            # Generate history for key sensors
            history_sensors = ['soil_moisture', 'soil_temperature', 'soil_ph']

            for sensor in history_sensors:
                if sensor in sensors:
                    history = self.iot_simulator.get_sensor_history(sensor, 24)

                    if history:
                        df = pd.DataFrame(history)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                        fig = px.line(
                            df,
                            x='timestamp',
                            y='value',
                            title=f"{sensor.replace('_', ' ').title()} Trend",
                            labels={'value': f"Value ({history[0].get('unit', '')})", 'timestamp': 'Time'}
                        )

                        st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            recommendations = agricultural_summary.get('recommendations', [])
            if recommendations:
                st.subheader("💡 IoT-Based Recommendations")
                for rec in recommendations:
                    st.info(f"• {rec}")

            # Active alerts
            alerts = self.iot_simulator.get_active_alerts()
            if alerts:
                st.subheader("🚨 Active Alerts")
                for alert in alerts:
                    severity = alert.get('severity', 'info')
                    if severity == 'warning':
                        st.warning(f"⚠️ {alert['message']} (Value: {alert['value']})")
                    elif severity == 'error':
                        st.error(f"🚨 {alert['message']} (Value: {alert['value']})")
                    else:
                        st.info(f"ℹ️ {alert['message']} (Value: {alert['value']})")
        else:
            st.warning("⚠️ No IoT sensor data available. Check sensor connectivity.")

            # Simulate some data for demo
            if st.button("🔄 Refresh Sensor Data"):
                st.rerun()

    def _render_data_analysis_tab(self):
        """Render data analysis and insights tab"""

        st.header("📈 Data Analysis & Insights")

        # Load sample data for analysis
        try:
            with open('data/yield_training_data.csv', 'r') as f:
                yield_data = pd.read_csv(f)

            st.subheader("🌾 Yield Performance Analysis")

            # Yield trends by region
            if 'region' in yield_data.columns and 'yield_tons_per_ha' in yield_data.columns:
                yield_by_region = yield_data.groupby('region')['yield_tons_per_ha'].mean().reset_index()

                fig = px.bar(
                    yield_by_region,
                    x='region',
                    y='yield_tons_per_ha',
                    title='Average Yield by Region',
                    color='yield_tons_per_ha',
                    color_continuous_scale='Greens'
                )

                st.plotly_chart(fig, use_container_width=True)

            # Correlation analysis
            if len(yield_data.columns) > 3:
                st.subheader("🔗 Correlation Analysis")

                numeric_columns = yield_data.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_columns) > 1:
                    correlation_matrix = yield_data[numeric_columns].corr()

                    fig = px.imshow(
                        correlation_matrix,
                        title="Variable Correlation Matrix",
                        aspect="auto",
                        color_continuous_scale='RdBu'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Data summary
            st.subheader("📊 Data Summary")
            st.dataframe(yield_data.describe(), use_container_width=True)

        except FileNotFoundError:
            st.info("📁 Training data not available for analysis.")

        # Fertilizer efficiency analysis
        st.subheader("💊 Fertilizer Efficiency Insights")

        if st.session_state.current_recommendation:
            recommendation = st.session_state.current_recommendation

            # Efficiency metrics
            efficiency_data = {
                'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
                'Required (kg/ha)': [
                    recommendation.nutrient_balance.total_n,
                    recommendation.nutrient_balance.total_p,
                    recommendation.nutrient_balance.total_k
                ],
                'Cost per kg': [2.5, 4.0, 3.0],  # Example costs
            }

            efficiency_df = pd.DataFrame(efficiency_data)
            efficiency_df['Cost Efficiency'] = efficiency_df['Required (kg/ha)'] / efficiency_df['Cost per kg']

            fig = px.scatter(
                efficiency_df,
                x='Required (kg/ha)',
                y='Cost per kg',
                size='Cost Efficiency',
                color='Nutrient',
                title='Nutrient Cost vs. Requirement Analysis',
                hover_data=['Cost Efficiency']
            )

            st.plotly_chart(fig, use_container_width=True)

        # Regional comparison
        st.subheader("🌍 Regional Performance Insights")

        # Mock regional data for visualization
        regional_performance = {
            'Region': ['West Africa', 'East Africa', 'Southern Africa', 'Central Africa'],
            'Avg Yield (tons/ha)': [4.2, 5.1, 3.8, 3.5],
            'Fertilizer Cost (USD/ha)': [180, 220, 160, 140],
            'ROI (%)': [15.2, 18.5, 12.8, 10.9]
        }

        regional_df = pd.DataFrame(regional_performance)

        fig = px.scatter(
            regional_df,
            x='Fertilizer Cost (USD/ha)',
            y='Avg Yield (tons/ha)',
            size='ROI (%)',
            color='Region',
            title='Regional Performance: Cost vs. Yield',
            hover_data=['ROI (%)']
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_help_support_tab(self):
        """Render help and support tab"""

        st.header("❓ Help & Support")

        # FAQ Section
        st.subheader("🤔 Frequently Asked Questions")

        faqs = [
            {
                "question": "How accurate are the fertilizer recommendations?",
                "answer": "Our recommendations are based on established agronomic principles and STCR methodology. Accuracy depends on the quality of soil test data provided. For best results, use recent laboratory soil test results."
            },
            {
                "question": "Can I use this system for organic farming?",
                "answer": "Yes! The system includes organic fertilizer options and can generate recommendations for organic farming systems. Select organic fertilizers in the advanced options."
            },
            {
                "question": "How often should I update my soil test data?",
                "answer": "We recommend updating soil test data annually, or every two years minimum. More frequent testing may be needed for intensive farming systems."
            },
            {
                "question": "What if my region is not listed?",
                "answer": "The system uses the closest regional data available. Contact support to add your specific region, or use the region with similar climate and soil conditions."
            },
            {
                "question": "How do I interpret the ROI calculation?",
                "answer": "ROI (Return on Investment) shows the expected profit percentage from fertilizer investment. It considers fertilizer costs, expected yield increase, and local crop prices."
            }
        ]

        for faq in faqs:
            with st.expander(f"❓ {faq['question']}"):
                st.write(faq['answer'])

        # User guides
        st.subheader("📚 User Guides")

        guides = [
            "🌱 **Getting Started Guide**: Learn how to use the Smart Fertilizer system",
            "🧪 **Soil Testing Guide**: How to collect and test soil samples",
            "💰 **Cost Optimization**: Tips for reducing fertilizer costs",
            "📊 **Interpreting Results**: Understanding your recommendations",
            "🌍 **Regional Considerations**: Local factors affecting fertilizer use"
        ]

        for guide in guides:
            st.info(guide)

        # Contact information
        st.subheader("📞 Contact & Support")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Technical Support**
            - 📧 Email: support@smartfertilizer.org
            - 📱 WhatsApp: +234-XXX-XXXX-XXX
            - 🕐 Hours: 8 AM - 6 PM WAT
            """)

        with col2:
            st.markdown("""
            **Agricultural Extension**
            - 🏢 Partner Organizations: FAO, ICAR, ICRISAT
            - 🌐 Website: www.smartfertilizer.org
            - 📱 Mobile App: Coming Soon
            """)

        # Feedback form
        st.subheader("💬 Feedback")

        with st.form("feedback_form"):
            feedback_type = st.selectbox(
                "Feedback Type:",
                options=['suggestion', 'bug_report', 'feature_request', 'general'],
                format_func=lambda x: x.replace('_', ' ').title()
            )

            feedback_text = st.text_area(
                "Your Feedback:",
                height=100,
                placeholder="Please share your thoughts, suggestions, or report any issues..."
            )

            contact_email = st.text_input(
                "Email (optional):",
                placeholder="your.email@example.com"
            )

            if st.form_submit_button("Submit Feedback"):
                if feedback_text:
                    st.success("✅ Thank you for your feedback! We'll review it and get back to you.")
                    # In production, this would actually send the feedback
                else:
                    st.error("Please enter your feedback before submitting.")

        # System information
        with st.expander("ℹ️ System Information"):
            st.write("**Smart Fertilizer System v1.0**")
            st.write("**Last Updated:** December 2024")
            st.write("**Supported Regions:** West Africa, East Africa, Southern Africa")
            st.write("**Data Sources:** FAO, ESDAC, ICAR/ICRISAT, NOAA/CHIRPS")
            st.write("**Methodology:** STCR (Soil Test Crop Response)")

    def _get_current_step(self) -> int:
        """Get current step in the recommendation process"""

        if not st.session_state.selected_region:
            return 0
        elif not st.session_state.soil_analysis_data:
            return 1
        elif not st.session_state.get('farm_details'):
            return 2
        elif not st.session_state.recommendation_generated:
            return 3
        else:
            return 4

    def _all_data_collected(self) -> bool:
        """Check if all required data has been collected"""

        return (
            st.session_state.selected_region is not None and
            bool(st.session_state.soil_analysis_data) and
            bool(st.session_state.crop_selection_data) and
            bool(st.session_state.get('farm_details'))
        )

    def _display_soil_interpretation(self):
        """Display soil analysis interpretation"""

        soil_data = st.session_state.soil_analysis_data

        st.subheader("🔍 Soil Analysis Interpretation")

        # pH interpretation
        ph = soil_data.get('ph', 6.5)
        if ph < 5.5:
            st.warning(f"🔴 **pH ({ph:.1f})**: Acidic - may limit nutrient availability. Consider liming.")
        elif ph > 7.5:
            st.warning(f"🔵 **pH ({ph:.1f})**: Alkaline - may cause micronutrient deficiencies.")
        else:
            st.success(f"✅ **pH ({ph:.1f})**: Optimal range for most crops.")

        # Organic matter interpretation
        om = soil_data.get('organic_matter', 3.0)
        if om < 2.0:
            st.warning(f"⚠️ **Organic Matter ({om:.1f}%)**: Low - consider organic amendments.")
        elif om > 4.0:
            st.success(f"✅ **Organic Matter ({om:.1f}%)**: High - excellent for soil health.")
        else:
            st.info(f"ℹ️ **Organic Matter ({om:.1f}%)**: Moderate - adequate for production.")

        # Nutrient status
        nutrients = {
            'Nitrogen': (soil_data.get('nitrogen', 250), 200, 400),
            'Phosphorus': (soil_data.get('phosphorus', 25), 10, 50),
            'Potassium': (soil_data.get('potassium', 200), 100, 400)
        }

        for nutrient, (value, low_thresh, high_thresh) in nutrients.items():
            if value < low_thresh:
                st.error(f"🔴 **{nutrient} ({value:.0f} ppm)**: Low - fertilizer application recommended.")
            elif value > high_thresh:
                st.info(f"🔵 **{nutrient} ({value:.0f} ppm)**: High - reduce application rates.")
            else:
                st.success(f"✅ **{nutrient} ({value:.0f} ppm)**: Adequate levels.")

    def _handle_export(self, format_type: str):
        """Handle export functionality"""

        if not st.session_state.current_recommendation:
            st.error("No recommendation available to export.")
            return

        try:
            recommendation = st.session_state.current_recommendation

            # Generate export
            if format_type == 'pdf':
                pdf_bytes = self.pdf_generator.generate_fertilizer_report(recommendation)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"fertilizer_recommendation_{recommendation.recommendation_id}.pdf",
                    mime="application/pdf"
                )
            else:
                export_data = self.export_utils.export_recommendation(
                    recommendation.__dict__, format_type
                )

                if isinstance(export_data, bytes):
                    st.download_button(
                        label=f"📄 Download {format_type.upper()} Report",
                        data=export_data,
                        file_name=f"fertilizer_recommendation_{recommendation.recommendation_id}.{format_type}",
                        mime=f"application/{format_type}"
                    )
                else:
                    st.download_button(
                        label=f"📄 Download {format_type.upper()} Report",
                        data=export_data,
                        file_name=f"fertilizer_recommendation_{recommendation.recommendation_id}.{format_type}",
                        mime="text/plain"
                    )

            st.success(f"✅ {format_type.upper()} report ready for download!")

        except Exception as e:
            st.error(f"Error generating {format_type} export: {str(e)}")
