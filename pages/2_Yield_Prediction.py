import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from utils.ml_models import YieldPredictor, prepare_prediction_data
from utils.data_processing1 import get_sample_agricultural_data
from utils import AdvancedAI


st.set_page_config(page_title="Yield Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Crop Yield Prediction")
st.markdown("### ML-powered agricultural yield forecasting")

# Initialize predictor
if 'yield_predictor' not in st.session_state:
    st.session_state.yield_predictor = YieldPredictor()

predictor = st.session_state.yield_predictor

# Sidebar for model configuration
st.sidebar.title("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Random Forest", "Linear Regression", "XGBoost"],
    help="Choose the machine learning model for predictions"
)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Make Prediction", "Model Training", "Model Performance", "Historical Analysis"])

with tab1:
    st.subheader("Generate Yield Prediction")

    # Input form for prediction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Crop Information**")
            crop_type = st.selectbox(
                "Crop Type",
                ["Wheat", "Corn", "Rice", "Soybeans", "Barley", "Cotton"],
                help="Select the type of crop to predict"
            )

            area = st.number_input(
                "Area (hectares)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0,
                step=0.1,
                help="Total area to be cultivated"
            )

            soil_ph = st.slider(
                "Soil pH",
                min_value=4.0,
                max_value=9.0,
                value=6.5,
                step=0.1,
                help="Soil pH level"
            )

            soil_nitrogen = st.number_input(
                "Soil Nitrogen (ppm)",
                min_value=0,
                max_value=200,
                value=50,
                help="Nitrogen content in soil"
            )

        with col2:
            st.markdown("**Environmental Conditions**")
            temperature = st.number_input(
                "Average Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=25.0,
                step=0.1,
                help="Average growing season temperature"
            )

            rainfall = st.number_input(
                "Total Rainfall (mm)",
                min_value=0,
                max_value=3000,
                value=800,
                help="Total rainfall during growing season"
            )

            humidity = st.slider(
                "Average Humidity (%)",
                min_value=20,
                max_value=100,
                value=65,
                help="Average relative humidity"
            )

            sunlight = st.slider(
                "Sunlight Hours/Day",
                min_value=6,
                max_value=14,
                value=10,
                help="Average daily sunlight hours"
            )

        submitted = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)

        if submitted:
            # Prepare input data
            input_data = {
                'crop_type': crop_type,
                'area': area,
                'soil_ph': soil_ph,
                'soil_nitrogen': soil_nitrogen,
                'temperature': temperature,
                'rainfall': rainfall,
                'humidity': humidity,
                'sunlight': sunlight
            }

            # Make prediction
            try:
                prediction_result = predictor.predict(input_data, model_type.lower().replace(" ", "_"))

                if prediction_result:
                    st.success("Prediction Generated Successfully!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Predicted Yield",
                            f"{prediction_result['yield']:.2f} tons/ha",
                            help="Expected yield per hectare"
                        )

                    with col2:
                        st.metric(
                            "Total Production",
                            f"{prediction_result['total_production']:.1f} tons",
                            help="Total expected production for the given area"
                        )

                    with col3:
                        st.metric(
                            "Confidence Score",
                            f"{prediction_result['confidence']:.1f}%",
                            help="Model confidence in the prediction"
                        )

                    # Additional insights
                    st.markdown("---")
                    st.subheader("Prediction Insights")

                    # Risk assessment
                    if prediction_result['yield'] < 2.0:
                        st.warning("⚠️ Low yield predicted. Consider reviewing soil conditions and weather factors.")
                    elif prediction_result['yield'] > 8.0:
                        st.success("🌟 High yield predicted! Excellent growing conditions.")
                    else:
                        st.info("📊 Moderate yield predicted. Standard agricultural practices recommended.")

                    # Recommendations
                    st.markdown("**Recommendations:**")
                    recommendations = predictor.get_recommendations(input_data, prediction_result)
                    for rec in recommendations:
                        st.write(f"• {rec}")

                else:
                    st.error("Unable to generate prediction. Please check your inputs and try again.")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

with tab2:
    st.subheader("Model Training")

    if 'agricultural_data' in st.session_state:
        data = st.session_state.agricultural_data

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Configuration**")
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42)

            if st.button("🔄 Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    training_result = predictor.train_model(data, model_type, test_size, random_state)

                    if training_result:
                        st.success("Model trained successfully!")
                        st.session_state.model_trained = True
                        st.session_state.training_results = training_result
                    else:
                        st.error("Model training failed. Please check your data.")

        with col2:
            st.markdown("**Dataset Information**")
            st.write(f"Total records: {len(data)}")
            st.write(f"Features: {len(data.columns)}")

            if 'yield' in data.columns:
                st.write(f"Yield range: {data['yield'].min():.2f} - {data['yield'].max():.2f}")
                st.write(f"Average yield: {data['yield'].mean():.2f}")

    else:
        st.warning("No training data available. Please upload data first.")
        if st.button("Upload Training Data"):
            st.switch_page("pages/5_Data_Upload.py")

with tab3:
    st.subheader("Model Performance Metrics")

    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        results = st.session_state.training_results

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("R² Score", f"{results['r2_score']:.3f}")
        with col2:
            st.metric("MAE", f"{results['mae']:.3f}")
        with col3:
            st.metric("Training Accuracy", f"{results['training_accuracy']:.1f}%")

        # Performance visualization
        if 'predictions' in results and 'actual' in results:
            col1, col2 = st.columns(2)

            with col1:
                # Actual vs Predicted scatter plot
                fig_scatter = px.scatter(
                    x=results['actual'],
                    y=results['predictions'],
                    title="Actual vs Predicted Yield",
                    labels={'x': 'Actual Yield', 'y': 'Predicted Yield'}
                )
                # Add perfect prediction line
                min_val = min(min(results['actual']), min(results['predictions']))
                max_val = max(max(results['actual']), max(results['predictions']))
                fig_scatter.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                # Residuals plot
                residuals = np.array(results['actual']) - np.array(results['predictions'])
                fig_residuals = px.scatter(
                    x=results['predictions'],
                    y=residuals,
                    title="Residuals Plot",
                    labels={'x': 'Predicted Yield', 'y': 'Residuals'}
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)

        # Feature importance (if available)
        if 'feature_importance' in results:
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': results['feature_names'],
                'Importance': results['feature_importance']
            }).sort_values('Importance', ascending=False)

            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Yield Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

    else:
        st.info("No trained model available. Please train a model first in the Model Training tab.")

with tab4:
    st.subheader("Historical Yield Analysis")

    if 'agricultural_data' in st.session_state:
        data = st.session_state.agricultural_data

        # Time series analysis if date column exists
        if 'date' in data.columns and 'yield' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'])

                # Yearly trends
                yearly_yield = data.groupby(data['date'].dt.year)['yield'].agg(['mean', 'std']).reset_index()

                fig_yearly = go.Figure()
                fig_yearly.add_trace(go.Scatter(
                    x=yearly_yield['date'],
                    y=yearly_yield['mean'],
                    mode='lines+markers',
                    name='Average Yield',
                    line=dict(color='blue')
                ))

                # Add error bars
                fig_yearly.add_trace(go.Scatter(
                    x=yearly_yield['date'],
                    y=yearly_yield['mean'] + yearly_yield['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig_yearly.add_trace(go.Scatter(
                    x=yearly_yield['date'],
                    y=yearly_yield['mean'] - yearly_yield['std'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name='Standard Deviation'
                ))

                fig_yearly.update_layout(
                    title="Historical Yield Trends",
                    xaxis_title="Year",
                    yaxis_title="Yield (tons/ha)"
                )
                st.plotly_chart(fig_yearly, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing historical data: {str(e)}")

        # Crop performance comparison
        if 'crop_type' in data.columns and 'yield' in data.columns:
            crop_performance = data.groupby('crop_type')['yield'].agg(['mean', 'count']).reset_index()
            crop_performance.columns = ['Crop Type', 'Average Yield', 'Number of Records']

            col1, col2 = st.columns(2)

            with col1:
                fig_performance = px.bar(
                    crop_performance,
                    x='Crop Type',
                    y='Average Yield',
                    title="Average Yield by Crop Type"
                )
                st.plotly_chart(fig_performance, use_container_width=True)

            with col2:
                st.dataframe(crop_performance, use_container_width=True)

    else:
        st.warning("No historical data available for analysis.")
