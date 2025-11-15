
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from utils import AdvancedAI


st.set_page_config(page_title="Climate Forecasting", page_icon="📡", layout="wide")

st.title("📡 Advanced Climate Forecasting")
st.markdown("### AI combined with satellite & IoT data to anticipate climate events")

# Sidebar controls
st.sidebar.title("Forecast Configuration")

location = st.sidebar.text_input(
    "Location",
    value="Paris, France",
    help="City, region or GPS coordinates"
)

forecast_range = st.sidebar.selectbox(
    "Forecast horizon",
    ["7 days", "15 days", "1 month", "3 months", "Seasonal"]
)

alert_sensitivity = st.sidebar.select_slider(
    "Alert sensitivity",
    options=["Low", "Normal", "High"],
    value="Normal"
)

# Data sources
data_sources = st.sidebar.multiselect(
    "Data sources",
    [
        "Weather satellites (GOES, MSG)",
        "Local IoT sensors",
        "Numerical models (GFS, ECMWF)",
        "Weather stations",
        "Precipitation radars",
        "Ocean buoys"
    ],
    default=["Weather satellites (GOES, MSG)", "Local IoT sensors", "Numerical models (GFS, ECMWF)"]
)

# Mapbox token & style
mapbox_token = st.secrets.get("MAPBOX_TOKEN", None)
if mapbox_token:
    try:
        px.set_mapbox_access_token(mapbox_token)
    except Exception:
        pass

map_styles = ["open-street-map"]
if mapbox_token:
    map_styles += [
        "carto-positron",
        "carto-darkmatter",
        "stamen-terrain",
        "basic",
        "streets",
        "outdoors",
        "light",
        "dark",
        "satellite",
        "satellite-streets",
    ]

map_style = st.sidebar.selectbox("Map style", map_styles, index=0)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Real-Time Forecasts",
    "Climate Alerts",
    "Seasonal Analysis",
    "Agricultural Impact",
    "Predictive Models"
])

with tab1:
    st.subheader("Real-Time Weather Forecasts")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate simulated forecast data
        days = 15 if "15 days" in forecast_range else 7
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]

        # Simulated data
        temperatures = np.random.normal(22, 8, days)
        humidity = np.random.normal(65, 15, days)
        precipitation = np.random.exponential(2, days)
        wind_speed = np.random.gamma(2, 3, days)
        pressure = np.random.normal(1015, 10, days)

        # Create multi-variable chart
        fig = go.Figure()

        # Temperature
        fig.add_trace(go.Scatter(
            x=dates,
            y=temperatures,
            mode='lines+markers',
            name='Temperature (°C)',
            line=dict(color='red', width=3),
            yaxis='y'
        ))

        # Precipitation (bars)
        fig.add_trace(go.Bar(
            x=dates,
            y=precipitation,
            name='Precipitation (mm)',
            marker_color='blue',
            opacity=0.6,
            yaxis='y2'
        ))

        # Humidity
        fig.add_trace(go.Scatter(
            x=dates,
            y=humidity,
            mode='lines',
            name='Humidity (%)',
            line=dict(color='green', dash='dash'),
            yaxis='y3'
        ))

        fig.update_layout(
            title="Multi-Variable Weather Forecasts",
            xaxis_title="Date",
            yaxis=dict(title="Temperature (°C)", side="left", color="red"),
            yaxis2=dict(title="Precipitation (mm)", side="right", overlaying="y", color="blue"),
            yaxis3=dict(title="Humidity (%)", side="right", overlaying="y", position=0.9, color="green"),
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

# Weather map
        st.markdown("**Regional Weather Map**")

# Simulate a temperature map
        lat = np.random.uniform(45, 50, 100)
        lon = np.random.uniform(1, 6, 100)
        temp_map = np.random.normal(20, 5, 100)

        fig_map = px.scatter_mapbox(
            lat=lat,
            lon=lon,
            color=temp_map,
            size=abs(temp_map - 20) + 5,
            hover_name=[f"Station {i}" for i in range(100)],
            hover_data={"Temperature": temp_map},
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=6,
            height=400,
            title="Real-Time Regional Temperature"
        )

        fig_map.update_layout(mapbox_style=map_style)
        st.plotly_chart(fig_map, use_container_width=True)


    with col2:
        st.markdown("**Current Conditions**")

        # Simulated current weather
        current_temp = np.random.normal(22, 5)
        current_humidity = np.random.uniform(40, 80)
        current_wind = np.random.uniform(5, 25)
        current_pressure = np.random.normal(1015, 8)

        # Weather metrics
        st.metric("🌡️ Temperature", f"{current_temp:.1f}°C", delta="2.3°C")
        st.metric("💧 Humidity", f"{current_humidity:.0f}%", delta="5%")
        st.metric("💨 Wind", f"{current_wind:.1f} km/h", delta="-3.2 km/h")
        st.metric("🧭 Pressure", f"{current_pressure:.0f} hPa", delta="2 hPa")

        st.markdown("**🎯 Agricultural Indices**")

        # Calculate agricultural indices
        gdd = max(0, current_temp - 10)  # Growing Degree Days
        et_rate = max(0, (current_temp - 5) * (1 - current_humidity/100) * 0.1)

        st.metric("🌱 GDD (Base 10°C)", f"{gdd:.1f}")
        st.metric("💦 Evapotranspiration", f"{et_rate:.2f} mm/day")

        # Simulated air quality
        air_quality_index = np.random.randint(25, 150)
        air_quality_status = "Good" if air_quality_index < 50 else "Moderate" if air_quality_index < 100 else "Poor"

        st.metric("🌬️ Air Quality", air_quality_status, f"AQI: {air_quality_index}")

        st.markdown("**📊 24h Trend**")

        # Mini trend chart
        hours = list(range(24))
        hourly_temp = current_temp + np.sin(np.array(hours) * np.pi / 12) * 5 + np.random.normal(0, 1, 24)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hours,
            y=hourly_temp,
            mode='lines',
            name='24h Temperature',
            line=dict(color='orange', width=2)
        ))

        fig_trend.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Hour",
            yaxis_title="°C"
        )

        st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Intelligent Climate Alert System")

    # Alert configuration
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Alert Configuration**")

        alert_types = st.multiselect(
            "Alert types to monitor",
            [
                "🌡️ Extreme temperatures",
                "🌧️ Heavy precipitation",
                "❄️ Frost risk",
                "🌪️ Strong winds",
                "⛈️ Severe storms",
                "🌵 Drought",
                "🌊 Flooding",
                "🌫️ Dense fog"
            ],
            default=["🌡️ Extreme temperatures", "❄️ Frost risk", "🌧️ Heavy precipitation"]
        )

        notification_methods = st.multiselect(
            "Notification methods",
            ["📧 Email", "📱 SMS", "🔔 Push", "📺 Dashboard", "📡 IoT Actuators"],
            default=["📧 Email", "📺 Dashboard"]
        )

        advance_warning = st.selectbox(
            "Desired advance warning",
            ["1 hour", "3 hours", "6 hours", "12 hours", "24 hours", "48 hours"],
            index=4
        )

    with col2:
        st.markdown("**Active Alerts**")

        # Simulate alerts
        active_alerts = [
            {
                "type": "❄️ Frost Alert",
                "severity": "🟡 Moderate",
                "time": "In 18h",
                "temp": "-2°C expected",
                "action": "Protect sensitive crops"
            },
            {
                "type": "🌧️ Heavy Rains",
                "severity": "🟠 High",
                "time": "In 6h",
                "amount": "25mm in 2h",
                "action": "Check drainage"
            },
            {
                "type": "💨 Strong Winds",
                "severity": "🟡 Moderate",
                "time": "In 12h",
                "speed": "45 km/h gusts",
                "action": "Secure equipment"
            }
        ]

        for alert in active_alerts:
            with st.container():
                st.markdown(f"**{alert['type']}** {alert['severity']}")
                st.write(f"⏰ {alert['time']}")
                st.write(f"📊 {alert.get('temp', alert.get('amount', alert.get('speed', 'N/A')))}")
                st.write(f"🎯 Action: {alert['action']}")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Acknowledge", key=f"ack_{alert['type']}"):
                        st.success("Alert acknowledged")
                with col_btn2:
                    if st.button("🔕 Disable", key=f"disable_{alert['type']}"):
                        st.info("Alert disabled")

                st.markdown("---")

    # Alert history
    st.markdown("**📈 Alert History (Last 30 days)**")

    # Generate history data
    alert_dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    alert_counts = np.random.poisson(1.5, 30)
    alert_severities = np.random.choice(['Low', 'Moderate', 'High', 'Critical'], 30, p=[0.4, 0.3, 0.2, 0.1])

    alert_df = pd.DataFrame({
        'Date': alert_dates,
        'Alert_Count': alert_counts,
        'Max_Severity': alert_severities
    })

    fig_alerts = px.bar(
        alert_df,
        x='Date',
        y='Alert_Count',
        color='Max_Severity',
        title="Weather Alert Evolution",
        color_discrete_map={
            'Low': 'green',
            'Moderate': 'yellow',
            'High': 'orange',
            'Critical': 'red'
        }
    )

    fig_alerts.update_layout(height=300)
    st.plotly_chart(fig_alerts, use_container_width=True)

with tab3:
    st.subheader("Seasonal Climate Analysis")

    # Analysis period selection
    col1, col2 = st.columns(2)

    with col1:
        analysis_year = st.selectbox(
            "Analysis year",
            [2024, 2023, 2022, 2021, 2020],
            index=0
        )

    with col2:
        comparison_year = st.selectbox(
            "Comparison year",
            [2023, 2022, 2021, 2020, 2019],
            index=0
        )

    # Simulated seasonal data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Monthly temperature
    temp_2024 = [2, 4, 9, 14, 18, 22, 25, 24, 20, 15, 8, 3]
    temp_2023 = [1, 3, 8, 13, 17, 21, 24, 23, 19, 14, 7, 2]

    # Monthly precipitation
    precip_2024 = [45, 38, 52, 67, 73, 45, 32, 48, 61, 78, 69, 54]
    precip_2023 = [52, 41, 48, 62, 68, 38, 28, 52, 58, 82, 74, 61]

    # Comparison charts
    fig_seasonal = go.Figure()

    # Temperatures
    fig_seasonal.add_trace(go.Scatter(
        x=months,
        y=temp_2024,
        mode='lines+markers',
        name=f'Temperature {analysis_year}',
        line=dict(color='red', width=3),
        yaxis='y'
    ))

    fig_seasonal.add_trace(go.Scatter(
        x=months,
        y=temp_2023,
        mode='lines+markers',
        name=f'Temperature {comparison_year}',
        line=dict(color='orange', width=2, dash='dash'),
        yaxis='y'
    ))

    # Precipitation
    fig_seasonal.add_trace(go.Bar(
        x=months,
        y=precip_2024,
        name=f'Precipitation {analysis_year}',
        marker_color='blue',
        opacity=0.7,
        yaxis='y2'
    ))

    fig_seasonal.add_trace(go.Bar(
        x=months,
        y=precip_2023,
        name=f'Precipitation {comparison_year}',
        marker_color='lightblue',
        opacity=0.5,
        yaxis='y2'
    ))

    fig_seasonal.update_layout(
        title="Seasonal Climate Comparison",
        xaxis_title="Month",
        yaxis=dict(title="Temperature (°C)", side="left"),
        yaxis2=dict(title="Precipitation (mm)", side="right", overlaying="y"),
        height=400
    )

    st.plotly_chart(fig_seasonal, use_container_width=True)

    # Anomaly analysis
    st.markdown("**🔍 Climate Anomaly Detection**")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp_anomaly = np.mean(temp_2024) - np.mean(temp_2023)
        st.metric(
            "Temperature Anomaly",
            f"{temp_anomaly:+.1f}°C",
            delta=f"vs {comparison_year}"
        )

    with col2:
        precip_anomaly = (np.sum(precip_2024) - np.sum(precip_2023)) / np.sum(precip_2023) * 100
        st.metric(
            "Precipitation Anomaly",
            f"{precip_anomaly:+.1f}%",
            delta=f"vs {comparison_year}"
        )

    with col3:
        extreme_days = np.random.randint(12, 28)
        st.metric(
            "Extreme Days",
            extreme_days,
            delta=f"+{extreme_days-20} vs normal"
        )

    # Climate projections
    st.markdown("**🔮 Climate Projections (AI)**")

    # Generate projections
    projection_months = ['Jan 2025', 'Feb 2025', 'Mar 2025', 'Apr 2025', 'May 2025', 'Jun 2025']
    projected_temp = [3, 5, 10, 15, 19, 23]
    confidence_bands = [1.5, 1.8, 2.1, 2.3, 2.0, 1.7]

    fig_projection = go.Figure()

    # Central projection
    fig_projection.add_trace(go.Scatter(
        x=projection_months,
        y=projected_temp,
        mode='lines+markers',
        name='AI Projection',
        line=dict(color='purple', width=3)
    ))

    # Confidence bands
    fig_projection.add_trace(go.Scatter(
        x=projection_months + projection_months[::-1],
        y=[t + c for t, c in zip(projected_temp, confidence_bands)] +
          [t - c for t, c in zip(projected_temp[::-1], confidence_bands[::-1])],
        fill='toself',
        fillcolor='rgba(128,0,128,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence Interval'
    ))

    fig_projection.update_layout(
        title="Temperature Projections - Next 6 Months",
        xaxis_title="Period",
        yaxis_title="Temperature (°C)",
        height=300
    )

    st.plotly_chart(fig_projection, use_container_width=True)

with tab4:
    st.subheader("Agricultural Impact Assessment")

    # Crop type selection
    crop_selection = st.selectbox(
        "Crop type to analyze",
        ["Wheat", "Corn", "Sunflower", "Sugar beet", "Rapeseed", "Barley", "Tomato", "Potato"]
    )

    growth_stage = st.selectbox(
        "Growth stage",
        ["Sowing/Planting", "Emergence", "Vegetative development", "Flowering", "Grain/fruit formation", "Maturation"]
    )

    # Climate impact matrix
    st.markdown("**📊 Climate Impact Matrix**")

    # Simulated impact data
    climate_factors = ['Temperature', 'Precipitation', 'Humidity', 'Wind', 'Solar radiation']
    impact_levels = ['Very Favorable', 'Favorable', 'Neutral', 'Unfavorable', 'Very Unfavorable']

    # Impact matrix for selected crop
    impact_matrix = np.random.choice([0, 1, 2, 3, 4], size=(len(climate_factors), len(impact_levels)),
                                   p=[0.1, 0.3, 0.3, 0.2, 0.1])

    # Create heatmap
    fig_impact = px.imshow(
        impact_matrix,
        x=impact_levels,
        y=climate_factors,
        color_continuous_scale='RdYlGn_r',
        title=f"Climate Impact on {crop_selection} - {growth_stage}",
        aspect='auto'
    )

    fig_impact.update_layout(height=300)
    st.plotly_chart(fig_impact, use_container_width=True)

    # Climate-based yield predictions
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**📈 Climate Yield Predictions**")

        # Generate climate scenarios
        scenarios = ['Optimistic', 'Probable', 'Pessimistic']
        base_yield = {'Wheat': 7.2, 'Corn': 9.8, 'Sunflower': 2.8}.get(crop_selection, 5.0)

        scenario_yields = [
            base_yield * 1.15,  # Optimistic
            base_yield * 1.0,   # Probable
            base_yield * 0.85   # Pessimistic
        ]

        fig_yield_scenarios = go.Figure()

        fig_yield_scenarios.add_trace(go.Bar(
            x=scenarios,
            y=scenario_yields,
            marker_color=['green', 'blue', 'red'],
            text=[f"{y:.1f} t/ha" for y in scenario_yields],
            textposition='auto'
        ))

        fig_yield_scenarios.update_layout(
            title=f"Yield Scenarios - {crop_selection}",
            yaxis_title="Yield (tonnes/ha)",
            height=300
        )

        st.plotly_chart(fig_yield_scenarios, use_container_width=True)

    with col2:
        st.markdown("**🎯 Adaptive Recommendations**")

        # Recommendations based on forecasts
        recommendations = {
            'Wheat': [
                "💧 Enhanced water monitoring",
                "🌡️ Protection against late frost",
                "🍄 Fungal disease prevention",
                "⏰ Adjust sowing dates"
            ],
            'Corn': [
                "🌱 Delay sowing if cold soil",
                "💦 Early irrigation if dry",
                "🌪️ Protection against lodging",
                "🦗 Pest monitoring"
            ]
        }

        crop_recommendations = recommendations.get(crop_selection, [
            "📊 Continuous monitoring required",
            "🔄 Adapt cultural practices",
            "⚠️ Weather alerts activated",
            "📈 Weekly yield tracking"
        ])

        for rec in crop_recommendations:
            st.write(rec)

        # Stress probability
        stress_probability = np.random.uniform(15, 45)
        stress_color = "green" if stress_probability < 25 else "orange" if stress_probability < 35 else "red"

        st.metric(
            "Stress Risk",
            f"{stress_probability:.0f}%",
            delta=f"{stress_probability-30:.0f}% vs normal"
        )

        # Optimal action window
        optimal_window = np.random.randint(3, 14)
        st.metric(
            "Action window",
            f"{optimal_window} days",
            help="Optimal period for interventions"
        )

with tab5:
    st.subheader("Predictive Models and Validation")

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Climate model type",
            [
                "Ensemble Neural Network",
                "Climate Random Forest",
                "LSTM Time Series",
                "Hybrid Satellite-IoT Model",
                "Weather Transformers"
            ]
        )

        model_resolution = st.selectbox(
            "Spatial resolution",
            ["1km x 1km", "5km x 5km", "10km x 10km", "25km x 25km"]
        )

    with col2:
        temporal_resolution = st.selectbox(
            "Temporal resolution",
            ["Hourly", "3 hours", "6 hours", "Daily"]
        )

        prediction_horizon = st.selectbox(
            "Prediction horizon",
            ["24h", "72h", "7 days", "15 days", "1 month"]
        )

    # Model performance
    st.markdown("**📊 Model Performance**")

    # Simulated performance metrics
    model_performance = {
        'Ensemble Neural Network': {'Precision': 94.2, 'Recall': 91.8, 'F1-Score': 93.0, 'RMSE': 1.12},
        'Climate Random Forest': {'Precision': 89.7, 'Recall': 87.3, 'F1-Score': 88.5, 'RMSE': 1.34},
        'LSTM Time Series': {'Precision': 91.5, 'Recall': 89.2, 'F1-Score': 90.3, 'RMSE': 1.21},
        'Hybrid Satellite-IoT Model': {'Precision': 96.1, 'Recall': 94.5, 'F1-Score': 95.3, 'RMSE': 0.98},
        'Weather Transformers': {'Precision': 93.8, 'Recall': 92.1, 'F1-Score': 92.9, 'RMSE': 1.15}
    }

    # Display metrics for selected model
    current_metrics = model_performance[model_type]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision", f"{current_metrics['Precision']:.1f}%")
    with col2:
        st.metric("Recall", f"{current_metrics['Recall']:.1f}%")
    with col3:
        st.metric("F1-Score", f"{current_metrics['F1-Score']:.1f}%")
    with col4:
        st.metric("RMSE", f"{current_metrics['RMSE']:.2f}°C")

    # Model comparison chart
    models = list(model_performance.keys())
    precisions = [model_performance[m]['Precision'] for m in models]
    rmse_values = [model_performance[m]['RMSE'] for m in models]

    fig_models = go.Figure()

    fig_models.add_trace(go.Bar(
        name='Precision (%)',
        x=models,
        y=precisions,
        yaxis='y',
        offsetgroup=1
    ))

    fig_models.add_trace(go.Scatter(
        name='RMSE (°C)',
        x=models,
        y=rmse_values,
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red', width=3)
    ))

    fig_models.update_layout(
        title="Climate Model Performance Comparison",
        yaxis=dict(title="Precision (%)", side="left"),
        yaxis2=dict(title="RMSE (°C)", side="right", overlaying="y"),
        height=400,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_models, use_container_width=True)

    # Cross-validation and tests
    st.markdown("**🔬 Model Validation and Tests**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Robustness Tests**")

        test_results = {
            "✅ Missing data test": "95.2% precision",
            "✅ Extreme conditions test": "87.8% precision",
            "✅ Spatial generalization test": "91.4% precision",
            "⚠️ Climate change test": "83.1% precision",
            "✅ Real-time test": "96.7% precision"
        }

        for test, result in test_results.items():
            st.write(f"{test}: {result}")

    with col2:
        st.markdown("**Model Update**")

        last_update = datetime.now() - timedelta(days=2)
        st.write(f"Last update: {last_update.strftime('%d/%m/%Y %H:%M')}")

        data_sources_count = len(data_sources)
        st.write(f"Active data sources: {data_sources_count}")

        st.write(f"Training samples: 2.3M")
        st.write(f"Re-training frequency: Weekly")

        if st.button("🔄 Trigger Re-training"):
            with st.spinner("Re-training in progress..."):
                import time
                time.sleep(3)
                st.success("✅ Model updated successfully!")

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")

st.sidebar.metric("Active sources", len(data_sources))
st.sidebar.metric("Predictions/day", "1,247")
st.sidebar.metric("Average precision", "94.1%")
st.sidebar.metric("Latency", "< 50ms")

# Footer
st.markdown("---")
st.markdown("**📡 Climate Forecasting Module** - Advanced AI for weather anticipation and agricultural alerts")
