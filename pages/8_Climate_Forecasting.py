
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from utils import AdvancedAI


st.set_page_config(page_title="Climate Forecasting", page_icon="📡", layout="wide")

st.title("📡 Prévision Climatique Avancée")
st.markdown("### IA combinée avec données satellites & IoT pour anticiper les événements climatiques")

# Sidebar controls
st.sidebar.title("Configuration Prévisions")

location = st.sidebar.text_input(
    "Localisation",
    value="Paris, France",
    help="Ville, région ou coordonnées GPS"
)

forecast_range = st.sidebar.selectbox(
    "Horizon de prévision",
    ["7 jours", "15 jours", "1 mois", "3 mois", "Saisonnier"]
)

alert_sensitivity = st.sidebar.select_slider(
    "Sensibilité des alertes",
    options=["Faible", "Normale", "Élevée"],
    value="Normale"
)

# Data sources
data_sources = st.sidebar.multiselect(
    "Sources de données",
    [
        "Satellites météo (GOES, MSG)",
        "Capteurs IoT locaux",
        "Modèles numériques (GFS, ECMWF)",
        "Stations météorologiques",
        "Radars précipitations",
        "Bouées océaniques"
    ],
    default=["Satellites météo (GOES, MSG)", "Capteurs IoT locaux", "Modèles numériques (GFS, ECMWF)"]
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

map_style = st.sidebar.selectbox("Style de carte", map_styles, index=0)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prévisions Temps Réel",
    "Alertes Climatiques",
    "Analyse Saisonnière",
    "Impact Agricole",
    "Modèles Prédictifs"
])

with tab1:
    st.subheader("Prévisions Météorologiques en Temps Réel")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Génération de données de prévision simulées
        days = 15 if "15 jours" in forecast_range else 7
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]

        # Données simulées
        temperatures = np.random.normal(22, 8, days)
        humidity = np.random.normal(65, 15, days)
        precipitation = np.random.exponential(2, days)
        wind_speed = np.random.gamma(2, 3, days)
        pressure = np.random.normal(1015, 10, days)

        # Création du graphique multi-variables
        fig = go.Figure()

        # Température
        fig.add_trace(go.Scatter(
            x=dates,
            y=temperatures,
            mode='lines+markers',
            name='Température (°C)',
            line=dict(color='red', width=3),
            yaxis='y'
        ))

        # Précipitations (barres)
        fig.add_trace(go.Bar(
            x=dates,
            y=precipitation,
            name='Précipitations (mm)',
            marker_color='blue',
            opacity=0.6,
            yaxis='y2'
        ))

        # Humidité
        fig.add_trace(go.Scatter(
            x=dates,
            y=humidity,
            mode='lines',
            name='Humidité (%)',
            line=dict(color='green', dash='dash'),
            yaxis='y3'
        ))

        fig.update_layout(
            title="Prévisions Météorologiques Multi-Variables",
            xaxis_title="Date",
            yaxis=dict(title="Température (°C)", side="left", color="red"),
            yaxis2=dict(title="Précipitations (mm)", side="right", overlaying="y", color="blue"),
            yaxis3=dict(title="Humidité (%)", side="right", overlaying="y", position=0.9, color="green"),
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

# Carte météorologique
        st.markdown("**Carte Météorologique Régionale**")

# Simulation d'une carte de températures
        lat = np.random.uniform(45, 50, 100)
        lon = np.random.uniform(1, 6, 100)
        temp_map = np.random.normal(20, 5, 100)

        fig_map = px.scatter_mapbox(
            lat=lat,
            lon=lon,
            color=temp_map,
            size=abs(temp_map - 20) + 5,
            hover_name=[f"Station {i}" for i in range(100)],
            hover_data={"Température": temp_map},
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=6,
            height=400,
            title="Température Régionale en Temps Réel"
        )

        fig_map.update_layout(mapbox_style=map_style)
        st.plotly_chart(fig_map, use_container_width=True)


    with col2:
        st.markdown("**Conditions Actuelles**")

        # Météo actuelle simulée
        current_temp = np.random.normal(22, 5)
        current_humidity = np.random.uniform(40, 80)
        current_wind = np.random.uniform(5, 25)
        current_pressure = np.random.normal(1015, 8)

        # Métriques météo
        st.metric("🌡️ Température", f"{current_temp:.1f}°C", delta="2.3°C")
        st.metric("💧 Humidité", f"{current_humidity:.0f}%", delta="5%")
        st.metric("💨 Vent", f"{current_wind:.1f} km/h", delta="-3.2 km/h")
        st.metric("🧭 Pression", f"{current_pressure:.0f} hPa", delta="2 hPa")

        st.markdown("**🎯 Indices Agricoles**")

        # Calcul d'indices agricoles
        gdd = max(0, current_temp - 10)  # Growing Degree Days
        et_rate = max(0, (current_temp - 5) * (1 - current_humidity/100) * 0.1)

        st.metric("🌱 GDD (Base 10°C)", f"{gdd:.1f}")
        st.metric("💦 Évapotranspiration", f"{et_rate:.2f} mm/jour")

        # Qualité de l'air simulée
        air_quality_index = np.random.randint(25, 150)
        air_quality_status = "Bon" if air_quality_index < 50 else "Modéré" if air_quality_index < 100 else "Mauvais"

        st.metric("🌬️ Qualité de l'Air", air_quality_status, f"AQI: {air_quality_index}")

        st.markdown("**📊 Tendance 24h**")

        # Mini graphique de tendance
        hours = list(range(24))
        hourly_temp = current_temp + np.sin(np.array(hours) * np.pi / 12) * 5 + np.random.normal(0, 1, 24)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hours,
            y=hourly_temp,
            mode='lines',
            name='Température 24h',
            line=dict(color='orange', width=2)
        ))

        fig_trend.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Heure",
            yaxis_title="°C"
        )

        st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Système d'Alertes Climatiques Intelligentes")

    # Configuration des alertes
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Configuration des Alertes**")

        alert_types = st.multiselect(
            "Types d'alertes à surveiller",
            [
                "🌡️ Températures extrêmes",
                "🌧️ Précipitations intenses",
                "❄️ Risque de gel",
                "🌪️ Vents violents",
                "⛈️ Orages sévères",
                "🌵 Sécheresse",
                "🌊 Inondations",
                "🌫️ Brouillard dense"
            ],
            default=["🌡️ Températures extrêmes", "❄️ Risque de gel", "🌧️ Précipitations intenses"]
        )

        notification_methods = st.multiselect(
            "Méthodes de notification",
            ["📧 Email", "📱 SMS", "🔔 Push", "📺 Dashboard", "📡 IoT Actuators"],
            default=["📧 Email", "📺 Dashboard"]
        )

        advance_warning = st.selectbox(
            "Préavis souhaité",
            ["1 heure", "3 heures", "6 heures", "12 heures", "24 heures", "48 heures"],
            index=4
        )

    with col2:
        st.markdown("**Alertes Actives**")

        # Simulation d'alertes
        active_alerts = [
            {
                "type": "❄️ Alerte Gel",
                "severity": "🟡 Modéré",
                "time": "Dans 18h",
                "temp": "-2°C attendu",
                "action": "Protéger cultures sensibles"
            },
            {
                "type": "🌧️ Pluies Intenses",
                "severity": "🟠 Élevé",
                "time": "Dans 6h",
                "amount": "25mm en 2h",
                "action": "Vérifier drainage"
            },
            {
                "type": "💨 Vents Forts",
                "severity": "🟡 Modéré",
                "time": "Dans 12h",
                "speed": "45 km/h rafales",
                "action": "Sécuriser équipements"
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
                    if st.button("✅ Acquitter", key=f"ack_{alert['type']}"):
                        st.success("Alerte acquittée")
                with col_btn2:
                    if st.button("🔕 Désactiver", key=f"disable_{alert['type']}"):
                        st.info("Alerte désactivée")

                st.markdown("---")

    # Historique des alertes
    st.markdown("**📈 Historique des Alertes (30 derniers jours)**")

    # Génération de données d'historique
    alert_dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    alert_counts = np.random.poisson(1.5, 30)
    alert_severities = np.random.choice(['Faible', 'Modéré', 'Élevé', 'Critique'], 30, p=[0.4, 0.3, 0.2, 0.1])

    alert_df = pd.DataFrame({
        'Date': alert_dates,
        'Nombre_Alertes': alert_counts,
        'Sévérité_Max': alert_severities
    })

    fig_alerts = px.bar(
        alert_df,
        x='Date',
        y='Nombre_Alertes',
        color='Sévérité_Max',
        title="Évolution des Alertes Météorologiques",
        color_discrete_map={
            'Faible': 'green',
            'Modéré': 'yellow',
            'Élevé': 'orange',
            'Critique': 'red'
        }
    )

    fig_alerts.update_layout(height=300)
    st.plotly_chart(fig_alerts, use_container_width=True)

with tab3:
    st.subheader("Analyse Climatique Saisonnière")

    # Sélection de la période d'analyse
    col1, col2 = st.columns(2)

    with col1:
        analysis_year = st.selectbox(
            "Année d'analyse",
            [2024, 2023, 2022, 2021, 2020],
            index=0
        )

    with col2:
        comparison_year = st.selectbox(
            "Année de comparaison",
            [2023, 2022, 2021, 2020, 2019],
            index=0
        )

    # Données saisonnières simulées
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
              'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

    # Température mensuelle
    temp_2024 = [2, 4, 9, 14, 18, 22, 25, 24, 20, 15, 8, 3]
    temp_2023 = [1, 3, 8, 13, 17, 21, 24, 23, 19, 14, 7, 2]

    # Précipitations mensuelles
    precip_2024 = [45, 38, 52, 67, 73, 45, 32, 48, 61, 78, 69, 54]
    precip_2023 = [52, 41, 48, 62, 68, 38, 28, 52, 58, 82, 74, 61]

    # Graphiques de comparaison
    fig_seasonal = go.Figure()

    # Températures
    fig_seasonal.add_trace(go.Scatter(
        x=months,
        y=temp_2024,
        mode='lines+markers',
        name=f'Température {analysis_year}',
        line=dict(color='red', width=3),
        yaxis='y'
    ))

    fig_seasonal.add_trace(go.Scatter(
        x=months,
        y=temp_2023,
        mode='lines+markers',
        name=f'Température {comparison_year}',
        line=dict(color='orange', width=2, dash='dash'),
        yaxis='y'
    ))

    # Précipitations
    fig_seasonal.add_trace(go.Bar(
        x=months,
        y=precip_2024,
        name=f'Précipitations {analysis_year}',
        marker_color='blue',
        opacity=0.7,
        yaxis='y2'
    ))

    fig_seasonal.add_trace(go.Bar(
        x=months,
        y=precip_2023,
        name=f'Précipitations {comparison_year}',
        marker_color='lightblue',
        opacity=0.5,
        yaxis='y2'
    ))

    fig_seasonal.update_layout(
        title="Comparaison Climatique Saisonnière",
        xaxis_title="Mois",
        yaxis=dict(title="Température (°C)", side="left"),
        yaxis2=dict(title="Précipitations (mm)", side="right", overlaying="y"),
        height=400
    )

    st.plotly_chart(fig_seasonal, use_container_width=True)

    # Analyse des anomalies
    st.markdown("**🔍 Détection d'Anomalies Climatiques**")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp_anomaly = np.mean(temp_2024) - np.mean(temp_2023)
        st.metric(
            "Anomalie Température",
            f"{temp_anomaly:+.1f}°C",
            delta=f"vs {comparison_year}"
        )

    with col2:
        precip_anomaly = (np.sum(precip_2024) - np.sum(precip_2023)) / np.sum(precip_2023) * 100
        st.metric(
            "Anomalie Précipitations",
            f"{precip_anomaly:+.1f}%",
            delta=f"vs {comparison_year}"
        )

    with col3:
        extreme_days = np.random.randint(12, 28)
        st.metric(
            "Jours Extrêmes",
            extreme_days,
            delta=f"+{extreme_days-20} vs normale"
        )

    # Projections climatiques
    st.markdown("**🔮 Projections Climatiques (IA)**")

    # Génération de projections
    projection_months = ['Jan 2025', 'Fév 2025', 'Mar 2025', 'Avr 2025', 'Mai 2025', 'Jun 2025']
    projected_temp = [3, 5, 10, 15, 19, 23]
    confidence_bands = [1.5, 1.8, 2.1, 2.3, 2.0, 1.7]

    fig_projection = go.Figure()

    # Projection centrale
    fig_projection.add_trace(go.Scatter(
        x=projection_months,
        y=projected_temp,
        mode='lines+markers',
        name='Projection IA',
        line=dict(color='purple', width=3)
    ))

    # Bandes de confiance
    fig_projection.add_trace(go.Scatter(
        x=projection_months + projection_months[::-1],
        y=[t + c for t, c in zip(projected_temp, confidence_bands)] +
          [t - c for t, c in zip(projected_temp[::-1], confidence_bands[::-1])],
        fill='toself',
        fillcolor='rgba(128,0,128,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Intervalle confiance 90%'
    ))

    fig_projection.update_layout(
        title="Projections Température - 6 Prochains Mois",
        xaxis_title="Période",
        yaxis_title="Température (°C)",
        height=300
    )

    st.plotly_chart(fig_projection, use_container_width=True)

with tab4:
    st.subheader("Évaluation d'Impact Agricole")

    # Sélection du type de culture
    crop_selection = st.selectbox(
        "Type de culture à analyser",
        ["Blé", "Maïs", "Tournesol", "Betterave", "Colza", "Orge", "Tomate", "Pomme de terre"]
    )

    growth_stage = st.selectbox(
        "Stade de croissance",
        ["Semis/Plantation", "Levée", "Développement végétatif", "Floraison", "Formation grains/fruits", "Maturation"]
    )

    # Matrice d'impact climatique
    st.markdown("**📊 Matrice d'Impact Climatique**")

    # Données d'impact simulées
    climate_factors = ['Température', 'Précipitations', 'Humidité', 'Vent', 'Radiation solaire']
    impact_levels = ['Très Favorable', 'Favorable', 'Neutre', 'Défavorable', 'Très Défavorable']

    # Matrice d'impact pour le crop sélectionné
    impact_matrix = np.random.choice([0, 1, 2, 3, 4], size=(len(climate_factors), len(impact_levels)),
                                   p=[0.1, 0.3, 0.3, 0.2, 0.1])

    # Création d'un heatmap
    fig_impact = px.imshow(
        impact_matrix,
        x=impact_levels,
        y=climate_factors,
        color_continuous_scale='RdYlGn_r',
        title=f"Impact Climatique sur {crop_selection} - {growth_stage}",
        aspect='auto'
    )

    fig_impact.update_layout(height=300)
    st.plotly_chart(fig_impact, use_container_width=True)

    # Prédictions de rendement basées sur le climat
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**📈 Prédictions de Rendement Climatique**")

        # Génération de scénarios climatiques
        scenarios = ['Optimiste', 'Probable', 'Pessimiste']
        base_yield = {'Blé': 7.2, 'Maïs': 9.8, 'Tournesol': 2.8}.get(crop_selection, 5.0)

        scenario_yields = [
            base_yield * 1.15,  # Optimiste
            base_yield * 1.0,   # Probable
            base_yield * 0.85   # Pessimiste
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
            title=f"Scénarios de Rendement - {crop_selection}",
            yaxis_title="Rendement (tonnes/ha)",
            height=300
        )

        st.plotly_chart(fig_yield_scenarios, use_container_width=True)

    with col2:
        st.markdown("**🎯 Recommandations Adaptatives**")

        # Recommandations basées sur les prévisions
        recommendations = {
            'Blé': [
                "💧 Surveillance hydrique renforcée",
                "🌡️ Protection contre gel tardif",
                "🍄 Prévention maladies fongiques",
                "⏰ Ajuster dates de semis"
            ],
            'Maïs': [
                "🌱 Retarder semis si sol froid",
                "💦 Irrigation précoce si sec",
                "🌪️ Protection contre verse",
                "🦗 Surveillance ravageurs"
            ]
        }

        crop_recommendations = recommendations.get(crop_selection, [
            "📊 Monitoring continu requis",
            "🔄 Adapter pratiques culturales",
            "⚠️ Alertes météo activées",
            "📈 Suivi rendement hebdomadaire"
        ])

        for rec in crop_recommendations:
            st.write(rec)

        # Probabilité de stress
        stress_probability = np.random.uniform(15, 45)
        stress_color = "green" if stress_probability < 25 else "orange" if stress_probability < 35 else "red"

        st.metric(
            "Risque de Stress",
            f"{stress_probability:.0f}%",
            delta=f"{stress_probability-30:.0f}% vs normale"
        )

        # Fenêtre d'action optimale
        optimal_window = np.random.randint(3, 14)
        st.metric(
            "Fenêtre d'action",
            f"{optimal_window} jours",
            help="Période optimale pour interventions"
        )

with tab5:
    st.subheader("Modèles Prédictifs et Validation")

    # Sélection du modèle
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Type de modèle climatique",
            [
                "Ensemble Neural Network",
                "Random Forest Climatique",
                "LSTM Séries Temporelles",
                "Modèle Hybride Satellite-IoT",
                "Transformers Météo"
            ]
        )

        model_resolution = st.selectbox(
            "Résolution spatiale",
            ["1km x 1km", "5km x 5km", "10km x 10km", "25km x 25km"]
        )

    with col2:
        temporal_resolution = st.selectbox(
            "Résolution temporelle",
            ["Horaire", "3 heures", "6 heures", "Quotidien"]
        )

        prediction_horizon = st.selectbox(
            "Horizon de prédiction",
            ["24h", "72h", "7 jours", "15 jours", "1 mois"]
        )

    # Performance des modèles
    st.markdown("**📊 Performance des Modèles**")

    # Métriques de performance simulées
    model_performance = {
        'Ensemble Neural Network': {'Précision': 94.2, 'Recall': 91.8, 'F1-Score': 93.0, 'RMSE': 1.12},
        'Random Forest Climatique': {'Précision': 89.7, 'Recall': 87.3, 'F1-Score': 88.5, 'RMSE': 1.34},
        'LSTM Séries Temporelles': {'Précision': 91.5, 'Recall': 89.2, 'F1-Score': 90.3, 'RMSE': 1.21},
        'Modèle Hybride Satellite-IoT': {'Précision': 96.1, 'Recall': 94.5, 'F1-Score': 95.3, 'RMSE': 0.98},
        'Transformers Météo': {'Précision': 93.8, 'Recall': 92.1, 'F1-Score': 92.9, 'RMSE': 1.15}
    }

    # Affichage des métriques pour le modèle sélectionné
    current_metrics = model_performance[model_type]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Précision", f"{current_metrics['Précision']:.1f}%")
    with col2:
        st.metric("Recall", f"{current_metrics['Recall']:.1f}%")
    with col3:
        st.metric("F1-Score", f"{current_metrics['F1-Score']:.1f}%")
    with col4:
        st.metric("RMSE", f"{current_metrics['RMSE']:.2f}°C")

    # Graphique de comparaison des modèles
    models = list(model_performance.keys())
    precisions = [model_performance[m]['Précision'] for m in models]
    rmse_values = [model_performance[m]['RMSE'] for m in models]

    fig_models = go.Figure()

    fig_models.add_trace(go.Bar(
        name='Précision (%)',
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
        title="Comparaison Performance Modèles Climatiques",
        yaxis=dict(title="Précision (%)", side="left"),
        yaxis2=dict(title="RMSE (°C)", side="right", overlaying="y"),
        height=400,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_models, use_container_width=True)

    # Validation croisée et tests
    st.markdown("**🔬 Validation et Tests du Modèle**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tests de Robustesse**")

        test_results = {
            "✅ Test données manquantes": "95.2% précision",
            "✅ Test conditions extrêmes": "87.8% précision",
            "✅ Test généralisation spatiale": "91.4% précision",
            "⚠️ Test changement climatique": "83.1% précision",
            "✅ Test temps réel": "96.7% précision"
        }

        for test, result in test_results.items():
            st.write(f"{test}: {result}")

    with col2:
        st.markdown("**Mise à Jour du Modèle**")

        last_update = datetime.now() - timedelta(days=2)
        st.write(f"Dernière mise à jour: {last_update.strftime('%d/%m/%Y %H:%M')}")

        data_sources_count = len(data_sources)
        st.write(f"Sources de données actives: {data_sources_count}")

        st.write(f"Échantillons d'entraînement: 2.3M")
        st.write(f"Fréquence re-entraînement: Hebdomadaire")

        if st.button("🔄 Déclencher Re-entraînement"):
            with st.spinner("Re-entraînement en cours..."):
                import time
                time.sleep(3)
                st.success("✅ Modèle mis à jour avec succès!")

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("**État du Système**")

st.sidebar.metric("Sources actives", len(data_sources))
st.sidebar.metric("Prédictions/jour", "1,247")
st.sidebar.metric("Précision moyenne", "94.1%")
st.sidebar.metric("Latence", "< 50ms")

# Footer
st.markdown("---")
st.markdown("**📡 Module Prévision Climatique** - IA avancée pour anticipation météorologique et alertes agricoles")
