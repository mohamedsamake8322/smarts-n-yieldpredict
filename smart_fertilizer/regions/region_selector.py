from typing import Dict, List, Optional, Tuple
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime, timedelta
from modules.smart_fertilizer.core.regional_context import RegionalContext
print("📁 Dossier courant :", os.getcwd())


class RegionSelector:
    """
    Advanced interactive region selector for African agricultural regions
    """

    def __init__(self):
        self.regional_context = RegionalContext()
        self.available_regions = self.regional_context.get_available_regions()

    def render_region_selector(self) -> Optional[Dict]:
        """
        Render advanced region selection interface with enhanced features
        """
        st.subheader("🌍 Sélection de Votre Région Agricole")

        # Tabs for different selection methods
        tab1, tab2, tab3 = st.tabs(["🗺️ Sélection par Carte", "📍 Sélection Détaillée", "🔍 Recherche Intelligente"])

        with tab1:
            selected_region = self._render_map_selection()

        with tab2:
            selected_region = self._render_detailed_selection()

        with tab3:
            selected_region = self._render_smart_search()

        # If region selected from any method, show comprehensive info
        if selected_region:
            st.divider()
            self._display_comprehensive_region_info(selected_region)
            self._render_regional_insights(selected_region)

        return selected_region

    def _render_map_selection(self) -> Optional[Dict]:
        """Render interactive map for region selection"""
        st.markdown("**🗺️ Cliquez sur votre région sur la carte interactive**")

        # Create interactive map data
        map_data = []
        for region in self.available_regions:
            # Approximate coordinates for African regions
            coordinates = self._get_region_coordinates(region['key'])
            if coordinates:
                map_data.append({
                    'région': region['name'],
                    'lat': coordinates[0],
                    'lon': coordinates[1],
                    'climat': region.get('climate_type', 'N/A'),
                    'pluviométrie': region.get('rainfall_pattern', 'N/A'),
                    'cultures_principales': ', '.join(region.get('major_crops', [])[:3])
                })

        if map_data:
            df_map = pd.DataFrame(map_data)

            # Create interactive map with Plotly
            fig = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                hover_name="région",
                hover_data=["climat", "pluviométrie", "cultures_principales"],
                color="climat",
                size_max=15,
                zoom=3,
                mapbox_style="open-street-map",
                title="Régions Agricoles d'Afrique",
                height=500
            )

            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=0, lon=20),  # Center on Africa
                    zoom=3
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Region selection based on map
            selected_region_name = st.selectbox(
                "Sélectionnez votre région depuis la carte:",
                options=[''] + [region['région'] for region in map_data],
                help="Choisissez la région qui correspond à votre localisation"
            )

            if selected_region_name:
                # Find the corresponding region data
                for region in self.available_regions:
                    if region['name'] == selected_region_name:
                        return self.regional_context.get_region_data(region['key'])

        return None

    def _render_detailed_selection(self) -> Optional[Dict]:
        """Render detailed region selection with filters"""
        st.markdown("**📍 Sélection détaillée par critères**")

        col1, col2 = st.columns(2)

        with col1:
            # Filter by climate type
            climate_types = list(set([region.get('climate_type', 'N/A') for region in self.available_regions]))
            selected_climate = st.selectbox(
                "Type de climat:",
                options=['Tous'] + climate_types,
                help="Filtrer par type de climat"
            )

        with col2:
            # Filter by rainfall pattern
            rainfall_patterns = list(set([region.get('rainfall_pattern', 'N/A') for region in self.available_regions]))
            selected_rainfall = st.selectbox(
                "Régime pluviométrique:",
                options=['Tous'] + rainfall_patterns,
                help="Filtrer par régime de précipitations"
            )

        # Filter regions based on selection
        filtered_regions = self.available_regions
        if selected_climate != 'Tous':
            filtered_regions = [r for r in filtered_regions if r.get('climate_type') == selected_climate]

        if selected_rainfall != 'Tous':
            filtered_regions = [r for r in filtered_regions if r.get('rainfall_pattern') == selected_rainfall]

        # Display filtered regions
        if filtered_regions:
            st.markdown(f"**{len(filtered_regions)} régions correspondent à vos critères:**")

            # Create detailed region cards
            for region in filtered_regions:
                with st.expander(f"🌍 {region['name']} - {region.get('climate_type', 'N/A')}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Climat:** {region.get('climate_type', 'N/A')}")
                        st.write(f"**Précipitations:** {region.get('rainfall_pattern', 'N/A')}")

                    with col2:
                        st.write(f"**Monnaie:** {region.get('currency', 'USD')}")
                        subsidies = "Oui" if region.get('fertilizer_subsidies', False) else "Non"
                        st.write(f"**Subventions:** {subsidies}")

                    with col3:
                        major_crops = region.get('major_crops', [])
                        if major_crops:
                            st.write(f"**Cultures principales:** {', '.join(major_crops[:5])}")

                    if st.button(f"Sélectionner {region['name']}", key=f"select_{region['key']}"):
                        return self.regional_context.get_region_data(region['key'])

        return None

    def _render_smart_search(self) -> Optional[Dict]:
        """Render intelligent search with recommendations"""
        st.markdown("**🔍 Recherche intelligente et recommandations**")

        # Smart search input
        search_query = st.text_input(
            "Rechercher par nom de pays, ville ou caractéristique:",
            placeholder="Ex: Kenya, Lagos, climat aride, maïs, riz...",
            help="Tapez n'importe quel terme lié à votre région"
        )

        if search_query:
            # Simple search logic
            matching_regions = []
            search_lower = search_query.lower()

            for region in self.available_regions:
                score = 0
                # Check name match
                if search_lower in region['name'].lower():
                    score += 10

                # Check climate match
                if region.get('climate_type') and search_lower in region['climate_type'].lower():
                    score += 5

                # Check crops match
                for crop in region.get('major_crops', []):
                    if search_lower in crop.lower():
                        score += 3

                # Check keywords
                keywords = region.get('keywords', [])
                for keyword in keywords:
                    if search_lower in keyword.lower():
                        score += 2

                if score > 0:
                    matching_regions.append((region, score))

            # Sort by relevance
            matching_regions.sort(key=lambda x: x[1], reverse=True)

            if matching_regions:
                st.success(f"✅ {len(matching_regions)} région(s) trouvée(s)")

                for region, score in matching_regions[:5]:  # Show top 5
                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**{region['name']}** (Score: {score})")
                            st.write(f"🌡️ {region.get('climate_type', 'N/A')} | 🌧️ {region.get('rainfall_pattern', 'N/A')}")

                            crops = region.get('major_crops', [])[:3]
                            if crops:
                                st.write(f"🌱 Cultures: {', '.join(crops)}")

                        with col2:
                            if st.button("Choisir", key=f"smart_select_{region['key']}"):
                                return self.regional_context.get_region_data(region['key'])
            else:
                st.warning("Aucune région trouvée. Essayez d'autres termes de recherche.")

        # Quick recommendations based on common scenarios
        st.markdown("### 🎯 Recommandations Rapides")

        quick_options = {
            "🌾 Producteur de céréales (maïs, blé, sorgho)": ["nigeria", "kenya", "ghana"],
            "🌾 Producteur de riz": ["ghana", "senegal", "madagascar"],
            "☕ Producteur de café/cacao": ["ethiopia", "ivory_coast", "uganda"],
            "🥜 Producteur d'arachides/légumineuses": ["senegal", "nigeria", "malawi"],
            "🌴 Région côtière": ["senegal", "ghana", "kenya"],
            "🏔️ Région montagneuse": ["ethiopia", "kenya", "rwanda"]
        }

        for description, region_keys in quick_options.items():
            if st.button(description, use_container_width=True):
                # Show regions for this category
                st.info(f"Régions recommandées pour {description}:")
                for key in region_keys:
                    for region in self.available_regions:
                        if region['key'] == key:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"• **{region['name']}** - {region.get('climate_type', 'N/A')}")
                            with col2:
                                if st.button("Sélectionner", key=f"quick_{key}"):
                                    return self.regional_context.get_region_data(key)

        return None

    def _get_region_coordinates(self, region_key: str) -> Optional[Tuple[float, float]]:
        """Get approximate coordinates for African regions"""
        coordinates_map = {
            'nigeria': (9.0820, 8.6753),
            'kenya': (-0.0236, 37.9062),
            'ghana': (7.9465, -1.0232),
            'senegal': (14.4974, -14.4524),
            'ethiopia': (9.1450, 40.4897),
            'uganda': (1.3733, 32.2903),
            'mali': (17.5707, -3.9962),
            'burkina_faso': (12.2383, -1.5616),
            'ivory_coast': (7.5400, -5.5471),
            'cameroon': (7.3697, 12.3547),
            'tanzania': (-6.3690, 34.8888),
            'madagascar': (-18.7669, 46.8691),
            'malawi': (-13.2543, 34.3015),
            'rwanda': (-1.9403, 29.8739),
            'burundi': (-3.3731, 29.9189),
            'zambia': (-13.1339, 27.8493),
            'mozambique': (-18.6657, 35.5296),
            'botswana': (-22.3285, 24.6849),
            'south_africa': (-30.5595, 22.9375)
        }
        return coordinates_map.get(region_key)

    def _display_comprehensive_region_info(self, region_data: Dict):
        """Display comprehensive region information with enhanced visuals"""
        st.markdown("### 📊 Informations Détaillées de la Région")

        # Key metrics in cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🌡️ Type de Climat",
                value=region_data.get('climate_type', 'N/A')
            )

        with col2:
            st.metric(
                label="🌧️ Précipitations",
                value=region_data.get('rainfall_pattern', 'N/A')
            )

        with col3:
            st.metric(
                label="💰 Monnaie",
                value=region_data.get('currency', 'USD')
            )

        with col4:
            subsidies = "Disponibles" if region_data.get('fertilizer_subsidies', False) else "Non disponibles"
            st.metric(
                label="🏛️ Subventions",
                value=subsidies
            )

        # Additional details in expandable sections
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("🌾 Cultures Principales"):
                major_crops = region_data.get('major_crops', [])
                if major_crops:
                    for i, crop in enumerate(major_crops, 1):
                        st.write(f"{i}. **{crop.title()}**")
                else:
                    st.info("Informations sur les cultures non disponibles")

        with col2:
            with st.expander("🏔️ Types de Sols Dominants"):
                dominant_soils = region_data.get('dominant_soils', [])
                if dominant_soils:
                    for i, soil in enumerate(dominant_soils, 1):
                        soil_info = self._get_soil_type_info(soil)
                        st.write(f"{i}. **{soil.title()}** - {soil_info}")
                else:
                    st.info("Informations sur les sols non disponibles")

    def _render_regional_insights(self, region_data: Dict):
        """Render regional agricultural insights and recommendations"""
        st.markdown("### 🎯 Aperçus Agricoles et Recommandations")

        # Seasonal calendar
        with st.expander("📅 Calendrier Cultural Régional", expanded=True):
            self._render_seasonal_calendar(region_data)

        # Market information
        with st.expander("💹 Informations du Marché"):
            self._render_market_info(region_data)

        # Climate and weather patterns
        with st.expander("🌤️ Conditions Climatiques"):
            self._render_climate_info(region_data)

        # Agricultural challenges and opportunities
        with st.expander("⚠️ Défis et Opportunités Agricoles"):
            self._render_challenges_opportunities(region_data)

    def _render_seasonal_calendar(self, region_data: Dict):
        """Render seasonal agricultural calendar"""
        st.markdown("**Calendrier des Principales Cultures**")

        # Create sample seasonal data
        seasons = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                  'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

        # Sample crop calendar data
        crop_calendar = {
            'Maïs': [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0],
            'Riz': [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
            'Sorgho': [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 0],
            'Millet': [0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 0, 0]
        }

        # Create calendar visualization with proper indexing
        calendar_df = pd.DataFrame(crop_calendar)
        calendar_df = calendar_df.set_axis(seasons, axis=0)

        # Color mapping
        color_map = {0: 'lightgray', 1: 'lightgreen', 2: 'yellow', 3: 'orange'}
        stage_labels = {0: 'Repos', 1: 'Semis', 2: 'Croissance', 3: 'Récolte'}

        fig = px.imshow(
            calendar_df.T,
            aspect='auto',
            color_continuous_scale='Viridis',
            title="Calendrier Cultural (0=Repos, 1=Semis, 2=Croissance, 3=Récolte)"
        )

        fig.update_layout(
            xaxis_title="Mois",
            yaxis_title="Cultures",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

        # Legend
        st.markdown("**Légende:** 🟢 Semis | 🟡 Croissance | 🟠 Récolte | ⚪ Repos")

    def _render_market_info(self, region_data: Dict):
        """Render market information"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**💰 Prix des Engrais (estimation)**")
            fertilizer_prices = region_data.get('fertilizer_prices', {})

            if fertilizer_prices:
                for fertilizer, price in fertilizer_prices.items():
                    currency = region_data.get('currency', 'USD')
                    st.write(f"• **{fertilizer}**: {currency} {price}/50kg")
            else:
                st.info("Prix des engrais non disponibles")

        with col2:
            st.markdown("**🌾 Prix des Cultures (estimation)**")
            crop_prices = region_data.get('crop_prices', {})

            if crop_prices:
                for crop, price in crop_prices.items():
                    currency = region_data.get('currency', 'USD')
                    st.write(f"• **{crop}**: {currency} {price}/tonne")
            else:
                st.info("Prix des cultures non disponibles")

    def _render_climate_info(self, region_data: Dict):
        """Render climate information"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🌡️ Température Moyenne**")
            climate_data = region_data.get('climate_data', {})

            if climate_data:
                temp_range = climate_data.get('temperature_range', 'N/A')
                st.write(f"Température: {temp_range}°C")

                humidity = climate_data.get('humidity', 'N/A')
                st.write(f"Humidité: {humidity}%")
            else:
                st.info("Données climatiques non disponibles")

        with col2:
            st.markdown("**🌧️ Pluviométrie**")

            if climate_data:
                rainfall = climate_data.get('annual_rainfall', 'N/A')
                st.write(f"Précipitations annuelles: {rainfall}mm")

                wet_season = climate_data.get('wet_season', 'N/A')
                st.write(f"Saison des pluies: {wet_season}")
            else:
                st.info("Données pluviométriques non disponibles")

    def _render_challenges_opportunities(self, region_data: Dict):
        """Render agricultural challenges and opportunities"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**⚠️ Défis Agricoles**")
            challenges = [
                "Variabilité climatique",
                "Accès limité aux engrais",
                "Dégradation des sols",
                "Parasites et maladies",
                "Accès aux marchés"
            ]

            for challenge in challenges:
                st.write(f"• {challenge}")

        with col2:
            st.markdown("**💡 Opportunités**")
            opportunities = [
                "Agriculture climato-intelligente",
                "Technologies numériques",
                "Certification biologique",
                "Coopératives agricoles",
                "Marchés d'exportation"
            ]

            for opportunity in opportunities:
                st.write(f"• {opportunity}")

    def _get_soil_type_info(self, soil_type: str) -> str:
        """Get information about soil types"""
        soil_info = {
            'sandy': 'Bien drainé, faible rétention d\'eau',
            'loamy': 'Équilibré, idéal pour la plupart des cultures',
            'clay': 'Rétention d\'eau élevée, peut être compact',
            'silt': 'Rétention moyenne, fertile',
            'laterite': 'Riche en fer, nécessite amendements',
            'vertisol': 'Très argileux, se fissure en saison sèche'
        }
        return soil_info.get(soil_type.lower(), 'Informations non disponibles')

    def _display_region_info(self, region_data: Dict):
        """Display detailed region information"""

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Climate Type:** {region_data.get('climate_type', 'N/A')}")
            st.info(f"**Rainfall Pattern:** {region_data.get('rainfall_pattern', 'N/A')}")

        with col2:
            st.info(f"**Currency:** {region_data.get('currency', 'USD')}")
            subsidies = "Yes" if region_data.get('fertilizer_subsidies', False) else "No"
            st.info(f"**Fertilizer Subsidies:** {subsidies}")

        # Major crops
        major_crops = region_data.get('major_crops', [])
        if major_crops:
            st.success(f"**Major Crops:** {', '.join(major_crops)}")

        # Dominant soils
        dominant_soils = region_data.get('dominant_soils', [])
        if dominant_soils:
            st.warning(f"**Dominant Soil Types:** {', '.join(dominant_soils)}")

    def get_agro_ecological_zones(self, region_key: str) -> List[str]:
        """Get agro-ecological zones for a region"""
        region_data = self.regional_context.get_region_data(region_key)
        return region_data.get('zones', ['mixed'])

    def render_zone_selector(self, region_key: str) -> Optional[str]:
        """Render agro-ecological zone selector"""

        zones = self.get_agro_ecological_zones(region_key)

        if len(zones) > 1:
            st.subheader("🏞️ Select Agro-Ecological Zone")

            zone_descriptions = {
                "guinea_savanna": "Guinea Savanna - High rainfall, longer growing season",
                "sudan_savanna": "Sudan Savanna - Moderate rainfall, medium growing season",
                "sahel": "Sahel - Low rainfall, short growing season",
                "forest": "Forest Zone - Very high rainfall, year-round growing",
                "highlands": "Highlands - Cool climate, altitude effects",
                "coast": "Coastal - Maritime influence, high humidity",
                "arid_semi_arid": "Arid/Semi-Arid - Low rainfall, drought risk"
            }

            zone_options = []
            for zone in zones:
                description = zone_descriptions.get(zone, zone.replace('_', ' ').title())
                zone_options.append(f"{zone.replace('_', ' ').title()} - {description.split(' - ')[1] if ' - ' in description else ''}")

            selected_zone = st.selectbox(
                "Select your specific agro-ecological zone:",
                options=zone_options,
                help="Choose the zone that best describes your farming area"
            )

            if selected_zone:
                # Extract zone key from selection
                zone_key = selected_zone.split(' - ')[0].lower().replace(' ', '_')
                return zone_key

        return zones[0] if zones else "mixed"

    def get_seasonal_recommendations(self, region_key: str, crop: str) -> Dict:
        """Get seasonal recommendations for region and crop"""
        from datetime import datetime

        return self.regional_context.get_seasonal_recommendations(
            region_key, crop, datetime.now()
        )

    def render_seasonal_info(self, region_key: str, crop: str):
        """Render seasonal information panel"""

        seasonal_info = self.get_seasonal_recommendations(region_key, crop)

        if seasonal_info.get('seasonal_advice'):
            st.subheader("📅 Seasonal Recommendations")

            for advice in seasonal_info['seasonal_advice']:
                st.info(f"🌱 {advice}")

            optimal_timing = seasonal_info.get('optimal_timing', {})
            if optimal_timing:
                st.success(f"⏰ **Fertilizer Timing:** {optimal_timing.get('fertilizer_application', 'Follow crop calendar')}")

    def get_extension_services(self, region_key: str, crop: str) -> List[str]:
        """Get extension service recommendations"""
        return self.regional_context.get_extension_recommendations(region_key, crop)

    def render_extension_info(self, region_key: str, crop: str):
        """Render extension services information"""

        recommendations = self.get_extension_services(region_key, crop)

        if recommendations:
            st.subheader("🤝 Extension Services & Support")

            for rec in recommendations:
                st.info(f"📋 {rec}")

    def get_market_info(self, region_key: str) -> Dict:
        """Get market access information"""
        return self.regional_context.get_market_access_info(region_key)

    def render_market_info(self, region_key: str):
        """Render market access information"""

        market_info = self.get_market_info(region_key)

        st.subheader("🏪 Market Access Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**💰 Financial Services**")
            financial = market_info.get('financial_services', {})
            for service, description in financial.items():
                st.write(f"• {service.replace('_', ' ').title()}: {description}")

        with col2:
            st.markdown("**🛠️ Support Services**")
            support = market_info.get('support_services', {})
            for service, description in support.items():
                st.write(f"• {service.replace('_', ' ').title()}: {description}")

        st.markdown("**🚚 Market Access**")
        market_access = market_info.get('market_access', {})
        for service, description in market_access.items():
            st.write(f"• {service.replace('_', ' ').title()}: {description}")

    def validate_region_crop_compatibility(self, region_key: str, crop: str) -> Dict:
        """Validate crop compatibility with region"""

        region_data = self.regional_context.get_region_data(region_key)
        major_crops = region_data.get('major_crops', [])

        validation = {
            "compatible": crop.lower() in major_crops,
            "confidence": "high" if crop.lower() in major_crops else "medium",
            "recommendations": []
        }

        if not validation["compatible"]:
            validation["recommendations"].append(
                f"{crop.title()} is not a major crop in {region_data.get('name', 'this region')}. "
                "Consider consulting local agricultural experts for suitability assessment."
            )
            validation["recommendations"].append(
                f"Major crops in this region are: {', '.join(major_crops)}"
            )
        else:
            validation["recommendations"].append(
                f"{crop.title()} is well-suited for {region_data.get('name', 'this region')}."
            )

        return validation

    def render_compatibility_check(self, region_key: str, crop: str):
        """Render crop-region compatibility check"""

        validation = self.validate_region_crop_compatibility(region_key, crop)

        if validation["compatible"]:
            st.success("✅ **Crop-Region Compatibility:** Excellent match!")
        else:
            st.warning("⚠️ **Crop-Region Compatibility:** Requires assessment")

        for rec in validation["recommendations"]:
            st.info(f"💡 {rec}")
