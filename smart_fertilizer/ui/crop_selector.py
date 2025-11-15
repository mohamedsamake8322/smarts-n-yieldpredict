from typing import Dict, List, Optional, Tuple
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json

class CropSelector:
    """
    Advanced crop selection interface with enhanced features
    """
    
    def __init__(self, fertilizer_engine):
        self.fertilizer_engine = fertilizer_engine
        self.crop_database = self._load_crop_database()
        
    def _load_crop_database(self) -> Dict:
        """Load comprehensive crop database"""
        try:
            with open('data/crop_profiles.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return self._get_default_crop_database()
    
    def _get_default_crop_database(self) -> Dict:
        """Get default crop database with African focus"""
        return {
            "cereals": {
                "maize": {
                    "name": "Maïs",
                    "varieties": ["Hybrid", "Local", "Sweet Corn", "Popcorn"],
                    "growth_duration": [90, 120, 150],
                    "yield_potential": [3.5, 8.0, 12.0],  # t/ha
                    "water_requirement": 600,  # mm
                    "soil_preferences": ["loamy", "sandy_loam"],
                    "climate_zones": ["tropical", "subtropical", "temperate"],
                    "nutritional_requirements": {
                        "N": 120, "P": 60, "K": 80
                    },
                    "diseases": ["Streak virus", "Stalk borer", "Gray leaf spot"],
                    "market_price": 300,  # USD/ton
                    "processing_options": ["Flour", "Animal feed", "Fresh consumption"]
                },
                "rice": {
                    "name": "Riz",
                    "varieties": ["Irrigated", "Upland", "Lowland", "Aromatic"],
                    "growth_duration": [120, 140, 160],
                    "yield_potential": [4.0, 7.0, 10.0],
                    "water_requirement": 1200,
                    "soil_preferences": ["clay", "clay_loam"],
                    "climate_zones": ["tropical", "subtropical"],
                    "nutritional_requirements": {
                        "N": 100, "P": 50, "K": 70
                    },
                    "diseases": ["Blast", "Bacterial blight", "Sheath rot"],
                    "market_price": 450,
                    "processing_options": ["Milled rice", "Parboiled", "Broken rice"]
                },
                "sorghum": {
                    "name": "Sorgho",
                    "varieties": ["Grain", "Sweet", "Forage", "Dual purpose"],
                    "growth_duration": [90, 120, 150],
                    "yield_potential": [2.0, 4.5, 7.0],
                    "water_requirement": 400,
                    "soil_preferences": ["sandy", "loamy", "clay"],
                    "climate_zones": ["arid", "semi_arid", "tropical"],
                    "nutritional_requirements": {
                        "N": 80, "P": 40, "K": 60
                    },
                    "diseases": ["Anthracnose", "Leaf blight", "Head smut"],
                    "market_price": 250,
                    "processing_options": ["Flour", "Syrup", "Animal feed", "Brewing"]
                },
                "millet": {
                    "name": "Mil",
                    "varieties": ["Pearl", "Finger", "Foxtail", "Proso"],
                    "growth_duration": [75, 90, 120],
                    "yield_potential": [1.5, 3.0, 5.0],
                    "water_requirement": 300,
                    "soil_preferences": ["sandy", "sandy_loam"],
                    "climate_zones": ["arid", "semi_arid"],
                    "nutritional_requirements": {
                        "N": 60, "P": 30, "K": 40
                    },
                    "diseases": ["Downy mildew", "Ergot", "Smut"],
                    "market_price": 280,
                    "processing_options": ["Flour", "Porridge", "Beer", "Animal feed"]
                }
            },
            "legumes": {
                "groundnuts": {
                    "name": "Arachides",
                    "varieties": ["Spanish", "Runner", "Virginia", "Valencia"],
                    "growth_duration": [90, 120, 140],
                    "yield_potential": [1.5, 3.0, 4.5],
                    "water_requirement": 450,
                    "soil_preferences": ["sandy", "sandy_loam"],
                    "climate_zones": ["tropical", "subtropical"],
                    "nutritional_requirements": {
                        "N": 25, "P": 50, "K": 75  # N-fixing crop
                    },
                    "diseases": ["Leaf spot", "Rust", "Rosette"],
                    "market_price": 800,
                    "processing_options": ["Oil extraction", "Roasted nuts", "Paste", "Flour"]
                },
                "cowpeas": {
                    "name": "Niébé",
                    "varieties": ["Dual purpose", "Grain", "Fodder", "Vegetable"],
                    "growth_duration": [60, 90, 120],
                    "yield_potential": [1.0, 2.5, 4.0],
                    "water_requirement": 350,
                    "soil_preferences": ["sandy", "loamy", "clay"],
                    "climate_zones": ["tropical", "subtropical", "arid"],
                    "nutritional_requirements": {
                        "N": 20, "P": 40, "K": 50
                    },
                    "diseases": ["Bacterial blight", "Virus", "Aphids"],
                    "market_price": 600,
                    "processing_options": ["Dried beans", "Fresh pods", "Flour", "Animal feed"]
                }
            },
            "cash_crops": {
                "cotton": {
                    "name": "Coton",
                    "varieties": ["Bt cotton", "Conventional", "Organic"],
                    "growth_duration": [120, 150, 180],
                    "yield_potential": [1.5, 2.5, 4.0],  # tons lint/ha
                    "water_requirement": 700,
                    "soil_preferences": ["loamy", "clay_loam"],
                    "climate_zones": ["tropical", "subtropical"],
                    "nutritional_requirements": {
                        "N": 120, "P": 60, "K": 100
                    },
                    "diseases": ["Bollworm", "Aphids", "Bacterial blight"],
                    "market_price": 1500,
                    "processing_options": ["Lint", "Seed oil", "Cake", "Textiles"]
                },
                "coffee": {
                    "name": "Café",
                    "varieties": ["Arabica", "Robusta", "Liberica"],
                    "growth_duration": [1095, 1460, 1825],  # 3-5 years to maturity
                    "yield_potential": [0.8, 1.5, 2.5],
                    "water_requirement": 1500,
                    "soil_preferences": ["volcanic", "loamy"],
                    "climate_zones": ["highland_tropical", "subtropical"],
                    "nutritional_requirements": {
                        "N": 200, "P": 50, "K": 150
                    },
                    "diseases": ["Coffee berry disease", "Leaf rust", "Nematodes"],
                    "market_price": 3000,
                    "processing_options": ["Green beans", "Roasted", "Instant", "Pulp"]
                }
            },
            "vegetables": {
                "tomato": {
                    "name": "Tomate",
                    "varieties": ["Determinate", "Indeterminate", "Cherry", "Processing"],
                    "growth_duration": [90, 120, 150],
                    "yield_potential": [15, 40, 80],  # tons/ha
                    "water_requirement": 550,
                    "soil_preferences": ["loamy", "sandy_loam"],
                    "climate_zones": ["tropical", "subtropical", "temperate"],
                    "nutritional_requirements": {
                        "N": 150, "P": 80, "K": 200
                    },
                    "diseases": ["Blight", "Wilt", "Whitefly"],
                    "market_price": 400,
                    "processing_options": ["Fresh", "Paste", "Sauce", "Dried"]
                }
            }
        }
    
    def render_crop_selection(self, selected_region: Optional[Dict] = None) -> Optional[Dict]:
        """
        Render advanced crop selection interface
        """
        st.subheader("🌱 Sélection Avancée des Cultures")
        
        # Tabs for different selection methods
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Sélection Guidée", 
            "📊 Comparaison", 
            "💰 Analyse Économique", 
            "🌍 Adaptation Régionale"
        ])
        
        with tab1:
            selected_crop = self._render_guided_selection(selected_region)
        
        with tab2:
            selected_crop = self._render_crop_comparison(selected_region)
        
        with tab3:
            selected_crop = self._render_economic_analysis(selected_region)
        
        with tab4:
            selected_crop = self._render_regional_adaptation(selected_region)
        
        return selected_crop
    
    def _render_guided_selection(self, selected_region: Optional[Dict]) -> Optional[Dict]:
        """Render guided crop selection with recommendations"""
        st.markdown("**🎯 Assistant de Sélection Guidée**")
        
        # Step 1: Farm purpose
        st.markdown("##### 1. Objectif de la Production")
        
        farm_purpose = st.radio(
            "Quel est votre objectif principal ?",
            options=[
                "subsistence", "commercial", "export", "mixed", "livestock_feed"
            ],
            format_func=lambda x: {
                "subsistence": "🏠 Subsistance familiale",
                "commercial": "💰 Commerce local",
                "export": "🌍 Export international",
                "mixed": "🔄 Usage mixte",
                "livestock_feed": "🐄 Alimentation animale"
            }[x],
            horizontal=True
        )
        
        # Step 2: Experience level
        st.markdown("##### 2. Niveau d'Expérience")
        
        experience_level = st.select_slider(
            "Votre niveau d'expérience agricole :",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Intermédiaire"
        )
        
        # Step 3: Available resources
        st.markdown("##### 3. Ressources Disponibles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            irrigation_available = st.checkbox("Irrigation disponible")
            mechanization_available = st.checkbox("Mécanisation disponible")
            
        with col2:
            processing_facilities = st.checkbox("Installations de transformation")
            market_access = st.selectbox(
                "Accès au marché:",
                options=["Local", "Régional", "National", "International"]
            )
        
        # Step 4: Risk tolerance
        st.markdown("##### 4. Tolérance au Risque")
        
        risk_tolerance = st.select_slider(
            "Votre tolérance au risque :",
            options=["Très faible", "Faible", "Moyenne", "Élevée", "Très élevée"],
            value="Moyenne"
        )
        
        # Generate recommendations based on inputs
        if st.button("🎯 Obtenir des Recommandations", type="primary"):
            recommendations = self._generate_crop_recommendations(
                farm_purpose, experience_level, {
                    'irrigation': irrigation_available,
                    'mechanization': mechanization_available,
                    'processing': processing_facilities,
                    'market_access': market_access
                }, risk_tolerance, selected_region
            )
            
            if recommendations:
                st.markdown("### 🌟 Cultures Recommandées")
                
                for i, (crop_key, crop_data, score) in enumerate(recommendations[:5], 1):
                    with st.expander(f"{i}. {crop_data['name']} (Score: {score:.1f}/10)", expanded=i==1):
                        self._display_crop_recommendation_card(crop_key, crop_data, score)
                        
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button(f"Choisir {crop_data['name']}", key=f"select_{crop_key}"):
                                return self._create_crop_selection_data(crop_key, crop_data)
        
        return None
    
    def _render_crop_comparison(self, selected_region: Optional[Dict]) -> Optional[Dict]:
        """Render crop comparison interface"""
        st.markdown("**📊 Comparaison des Cultures**")
        
        # Select crops to compare
        available_crops = []
        for category, crops in self.crop_database.items():
            for crop_key, crop_data in crops.items():
                available_crops.append((f"{crop_data['name']} ({category})", crop_key, crop_data))
        
        selected_crops = st.multiselect(
            "Sélectionnez les cultures à comparer (2-5 cultures):",
            options=available_crops,
            format_func=lambda x: x[0],
            max_selections=5
        )
        
        if len(selected_crops) >= 2:
            # Comparison criteria
            st.markdown("##### Critères de Comparaison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_yield = st.checkbox("Rendement potentiel", value=True)
                show_duration = st.checkbox("Durée de croissance", value=True)
                
            with col2:
                show_water = st.checkbox("Besoin en eau", value=True)
                show_nutrients = st.checkbox("Besoins nutritionnels", value=True)
                
            with col3:
                show_price = st.checkbox("Prix du marché", value=True)
                show_risk = st.checkbox("Facteurs de risque", value=False)
            
            # Create comparison table
            comparison_data = []
            
            for crop_name, crop_key, crop_data in selected_crops:
                row = {
                    'Culture': crop_data['name'],
                    'Catégorie': crop_name.split('(')[1].rstrip(')')
                }
                
                if show_yield:
                    yield_range = crop_data.get('yield_potential', [0, 0, 0])
                    row['Rendement (t/ha)'] = f"{yield_range[0]}-{yield_range[2]}"
                
                if show_duration:
                    duration_range = crop_data.get('growth_duration', [0, 0, 0])
                    row['Durée (jours)'] = f"{duration_range[0]}-{duration_range[2]}"
                
                if show_water:
                    water_req = crop_data.get('water_requirement', 0)
                    row['Eau (mm)'] = water_req
                
                if show_nutrients:
                    nutrients = crop_data.get('nutritional_requirements', {})
                    row['NPK'] = f"{nutrients.get('N', 0)}-{nutrients.get('P', 0)}-{nutrients.get('K', 0)}"
                
                if show_price:
                    price = crop_data.get('market_price', 0)
                    row['Prix (USD/t)'] = price
                
                if show_risk:
                    diseases = crop_data.get('diseases', [])
                    row['Risques'] = len(diseases)
                
                comparison_data.append(row)
            
            # Display comparison table
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Visualization
            if show_yield and show_price:
                self._render_yield_price_scatter(selected_crops)
            
            if show_water and show_duration:
                self._render_water_duration_comparison(selected_crops)
            
            # Selection from comparison
            st.markdown("##### Faire une Sélection")
            
            selected_crop_name = st.selectbox(
                "Choisissez une culture d'après la comparaison:",
                options=[crop[0] for crop in selected_crops],
                key="comparison_selection"
            )
            
            if selected_crop_name and st.button("Confirmer la sélection"):
                # Find the selected crop data
                for crop_name, crop_key, crop_data in selected_crops:
                    if crop_name == selected_crop_name:
                        return self._create_crop_selection_data(crop_key, crop_data)
        
        return None
    
    def _render_economic_analysis(self, selected_region: Optional[Dict]) -> Optional[Dict]:
        """Render economic analysis interface"""
        st.markdown("**💰 Analyse Économique**")
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            farm_size = st.number_input(
                "Superficie de la ferme (hectares):",
                min_value=0.1,
                max_value=1000.0,
                value=2.0,
                step=0.1
            )
            
            available_capital = st.number_input(
                "Capital disponible (USD):",
                min_value=100,
                max_value=100000,
                value=2000,
                step=100
            )
        
        with col2:
            labor_cost = st.number_input(
                "Coût de la main-d'œuvre (USD/jour):",
                min_value=1.0,
                max_value=50.0,
                value=5.0,
                step=0.5
            )
            
            target_roi = st.number_input(
                "ROI cible (%):",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
        
        if st.button("📊 Analyser la Rentabilité"):
            # Calculate economic metrics for all crops
            economic_analysis = []
            
            for category, crops in self.crop_database.items():
                for crop_key, crop_data in crops.items():
                    analysis = self._calculate_economic_metrics(
                        crop_data, farm_size, available_capital, labor_cost, target_roi
                    )
                    analysis['crop_key'] = crop_key
                    analysis['crop_name'] = crop_data['name']
                    analysis['category'] = category
                    economic_analysis.append(analysis)
            
            # Sort by profitability
            economic_analysis.sort(key=lambda x: x['roi'], reverse=True)
            
            # Display top profitable crops
            st.markdown("### 💎 Cultures les Plus Rentables")
            
            profitable_crops = [crop for crop in economic_analysis if crop['roi'] >= target_roi]
            
            if profitable_crops:
                for i, crop in enumerate(profitable_crops[:5], 1):
                    with st.expander(f"{i}. {crop['crop_name']} - ROI: {crop['roi']:.1f}%"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Coût Total", f"${crop['total_cost']:.0f}")
                            st.metric("Revenus", f"${crop['total_revenue']:.0f}")
                            
                        with col2:
                            st.metric("Profit Net", f"${crop['net_profit']:.0f}")
                            st.metric("ROI", f"{crop['roi']:.1f}%")
                            
                        with col3:
                            st.metric("Seuil de Rentabilité", f"{crop['breakeven_yield']:.1f} t/ha")
                            st.metric("Période de Retour", f"{crop['payback_months']:.1f} mois")
                        
                        if st.button(f"Sélectionner {crop['crop_name']}", key=f"econ_select_{crop['crop_key']}"):
                            crop_data = None
                            for cat, crops in self.crop_database.items():
                                if crop['crop_key'] in crops:
                                    crop_data = crops[crop['crop_key']]
                                    break
                            
                            if crop_data:
                                return self._create_crop_selection_data(crop['crop_key'], crop_data)
            else:
                st.warning("Aucune culture ne répond aux critères de rentabilité spécifiés.")
                st.info("Essayez de réduire le ROI cible ou d'augmenter le capital disponible.")
        
        return None
    
    def _render_regional_adaptation(self, selected_region: Optional[Dict]) -> Optional[Dict]:
        """Render regional adaptation interface"""
        st.markdown("**🌍 Adaptation Régionale**")
        
        if not selected_region:
            st.info("Veuillez d'abord sélectionner une région pour voir les cultures adaptées.")
            return None
        
        region_name = selected_region.get('name', 'Région Inconnue')
        climate_type = selected_region.get('climate_type', 'N/A')
        
        st.markdown(f"**Région Sélectionnée:** {region_name} ({climate_type})")
        
        # Filter crops based on regional suitability
        suitable_crops = self._filter_crops_by_region(selected_region)
        
        if suitable_crops:
            st.markdown(f"### 🌿 Cultures Adaptées à {region_name}")
            
            # Categorize by suitability
            highly_suitable = [crop for crop in suitable_crops if crop['suitability_score'] >= 8]
            moderately_suitable = [crop for crop in suitable_crops if 6 <= crop['suitability_score'] < 8]
            marginally_suitable = [crop for crop in suitable_crops if crop['suitability_score'] < 6]
            
            # Display categories
            if highly_suitable:
                st.markdown("#### 🌟 Très Adaptées")
                for crop in highly_suitable:
                    self._render_regional_crop_card(crop, "success")
            
            if moderately_suitable:
                st.markdown("#### ⭐ Moyennement Adaptées")
                for crop in moderately_suitable:
                    self._render_regional_crop_card(crop, "info")
            
            if marginally_suitable:
                st.markdown("#### ⚠️ Faiblement Adaptées")
                for crop in marginally_suitable:
                    self._render_regional_crop_card(crop, "warning")
        
        else:
            st.error("Aucune culture adaptée trouvée dans la base de données.")
        
        return None
    
    def _generate_crop_recommendations(self, purpose: str, experience: str, 
                                     resources: Dict, risk_tolerance: str, 
                                     region: Optional[Dict]) -> List[Tuple]:
        """Generate crop recommendations based on user inputs"""
        recommendations = []
        
        # Scoring weights
        weights = {
            'purpose_match': 0.25,
            'experience_match': 0.15,
            'resource_match': 0.20,
            'risk_match': 0.15,
            'regional_suitability': 0.25
        }
        
        for category, crops in self.crop_database.items():
            for crop_key, crop_data in crops.items():
                score = 0
                
                # Purpose matching
                purpose_score = self._calculate_purpose_score(crop_data, purpose)
                score += purpose_score * weights['purpose_match']
                
                # Experience matching
                experience_score = self._calculate_experience_score(crop_data, experience)
                score += experience_score * weights['experience_match']
                
                # Resource matching
                resource_score = self._calculate_resource_score(crop_data, resources)
                score += resource_score * weights['resource_match']
                
                # Risk matching
                risk_score = self._calculate_risk_score(crop_data, risk_tolerance)
                score += risk_score * weights['risk_match']
                
                # Regional suitability
                if region:
                    regional_score = self._calculate_regional_score(crop_data, region)
                    score += regional_score * weights['regional_suitability']
                else:
                    score += 5 * weights['regional_suitability']  # Neutral score
                
                recommendations.append((crop_key, crop_data, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        return recommendations
    
    def _calculate_purpose_score(self, crop_data: Dict, purpose: str) -> float:
        """Calculate purpose matching score"""
        # Simplified scoring logic
        if purpose == "subsistence":
            # Prefer staple crops
            if crop_data['name'].lower() in ['maïs', 'riz', 'sorgho', 'mil']:
                return 9
            elif crop_data['name'].lower() in ['arachides', 'niébé']:
                return 7
            else:
                return 4
        
        elif purpose == "commercial":
            # Prefer high-value crops
            price = crop_data.get('market_price', 0)
            if price > 800:
                return 9
            elif price > 400:
                return 7
            else:
                return 5
        
        elif purpose == "export":
            # Prefer export crops
            if crop_data['name'].lower() in ['café', 'coton', 'arachides']:
                return 9
            else:
                return 3
        
        return 5  # Default neutral score
    
    def _calculate_experience_score(self, crop_data: Dict, experience: str) -> float:
        """Calculate experience matching score"""
        # Simplified scoring - more complex crops need more experience
        complex_crops = ['café', 'coton', 'tomate']
        
        if experience == "Débutant":
            if crop_data['name'].lower() in complex_crops:
                return 2
            else:
                return 8
        
        elif experience == "Expert":
            if crop_data['name'].lower() in complex_crops:
                return 9
            else:
                return 7
        
        return 6  # Default for intermediate
    
    def _calculate_resource_score(self, crop_data: Dict, resources: Dict) -> float:
        """Calculate resource matching score"""
        score = 5  # Base score
        
        water_req = crop_data.get('water_requirement', 500)
        
        # Irrigation requirement
        if water_req > 800 and not resources.get('irrigation', False):
            score -= 3
        elif water_req < 400 and resources.get('irrigation', False):
            score += 1
        
        # Processing facilities
        processing_options = crop_data.get('processing_options', [])
        if len(processing_options) > 2 and resources.get('processing', False):
            score += 2
        elif len(processing_options) > 2 and not resources.get('processing', False):
            score -= 1
        
        return max(0, min(10, score))
    
    def _calculate_risk_score(self, crop_data: Dict, risk_tolerance: str) -> float:
        """Calculate risk matching score"""
        diseases = crop_data.get('diseases', [])
        num_diseases = len(diseases)
        
        if risk_tolerance == "Très faible":
            return max(0, 8 - num_diseases)
        elif risk_tolerance == "Très élevée":
            return min(10, 5 + num_diseases)
        else:
            return 5  # Neutral
    
    def _calculate_regional_score(self, crop_data: Dict, region: Dict) -> float:
        """Calculate regional suitability score"""
        score = 5  # Base score
        
        climate_zones = crop_data.get('climate_zones', [])
        region_climate = region.get('climate_type', '').lower()
        
        if region_climate in [zone.lower() for zone in climate_zones]:
            score += 3
        
        # Check if crop is in region's major crops
        major_crops = region.get('major_crops', [])
        if crop_data['name'].lower() in [crop.lower() for crop in major_crops]:
            score += 2
        
        return min(10, score)
    
    def _display_crop_recommendation_card(self, crop_key: str, crop_data: Dict, score: float):
        """Display a detailed crop recommendation card"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Informations Générales**")
            varieties = crop_data.get('varieties', [])
            st.write(f"Variétés: {', '.join(varieties[:3])}")
            
            duration = crop_data.get('growth_duration', [0, 0, 0])
            st.write(f"Durée: {duration[0]}-{duration[2]} jours")
            
        with col2:
            st.markdown("**🌱 Production**")
            yield_pot = crop_data.get('yield_potential', [0, 0, 0])
            st.write(f"Rendement: {yield_pot[0]}-{yield_pot[2]} t/ha")
            
            price = crop_data.get('market_price', 0)
            st.write(f"Prix: ${price}/tonne")
            
        with col3:
            st.markdown("**⚠️ Considérations**")
            water_req = crop_data.get('water_requirement', 0)
            st.write(f"Eau: {water_req} mm")
            
            diseases = crop_data.get('diseases', [])
            st.write(f"Maladies: {len(diseases)} principales")
    
    def _render_yield_price_scatter(self, selected_crops: List):
        """Render yield vs price scatter plot"""
        st.markdown("#### 📈 Rendement vs Prix")
        
        scatter_data = []
        for crop_name, crop_key, crop_data in selected_crops:
            yield_avg = sum(crop_data.get('yield_potential', [0, 0, 0])) / 3
            price = crop_data.get('market_price', 0)
            
            scatter_data.append({
                'Culture': crop_data['name'],
                'Rendement (t/ha)': yield_avg,
                'Prix (USD/t)': price
            })
        
        df_scatter = pd.DataFrame(scatter_data)
        
        fig = px.scatter(
            df_scatter,
            x='Rendement (t/ha)',
            y='Prix (USD/t)',
            text='Culture',
            title="Rendement vs Prix des Cultures",
            size_max=20
        )
        
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_water_duration_comparison(self, selected_crops: List):
        """Render water requirement vs duration comparison"""
        st.markdown("#### 💧 Besoins en Eau vs Durée")
        
        comparison_data = []
        for crop_name, crop_key, crop_data in selected_crops:
            duration_avg = sum(crop_data.get('growth_duration', [0, 0, 0])) / 3
            water_req = crop_data.get('water_requirement', 0)
            
            comparison_data.append({
                'Culture': crop_data['name'],
                'Durée (jours)': duration_avg,
                'Eau (mm)': water_req
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            df_comparison,
            x='Culture',
            y=['Durée (jours)', 'Eau (mm)'],
            title="Comparaison des Besoins",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _calculate_economic_metrics(self, crop_data: Dict, farm_size: float, 
                                  capital: float, labor_cost: float, target_roi: float) -> Dict:
        """Calculate economic metrics for a crop"""
        # Simplified economic calculation
        yield_avg = sum(crop_data.get('yield_potential', [1, 2, 3])) / 3
        price = crop_data.get('market_price', 300)
        duration_avg = sum(crop_data.get('growth_duration', [90, 120, 150])) / 3
        
        # Costs
        seed_cost = farm_size * 50  # USD per hectare
        fertilizer_cost = farm_size * 200  # USD per hectare
        labor_days = duration_avg / 10  # Simplified
        labor_total = labor_days * labor_cost * farm_size
        other_costs = farm_size * 100  # Other costs
        
        total_cost = seed_cost + fertilizer_cost + labor_total + other_costs
        
        # Revenue
        total_yield = yield_avg * farm_size
        total_revenue = total_yield * price
        
        # Metrics
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        breakeven_yield = total_cost / (price * farm_size) if price > 0 else 0
        payback_months = (duration_avg / 30) if roi > 0 else float('inf')
        
        return {
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'net_profit': net_profit,
            'roi': roi,
            'breakeven_yield': breakeven_yield,
            'payback_months': payback_months
        }
    
    def _filter_crops_by_region(self, region: Dict) -> List[Dict]:
        """Filter crops based on regional suitability"""
        suitable_crops = []
        
        for category, crops in self.crop_database.items():
            for crop_key, crop_data in crops.items():
                suitability_score = self._calculate_regional_score(crop_data, region)
                
                if suitability_score >= 4:  # Minimum threshold
                    suitable_crops.append({
                        'crop_key': crop_key,
                        'crop_data': crop_data,
                        'category': category,
                        'suitability_score': suitability_score
                    })
        
        return sorted(suitable_crops, key=lambda x: x['suitability_score'], reverse=True)
    
    def _render_regional_crop_card(self, crop_info: Dict, status_type: str):
        """Render a regional crop suitability card"""
        crop_data = crop_info['crop_data']
        score = crop_info['suitability_score']
        
        with st.container():
            if status_type == "success":
                st.success(f"🌟 **{crop_data['name']}** (Score: {score:.1f}/10)")
            elif status_type == "info":
                st.info(f"⭐ **{crop_data['name']}** (Score: {score:.1f}/10)")
            else:
                st.warning(f"⚠️ **{crop_data['name']}** (Score: {score:.1f}/10)")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                varieties = crop_data.get('varieties', [])[:2]
                st.write(f"Variétés: {', '.join(varieties)}")
                
            with col2:
                yield_range = crop_data.get('yield_potential', [0, 0, 0])
                st.write(f"Rendement: {yield_range[0]}-{yield_range[2]} t/ha")
                
            with col3:
                if st.button("Choisir", key=f"regional_{crop_info['crop_key']}"):
                    return self._create_crop_selection_data(crop_info['crop_key'], crop_data)
    
    def _create_crop_selection_data(self, crop_key: str, crop_data: Dict) -> Dict:
        """Create crop selection data structure"""
        return {
            'crop_type': crop_key,
            'crop_name': crop_data['name'],
            'variety': crop_data.get('varieties', ['Standard'])[0],
            'planting_season': 'wet_season',  # Default
            'growth_duration': crop_data.get('growth_duration', [120])[1],  # Medium duration
            'yield_potential': crop_data.get('yield_potential', [0, 0, 0]),
            'nutritional_requirements': crop_data.get('nutritional_requirements', {}),
            'water_requirement': crop_data.get('water_requirement', 500),
            'market_price': crop_data.get('market_price', 300),
            'processing_options': crop_data.get('processing_options', []),
            'diseases': crop_data.get('diseases', [])
        }