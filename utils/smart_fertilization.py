
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CropDatabase:
    """Base de données des cultures et stades de croissance"""
    
    def __init__(self):
        self.crops_data = {
            'wheat': {
                'name': 'Blé',
                'growth_stages': {
                    'germination': {
                        'duration_days': 7,
                        'nutrients': {'N': 15, 'P': 20, 'K': 10},
                        'description': 'Émergence et établissement'
                    },
                    'tallage': {
                        'duration_days': 30,
                        'nutrients': {'N': 40, 'P': 15, 'K': 20},
                        'description': 'Développement des talles'
                    },
                    'montaison': {
                        'duration_days': 45,
                        'nutrients': {'N': 60, 'P': 10, 'K': 30},
                        'description': 'Élongation des tiges'
                    },
                    'epiaison': {
                        'duration_days': 20,
                        'nutrients': {'N': 30, 'P': 25, 'K': 40},
                        'description': 'Formation des épis'
                    },
                    'floraison': {
                        'duration_days': 15,
                        'nutrients': {'N': 20, 'P': 30, 'K': 35},
                        'description': 'Floraison et fécondation'
                    },
                    'maturation': {
                        'duration_days': 35,
                        'nutrients': {'N': 10, 'P': 15, 'K': 20},
                        'description': 'Remplissage des grains'
                    }
                },
                'total_nutrients': {'N': 175, 'P': 115, 'K': 155},
                'micro_elements': {'Zn': 2, 'Mn': 1.5, 'Cu': 1, 'B': 0.5}
            },
            'corn': {
                'name': 'Maïs',
                'growth_stages': {
                    'emergence': {
                        'duration_days': 10,
                        'nutrients': {'N': 20, 'P': 25, 'K': 15},
                        'description': 'Émergence et première feuille'
                    },
                    'vegetatif': {
                        'duration_days': 50,
                        'nutrients': {'N': 80, 'P': 20, 'K': 40},
                        'description': 'Croissance végétative'
                    },
                    'floraison': {
                        'duration_days': 20,
                        'nutrients': {'N': 60, 'P': 30, 'K': 60},
                        'description': 'Floraison mâle et femelle'
                    },
                    'remplissage': {
                        'duration_days': 45,
                        'nutrients': {'N': 40, 'P': 25, 'K': 50},
                        'description': 'Remplissage des grains'
                    },
                    'maturation': {
                        'duration_days': 30,
                        'nutrients': {'N': 15, 'P': 10, 'K': 25},
                        'description': 'Maturation physiologique'
                    }
                },
                'total_nutrients': {'N': 215, 'P': 110, 'K': 190},
                'micro_elements': {'Zn': 3, 'Mn': 2, 'Cu': 1.5, 'B': 0.8}
            },
            'rice': {
                'name': 'Riz',
                'growth_stages': {
                    'semis': {
                        'duration_days': 15,
                        'nutrients': {'N': 25, 'P': 30, 'K': 20},
                        'description': 'Semis et germination'
                    },
                    'tallage': {
                        'duration_days': 35,
                        'nutrients': {'N': 70, 'P': 25, 'K': 35},
                        'description': 'Formation des talles'
                    },
                    'montaison': {
                        'duration_days': 30,
                        'nutrients': {'N': 50, 'P': 15, 'K': 45},
                        'description': 'Élongation des tiges'
                    },
                    'paniculation': {
                        'duration_days': 25,
                        'nutrients': {'N': 40, 'P': 35, 'K': 55},
                        'description': 'Formation des panicules'
                    },
                    'remplissage': {
                        'duration_days': 30,
                        'nutrients': {'N': 25, 'P': 20, 'K': 40},
                        'description': 'Remplissage des grains'
                    }
                },
                'total_nutrients': {'N': 210, 'P': 125, 'K': 195},
                'micro_elements': {'Zn': 4, 'Mn': 3, 'Cu': 1, 'B': 1}
            },
            'soybeans': {
                'name': 'Soja',
                'growth_stages': {
                    'emergence': {
                        'duration_days': 8,
                        'nutrients': {'N': 10, 'P': 25, 'K': 15},
                        'description': 'Émergence des cotylédons'
                    },
                    'croissance': {
                        'duration_days': 40,
                        'nutrients': {'N': 20, 'P': 30, 'K': 40},
                        'description': 'Croissance végétative'
                    },
                    'floraison': {
                        'duration_days': 25,
                        'nutrients': {'N': 15, 'P': 40, 'K': 50},
                        'description': 'Floraison et formation gousses'
                    },
                    'remplissage': {
                        'duration_days': 35,
                        'nutrients': {'N': 25, 'P': 35, 'K': 45},
                        'description': 'Remplissage des graines'
                    },
                    'maturation': {
                        'duration_days': 20,
                        'nutrients': {'N': 10, 'P': 15, 'K': 25},
                        'description': 'Maturation des graines'
                    }
                },
                'total_nutrients': {'N': 80, 'P': 145, 'K': 175},
                'micro_elements': {'Zn': 2, 'Mn': 2.5, 'Cu': 1, 'B': 1.5}
            }
        }
    
    def get_crop_info(self, crop_type: str) -> Dict:
        """Récupère les informations d'une culture"""
        return self.crops_data.get(crop_type.lower(), {})
    
    def get_current_stage(self, crop_type: str, planting_date: str) -> Dict:
        """Détermine le stade actuel de la culture"""
        crop_info = self.get_crop_info(crop_type)
        if not crop_info:
            return {}
        
        try:
            planting_dt = datetime.fromisoformat(planting_date)
            days_since_planting = (datetime.now() - planting_dt).days
            
            current_days = 0
            for stage_name, stage_info in crop_info['growth_stages'].items():
                stage_duration = stage_info['duration_days']
                if days_since_planting <= current_days + stage_duration:
                    return {
                        'stage_name': stage_name,
                        'stage_info': stage_info,
                        'days_in_stage': days_since_planting - current_days,
                        'days_remaining': current_days + stage_duration - days_since_planting
                    }
                current_days += stage_duration
            
            # Si au-delà de tous les stades
            return {
                'stage_name': 'mature',
                'stage_info': {'nutrients': {'N': 0, 'P': 0, 'K': 0}},
                'days_in_stage': days_since_planting - current_days,
                'days_remaining': 0
            }
            
        except:
            return {}

class SmartFertilizationSystem:
    """Système de fertilisation intelligent avec IA"""
    
    def __init__(self):
        self.crop_db = CropDatabase()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.fertilization_history = []
        
    def analyze_soil_conditions(self, soil_data: Dict) -> Dict:
        """Analyse les conditions du sol"""
        ph = soil_data.get('ph', 6.5)
        nitrogen = soil_data.get('nitrogen', 40)
        phosphorus = soil_data.get('phosphorus', 25)
        potassium = soil_data.get('potassium', 200)
        organic_matter = soil_data.get('organic_matter', 3.0)
        moisture = soil_data.get('moisture', 55)
        
        # Évaluation de la disponibilité des nutriments
        nutrient_availability = {
            'N': self._calculate_n_availability(ph, organic_matter, moisture),
            'P': self._calculate_p_availability(ph, phosphorus),
            'K': self._calculate_k_availability(ph, potassium)
        }
        
        # Facteurs de correction basés sur les conditions
        correction_factors = {
            'ph_factor': self._get_ph_correction(ph),
            'moisture_factor': self._get_moisture_correction(moisture),
            'organic_matter_factor': self._get_om_correction(organic_matter)
        }
        
        return {
            'nutrient_availability': nutrient_availability,
            'correction_factors': correction_factors,
            'soil_quality_score': self._calculate_soil_quality(soil_data)
        }
    
    def _calculate_n_availability(self, ph: float, om: float, moisture: float) -> float:
        """Calcule la disponibilité de l'azote"""
        base_availability = 0.7
        
        # Correction pH
        if 6.0 <= ph <= 7.5:
            ph_factor = 1.0
        elif ph < 6.0:
            ph_factor = 0.8 - (6.0 - ph) * 0.1
        else:
            ph_factor = 0.9 - (ph - 7.5) * 0.05
        
        # Correction matière organique
        om_factor = min(1.2, 0.6 + om * 0.15)
        
        # Correction humidité
        moisture_factor = 1.0 if 40 <= moisture <= 70 else 0.8
        
        return base_availability * ph_factor * om_factor * moisture_factor
    
    def _calculate_p_availability(self, ph: float, p_content: float) -> float:
        """Calcule la disponibilité du phosphore"""
        base_availability = 0.6
        
        # Le phosphore est mieux disponible à pH neutre
        if 6.5 <= ph <= 7.0:
            ph_factor = 1.0
        else:
            ph_factor = 0.7 - abs(ph - 6.75) * 0.1
        
        # Facteur basé sur le contenu
        content_factor = min(1.5, p_content / 30)
        
        return base_availability * ph_factor * content_factor
    
    def _calculate_k_availability(self, ph: float, k_content: float) -> float:
        """Calcule la disponibilité du potassium"""
        base_availability = 0.8
        
        # Le potassium est généralement bien disponible
        ph_factor = 0.95 if ph < 5.5 else 1.0
        content_factor = min(1.3, k_content / 200)
        
        return base_availability * ph_factor * content_factor
    
    def _get_ph_correction(self, ph: float) -> float:
        """Facteur de correction basé sur le pH"""
        if 6.0 <= ph <= 7.5:
            return 1.0
        elif ph < 6.0:
            return 0.8 - (6.0 - ph) * 0.05
        else:
            return 0.9 - (ph - 7.5) * 0.03
    
    def _get_moisture_correction(self, moisture: float) -> float:
        """Facteur de correction basé sur l'humidité"""
        if 40 <= moisture <= 70:
            return 1.0
        elif moisture < 40:
            return 0.7 + moisture * 0.0075
        else:
            return 1.1 - (moisture - 70) * 0.005
    
    def _get_om_correction(self, organic_matter: float) -> float:
        """Facteur de correction basé sur la matière organique"""
        return min(1.2, 0.8 + organic_matter * 0.1)
    
    def _calculate_soil_quality(self, soil_data: Dict) -> float:
        """Calcule un score de qualité du sol (0-100)"""
        ph = soil_data.get('ph', 6.5)
        organic_matter = soil_data.get('organic_matter', 3.0)
        moisture = soil_data.get('moisture', 55)
        
        # Score pH (optimal 6.0-7.5)
        if 6.0 <= ph <= 7.5:
            ph_score = 100
        elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
            ph_score = 80
        else:
            ph_score = max(0, 60 - abs(ph - 6.75) * 20)
        
        # Score matière organique (optimal >3%)
        om_score = min(100, organic_matter * 25)
        
        # Score humidité (optimal 40-70%)
        if 40 <= moisture <= 70:
            moisture_score = 100
        else:
            moisture_score = max(0, 100 - abs(moisture - 55) * 2)
        
        return (ph_score + om_score + moisture_score) / 3
    
    def generate_fertilization_plan(self, 
                                  crop_type: str,
                                  planting_date: str,
                                  area: float,
                                  soil_data: Dict,
                                  weather_forecast: List[Dict] = None,
                                  target_yield: float = None) -> Dict:
        """Génère un plan de fertilisation personnalisé"""
        
        # Analyse du sol
        soil_analysis = self.analyze_soil_conditions(soil_data)
        
        # Informations de la culture
        crop_info = self.crop_db.get_crop_info(crop_type)
        current_stage = self.crop_db.get_current_stage(crop_type, planting_date)
        
        if not crop_info or not current_stage:
            return {'error': 'Culture ou stade non reconnu'}
        
        # Calcul des besoins ajustés
        adjusted_nutrients = self._calculate_adjusted_nutrients(
            crop_info, current_stage, soil_analysis, target_yield
        )
        
        # Génération du calendrier de fertilisation
        fertilization_schedule = self._create_fertilization_schedule(
            crop_info, current_stage, adjusted_nutrients, area, weather_forecast
        )
        
        # Recommandations spécifiques
        recommendations = self._generate_recommendations(
            soil_analysis, current_stage, weather_forecast
        )
        
        return {
            'crop_info': {
                'name': crop_info['name'],
                'current_stage': current_stage['stage_name'],
                'days_remaining_stage': current_stage['days_remaining']
            },
            'soil_analysis': soil_analysis,
            'adjusted_nutrients': adjusted_nutrients,
            'fertilization_schedule': fertilization_schedule,
            'recommendations': recommendations,
            'total_cost_estimate': self._estimate_costs(fertilization_schedule),
            'plan_generated_date': datetime.now().isoformat()
        }
    
    def _calculate_adjusted_nutrients(self, crop_info: Dict, current_stage: Dict, 
                                    soil_analysis: Dict, target_yield: float = None) -> Dict:
        """Calcule les besoins en nutriments ajustés"""
        base_nutrients = crop_info['total_nutrients'].copy()
        availability = soil_analysis['nutrient_availability']
        
        # Ajustement basé sur la disponibilité
        adjusted = {}
        for nutrient, base_amount in base_nutrients.items():
            # Facteur de disponibilité
            avail_factor = availability.get(nutrient, 0.7)
            
            # Ajustement pour le rendement cible
            yield_factor = 1.0
            if target_yield:
                # Supposons un rendement de base de 5 t/ha
                yield_factor = min(1.5, target_yield / 5.0)
            
            adjusted[nutrient] = base_amount / avail_factor * yield_factor
        
        return adjusted
    
    def _create_fertilization_schedule(self, crop_info: Dict, current_stage: Dict,
                                     adjusted_nutrients: Dict, area: float,
                                     weather_forecast: List[Dict] = None) -> List[Dict]:
        """Crée le calendrier de fertilisation"""
        schedule = []
        
        # Répartition des nutriments par stade
        for stage_name, stage_info in crop_info['growth_stages'].items():
            stage_nutrients = stage_info['nutrients']
            
            # Calcul des quantités par hectare
            applications = []
            for nutrient, stage_need in stage_nutrients.items():
                if nutrient in adjusted_nutrients:
                    # Pourcentage de ce nutriment pour ce stade
                    percentage = stage_need / crop_info['total_nutrients'][nutrient]
                    total_needed = adjusted_nutrients[nutrient] * percentage
                    
                    # Répartition en plusieurs applications si nécessaire
                    if nutrient == 'N' and total_needed > 60:
                        # Azote : fractionner si > 60 kg/ha
                        num_apps = min(3, int(total_needed / 40) + 1)
                        app_amount = total_needed / num_apps
                        
                        for i in range(num_apps):
                            applications.append({
                                'nutrient': nutrient,
                                'amount_per_ha': app_amount,
                                'total_amount': app_amount * area,
                                'application_number': i + 1,
                                'fertilizer_type': self._get_fertilizer_type(nutrient, stage_name)
                            })
                    else:
                        applications.append({
                            'nutrient': nutrient,
                            'amount_per_ha': total_needed,
                            'total_amount': total_needed * area,
                            'application_number': 1,
                            'fertilizer_type': self._get_fertilizer_type(nutrient, stage_name)
                        })
            
            if applications:
                schedule.append({
                    'stage': stage_name,
                    'stage_description': stage_info['description'],
                    'applications': applications,
                    'timing_recommendations': self._get_timing_recommendations(stage_name)
                })
        
        return schedule
    
    def _get_fertilizer_type(self, nutrient: str, stage: str) -> str:
        """Recommande le type d'engrais"""
        fertilizer_types = {
            'N': {
                'early': 'Urée (46-0-0)',
                'mid': 'Nitrate d\'ammonium (33-0-0)',
                'late': 'Solution azotée (30-0-0)'
            },
            'P': {
                'early': 'Superphosphate (0-45-0)',
                'mid': 'DAP (18-46-0)',
                'late': 'MAP (11-52-0)'
            },
            'K': {
                'early': 'Chlorure de potassium (0-0-60)',
                'mid': 'Sulfate de potassium (0-0-50)',
                'late': 'Nitrate de potassium (13-0-46)'
            }
        }
        
        stage_timing = 'early' if stage in ['germination', 'emergence', 'semis'] else \
                      'late' if stage in ['maturation', 'mature'] else 'mid'
        
        return fertilizer_types.get(nutrient, {}).get(stage_timing, f'Engrais {nutrient}')
    
    def _get_timing_recommendations(self, stage: str) -> List[str]:
        """Recommandations de timing pour l'application"""
        recommendations = {
            'germination': ['Appliquer 2-3 jours avant semis', 'Incorporer au sol'],
            'tallage': ['Appliquer en début de stade', 'Éviter les périodes de gel'],
            'montaison': ['Fractionner les apports', 'Surveiller les prévisions météo'],
            'floraison': ['Application foliaire possible', 'Éviter les heures chaudes'],
            'remplissage': ['Derniers apports avant arrêt', 'Privilégier le matin'],
            'maturation': ['Arrêter la fertilisation azotée', 'Favoriser K et P']
        }
        
        return recommendations.get(stage, ['Suivre les recommandations générales'])
    
    def _generate_recommendations(self, soil_analysis: Dict, current_stage: Dict, 
                                weather_forecast: List[Dict] = None) -> List[str]:
        """Génère des recommandations spécifiques"""
        recommendations = []
        
        # Recommandations basées sur le sol
        soil_quality = soil_analysis['soil_quality_score']
        if soil_quality < 60:
            recommendations.append("⚠️ Qualité du sol faible - Considérer un amendement organique")
        
        availability = soil_analysis['nutrient_availability']
        for nutrient, avail in availability.items():
            if avail < 0.6:
                recommendations.append(f"📉 Faible disponibilité {nutrient} - Augmenter les doses")
        
        # Recommandations basées sur le stade
        stage_name = current_stage['stage_name']
        if stage_name in ['floraison', 'remplissage']:
            recommendations.append("🌸 Stade critique - Surveiller étroitement la nutrition")
        
        # Recommandations météo
        if weather_forecast:
            for forecast in weather_forecast[:7]:  # 7 prochains jours
                if forecast.get('rainfall', 0) > 20:
                    recommendations.append("🌧️ Pluies prévues - Reporter l'application d'engrais")
                    break
        
        recommendations.append("📅 Programmer des analyses de sol régulières")
        recommendations.append("🔄 Ajuster selon l'évolution de la culture")
        
        return recommendations
    
    def _estimate_costs(self, schedule: List[Dict]) -> Dict:
        """Estime les coûts de fertilisation"""
        # Prix approximatifs par kg de nutriment (€)
        nutrient_prices = {'N': 1.2, 'P': 2.5, 'K': 1.0}
        
        total_cost = 0
        cost_breakdown = {}
        
        for stage in schedule:
            stage_cost = 0
            for app in stage['applications']:
                nutrient = app['nutrient']
                amount = app['total_amount']
                cost = amount * nutrient_prices.get(nutrient, 1.5)
                stage_cost += cost
                
                if nutrient not in cost_breakdown:
                    cost_breakdown[nutrient] = 0
                cost_breakdown[nutrient] += cost
            
            total_cost += stage_cost
        
        return {
            'total_cost_euros': round(total_cost, 2),
            'cost_breakdown': {k: round(v, 2) for k, v in cost_breakdown.items()},
            'cost_per_hectare': round(total_cost / max(1, len(schedule)), 2)
        }
    
    def train_optimization_model(self, historical_data: List[Dict]):
        """Entraîne le modèle IA d'optimisation"""
        if len(historical_data) < 10:
            return False
        
        try:
            # Préparation des données d'entraînement
            features = []
            targets = []
            
            for record in historical_data:
                feature_row = [
                    record.get('soil_ph', 6.5),
                    record.get('soil_nitrogen', 40),
                    record.get('soil_phosphorus', 25),
                    record.get('soil_potassium', 200),
                    record.get('organic_matter', 3.0),
                    record.get('fertilizer_n_applied', 150),
                    record.get('fertilizer_p_applied', 80),
                    record.get('fertilizer_k_applied', 120),
                    record.get('rainfall_season', 500),
                    record.get('temperature_avg', 20)
                ]
                features.append(feature_row)
                targets.append(record.get('yield_achieved', 5.0))
            
            # Normalisation
            X = self.scaler.fit_transform(features)
            y = np.array(targets)
            
            # Entraînement
            self.ml_model.fit(X, y)
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"Erreur d'entraînement: {e}")
            return False
    
    def optimize_fertilization_ai(self, base_plan: Dict, soil_data: Dict, 
                                 weather_data: Dict) -> Dict:
        """Optimise le plan avec l'IA"""
        if not self.is_trained:
            return base_plan
        
        try:
            # Prédiction avec le modèle actuel
            current_features = [
                soil_data.get('ph', 6.5),
                soil_data.get('nitrogen', 40),
                soil_data.get('phosphorus', 25),
                soil_data.get('potassium', 200),
                soil_data.get('organic_matter', 3.0),
                base_plan['adjusted_nutrients'].get('N', 150),
                base_plan['adjusted_nutrients'].get('P', 80),
                base_plan['adjusted_nutrients'].get('K', 120),
                weather_data.get('total_rainfall', 500),
                weather_data.get('avg_temperature', 20)
            ]
            
            X_scaled = self.scaler.transform([current_features])
            predicted_yield = self.ml_model.predict(X_scaled)[0]
            
            # Optimisation simple par ajustement des doses
            optimized_nutrients = base_plan['adjusted_nutrients'].copy()
            
            # Si le rendement prédit est faible, augmenter légèrement les doses
            if predicted_yield < 4.0:
                for nutrient in optimized_nutrients:
                    optimized_nutrients[nutrient] *= 1.1
            elif predicted_yield > 7.0:
                # Si très bon rendement prédit, peut-être réduire
                for nutrient in optimized_nutrients:
                    optimized_nutrients[nutrient] *= 0.95
            
            base_plan['adjusted_nutrients'] = optimized_nutrients
            base_plan['ai_optimization'] = {
                'predicted_yield': round(predicted_yield, 2),
                'optimization_applied': True,
                'confidence_score': 0.85
            }
            
            return base_plan
            
        except Exception as e:
            print(f"Erreur optimisation IA: {e}")
            return base_plan

# Instance globale
smart_fertilization = SmartFertilizationSystem()
