from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from modules.smart_fertilizer.api.models import SoilAnalysis, CropSelection, ApplicationTiming
class SmartFertilization:
    """
    Smart fertilization system that provides intelligent timing and application strategies
    """

    def __init__(self):
        self.fertilizer_interactions = self._load_fertilizer_interactions()
        self.soil_mobility_factors = self._load_mobility_factors()
        self.crop_uptake_patterns = self._load_uptake_patterns()

    def _load_fertilizer_interactions(self) -> Dict:
        """Load fertilizer interaction data"""
        return {
            "antagonistic": [
                ("phosphorus", "zinc"),
                ("potassium", "magnesium"),
                ("calcium", "potassium"),
                ("iron", "phosphorus")
            ],
            "synergistic": [
                ("nitrogen", "phosphorus"),
                ("calcium", "phosphorus"),
                ("sulfur", "nitrogen"),
                ("magnesium", "phosphorus")
            ],
            "neutral": [
                ("nitrogen", "potassium"),
                ("phosphorus", "potassium")
            ]
        }

    def _load_mobility_factors(self) -> Dict:
        """Load nutrient mobility factors in soil"""
        return {
            "nitrogen": {
                "mobility": "high",
                "leaching_risk": "high",
                "optimal_split": 3,
                "rain_delay_hours": 24
            },
            "phosphorus": {
                "mobility": "low",
                "leaching_risk": "low",
                "optimal_split": 1,
                "rain_delay_hours": 6
            },
            "potassium": {
                "mobility": "medium",
                "leaching_risk": "medium",
                "optimal_split": 2,
                "rain_delay_hours": 12
            }
        }

    def _load_uptake_patterns(self) -> Dict:
        """Load crop nutrient uptake patterns"""
        return {
            "maize": {
                "n_uptake_curve": [0, 15, 45, 25, 10, 5],  # % uptake by growth stage
                "p_uptake_curve": [5, 25, 35, 25, 10, 0],
                "k_uptake_curve": [10, 20, 30, 25, 15, 0],
                "critical_stages": ["v6", "tasseling", "grain_filling"]
            },
            "rice": {
                "n_uptake_curve": [0, 20, 40, 25, 15, 0],
                "p_uptake_curve": [10, 30, 30, 20, 10, 0],
                "k_uptake_curve": [15, 25, 25, 20, 15, 0],
                "critical_stages": ["tillering", "panicle_initiation", "grain_filling"]
            },
            "wheat": {
                "n_uptake_curve": [0, 10, 35, 30, 20, 5],
                "p_uptake_curve": [5, 20, 35, 25, 15, 0],
                "k_uptake_curve": [10, 25, 30, 20, 15, 0],
                "critical_stages": ["tillering", "stem_elongation", "grain_filling"]
            }
        }

    def optimize_application_timing(self, crop_selection: CropSelection,
                                  fertilizer_schedule: List[ApplicationTiming],
                                  weather_data: Optional[Dict] = None,
                                  soil_analysis: Optional[SoilAnalysis] = None) -> List[ApplicationTiming]:
        """
        Optimize fertilizer application timing based on crop needs and environmental conditions
        """

        crop_type = crop_selection.crop_type.lower()
        uptake_pattern = self.crop_uptake_patterns.get(crop_type, self.crop_uptake_patterns["maize"])

        optimized_schedule = []

        for application in fertilizer_schedule:
            # Get the dominant nutrient in this application
            dominant_nutrient = self._identify_dominant_nutrient(application.fertilizer_type)

            # Apply timing adjustments
            adjusted_application = self._adjust_timing_for_conditions(
                application, dominant_nutrient, weather_data, soil_analysis
            )

            # Check for nutrient interactions
            adjusted_application = self._check_nutrient_interactions(
                adjusted_application, optimized_schedule
            )

            optimized_schedule.append(adjusted_application)

        return optimized_schedule

    def _identify_dominant_nutrient(self, fertilizer_type: str) -> str:
        """Identify the dominant nutrient in a fertilizer"""
        fertilizer_lower = fertilizer_type.lower()

        if "urea" in fertilizer_lower or "nitrogen" in fertilizer_lower:
            return "nitrogen"
        elif "phosphor" in fertilizer_lower or "dap" in fertilizer_lower or "tsp" in fertilizer_lower:
            return "phosphorus"
        elif "potash" in fertilizer_lower or "kcl" in fertilizer_lower:
            return "potassium"
        elif "npk" in fertilizer_lower:
            return "nitrogen"  # Default to nitrogen for NPK
        else:
            return "nitrogen"  # Default fallback

    def _adjust_timing_for_conditions(self, application: ApplicationTiming,
                                    dominant_nutrient: str,
                                    weather_data: Optional[Dict],
                                    soil_analysis: Optional[SoilAnalysis]) -> ApplicationTiming:
        """Adjust application timing based on environmental conditions"""

        adjusted_application = application.model_copy()
        mobility_data = self.soil_mobility_factors.get(dominant_nutrient, {})

        # Weather-based adjustments
        if weather_data:
            rainfall_forecast = weather_data.get("rainfall_forecast", [])

            # Check for heavy rain in the next few days
            rain_delay_hours = mobility_data.get("rain_delay_hours", 12)
            if self._heavy_rain_expected(rainfall_forecast, rain_delay_hours):
                adjusted_application.notes += f" | Weather advisory: Delay application due to expected heavy rainfall"

        # Soil-based adjustments
        if soil_analysis:
            # Adjust for soil texture
            if hasattr(soil_analysis, 'texture'):
                if soil_analysis.texture == "sandy" and dominant_nutrient == "nitrogen":
                    adjusted_application.notes += " | Sandy soil: Consider split application to reduce leaching"
                elif soil_analysis.texture == "clay" and dominant_nutrient == "phosphorus":
                    adjusted_application.notes += " | Clay soil: Apply phosphorus with organic matter for better availability"

            # Adjust for soil pH
            if hasattr(soil_analysis, 'ph'):
                if soil_analysis.ph < 6.0 and dominant_nutrient == "phosphorus":
                    adjusted_application.notes += " | Low pH: Consider liming before phosphorus application"
                elif soil_analysis.ph > 7.5 and dominant_nutrient in ["iron", "zinc", "manganese"]:
                    adjusted_application.notes += " | High pH: Use chelated forms of micronutrients"

        return adjusted_application

    def _heavy_rain_expected(self, rainfall_forecast: List[Dict], hours_ahead: int) -> bool:
        """Check if heavy rain is expected within specified hours"""
        if not rainfall_forecast:
            return False

        current_time = datetime.now()
        threshold_time = current_time + timedelta(hours=hours_ahead)

        for forecast in rainfall_forecast:
            forecast_time = datetime.fromisoformat(forecast.get("datetime", ""))
            if forecast_time <= threshold_time:
                rainfall_mm = forecast.get("rainfall_mm", 0)
                if rainfall_mm > 10:  # Heavy rain threshold
                    return True

        return False

    def _check_nutrient_interactions(self, current_application: ApplicationTiming,
                                   existing_schedule: List[ApplicationTiming]) -> ApplicationTiming:
        """Check for nutrient interactions and adjust accordingly"""

        current_nutrient = self._identify_dominant_nutrient(current_application.fertilizer_type)

        # Check for antagonistic interactions with recent applications
        for existing_app in existing_schedule:
            existing_nutrient = self._identify_dominant_nutrient(existing_app.fertilizer_type)

            # Check if applications are close in time (within 7 days)
            time_diff = abs(current_application.days_after_planting - existing_app.days_after_planting)

            if time_diff <= 7:
                interaction_type = self._get_interaction_type(current_nutrient, existing_nutrient)

                if interaction_type == "antagonistic":
                    current_application.notes += f" | Interaction warning: Separate from {existing_nutrient} application by at least 7 days"
                elif interaction_type == "synergistic":
                    current_application.notes += f" | Synergistic: Can be applied together with {existing_nutrient}"

        return current_application

    def _get_interaction_type(self, nutrient1: str, nutrient2: str) -> str:
        """Get interaction type between two nutrients"""
        interactions = self.fertilizer_interactions

        for interaction in interactions["antagonistic"]:
            if (nutrient1 in interaction and nutrient2 in interaction):
                return "antagonistic"

        for interaction in interactions["synergistic"]:
            if (nutrient1 in interaction and nutrient2 in interaction):
                return "synergistic"

        return "neutral"

    def calculate_split_application_strategy(self, total_amount: float,
                                           nutrient_type: str,
                                           crop_type: str,
                                           growth_duration: int) -> List[Dict]:
        """Calculate optimal split application strategy"""

        mobility_data = self.soil_mobility_factors.get(nutrient_type, {})
        optimal_splits = mobility_data.get("optimal_split", 2)

        uptake_pattern = self.crop_uptake_patterns.get(crop_type, {})
        uptake_curve = uptake_pattern.get(f"{nutrient_type}_uptake_curve", [0, 25, 50, 25])

        split_strategy = []
        cumulative_percent = 0

        for i in range(optimal_splits):
            if i < len(uptake_curve):
                percent_this_split = uptake_curve[i]
            else:
                percent_this_split = (100 - cumulative_percent) / (optimal_splits - i)

            amount_this_split = (percent_this_split / 100) * total_amount
            timing_days = (growth_duration / optimal_splits) * (i + 1)

            split_strategy.append({
                "split_number": i + 1,
                "amount_kg_per_ha": amount_this_split,
                "timing_days_after_planting": int(timing_days),
                "percent_of_total": percent_this_split
            })

            cumulative_percent += percent_this_split

        return split_strategy

    def generate_precision_recommendations(self, soil_variability: Dict,
                                         field_zones: List[Dict]) -> List[Dict]:
        """Generate precision agriculture recommendations for variable rate application"""

        zone_recommendations = []

        for zone in field_zones:
            zone_id = zone.get("zone_id")
            zone_soil = zone.get("soil_properties", {})
            zone_size = zone.get("area_hectares", 1.0)

            # Adjust fertilizer rates based on zone characteristics
            base_n_rate = zone.get("base_n_rate", 150)
            base_p_rate = zone.get("base_p_rate", 60)
            base_k_rate = zone.get("base_k_rate", 80)

            # Soil-based adjustments
            om_factor = max(0.8, min(1.2, zone_soil.get("organic_matter", 3.0) / 3.0))
            ph_factor = 1.0

            soil_ph = zone_soil.get("ph", 6.5)
            if soil_ph < 6.0:
                ph_factor = 0.9  # Reduce rates for acidic soils
            elif soil_ph > 7.5:
                ph_factor = 1.1  # Increase rates for alkaline soils

            # Calculate adjusted rates
            adjusted_n = base_n_rate * om_factor * ph_factor
            adjusted_p = base_p_rate * ph_factor
            adjusted_k = base_k_rate * om_factor

            zone_recommendation = {
                "zone_id": zone_id,
                "area_hectares": zone_size,
                "n_rate_kg_per_ha": round(adjusted_n, 1),
                "p_rate_kg_per_ha": round(adjusted_p, 1),
                "k_rate_kg_per_ha": round(adjusted_k, 1),
                "adjustment_factors": {
                    "organic_matter": om_factor,
                    "ph": ph_factor
                },
                "special_considerations": self._generate_zone_considerations(zone_soil)
            }

            zone_recommendations.append(zone_recommendation)

        return zone_recommendations

    def _generate_zone_considerations(self, soil_properties: Dict) -> List[str]:
        """Generate special considerations for each zone"""
        considerations = []

        ph = soil_properties.get("ph", 6.5)
        if ph < 5.5:
            considerations.append("Consider liming to improve nutrient availability")
        elif ph > 8.0:
            considerations.append("Monitor for micronutrient deficiencies")

        om = soil_properties.get("organic_matter", 3.0)
        if om < 2.0:
            considerations.append("Increase organic matter through compost or cover crops")
        elif om > 5.0:
            considerations.append("Reduce nitrogen rates due to high organic matter")

        cec = soil_properties.get("cec", 15)
        if cec < 10:
            considerations.append("Split applications to reduce nutrient leaching")

        return considerations

    def validate_application_feasibility(self, application_schedule: List[ApplicationTiming],
                                       field_conditions: Dict) -> List[Dict]:
        """Validate the feasibility of the application schedule"""

        validation_results = []

        for application in application_schedule:
            result = {
                "application": application,
                "feasible": True,
                "warnings": [],
                "recommendations": []
            }

            # Check equipment availability
            if field_conditions.get("equipment_available", True) == False:
                result["feasible"] = False
                result["warnings"].append("Equipment not available for scheduled application")

            # Check field accessibility
            if field_conditions.get("field_accessible", True) == False:
                result["warnings"].append("Field may not be accessible due to wet conditions")

            # Check labor availability
            if field_conditions.get("labor_available", True) == False:
                result["warnings"].append("Labor may not be available for application")

            # Check storage requirements
            storage_capacity = field_conditions.get("storage_capacity_kg", 10000)
            if application.amount_kg_per_ha * field_conditions.get("total_area_ha", 1) > storage_capacity:
                result["warnings"].append("Insufficient storage capacity for recommended amount")

            validation_results.append(result)

        return validation_results
