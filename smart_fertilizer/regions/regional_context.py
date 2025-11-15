from typing import Dict, List, Optional
from datetime import datetime
import json


class RegionalContextManager:
    """
    Enhanced regional context manager with detailed African agricultural data
    """

    def __init__(self):
        self.african_regions = self._initialize_african_regions()
        self.agro_ecological_zones = self._initialize_aez_data()
        self.soil_types = self._initialize_soil_types()
        self.climate_zones = self._initialize_climate_zones()

    def _initialize_african_regions(self) -> Dict:
        """Initialize detailed African regional data"""
        return {
            "west_africa": {
                "countries": ["nigeria", "ghana", "burkina_faso", "mali", "senegal", "ivory_coast"],
                "dominant_crops": ["maize", "rice", "yam", "cassava", "millet", "sorghum"],
                "soil_challenges": ["low_fertility", "acidity", "erosion"],
                "climate_pattern": "tropical_savanna",
                "rainfall_seasons": ["wet_season", "dry_season"],
                "fertilizer_access": "moderate",
                "extension_coverage": "limited"
            },
            "east_africa": {
                "countries": ["kenya", "tanzania", "uganda", "ethiopia", "rwanda", "burundi"],
                "dominant_crops": ["maize", "wheat", "barley", "teff", "beans", "bananas"],
                "soil_challenges": ["volcanic_soils", "phosphorus_fixation", "erosion"],
                "climate_pattern": "tropical_highland",
                "rainfall_seasons": ["long_rains", "short_rains"],
                "fertilizer_access": "good",
                "extension_coverage": "moderate"
            },
            "southern_africa": {
                "countries": ["south_africa", "zimbabwe", "zambia", "malawi", "botswana", "namibia"],
                "dominant_crops": ["maize", "wheat", "sorghum", "sugarcane", "tobacco"],
                "soil_challenges": ["drought", "low_organic_matter", "salinity"],
                "climate_pattern": "semi_arid",
                "rainfall_seasons": ["summer_rains"],
                "fertilizer_access": "good",
                "extension_coverage": "good"
            },
            "central_africa": {
                "countries": ["cameroon", "central_african_republic", "chad", "congo", "drc"],
                "dominant_crops": ["cassava", "plantain", "maize", "rice", "yam"],
                "soil_challenges": ["acidity", "aluminum_toxicity", "low_phosphorus"],
                "climate_pattern": "tropical_humid",
                "rainfall_seasons": ["bimodal"],
                "fertilizer_access": "limited",
                "extension_coverage": "very_limited"
            },
            "north_africa": {
                "countries": ["egypt", "morocco", "algeria", "tunisia", "libya", "sudan"],
                "dominant_crops": ["wheat", "barley", "rice", "maize", "cotton"],
                "soil_challenges": ["salinity", "water_scarcity", "sand_movement"],
                "climate_pattern": "arid_mediterranean",
                "rainfall_seasons": ["winter_rains"],
                "fertilizer_access": "good",
                "extension_coverage": "moderate"
            }
        }

    def _initialize_aez_data(self) -> Dict:
        """Initialize agro-ecological zone data"""
        return {
            "sahel": {
                "rainfall_range": "200-600mm",
                "temperature_range": "25-40°C",
                "growing_season": "90-120 days",
                "dominant_crops": ["millet", "sorghum", "cowpea"],
                "soil_types": ["arenosols", "lixisols"],
                "constraints": ["drought", "low_fertility", "wind_erosion"],
                "fertilizer_priority": ["nitrogen", "phosphorus"]
            },
            "sudan_savanna": {
                "rainfall_range": "600-1000mm",
                "temperature_range": "20-35°C",
                "growing_season": "120-180 days",
                "dominant_crops": ["sorghum", "millet", "maize", "cotton"],
                "soil_types": ["lixisols", "luvisols"],
                "constraints": ["irregular_rainfall", "soil_crusting"],
                "fertilizer_priority": ["nitrogen", "phosphorus", "potassium"]
            },
            "guinea_savanna": {
                "rainfall_range": "1000-1500mm",
                "temperature_range": "20-32°C",
                "growing_season": "180-240 days",
                "dominant_crops": ["maize", "yam", "rice", "soybeans"],
                "soil_types": ["luvisols", "acrisols"],
                "constraints": ["soil_acidity", "aluminum_toxicity"],
                "fertilizer_priority": ["nitrogen", "phosphorus", "lime"]
            },
            "forest_zone": {
                "rainfall_range": "1500-2500mm",
                "temperature_range": "22-30°C",
                "growing_season": "240-365 days",
                "dominant_crops": ["cassava", "plantain", "cocoa", "oil_palm"],
                "soil_types": ["ferralsols", "acrisols"],
                "constraints": ["high_acidity", "aluminum_toxicity", "low_cec"],
                "fertilizer_priority": ["lime", "phosphorus", "potassium"]
            },
            "highland": {
                "rainfall_range": "800-1800mm",
                "temperature_range": "10-25°C",
                "growing_season": "180-300 days",
                "dominant_crops": ["wheat", "barley", "potato", "beans"],
                "soil_types": ["andosols", "cambisols"],
                "constraints": ["phosphorus_fixation", "frost_risk"],
                "fertilizer_priority": ["phosphorus", "nitrogen", "micronutrients"]
            }
        }

    def _initialize_soil_types(self) -> Dict:
        """Initialize African soil type characteristics"""
        return {
            "ferralsols": {
                "description": "Highly weathered soils with low fertility",
                "ph_range": "4.5-6.0",
                "organic_matter": "low",
                "cec": "low",
                "phosphorus_status": "very_low",
                "aluminum_saturation": "high",
                "management_needs": ["liming", "organic_matter", "phosphorus"]
            },
            "acrisols": {
                "description": "Acidic soils with clay accumulation",
                "ph_range": "4.0-5.5",
                "organic_matter": "low_to_medium",
                "cec": "low_to_medium",
                "phosphorus_status": "low",
                "aluminum_saturation": "medium_to_high",
                "management_needs": ["liming", "phosphorus", "organic_matter"]
            },
            "luvisols": {
                "description": "Soils with clay illuviation",
                "ph_range": "6.0-7.5",
                "organic_matter": "medium",
                "cec": "medium_to_high",
                "phosphorus_status": "medium",
                "aluminum_saturation": "low",
                "management_needs": ["nitrogen", "phosphorus", "potassium"]
            },
            "vertisols": {
                "description": "Clay-rich soils with shrink-swell properties",
                "ph_range": "6.5-8.5",
                "organic_matter": "medium",
                "cec": "high",
                "phosphorus_status": "medium_to_high",
                "aluminum_saturation": "very_low",
                "management_needs": ["nitrogen", "drainage", "timing"]
            },
            "andosols": {
                "description": "Volcanic soils with high phosphorus fixation",
                "ph_range": "5.0-6.5",
                "organic_matter": "high",
                "cec": "high",
                "phosphorus_status": "low",
                "aluminum_saturation": "low",
                "management_needs": ["phosphorus", "sulfur", "micronutrients"]
            },
            "arenosols": {
                "description": "Sandy soils with low water holding capacity",
                "ph_range": "5.5-7.0",
                "organic_matter": "very_low",
                "cec": "very_low",
                "phosphorus_status": "low",
                "aluminum_saturation": "low",
                "management_needs": ["organic_matter", "frequent_applications", "all_nutrients"]
            }
        }

    def _initialize_climate_zones(self) -> Dict:
        """Initialize climate zone data"""
        return {
            "arid": {
                "annual_rainfall": "<400mm",
                "evapotranspiration": ">2000mm",
                "constraints": ["water_stress", "high_temperatures", "salinity"],
                "fertilizer_considerations": ["efficient_use", "timing_critical", "organic_matter"]
            },
            "semi_arid": {
                "annual_rainfall": "400-800mm",
                "evapotranspiration": "1500-2000mm",
                "constraints": ["drought_risk", "irregular_rainfall"],
                "fertilizer_considerations": ["split_applications", "drought_resistant_varieties"]
            },
            "sub_humid": {
                "annual_rainfall": "800-1200mm",
                "evapotranspiration": "1200-1500mm",
                "constraints": ["seasonal_drought", "soil_degradation"],
                "fertilizer_considerations": ["balanced_nutrition", "soil_conservation"]
            },
            "humid": {
                "annual_rainfall": ">1200mm",
                "evapotranspiration": "1000-1200mm",
                "constraints": ["leaching", "acidity", "pests_diseases"],
                "fertilizer_considerations": ["split_nitrogen", "lime_application", "ipm"]
            }
        }

    def get_regional_fertilizer_strategy(self, region: str, aez: str, soil_type: str) -> Dict:
        """Get comprehensive fertilizer strategy for specific regional context"""

        region_data = self.african_regions.get(region, {})
        aez_data = self.agro_ecological_zones.get(aez, {})
        soil_data = self.soil_types.get(soil_type, {})

        strategy = {
            "region": region,
            "agro_ecological_zone": aez,
            "soil_type": soil_type,
            "priority_nutrients": [],
            "application_strategy": {},
            "timing_recommendations": {},
            "risk_management": {},
            "cost_optimization": {}
        }

        # Determine priority nutrients
        aez_priorities = aez_data.get("fertilizer_priority", [])
        soil_needs = soil_data.get("management_needs", [])

        # Combine and prioritize nutrients
        all_priorities = aez_priorities + soil_needs
        nutrient_priority = []

        # Priority order based on African conditions
        priority_order = ["lime", "nitrogen", "phosphorus", "potassium", "organic_matter", "micronutrients"]

        for nutrient in priority_order:
            if nutrient in all_priorities and nutrient not in nutrient_priority:
                nutrient_priority.append(nutrient)

        strategy["priority_nutrients"] = nutrient_priority

        # Application strategy
        strategy["application_strategy"] = self._get_application_strategy(region_data, aez_data, soil_data)

        # Timing recommendations
        strategy["timing_recommendations"] = self._get_timing_recommendations(region_data, aez_data)

        # Risk management
        strategy["risk_management"] = self._get_risk_management(region_data, aez_data, soil_data)

        # Cost optimization
        strategy["cost_optimization"] = self._get_cost_optimization(region_data)

        return strategy

    def _get_application_strategy(self, region_data: Dict, aez_data: Dict, soil_data: Dict) -> Dict:
        """Determine optimal application strategy"""

        strategy = {
            "split_applications": True,
            "application_method": "broadcast_incorporate",
            "special_considerations": []
        }

        # Determine split strategy based on rainfall
        rainfall_pattern = region_data.get("rainfall_seasons", [])
        if len(rainfall_pattern) > 1:
            strategy["nitrogen_splits"] = 3
            strategy["phosphorus_splits"] = 1
            strategy["potassium_splits"] = 2
        else:
            strategy["nitrogen_splits"] = 2
            strategy["phosphorus_splits"] = 1
            strategy["potassium_splits"] = 1

        # Soil-specific considerations
        if "low_cec" in soil_data.get("management_needs", []):
            strategy["special_considerations"].append("Frequent light applications for low CEC soils")

        if "phosphorus_fixation" in aez_data.get("constraints", []):
            strategy["special_considerations"].append("Band phosphorus application to reduce fixation")

        if "aluminum_toxicity" in aez_data.get("constraints", []):
            strategy["special_considerations"].append("Apply lime 2-4 weeks before other fertilizers")

        return strategy

    def _get_timing_recommendations(self, region_data: Dict, aez_data: Dict) -> Dict:
        """Get timing recommendations based on regional patterns"""

        timing = {
            "land_preparation": "2-4 weeks before planting",
            "basal_application": "at_planting",
            "top_dressing": [],
            "foliar_applications": []
        }

        growing_season = aez_data.get("growing_season", "120-180 days")
        season_length = int(growing_season.split("-")[0])

        # Calculate top dressing timings
        if season_length > 90:
            timing["top_dressing"].append({"timing": "3-4 weeks after planting", "nutrients": ["nitrogen"]})

        if season_length > 120:
            timing["top_dressing"].append({"timing": "6-8 weeks after planting", "nutrients": ["nitrogen", "potassium"]})

        if season_length > 150:
            timing["top_dressing"].append({"timing": "10-12 weeks after planting", "nutrients": ["potassium"]})

        # Foliar applications for micronutrients
        timing["foliar_applications"] = [
            {"timing": "vegetative_stage", "nutrients": ["zinc", "boron"]},
            {"timing": "reproductive_stage", "nutrients": ["boron", "calcium"]}
        ]

        return timing

    def _get_risk_management(self, region_data: Dict, aez_data: Dict, soil_data: Dict) -> Dict:
        """Get risk management strategies"""

        risks = {
            "climate_risks": [],
            "soil_risks": [],
            "market_risks": [],
            "mitigation_strategies": []
        }

        # Climate risks
        constraints = aez_data.get("constraints", [])
        if "drought" in constraints:
            risks["climate_risks"].append("drought_stress")
            risks["mitigation_strategies"].append("Use drought-tolerant varieties and efficient irrigation")

        if "irregular_rainfall" in constraints:
            risks["climate_risks"].append("rainfall_variability")
            risks["mitigation_strategies"].append("Split fertilizer applications to match rainfall")

        # Soil risks
        if "low_fertility" in constraints:
            risks["soil_risks"].append("nutrient_deficiency")
            risks["mitigation_strategies"].append("Regular soil testing and balanced fertilization")

        if "erosion" in constraints:
            risks["soil_risks"].append("soil_loss")
            risks["mitigation_strategies"].append("Contour farming and cover crops")

        # Market risks
        fertilizer_access = region_data.get("fertilizer_access", "moderate")
        if fertilizer_access in ["limited", "very_limited"]:
            risks["market_risks"].append("fertilizer_availability")
            risks["mitigation_strategies"].append("Early procurement and storage")

        return risks

    def _get_cost_optimization(self, region_data: Dict) -> Dict:
        """Get cost optimization strategies"""

        optimization = {
            "procurement_strategy": "group_purchasing",
            "storage_recommendations": "proper_storage_facilities",
            "alternatives": [],
            "subsidies": []
        }

        # Check for subsidy programs
        fertilizer_access = region_data.get("fertilizer_access", "moderate")
        if fertilizer_access == "good":
            optimization["subsidies"].append("Government fertilizer subsidy programs may be available")

        # Alternative strategies
        optimization["alternatives"] = [
            "Organic fertilizers and compost",
            "Crop residue management",
            "Legume intercropping",
            "Microbial fertilizers"
        ]

        return optimization

    def get_crop_specific_recommendations(self, region: str, crop: str, season: str) -> Dict:
        """Get crop-specific recommendations for region and season"""

        region_data = self.african_regions.get(region, {})
        dominant_crops = region_data.get("dominant_crops", [])

        recommendations = {
            "crop_suitability": "unknown",
            "variety_recommendations": [],
            "nutrient_requirements": {},
            "growth_calendar": {},
            "pest_disease_risks": [],
            "market_potential": "unknown"
        }

        # Assess crop suitability
        if crop.lower() in dominant_crops:
            recommendations["crop_suitability"] = "high"
            recommendations["variety_recommendations"] = self._get_variety_recommendations(region, crop)
        else:
            recommendations["crop_suitability"] = "medium"
            recommendations["variety_recommendations"] = ["Consult local extension services for suitable varieties"]

        # Get nutrient requirements
        recommendations["nutrient_requirements"] = self._get_crop_nutrient_requirements(crop, region, season)

        return recommendations

    def _get_variety_recommendations(self, region: str, crop: str) -> List[str]:
        """Get variety recommendations for specific region and crop"""

        variety_database = {
            "west_africa": {
                "maize": ["SAMMAZ-15", "SAMMAZ-16", "BR-9928-DMRSR", "TZE-W"],
                "rice": ["NERICA-1", "NERICA-2", "FARO-44", "FARO-52"],
                "sorghum": ["ICSV-400", "CSR-01", "SAMSORG-17"]
            },
            "east_africa": {
                "maize": ["H-513", "H-614", "DK-8053", "SC-627"],
                "wheat": ["Digalu", "Hidase", "Kakaba", "Sofumer"],
                "barley": ["EH-1493", "Traveler", "HB-1307"]
            },
            "southern_africa": {
                "maize": ["SC-403", "PAN-53", "ZM-421", "ZM-623"],
                "wheat": ["SST-027", "SST-015", "Tugela-DN"],
                "sorghum": ["NK-283", "PAN-8564", "MR-Buster"]
            }
        }

        return variety_database.get(region, {}).get(crop, ["Consult local seed suppliers"])

    def _get_crop_nutrient_requirements(self, crop: str, region: str, season: str) -> Dict:
        """Get crop nutrient requirements adjusted for region and season"""

        base_requirements = {
            "maize": {"n": 120, "p": 40, "k": 60},
            "rice": {"n": 100, "p": 30, "k": 50},
            "wheat": {"n": 100, "p": 35, "k": 40},
            "sorghum": {"n": 80, "p": 25, "k": 40},
            "millet": {"n": 60, "p": 20, "k": 30}
        }

        crop_requirement = base_requirements.get(crop.lower(), {"n": 100, "p": 30, "k": 50})

        # Regional adjustments
        region_data = self.african_regions.get(region, {})
        soil_challenges = region_data.get("soil_challenges", [])

        adjusted_requirement = crop_requirement.copy()

        if "low_fertility" in soil_challenges:
            adjusted_requirement["n"] *= 1.2
            adjusted_requirement["p"] *= 1.3
            adjusted_requirement["k"] *= 1.1

        if "phosphorus_fixation" in soil_challenges:
            adjusted_requirement["p"] *= 1.5

        if "acidity" in soil_challenges:
            adjusted_requirement["lime"] = 1000  # kg/ha

        return adjusted_requirement



def get_regional_config(region_name: str) -> dict:
    """
    Retourne les données régionales pour une région africaine donnée.
    Exemple : 'west_africa', 'east_africa', etc.
        """
    manager = RegionalContextManager()
    return manager.african_regions.get(region_name.lower(), {})
