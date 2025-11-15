import json
from typing import Dict, List, Optional, Tuple
import numpy as np

class AgronomicKnowledgeBase:
    """
    Comprehensive agronomic knowledge base containing fertilizer and crop nutrition data
    """
    
    def __init__(self):
        self.stcr_coefficients = self._load_stcr_coefficients()
        self.crop_nutrient_responses = self._load_crop_responses()
        self.soil_test_correlations = self._load_soil_correlations()
        self.fertilizer_efficiency_factors = self._load_efficiency_factors()
        self.critical_nutrient_levels = self._load_critical_levels()
        
    def _load_stcr_coefficients(self) -> Dict:
        """
        Load Soil Test Crop Response (STCR) coefficients based on ICAR/ICRISAT data
        """
        return {
            "maize": {
                "sandy_loam": {
                    "n": {"a": 4.5, "b": 0.65, "c": 0.85},  # Y = a + b*SN + c*FN
                    "p": {"a": 3.2, "b": 0.45, "c": 0.78},
                    "k": {"a": 3.8, "b": 0.35, "c": 0.70}
                },
                "clay_loam": {
                    "n": {"a": 4.2, "b": 0.70, "c": 0.80},
                    "p": {"a": 3.5, "b": 0.50, "c": 0.75},
                    "k": {"a": 4.0, "b": 0.40, "c": 0.68}
                },
                "alluvial": {
                    "n": {"a": 4.8, "b": 0.60, "c": 0.88},
                    "p": {"a": 3.0, "b": 0.42, "c": 0.80},
                    "k": {"a": 3.6, "b": 0.38, "c": 0.72}
                }
            },
            "rice": {
                "sandy_loam": {
                    "n": {"a": 3.8, "b": 0.55, "c": 0.75},
                    "p": {"a": 2.8, "b": 0.40, "c": 0.70},
                    "k": {"a": 3.2, "b": 0.45, "c": 0.65}
                },
                "clay_loam": {
                    "n": {"a": 4.0, "b": 0.60, "c": 0.78},
                    "p": {"a": 3.0, "b": 0.45, "c": 0.72},
                    "k": {"a": 3.5, "b": 0.48, "c": 0.68}
                }
            },
            "wheat": {
                "sandy_loam": {
                    "n": {"a": 3.5, "b": 0.58, "c": 0.82},
                    "p": {"a": 2.5, "b": 0.35, "c": 0.75},
                    "k": {"a": 3.0, "b": 0.30, "c": 0.70}
                },
                "clay_loam": {
                    "n": {"a": 3.8, "b": 0.62, "c": 0.80},
                    "p": {"a": 2.8, "b": 0.40, "c": 0.72},
                    "k": {"a": 3.2, "b": 0.35, "c": 0.68}
                }
            }
        }
    
    def _load_crop_responses(self) -> Dict:
        """Load crop nutrient response functions"""
        return {
            "maize": {
                "n_response": {
                    "linear_range": (0, 150),
                    "quadratic_range": (150, 250),
                    "plateau_range": (250, 300),
                    "toxicity_threshold": 350
                },
                "p_response": {
                    "linear_range": (0, 60),
                    "quadratic_range": (60, 100),
                    "plateau_range": (100, 120),
                    "toxicity_threshold": 150
                },
                "k_response": {
                    "linear_range": (0, 80),
                    "quadratic_range": (80, 120),
                    "plateau_range": (120, 150),
                    "toxicity_threshold": 200
                }
            },
            "rice": {
                "n_response": {
                    "linear_range": (0, 120),
                    "quadratic_range": (120, 180),
                    "plateau_range": (180, 220),
                    "toxicity_threshold": 280
                },
                "p_response": {
                    "linear_range": (0, 40),
                    "quadratic_range": (40, 80),
                    "plateau_range": (80, 100),
                    "toxicity_threshold": 120
                },
                "k_response": {
                    "linear_range": (0, 60),
                    "quadratic_range": (60, 100),
                    "plateau_range": (100, 140),
                    "toxicity_threshold": 180
                }
            }
        }
    
    def _load_soil_correlations(self) -> Dict:
        """Load soil test correlation data"""
        return {
            "nitrogen": {
                "test_methods": {
                    "kjeldahl": {"factor": 1.0, "reliability": 0.85},
                    "alkaline_permanganate": {"factor": 0.75, "reliability": 0.70},
                    "hot_water_extractable": {"factor": 0.80, "reliability": 0.75}
                },
                "critical_levels": {
                    "low": 200,      # ppm
                    "medium": 280,
                    "high": 400
                }
            },
            "phosphorus": {
                "test_methods": {
                    "bray_p1": {"factor": 1.0, "reliability": 0.90},
                    "olsen": {"factor": 1.2, "reliability": 0.88},
                    "mehlich_3": {"factor": 0.95, "reliability": 0.85}
                },
                "critical_levels": {
                    "low": 10,
                    "medium": 25,
                    "high": 50
                }
            },
            "potassium": {
                "test_methods": {
                    "ammonium_acetate": {"factor": 1.0, "reliability": 0.88},
                    "mehlich_3": {"factor": 0.90, "reliability": 0.85},
                    "morgan": {"factor": 0.85, "reliability": 0.80}
                },
                "critical_levels": {
                    "low": 100,
                    "medium": 200,
                    "high": 400
                }
            }
        }
    
    def _load_efficiency_factors(self) -> Dict:
        """Load fertilizer efficiency factors"""
        return {
            "nitrogen": {
                "urea": {
                    "efficiency": 0.60,
                    "loss_mechanisms": ["volatilization", "leaching", "denitrification"],
                    "conditions": {
                        "pH_sensitive": True,
                        "temperature_sensitive": True,
                        "moisture_sensitive": True
                    }
                },
                "ammonium_sulfate": {
                    "efficiency": 0.70,
                    "loss_mechanisms": ["leaching", "denitrification"],
                    "conditions": {
                        "pH_sensitive": False,
                        "temperature_sensitive": False,
                        "moisture_sensitive": True
                    }
                },
                "calcium_ammonium_nitrate": {
                    "efficiency": 0.75,
                    "loss_mechanisms": ["leaching", "denitrification"],
                    "conditions": {
                        "pH_sensitive": False,
                        "temperature_sensitive": False,
                        "moisture_sensitive": True
                    }
                }
            },
            "phosphorus": {
                "dap": {
                    "efficiency": 0.25,
                    "conditions": {
                        "pH_dependent": True,
                        "soil_fixation": True,
                        "optimal_pH_range": (6.0, 7.0)
                    }
                },
                "tsp": {
                    "efficiency": 0.20,
                    "conditions": {
                        "pH_dependent": True,
                        "soil_fixation": True,
                        "optimal_pH_range": (6.0, 7.0)
                    }
                },
                "rock_phosphate": {
                    "efficiency": 0.15,
                    "conditions": {
                        "pH_dependent": True,
                        "slow_release": True,
                        "optimal_pH_range": (5.0, 6.0)
                    }
                }
            },
            "potassium": {
                "kcl": {
                    "efficiency": 0.80,
                    "conditions": {
                        "leaching_susceptible": True,
                        "salt_index": "high"
                    }
                },
                "k2so4": {
                    "efficiency": 0.85,
                    "conditions": {
                        "leaching_susceptible": True,
                        "salt_index": "medium",
                        "sulfur_bonus": True
                    }
                }
            }
        }
    
    def _load_critical_levels(self) -> Dict:
        """Load critical nutrient levels for different crops and soils"""
        return {
            "soil_critical_levels": {
                "organic_carbon": {
                    "very_low": 0.3,
                    "low": 0.5,
                    "medium": 0.75,
                    "high": 1.0
                },
                "available_nitrogen": {
                    "very_low": 150,
                    "low": 200,
                    "medium": 280,
                    "high": 400
                },
                "available_phosphorus": {
                    "very_low": 5,
                    "low": 10,
                    "medium": 25,
                    "high": 50
                },
                "available_potassium": {
                    "very_low": 80,
                    "low": 120,
                    "medium": 200,
                    "high": 400
                }
            },
            "plant_critical_levels": {
                "maize": {
                    "n_percent": {"deficient": 2.5, "sufficient": 3.0, "excess": 4.5},
                    "p_percent": {"deficient": 0.25, "sufficient": 0.35, "excess": 0.80},
                    "k_percent": {"deficient": 1.5, "sufficient": 2.0, "excess": 3.5}
                },
                "rice": {
                    "n_percent": {"deficient": 2.0, "sufficient": 2.8, "excess": 4.0},
                    "p_percent": {"deficient": 0.20, "sufficient": 0.30, "excess": 0.70},
                    "k_percent": {"deficient": 1.2, "sufficient": 1.8, "excess": 3.0}
                }
            }
        }
    
    def calculate_fertilizer_requirement(self, crop_type: str, soil_type: str, 
                                       target_yield: float, soil_nutrients: Dict) -> Dict:
        """
        Calculate fertilizer requirements using STCR approach
        """
        
        crop_type = crop_type.lower()
        soil_type = soil_type.lower()
        
        # Get STCR coefficients
        stcr_data = self.stcr_coefficients.get(crop_type, {}).get(soil_type, {})
        
        if not stcr_data:
            # Fallback to general coefficients
            stcr_data = self.stcr_coefficients.get("maize", {}).get("sandy_loam", {})
        
        requirements = {}
        
        for nutrient in ["n", "p", "k"]:
            if nutrient in stcr_data:
                coeff = stcr_data[nutrient]
                soil_supply = soil_nutrients.get(nutrient, 0)
                
                # STCR equation: Fertilizer needed = (Target - a - b*Soil_test) / c
                fertilizer_needed = max(0, (target_yield - coeff["a"] - coeff["b"] * soil_supply) / coeff["c"])
                requirements[nutrient] = round(fertilizer_needed, 2)
        
        return requirements
    
    def assess_nutrient_status(self, soil_test_values: Dict, crop_type: str) -> Dict:
        """Assess soil nutrient status based on critical levels"""
        
        status = {}
        critical_levels = self.critical_nutrient_levels["soil_critical_levels"]
        
        for nutrient, value in soil_test_values.items():
            if nutrient in critical_levels:
                levels = critical_levels[nutrient]
                
                if value <= levels["very_low"]:
                    status[nutrient] = "very_low"
                elif value <= levels["low"]:
                    status[nutrient] = "low"
                elif value <= levels["medium"]:
                    status[nutrient] = "medium"
                else:
                    status[nutrient] = "high"
            else:
                status[nutrient] = "unknown"
        
        return status
    
    def calculate_nutrient_efficiency(self, fertilizer_type: str, nutrient: str, 
                                    soil_conditions: Dict) -> float:
        """Calculate nutrient use efficiency based on fertilizer type and conditions"""
        
        efficiency_data = self.fertilizer_efficiency_factors.get(nutrient, {}).get(fertilizer_type, {})
        
        if not efficiency_data:
            return 0.50  # Default efficiency
        
        base_efficiency = efficiency_data.get("efficiency", 0.50)
        conditions = efficiency_data.get("conditions", {})
        
        # Adjust efficiency based on soil conditions
        adjusted_efficiency = base_efficiency
        
        if conditions.get("pH_sensitive", False):
            soil_ph = soil_conditions.get("ph", 6.5)
            if soil_ph < 5.5 or soil_ph > 8.0:
                adjusted_efficiency *= 0.85
        
        if conditions.get("pH_dependent", False):
            soil_ph = soil_conditions.get("ph", 6.5)
            optimal_range = conditions.get("optimal_pH_range", (6.0, 7.0))
            if soil_ph < optimal_range[0] or soil_ph > optimal_range[1]:
                adjusted_efficiency *= 0.75
        
        if conditions.get("moisture_sensitive", False):
            moisture_status = soil_conditions.get("moisture", "adequate")
            if moisture_status in ["dry", "waterlogged"]:
                adjusted_efficiency *= 0.80
        
        return round(adjusted_efficiency, 3)
    
    def get_nutrient_interactions(self, applied_nutrients: List[str]) -> Dict:
        """Get nutrient interactions for applied nutrients"""
        
        interactions = {
            "positive": [],
            "negative": [],
            "neutral": []
        }
        
        interaction_matrix = {
            ("nitrogen", "phosphorus"): "positive",
            ("nitrogen", "potassium"): "neutral",
            ("nitrogen", "sulfur"): "positive",
            ("phosphorus", "potassium"): "neutral",
            ("phosphorus", "zinc"): "negative",
            ("phosphorus", "iron"): "negative",
            ("potassium", "magnesium"): "negative",
            ("potassium", "calcium"): "negative",
            ("calcium", "magnesium"): "positive",
            ("sulfur", "nitrogen"): "positive"
        }
        
        for i, nutrient1 in enumerate(applied_nutrients):
            for j, nutrient2 in enumerate(applied_nutrients[i+1:], i+1):
                interaction_key = tuple(sorted([nutrient1.lower(), nutrient2.lower()]))
                interaction_type = interaction_matrix.get(interaction_key, "neutral")
                
                interaction_info = {
                    "nutrients": [nutrient1, nutrient2],
                    "type": interaction_type,
                    "description": self._get_interaction_description(nutrient1, nutrient2, interaction_type)
                }
                
                interactions[interaction_type].append(interaction_info)
        
        return interactions
    
    def _get_interaction_description(self, nutrient1: str, nutrient2: str, interaction_type: str) -> str:
        """Get description of nutrient interaction"""
        
        descriptions = {
            ("nitrogen", "phosphorus", "positive"): "Nitrogen enhances phosphorus uptake and utilization",
            ("phosphorus", "zinc", "negative"): "High phosphorus can induce zinc deficiency",
            ("potassium", "magnesium", "negative"): "Excess potassium can reduce magnesium uptake",
            ("sulfur", "nitrogen", "positive"): "Sulfur is essential for nitrogen metabolism and protein synthesis"
        }
        
        key = tuple(sorted([nutrient1.lower(), nutrient2.lower()]) + [interaction_type])
        return descriptions.get(key, f"{interaction_type.title()} interaction between {nutrient1} and {nutrient2}")
    
    def get_regional_adjustments(self, region: str, crop_type: str) -> Dict:
        """Get regional adjustments for fertilizer recommendations"""
        
        # Regional adjustment factors based on climate and soil types
        regional_factors = {
            "west_africa": {
                "climate": "tropical_savanna",
                "soil_dominant": "ferrasols",
                "adjustments": {
                    "n_factor": 1.1,  # Higher N requirement due to leaching
                    "p_factor": 1.2,  # Higher P requirement due to fixation
                    "k_factor": 0.9,  # Lower K requirement
                    "timing_adjustment": "split_more"
                }
            },
            "east_africa": {
                "climate": "tropical_highland",
                "soil_dominant": "andosols",
                "adjustments": {
                    "n_factor": 1.0,
                    "p_factor": 1.3,  # Very high P fixation in volcanic soils
                    "k_factor": 1.0,
                    "timing_adjustment": "normal"
                }
            },
            "southern_africa": {
                "climate": "semi_arid",
                "soil_dominant": "aridisols",
                "adjustments": {
                    "n_factor": 0.9,
                    "p_factor": 1.1,
                    "k_factor": 1.1,
                    "timing_adjustment": "concentrate"
                }
            }
        }
        
        return regional_factors.get(region.lower(), regional_factors["west_africa"])
    
    def validate_recommendation(self, recommendation: Dict, soil_analysis: Dict) -> Dict:
        """Validate fertilizer recommendation against agronomic principles"""
        
        validation = {
            "valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Check for excessive rates
        n_rate = recommendation.get("n_kg_per_ha", 0)
        p_rate = recommendation.get("p_kg_per_ha", 0)
        k_rate = recommendation.get("k_kg_per_ha", 0)
        
        if n_rate > 300:
            validation["warnings"].append(f"Nitrogen rate ({n_rate} kg/ha) is very high - consider split application")
        
        if p_rate > 150:
            validation["warnings"].append(f"Phosphorus rate ({p_rate} kg/ha) is excessive - reduce to prevent fixation")
        
        if k_rate > 200:
            validation["warnings"].append(f"Potassium rate ({k_rate} kg/ha) is high - monitor for luxury consumption")
        
        # Check soil pH compatibility
        soil_ph = soil_analysis.get("ph", 6.5)
        if soil_ph < 5.5:
            validation["suggestions"].append("Consider liming before fertilizer application")
            if p_rate > 0:
                validation["suggestions"].append("Use acidulated phosphorus sources for low pH soils")
        
        if soil_ph > 7.5:
            validation["suggestions"].append("Consider micronutrient supplementation for alkaline soils")
        
        # Check nutrient ratios
        if n_rate > 0 and p_rate > 0:
            np_ratio = n_rate / p_rate
            if np_ratio > 10:
                validation["warnings"].append("N:P ratio is very high - may cause phosphorus deficiency")
            elif np_ratio < 2:
                validation["warnings"].append("N:P ratio is low - may cause nitrogen deficiency")
        
        return validation
