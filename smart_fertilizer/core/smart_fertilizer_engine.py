import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np # type: ignore
from .agronomic_knowledge_base import AgronomicKnowledgeBase
from .fertilizer_optimizer import FertilizerOptimizer
from .smart_fertilization import SmartFertilization
from modules.smart_fertilizer.api.models import (
    SoilAnalysis,
    CropSelection,
    FertilizerRecommendation,
    NutrientBalance,
    ApplicationTiming,
    CostAnalysis,
    FertilizerType
)

class SmartFertilizerEngine:
    """
    Core engine for generating intelligent fertilizer recommendations
    """

    def __init__(self):
        self.knowledge_base = AgronomicKnowledgeBase()
        self.optimizer = FertilizerOptimizer()
        self.smart_fert = SmartFertilization()
        self.crop_data = self._load_crop_data()
        self.fertilizer_database = self._load_fertilizer_database()

    def _load_crop_data(self) -> Dict:
        """Load crop nutrient requirements and characteristics"""
        try:
            with open('data/crop_profiles.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_crop_data()

    def _load_fertilizer_database(self) -> Dict:
        """Load available fertilizers database"""
        try:
            with open('data/regional_prices.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_fertilizer_data()

    def _get_default_crop_data(self) -> Dict:
        """Default crop data based on FAO guidelines"""
        return {
            "maize": {
                "nutrient_requirements": {
                    "n_kg_per_ton": 25,
                    "p_kg_per_ton": 4,
                    "k_kg_per_ton": 20,
                    "ca_kg_per_ton": 3,
                    "mg_kg_per_ton": 2,
                    "s_kg_per_ton": 3
                },
                "growth_stages": [
                    {"stage": "planting", "days": 0, "n_percent": 0},
                    {"stage": "vegetative", "days": 25, "n_percent": 40},
                    {"stage": "tasseling", "days": 60, "n_percent": 35},
                    {"stage": "grain_filling", "days": 90, "n_percent": 25}
                ],
                "typical_yield": 6.0,
                "ph_optimum": {"min": 6.0, "max": 7.0}
            },
            "rice": {
                "nutrient_requirements": {
                    "n_kg_per_ton": 20,
                    "p_kg_per_ton": 4,
                    "k_kg_per_ton": 25,
                    "ca_kg_per_ton": 2,
                    "mg_kg_per_ton": 1.5,
                    "s_kg_per_ton": 2
                },
                "growth_stages": [
                    {"stage": "transplanting", "days": 0, "n_percent": 0},
                    {"stage": "tillering", "days": 20, "n_percent": 50},
                    {"stage": "panicle_initiation", "days": 50, "n_percent": 30},
                    {"stage": "grain_filling", "days": 80, "n_percent": 20}
                ],
                "typical_yield": 5.0,
                "ph_optimum": {"min": 5.5, "max": 6.5}
            },
            "wheat": {
                "nutrient_requirements": {
                    "n_kg_per_ton": 30,
                    "p_kg_per_ton": 5,
                    "k_kg_per_ton": 15,
                    "ca_kg_per_ton": 3,
                    "mg_kg_per_ton": 2,
                    "s_kg_per_ton": 4
                },
                "growth_stages": [
                    {"stage": "sowing", "days": 0, "n_percent": 0},
                    {"stage": "tillering", "days": 30, "n_percent": 40},
                    {"stage": "stem_elongation", "days": 60, "n_percent": 35},
                    {"stage": "grain_filling", "days": 90, "n_percent": 25}
                ],
                "typical_yield": 4.5,
                "ph_optimum": {"min": 6.0, "max": 7.5}
            }
        }

    def _get_default_fertilizer_data(self) -> Dict:
        """Default fertilizer data with African market prices"""
        return {
            "urea": {
                "n_content": 46,
                "p_content": 0,
                "k_content": 0,
                "price_usd_per_kg": 0.45,
                "availability": "high"
            },
            "dap": {
                "n_content": 18,
                "p_content": 46,
                "k_content": 0,
                "price_usd_per_kg": 0.55,
                "availability": "high"
            },
            "tsp": {
                "n_content": 0,
                "p_content": 46,
                "k_content": 0,
                "price_usd_per_kg": 0.50,
                "availability": "medium"
            },
            "kcl": {
                "n_content": 0,
                "p_content": 0,
                "k_content": 60,
                "price_usd_per_kg": 0.40,
                "availability": "medium"
            },
            "npk_15_15_15": {
                "n_content": 15,
                "p_content": 15,
                "k_content": 15,
                "price_usd_per_kg": 0.60,
                "availability": "high"
            },
            "organic_compost": {
                "n_content": 1.5,
                "p_content": 1.0,
                "k_content": 1.0,
                "price_usd_per_kg": 0.05,
                "availability": "high"
            }
        }

    def generate_recommendation(self, soil_analysis: SoilAnalysis, crop_selection: CropSelection,
                              region_data: Dict, area_hectares: float, target_yield: float,
                              currency: str = "USD") -> FertilizerRecommendation:
        """
        Generate comprehensive fertilizer recommendation
        """

        # Get crop nutrient requirements
        crop_data = self.crop_data.get(crop_selection.crop_type.lower(), self.crop_data["maize"])

        # Calculate nutrient requirements based on target yield
        nutrient_requirements = self._calculate_nutrient_requirements(crop_data, target_yield)

        # Assess soil nutrient availability
        available_nutrients = self._assess_soil_nutrients(soil_analysis)

        # Calculate nutrient deficits
        nutrient_deficits = self._calculate_nutrient_deficits(nutrient_requirements, available_nutrients)

        # Optimize fertilizer selection
        optimized_fertilizers = self.optimizer.optimize_fertilizer_selection(
            nutrient_deficits, self.fertilizer_database, region_data
        )

        # Generate application schedule
        application_schedule = self._generate_application_schedule(
            crop_data, optimized_fertilizers, area_hectares
        )

        # Calculate costs
        cost_analysis = self._calculate_costs(optimized_fertilizers, area_hectares, currency)

        # Create nutrient balance
        nutrient_balance = NutrientBalance(
            total_n=nutrient_deficits.get("n", 0),
            total_p=nutrient_deficits.get("p", 0),
            total_k=nutrient_deficits.get("k", 0),
            secondary_nutrients={
                "ca": nutrient_deficits.get("ca", 0),
                "mg": nutrient_deficits.get("mg", 0),
                "s": nutrient_deficits.get("s", 0)
            },
            micronutrients={
                "zn": nutrient_deficits.get("zn", 0),
                "fe": nutrient_deficits.get("fe", 0),
                "mn": nutrient_deficits.get("mn", 0),
                "b": nutrient_deficits.get("b", 0)
            }
        )

        # Generate climate considerations
        climate_considerations = self._generate_climate_considerations(region_data, crop_selection)

        # Calculate expected yield and ROI
        expected_yield = self._calculate_expected_yield(crop_data, soil_analysis, nutrient_deficits)
        roi_percentage = self._calculate_roi(cost_analysis.total_cost, expected_yield, target_yield, region_data)

        return FertilizerRecommendation(
            recommendation_id=f"SF_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            soil_analysis=soil_analysis,
            crop_selection=crop_selection,
            region=region_data.get("name", "Unknown"),
            area_hectares=area_hectares,
            target_yield=target_yield,
            nutrient_balance=nutrient_balance,
            application_schedule=application_schedule,
            recommended_fertilizers=optimized_fertilizers,
            cost_analysis=cost_analysis,
            expected_yield=expected_yield,
            roi_percentage=roi_percentage,
            climate_considerations=climate_considerations,
            risk_factors=self._identify_risk_factors(soil_analysis, region_data),
            alternative_options=self._generate_alternatives(optimized_fertilizers)
        )

    def _calculate_nutrient_requirements(self, crop_data: Dict, target_yield: float) -> Dict:
        """Calculate total nutrient requirements based on target yield"""
        requirements = crop_data["nutrient_requirements"]
        return {
            "n": requirements["n_kg_per_ton"] * target_yield,
            "p": requirements["p_kg_per_ton"] * target_yield,
            "k": requirements["k_kg_per_ton"] * target_yield,
            "ca": requirements.get("ca_kg_per_ton", 0) * target_yield,
            "mg": requirements.get("mg_kg_per_ton", 0) * target_yield,
            "s": requirements.get("s_kg_per_ton", 0) * target_yield
        }

    def _assess_soil_nutrients(self, soil_analysis: SoilAnalysis) -> Dict:
        """Convert soil test values to available nutrients"""
        # Conversion factors from ppm to kg/ha (assuming 20cm depth, 2.2 million kg soil)
        conversion_factor = 2.2

        return {
            "n": soil_analysis.nitrogen * conversion_factor * 0.001,
            "p": soil_analysis.phosphorus * conversion_factor * 0.001,
            "k": soil_analysis.potassium * conversion_factor * 0.001,
            "ca": (soil_analysis.calcium or 0) * conversion_factor * 0.001,
            "mg": (soil_analysis.magnesium or 0) * conversion_factor * 0.001,
            "s": (soil_analysis.sulfur or 0) * conversion_factor * 0.001
        }

    def _calculate_nutrient_deficits(self, requirements: Dict, available: Dict) -> Dict:
        """Calculate nutrient deficits accounting for soil supply"""
        deficits = {}
        for nutrient in requirements:
            deficit = max(0, requirements[nutrient] - available.get(nutrient, 0))
            deficits[nutrient] = deficit
        return deficits

    def _generate_application_schedule(self, crop_data: Dict, fertilizers: List[FertilizerType],
                                     area_hectares: float) -> List[ApplicationTiming]:
        """Generate detailed application schedule based on crop growth stages"""
        schedule = []
        growth_stages = crop_data.get("growth_stages", [])

        for i, stage in enumerate(growth_stages):
            if stage["n_percent"] > 0:  # Only create applications where nutrients are needed
                # Determine primary fertilizer for this stage
                fertilizer_name = "npk_15_15_15" if i == 0 else "urea"

                # Find matching fertilizer
                selected_fertilizer = next(
                    (f for f in fertilizers if fertilizer_name in f.name.lower()),
                    fertilizers[0] if fertilizers else None
                )

                if selected_fertilizer:
                    # Calculate application rate based on stage requirements
                    base_rate = 200  # Base rate kg/ha
                    stage_rate = base_rate * (stage["n_percent"] / 100)

                    application = ApplicationTiming(
                        stage=stage["stage"],
                        days_after_planting=stage["days"],
                        fertilizer_type=selected_fertilizer.name,
                        amount_kg_per_ha=stage_rate,
                        application_method="broadcast" if i == 0 else "side_dress",
                        notes=f"Apply during {stage['stage']} stage for optimal nutrient uptake"
                    )
                    schedule.append(application)

        return schedule

    def _calculate_costs(self, fertilizers: List[FertilizerType], area_hectares: float,
                        currency: str) -> CostAnalysis:
        """Calculate total fertilization costs"""
        total_cost = 0
        breakdown = {}

        for fertilizer in fertilizers:
            fertilizer_cost = fertilizer.price_per_kg * 200 * area_hectares  # Assuming 200kg/ha average
            total_cost += fertilizer_cost
            breakdown[fertilizer.name] = fertilizer_cost

        return CostAnalysis(
            total_cost=total_cost,
            cost_per_hectare=total_cost / area_hectares if area_hectares > 0 else 0,
            currency=currency,
            fertilizer_breakdown=breakdown
        )

    def _generate_climate_considerations(self, region_data: Dict, crop_selection: CropSelection) -> List[str]:
        """Generate climate-specific recommendations"""
        considerations = []

        # Seasonal considerations
        if crop_selection.planting_season == "wet":
            considerations.append("Apply nitrogen in split doses to reduce leaching during rainy season")
            considerations.append("Ensure proper drainage to prevent nutrient loss")
        else:
            considerations.append("Consider irrigation scheduling with fertilizer applications")
            considerations.append("Monitor soil moisture before fertilizer application")

        # Regional climate considerations
        if region_data.get("climate_type") == "arid":
            considerations.append("Use slow-release fertilizers to improve efficiency")
            considerations.append("Apply fertilizers before expected rainfall")
        elif region_data.get("climate_type") == "humid":
            considerations.append("Split nitrogen applications to prevent leaching")
            considerations.append("Consider organic matter additions to improve soil structure")

        return considerations

    def _calculate_expected_yield(self, crop_data: Dict, soil_analysis: SoilAnalysis,
                                nutrient_deficits: Dict) -> float:
        """Estimate expected yield with fertilizer application"""
        base_yield = crop_data.get("typical_yield", 5.0)

        # Adjust for soil pH
        ph_optimum = crop_data.get("ph_optimum", {"min": 6.0, "max": 7.0})
        if soil_analysis.ph < ph_optimum["min"] or soil_analysis.ph > ph_optimum["max"]:
            base_yield *= 0.85  # Reduce yield for sub-optimal pH

        # Adjust for organic matter
        if soil_analysis.organic_matter < 2.0:
            base_yield *= 0.90  # Reduce yield for low organic matter
        elif soil_analysis.organic_matter > 4.0:
            base_yield *= 1.10  # Increase yield for good organic matter

        # Adjust for fertilizer application (simplified model)
        if sum(nutrient_deficits.values()) > 0:
            base_yield *= 1.20  # Assume 20% yield increase with proper fertilization

        return round(base_yield, 2)

    def _calculate_roi(self, fertilizer_cost: float, expected_yield: float,
                      target_yield: float, region_data: Dict) -> float:
        """Calculate return on investment for fertilizer application"""
        crop_price = region_data.get("crop_price_per_ton", 200)  # Default price per ton
        yield_increase = max(0, expected_yield - (target_yield * 0.7))  # Assume 70% yield without fertilizer
        additional_revenue = yield_increase * crop_price

        if fertilizer_cost > 0:
            roi = ((additional_revenue - fertilizer_cost) / fertilizer_cost) * 100
            return round(roi, 2)
        return 0.0

    def _identify_risk_factors(self, soil_analysis: SoilAnalysis, region_data: Dict) -> List[str]:
        """Identify potential risks and limitations"""
        risks = []

        if soil_analysis.ph < 5.5:
            risks.append("Low soil pH may limit nutrient availability - consider liming")
        if soil_analysis.ph > 8.0:
            risks.append("High soil pH may cause micronutrient deficiencies")
        if soil_analysis.organic_matter < 2.0:
            risks.append("Low organic matter may affect nutrient retention")
        if soil_analysis.cec < 10:
            risks.append("Low CEC may result in nutrient leaching")
        if region_data.get("rainfall_variability", "low") == "high":
            risks.append("High rainfall variability may affect fertilizer timing")

        return risks

    def _generate_alternatives(self, primary_fertilizers: List[FertilizerType]) -> List[str]:
        """Generate alternative fertilization options"""
        alternatives = [
            "Consider organic fertilizers like compost or manure for long-term soil health",
            "Explore precision agriculture techniques for variable rate application",
            "Investigate bio-fertilizers and microbial inoculants",
            "Consider cover crops and green manures for sustainable nutrition"
        ]
        return alternatives

    def get_available_crops(self) -> List[str]:
        """Get list of supported crops"""
        return list(self.crop_data.keys())
