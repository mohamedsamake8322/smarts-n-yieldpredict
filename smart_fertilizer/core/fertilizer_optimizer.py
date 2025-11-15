import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from modules.smart_fertilizer.api.models import FertilizerType


class FertilizerOptimizer:
    """
    Optimizer for selecting the most cost-effective fertilizer combinations
    """

    def __init__(self):
        self.optimization_method = "cost_minimization"
        self.constraints = {
            "max_single_fertilizer_ratio": 0.8,  # Max 80% of nutrients from single source
            "min_efficiency_threshold": 0.6,      # Minimum nutrient use efficiency
            "availability_weight": 0.3            # Weight for fertilizer availability
        }

    def optimize_fertilizer_selection(self, nutrient_deficits: Dict,
                                    fertilizer_database: Dict,
                                    region_data: Dict) -> List[FertilizerType]:
        """
        Optimize fertilizer selection based on nutrient needs and local availability
        """

        # Convert fertilizer database to optimization format
        fertilizers = self._prepare_fertilizer_data(fertilizer_database, region_data)

        if not fertilizers:
            return []

        # Set up optimization problem
        n_fertilizers = len(fertilizers)
        nutrient_requirements = [
            nutrient_deficits.get("n", 0),
            nutrient_deficits.get("p", 0),
            nutrient_deficits.get("k", 0)
        ]

        # Create nutrient content matrix
        nutrient_matrix = np.array([
            [f["n_content"]/100, f["p_content"]/100, f["k_content"]/100]
            for f in fertilizers
        ])

        # Create cost vector
        cost_vector = np.array([f["price_per_kg"] for f in fertilizers])

        # Initial guess - equal distribution
        x0 = np.ones(n_fertilizers) * 100

        # Bounds - minimum 0, maximum reasonable application rate
        bounds = [(0, 1000) for _ in range(n_fertilizers)]

        # Constraints
        constraints = self._create_optimization_constraints(
            nutrient_matrix, nutrient_requirements, n_fertilizers
        )

        # Solve optimization problem
        try:
            result = minimize(
                fun=lambda x: np.dot(x, cost_vector),  # Minimize cost
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                optimal_amounts = result.x
                return self._create_fertilizer_recommendations(fertilizers, optimal_amounts)
            else:
                # Fallback to heuristic solution
                return self._heuristic_selection(fertilizers, nutrient_requirements)

        except Exception as e:
            print(f"Optimization failed: {e}")
            return self._heuristic_selection(fertilizers, nutrient_requirements)

    def _prepare_fertilizer_data(self, fertilizer_database: Dict, region_data: Dict) -> List[Dict]:
        """Prepare fertilizer data for optimization"""
        fertilizers = []

        for name, data in fertilizer_database.items():
            # Adjust price based on regional factors
            regional_price_factor = region_data.get("price_factor", 1.0)
            adjusted_price = data["price_usd_per_kg"] * regional_price_factor

            # Adjust availability
            availability_score = self._calculate_availability_score(data["availability"])

            fertilizer = {
                "name": name,
                "n_content": data["n_content"],
                "p_content": data["p_content"],
                "k_content": data["k_content"],
                "price_per_kg": adjusted_price,
                "availability_score": availability_score,
                "original_data": data
            }
            fertilizers.append(fertilizer)

        # Sort by availability and cost-effectiveness
        fertilizers.sort(key=lambda x: (x["availability_score"], -x["price_per_kg"]), reverse=True)

        return fertilizers

    def _calculate_availability_score(self, availability: str) -> float:
        """Convert availability string to numeric score"""
        availability_map = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
            "very_low": 0.2
        }
        return availability_map.get(availability.lower(), 0.5)

    def _create_optimization_constraints(self, nutrient_matrix: np.ndarray,
                                       requirements: List[float],
                                       n_fertilizers: int) -> List[Dict]:
        """Create optimization constraints"""
        constraints = []

        # Nutrient requirement constraints (must meet at least 90% of requirements)
        for i, req in enumerate(requirements):
            if req > 0:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i, requirement=req:
                           np.dot(x, nutrient_matrix[:, idx]) - 0.9 * requirement
                })

        # Maximum single fertilizer constraint
        max_ratio = self.constraints["max_single_fertilizer_ratio"]
        total_req = sum(requirements)
        if total_req > 0:
            for i in range(n_fertilizers):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i, max_contrib=max_ratio * total_req:
                           max_contrib - x[idx]
                })

        return constraints

    def _heuristic_selection(self, fertilizers: List[Dict],
                           requirements: List[float]) -> List[FertilizerType]:
        """Fallback heuristic fertilizer selection"""
        selected = []
        n_req, p_req, k_req = requirements

        # Select NPK fertilizer if balanced nutrition needed
        if n_req > 0 and p_req > 0 and k_req > 0:
            npk_fertilizer = next((f for f in fertilizers if "npk" in f["name"].lower()), None)
            if npk_fertilizer:
                selected.append(self._create_fertilizer_type(npk_fertilizer, 150))

        # Add nitrogen source if needed
        if n_req > 0:
            n_fertilizer = max(fertilizers, key=lambda x: x["n_content"])
            if n_fertilizer["n_content"] > 0:
                selected.append(self._create_fertilizer_type(n_fertilizer, 100))

        # Add phosphorus source if needed
        if p_req > 0:
            p_fertilizer = max(fertilizers, key=lambda x: x["p_content"])
            if p_fertilizer["p_content"] > 0 and p_fertilizer not in [s.name for s in selected]:
                selected.append(self._create_fertilizer_type(p_fertilizer, 80))

        # Add potassium source if needed
        if k_req > 0:
            k_fertilizer = max(fertilizers, key=lambda x: x["k_content"])
            if k_fertilizer["k_content"] > 0 and k_fertilizer not in [s.name for s in selected]:
                selected.append(self._create_fertilizer_type(k_fertilizer, 60))

        return selected

    def _create_fertilizer_recommendations(self, fertilizers: List[Dict],
                                         amounts: np.ndarray) -> List[FertilizerType]:
        """Create fertilizer recommendations from optimization results"""
        recommendations = []

        for i, amount in enumerate(amounts):
            if amount > 10:  # Only include significant amounts
                fertilizer_data = fertilizers[i]
                recommendations.append(self._create_fertilizer_type(fertilizer_data, amount))

        return recommendations

    def _create_fertilizer_type(self, fertilizer_data: Dict, amount: float = 0) -> FertilizerType:
        """Create FertilizerType object from fertilizer data"""
        return FertilizerType(
            name=fertilizer_data["name"],
            n_content=fertilizer_data["n_content"],
            p_content=fertilizer_data["p_content"],
            k_content=fertilizer_data["k_content"],
            price_per_kg=fertilizer_data["price_per_kg"],
            availability=fertilizer_data["original_data"]["availability"]
        )

    def calculate_nutrient_efficiency(self, selected_fertilizers: List[FertilizerType],
                                    requirements: Dict) -> Dict[str, float]:
        """Calculate nutrient use efficiency for selected fertilizers"""
        total_n = sum(f.n_content * 2 for f in selected_fertilizers)  # Assuming 200kg/ha average
        total_p = sum(f.p_content * 2 for f in selected_fertilizers)
        total_k = sum(f.k_content * 2 for f in selected_fertilizers)

        efficiency = {}
        if requirements.get("n", 0) > 0:
            efficiency["n"] = min(1.0, total_n / requirements["n"])
        if requirements.get("p", 0) > 0:
            efficiency["p"] = min(1.0, total_p / requirements["p"])
        if requirements.get("k", 0) > 0:
            efficiency["k"] = min(1.0, total_k / requirements["k"])

        return efficiency

    def suggest_application_optimization(self, fertilizers: List[FertilizerType],
                                       soil_analysis, crop_data: Dict) -> List[str]:
        """Suggest application optimization strategies"""
        suggestions = []

        # Timing suggestions
        suggestions.append("Split nitrogen applications to improve uptake efficiency")

        # Method suggestions
        if any(f.n_content > 30 for f in fertilizers):
            suggestions.append("Consider banding high-nitrogen fertilizers to reduce losses")

        # Soil-specific suggestions
        if hasattr(soil_analysis, 'ph') and soil_analysis.ph < 6.0:
            suggestions.append("Apply phosphorus fertilizers with organic matter to improve availability")

        # Crop-specific suggestions
        if crop_data.get("crop_type") == "rice":
            suggestions.append("Apply potassium before flooding for better root uptake")

        return suggestions
