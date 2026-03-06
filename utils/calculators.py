"""
Calculateurs agricoles pour Agro-Scan
"""

class FertilizerCalculator:
    """Calculateur d'engrais"""
    
    @staticmethod
    def calculate_npk(area_hectares, crop_type, soil_analysis=None):
        """Calcule les besoins en NPK"""
        # Recommandations de base par culture
        recommendations = {
            "Tomate": {"N": 120, "P": 60, "K": 150},
            "Maïs": {"N": 150, "P": 50, "K": 100},
            "Riz": {"N": 100, "P": 40, "K": 60},
            "Manioc": {"N": 80, "P": 40, "K": 100},
        }
        
        base = recommendations.get(crop_type, {"N": 100, "P": 50, "K": 100})
        
        # Calcul par hectare
        n_kg = base["N"] * area_hectares
        p_kg = base["P"] * area_hectares
        k_kg = base["K"] * area_hectares
        
        return {
            "N": round(n_kg, 2),
            "P": round(p_kg, 2),
            "K": round(k_kg, 2),
            "total": round(n_kg + p_kg + k_kg, 2)
        }
    
    @staticmethod
    def calculate_fertilizer_amount(npk_needs, fertilizer_type, npk_ratio):
        """Calcule la quantité d'engrais nécessaire"""
        # npk_ratio = {"N": 15, "P": 15, "K": 15} pour NPK 15-15-15
        n_amount = npk_needs["N"] / (npk_ratio["N"] / 100)
        p_amount = npk_needs["P"] / (npk_ratio["P"] / 100)
        k_amount = npk_needs["K"] / (npk_ratio["K"] / 100)
        
        # Prendre le maximum
        total_amount = max(n_amount, p_amount, k_amount)
        
        return round(total_amount, 2)


class PesticideCalculator:
    """Calculateur de pesticides"""
    
    @staticmethod
    def calculate_dose(area_hectares, pesticide_type, concentration, recommended_rate):
        """
        Calcule la dose de pesticide nécessaire
        recommended_rate en L/ha ou kg/ha
        """
        total_dose = area_hectares * recommended_rate
        
        # Calcul de la quantité d'eau nécessaire (généralement 200-400 L/ha)
        water_per_hectare = 300  # L/ha
        total_water = area_hectares * water_per_hectare
        
        return {
            "pesticide_amount": round(total_dose, 2),
            "water_amount": round(total_water, 2),
            "concentration": concentration,
            "unit": "L" if pesticide_type == "liquid" else "kg"
        }
    
    @staticmethod
    def calculate_cost(pesticide_amount, price_per_unit):
        """Calcule le coût du traitement"""
        return round(pesticide_amount * price_per_unit, 2)


class FarmingCalculator:
    """Calculateur agricole général"""
    
    @staticmethod
    def calculate_yield(area_hectares, plants_per_hectare, fruits_per_plant, avg_fruit_weight_kg):
        """Calcule le rendement estimé"""
        total_plants = area_hectares * plants_per_hectare
        total_fruits = total_plants * fruits_per_plant
        total_yield_kg = total_fruits * avg_fruit_weight_kg
        total_yield_tons = total_yield_kg / 1000
        
        return {
            "total_plants": int(total_plants),
            "estimated_yield_kg": round(total_yield_kg, 2),
            "estimated_yield_tons": round(total_yield_tons, 2)
        }
    
    @staticmethod
    def calculate_irrigation(area_hectares, crop_water_needs_mm, irrigation_efficiency=0.8):
        """Calcule les besoins en irrigation"""
        # Conversion mm en m3/ha (1 mm = 10 m3/ha)
        water_needs_m3 = area_hectares * crop_water_needs_mm * 10
        irrigation_needs = water_needs_m3 / irrigation_efficiency
        
        return {
            "water_needs_m3": round(water_needs_m3, 2),
            "irrigation_needs_m3": round(irrigation_needs, 2),
            "irrigation_efficiency": irrigation_efficiency
        }
    
    @staticmethod
    def calculate_planting_density(length_m, width_m, spacing_row_m, spacing_plant_m):
        """Calcule la densité de plantation"""
        area_ha = (length_m * width_m) / 10000
        plants_per_row = length_m / spacing_plant_m
        rows = width_m / spacing_row_m
        total_plants = plants_per_row * rows
        plants_per_hectare = total_plants / area_ha
        
        return {
            "area_hectares": round(area_ha, 4),
            "total_plants": int(total_plants),
            "plants_per_hectare": int(plants_per_hectare)
        }
