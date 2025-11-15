from smart_agro_tools.input_recommender.stress_detector import detect_stress_from_ndvi
from smart_agro_tools.input_recommender.soil_matcher import adjust_for_soil
from smart_agro_tools.input_recommender.climate_filter import adjust_for_climate

def get_base_requirements(crop_target):
    base_npk = {
        "maize": {"N": 50, "P": 30, "K": 20},
        "mil": {"N": 35, "P": 20, "K": 10},
        "Bananas": {"N": 60, "P": 40, "K": 40},
        # À enrichir...
    }
    return base_npk.get(crop_target, {"N": 40, "P": 20, "K": 20})

def suggest_inputs(ndvi_profile, soil_data, climate_data, crop_target, yield_target=None):
    stress = detect_stress_from_ndvi(ndvi_profile)
    soil_factor = adjust_for_soil(soil_data)
    climate_factor = adjust_for_climate(climate_data)
    base = get_base_requirements(crop_target)

    npk = {
        "N": base["N"] * stress * soil_factor * climate_factor,
        "P": base["P"] * stress * soil_factor,
        "K": base["K"] * climate_factor
    }

    # Ajustement si rendement visé est connu
    if yield_target:
        boost = yield_target / 1000
        npk = {k: v * boost for k, v in npk.items()}

    return {k: round(v, 1) for k, v in npk.items()}
