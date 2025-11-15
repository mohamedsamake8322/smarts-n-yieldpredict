import numpy as np # type: ignore
from soil_matcher import adjust_for_soil
from climate_filter import adjust_for_climate
from stress_detector import detect_stress_from_ndvi

# Simulated ML model (à remplacer par ton vrai modèle)
def predict_npk(base_features):
    """
    Dummy NPK predictor – replace with your real ML model.
    Args:
        base_features (dict): base features before adjustment

    Returns:
        dict: {'N': value, 'P': value, 'K': value}
    """
    return {
        'N': 100 + base_features.get('soil_score', 0),
        'P': 60 + base_features.get('climate_factor', 0),
        'K': 80 + base_features.get('stress_factor', 0),
    }

def main():
    # 🌱 1. Input Data Simulation (à remplacer par des vraies entrées utilisateur)
    soil_data = {'moisture': 25}  # en pourcentage
    climate_data = {'WD10M': 110, 'WS10M_RANGE': 60}
    ndvi_profile = [0.52, 0.49, 0.51, 0.55]

    # 🛠 2. Ajustements
    soil_score = adjust_for_soil(soil_data)
    climate_factor = adjust_for_climate(climate_data)
    stress_factor = detect_stress_from_ndvi(ndvi_profile)

    # 🔍 3. Compilation des features pour le modèle
    features = {
        'soil_score': soil_score,
        'climate_factor': climate_factor,
        'stress_factor': stress_factor
    }

    # 🤖 4. Prédiction NPK
    recommendation = predict_npk(features)

    # 📊 5. Résultat
    print("✅ Recommended Fertilization (in kg/ha):")
    print(f"🧪 Nitrogen (N): {recommendation['N']:.1f}")
    print(f"🧪 Phosphorus (P): {recommendation['P']:.1f}")
    print(f"🧪 Potassium (K): {recommendation['K']:.1f}")

if __name__ == "__main__":
    main()
