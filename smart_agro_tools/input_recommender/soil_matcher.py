
def adjust_for_soil(soil_data: dict) -> float:
    """
    Adjust the NPK recommendation factor based on soil moisture.

    Args:
        soil_data (dict): Must contain 'GWETPROF', 'GWETROOT', 'GWETTOP' moisture values (0.0 to 1.0).

    Returns:
        float: A correction factor between 0.8 and 1.2.
               - < 1.0 indicates lower moisture → reduce dosage
               - > 1.0 indicates good moisture → allow boost
    """
    required_keys = ["GWETPROF", "GWETROOT", "GWETTOP"]

    missing_keys = [k for k in required_keys if k not in soil_data or soil_data[k] is None]
    if missing_keys:
        raise ValueError(f"Missing required soil moisture keys: {', '.join(missing_keys)}")

    try:
        gwet_values = [float(soil_data[k]) for k in required_keys]
        gwet_avg = sum(gwet_values) / len(gwet_values)

        # Empirical logic for adjustment based on average moisture
        if gwet_avg < 0.2:
            factor = 0.85  # Very dry soil
        elif gwet_avg < 0.4:
            factor = 1.0   # Moderate
        else:
            factor = 1.15  # Wet soil, good absorption

        # Clamp to defined range
        return max(0.8, min(factor, 1.2))

    except Exception as e:
        print(f"⚠️ Soil adjustment error: {e}")
        return 1.0  # Neutral fallback in case of failure


# 🧪 Example usage
if __name__ == "__main__":
    test_inputs = [
        {"GWETPROF": 0.1, "GWETROOT": 0.15, "GWETTOP": 0.18},  # Very dry
        {"GWETPROF": 0.35, "GWETROOT": 0.38, "GWETTOP": 0.36},  # Moderate
        {"GWETPROF": 0.5, "GWETROOT": 0.55, "GWETTOP": 0.6},    # Moist
    ]

    for idx, data in enumerate(test_inputs, 1):
        factor = adjust_for_soil(data)
        print(f"Test #{idx} - Adjustment factor: {factor:.2f}")
