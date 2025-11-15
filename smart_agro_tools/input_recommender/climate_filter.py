def adjust_for_climate(climate_data):
    """
    Corrige selon les conditions climatiques (vent, stress météo).

    Args:
        climate_data (dict): WD10M, WS10M_RANGE...

    Returns:
        float: facteur de stress climatique (0.8 à 1.2)
    """
    wind_speed = climate_data.get("WD10M", 0)
    variability = climate_data.get("WS10M_RANGE", 0)
    score = (wind_speed + variability) / 2

    if score > 150:
        return 0.85  # stress élevé → réduction
    elif score > 100:
        return 1.0   # neutre
    else:
        return 1.1   # conditions favorables → petit boost
if __name__ == "__main__":
    test = {"WD10M": 90, "WS10M_RANGE": 80}
    print("Facteur climatique :", adjust_for_climate(test))
