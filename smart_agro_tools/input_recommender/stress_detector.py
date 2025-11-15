import numpy as np # type: ignore

def detect_stress_from_ndvi(ndvi_profile):
    """
    Analyze NDVI time series to detect crop vegetative stress.

    Args:
        ndvi_profile (list or np.array): NDVI time series (0 to 1 scale).

    Returns:
        float: Stress factor (range: 0.5 to 1.5)
               - < 1.0 means likely stress
               - = 1.0 means neutral
               - > 1.0 means healthy vegetation
    """
    if not ndvi_profile or len(ndvi_profile) == 0:
        return 1.0  # Neutral fallback if profile is missing

    try:
        ndvi_array = np.array(ndvi_profile, dtype=np.float32)
        avg_ndvi = np.mean(ndvi_array)
        std_ndvi = np.std(ndvi_array)

        # NDVI thresholds can be adapted to local calibration
        if avg_ndvi < 0.2:
            return 0.5  # Very low → critical stress
        elif avg_ndvi < 0.35:
            return 0.75  # Moderate stress
        elif avg_ndvi < 0.5:
            return 0.9  # Slight stress
        elif avg_ndvi < 0.65:
            return 1.1  # Healthy
        else:
            return 1.3  # Very healthy (e.g. irrigated zones)

    except Exception as e:
        print(f"⚠️ Error in detect_stress_from_ndvi: {e}")
        return 1.0  # Fallback in case of unexpected issue
