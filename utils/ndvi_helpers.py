import numpy as np

def ndvi_to_monthly_profile(ndvi_da):
    """
    Convertit un NDVI xarray en série mensuelle moyenne.
    """
    monthly_means = ndvi_da.groupby("time.month").mean().values
    return np.round(monthly_means, 3).tolist()
