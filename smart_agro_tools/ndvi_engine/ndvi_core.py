# smart_agro_tools/ndvi_core.py

def calculate_indices(ds, index='NDVI'):
    """
    Calcule des indices de végétation à partir d'un dataset xarray.

    Args:
        ds (xarray.Dataset): Jeu de données avec au moins les bandes nécessaires.
        index (str): Nom de l'indice à calculer. Actuellement supporté : 'NDVI'.

    Returns:
        xarray.DataArray: L'indice calculé.
    """
    if index.upper() == 'NDVI':
        if 'B8' in ds and 'B4' in ds:
            nir = ds['B8']
            red = ds['B4']
        elif 'nir' in ds and 'red' in ds:
            nir = ds['nir']
            red = ds['red']
        else:
            raise ValueError("❌ Les bandes nécessaires au calcul NDVI sont absentes.")

        ndvi = (nir - red) / (nir + red)
        return ndvi

    else:
        raise NotImplementedError(f"⚠️ L'indice '{index}' n'est pas supporté.")
