from smart_agro_tools.dataset_loader import load_ard

def check(ndvi_array):
    """
    Vérifie la validité d'un tableau NDVI.

    Args:
        ndvi_array (xarray.DataArray | np.ndarray | list): Données NDVI à valider.

    Returns:
        bool: True si les données sont valides, False sinon.
    """
    try:
        # Vérifie que les données ne sont pas None
        if ndvi_array is None:
            return False

        # Vérifie que les données ont un attribut 'shape' et une taille non nulle
        if not hasattr(ndvi_array, 'shape') or getattr(ndvi_array, 'size', 0) == 0:
            return False

        # Vérifie que toutes les valeurs ne sont pas nulles ou masquées
        if hasattr(ndvi_array, 'values'):
            data = ndvi_array.values
        else:
            data = ndvi_array

        # Vérifie que le tableau contient au moins quelques valeurs numériques valides
        valid_pixels = (data is not None) and (data != 0) and (data == data)  # élimine NaN
        if hasattr(valid_pixels, 'sum'):
            return valid_pixels.sum() > 0
        elif isinstance(valid_pixels, list):
            return sum(1 for v in valid_pixels if v) > 0

        return True
    except Exception as e:
        print(f"[NDVI Validation Error] {e}")
        return False

