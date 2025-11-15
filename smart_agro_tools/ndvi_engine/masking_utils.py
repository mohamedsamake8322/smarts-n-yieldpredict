# smart_agro_tools/masking_utils.py

import numpy as np # type: ignore

def masking(image_data, cloud_mask=None, nodata_value=None):
    """
    Applique un masque sur les données image (par exemple NDVI)
    pour exclure les zones de nuages, d'ombre ou les valeurs invalides.

    Args:
        image_data (np.ndarray): tableau 2D de données NDVI ou autre indice.
        cloud_mask (np.ndarray, optional): masque booléen où True indique un pixel masqué.
        nodata_value (float or int, optional): valeur à exclure (ex : -9999 ou 0).

    Returns:
        np.ndarray: image masquée (valeurs exclues remplacées par np.nan).
    """

    # Copie des données pour éviter de modifier l'original
    masked_data = image_data.copy().astype(np.float32)

    # Masquage des pixels avec la valeur nodata
    if nodata_value is not None:
        masked_data[masked_data == nodata_value] = np.nan

    # Masquage des pixels définis par un masque de nuages ou autre
    if cloud_mask is not None:
        if cloud_mask.shape != masked_data.shape:
            raise ValueError("La forme du cloud_mask ne correspond pas à celle de l'image.")
        masked_data[cloud_mask] = np.nan

    return masked_data
