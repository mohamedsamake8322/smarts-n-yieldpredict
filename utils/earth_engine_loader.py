import ee
import pandas as pd
from datetime import datetime, timedelta

ee.Initialize(project='plant-ai-mohamed-tpu')

# Zone d’intérêt (exemple : rectangle autour de Manisa, Türkiye)
geometry = ee.Geometry.Rectangle([27.3, 38.5, 27.7, 38.7])

def get_indicator(image_collection_id, band, start_date, end_date, geometry, scale=250):
    collection = ee.ImageCollection(image_collection_id) \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry) \
        .select(band)
    image = collection.mean()
    value = image.reduceRegion(ee.Reducer.mean(), geometry, scale).get(band)
    return value.getInfo() if value else None

def get_agro_indicators(n_days=30):
    start_date = datetime.utcnow().date() - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    results = []

    for date in dates:
        ee_date = ee.Date(str(date))
        ee_next = ee_date.advance(1, 'day')

        indicators = {
            "date": date,
            "NDVI": get_indicator("MODIS/006/MOD13Q1", "NDVI", ee_date, ee_next.advance(16, 'day'), geometry),
            "EVI": get_indicator("MODIS/006/MOD13Q1", "EVI", ee_date, ee_next.advance(16, 'day'), geometry),
            "NDMI": get_indicator("MODIS/061/MOD09GA", "NDMI", ee_date, ee_next, geometry),
            "ETa": get_indicator("FAO/WAPOR/3/L1_AETI_D", "AETI", ee_date, ee_next, geometry),
            "RET": get_indicator("FAO/WAPOR/3/L1_RET_E", "RET", ee_date, ee_next, geometry),
            "SPEI": get_indicator("IDAHO_EPSCOR/TERRACLIMATE", "spei", ee_date, ee_next, geometry),
            "SMAP": get_indicator("NASA_USDA/HSL/SMAP10KM_soil_moisture", "ssm", ee_date, ee_next, geometry)
        }

        results.append(indicators)

    return pd.DataFrame(results)
