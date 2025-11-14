from shapely.geometry import Point
import geopandas as gpd

def latlon_to_geom(lat, lon, crs="EPSG:4326"):
    """
    Convertit des coordonnées lat/lon en géométrie PostGIS.
    """
    geom = Point(float(lon), float(lat))
    return gpd.GeoSeries([geom], crs=crs).geometry[0]
