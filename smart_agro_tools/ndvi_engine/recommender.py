def suggest_inputs_from_dataset(df, lat, lon, crop_target):
    sub_df = df[(df['latitude'] == lat) & (df['longitude'] == lon) & (df['culture'] == crop_target)]
    if sub_df.empty:
        raise ValueError("Pas de données pour ce point et cette culture.")

    # Extraction des signaux biophysiques
    soil = {
        "GWETPROF": sub_df["GWETPROF"].mean(),
        "GWETROOT": sub_df["GWETROOT"].mean(),
        "GWETTOP": sub_df["GWETTOP"].mean()
    }
    climate = {
        "WD10M": sub_df["WD10M"].mean(),
        "WS10M_RANGE": sub_df["WS10M_RANGE"].mean()
    }
    yield_target = sub_df["yield_target"].mean()

    # Appel à la fonction principale
    return suggest_inputs(ndvi_profile=[], soil_data=soil, climate_data=climate, crop_target=crop_target, yield_target=yield_target) # type: ignore
