import pandas as pd

def standardize_columns(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def drop_missing_critical(df, required=["latitude", "longitude", "culture", "yield_target"]):
    return df.dropna(subset=required)
