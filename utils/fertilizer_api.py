import pandas as pd
import xgboost as xgb  # type: ignore
import json
import numpy as np

# 📦 Chemins vers le modèle et les métadonnées
MODEL_PATH = r"C:\plateforme-agricole-complete-v2\models\fertilizer_model.bin"
COLS_PATH = r"C:\plateforme-agricole-complete-v2\models\fertilizer_columns.json"
LABELS_PATH = r"C:\plateforme-agricole-complete-v2\models\fertilizer_labels.json"

# 🔄 Chargement du modèle
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# 🔄 Chargement des colonnes et labels
with open(COLS_PATH, "r", encoding="utf-8") as f:
    model_cols = json.load(f)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# 🔮 Fonction de prédiction
def predict_fertilizer(user_inputs: dict) -> str:
    try:
        # Initialiser les colonnes à 0
        X_input_dict = {col: 0 for col in model_cols}

        for k, v in user_inputs.items():
            key = k.lower().strip().replace(" ", "_")
            val = str(v).lower().strip()

            # One-hot encoding pour les colonnes catégorielles
            for col in model_cols:
                if col.startswith(key + "_") and val in col.lower():
                    X_input_dict[col] = 1

            # Valeur numérique directe
            if key in model_cols:
                X_input_dict[key] = v

        # Créer le DataFrame
        X_input = pd.DataFrame([X_input_dict], columns=model_cols)

        # Prédiction
        pred = model.predict(X_input)
        pred_label = int(np.ravel(pred)[0])

        if 0 <= pred_label < len(label_map):
            return label_map[pred_label]
        else:
            return "Fertilizer inconnu"
    except Exception as e:
        return f"❌ Erreur lors de la prédiction : {str(e)}"
