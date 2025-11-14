# 📦 Import des bibliothèques
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ================================
# 🧪 ÉTAPE 1 : Chargement du Dataset
# ================================

df = pd.read_csv("dataset_agricole_prepared.csv")

# 🧼 Normaliser les noms de colonnes
df.columns = [col.strip().lower() for col in df.columns]

# 📋 Vérification des colonnes
print("\n🧠 Colonnes disponibles :")
print(df.columns.tolist())

# 🎯 Sélection des variables explicatives
features = [
    "production", "pesticides_use",
    "prectotcorr", "ws10m_range", "t2m_max",
    "t2m_min", "qv2m", "rh2m",
    "ph", "carbon_organic", "nitrogen_total"
]

# 🔍 Vérification des colonnes existantes
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"\n⚠️ Colonnes manquantes : {missing_features}")
    features = [f for f in features if f in df.columns]
    print(f"✅ Utilisation des colonnes disponibles : {features}")

# 🎯 Extraction des données
df = df.dropna(subset=features + ["yield_target"])
X = df[features]
y = df["yield_target"]

# 🎓 Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 🚀 ÉTAPE 2 : Entraînement du Modèle
# ================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    verbosity=1
)

model.fit(X_train, y_train)

# 📈 Évaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n📊 Évaluation du modèle :")
print(f"✅ RMSE : {rmse:.2f}")
print(f"✅ R²    : {r2:.2f}")

# ================================
# 📊 ÉTAPE 3 : Visualisation des Importances
# ================================

importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("🎯 Importance des variables dans la prédiction de rendement")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
