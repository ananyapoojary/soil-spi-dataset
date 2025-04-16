import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ✅ Load Dataset
file_path = "your_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path).dropna()

# ✅ Load Nitrogen Model and Add Predictions
rf_nitrogen, xgb_nitrogen, scaler_nitrogen, rf_w, xgb_w = joblib.load("weighted_ensemble_nitrogen_model.pkl")
X_nitrogen = scaler_nitrogen.transform(data.drop(columns=['phosphorus', 'potassium']))
data["nitrogen"] = (rf_w * rf_nitrogen.predict(X_nitrogen)) + (xgb_w * xgb_nitrogen.predict(X_nitrogen))

# ✅ Define Features and Target
features = [
    "Latitude", "Longitude", "clay", "cec", "ocs", "sand", "silt", "ocd",
    "wv1500", "cfvo", "wv0033", "wv0010", "soc", "bdod", "phh2o",
    "Elevation", "Mean Temperature", "Humidity", "Rainfall", "nitrogen"
]

target = 'phosphorus'

X = data[features]
y = data[target]

# ✅ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train Models
rf_model = RandomForestRegressor(n_estimators=300, max_depth=35, random_state=42, n_jobs=-1)
xgb_model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1)

print("\n🚀 Training Phosphorus Models...")
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# ✅ Weighted Ensemble
rf_weight = 0.6
xgb_weight = 0.4

rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
ensemble_preds = (rf_weight * rf_preds) + (xgb_weight * xgb_preds)

# ✅ Evaluate the Model
mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
r2 = r2_score(y_test, ensemble_preds)

print("\n🔹 Phosphorus Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the Model
joblib.dump((rf_model, xgb_model, scaler, rf_weight, xgb_weight), "weighted_ensemble_phosphorus_model.pkl")
print("\n✅ Phosphorus Model saved successfully!")
