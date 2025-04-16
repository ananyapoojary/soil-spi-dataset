import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ✅ Load dataset
file_path = "your_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path).dropna()

# ✅ Define features and target
target = 'nitrogen'
features = [
    "Latitude", "Longitude", "clay", "cec", "ocs", "sand", "silt", "ocd",
    "wv1500", "cfvo", "wv0033", "wv0010", "soc", "bdod", "phh2o",
    "Elevation", "Mean Temperature", "Humidity", "Rainfall"
]

X = data[features]
y = data[target]

# ✅ Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ RandomForest Hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=35, min_samples_split=2, 
    min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
)

# ✅ XGBoost Hyperparameters
xgb_model = XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=10, 
    colsample_bytree=0.8, subsample=0.9, random_state=42, n_jobs=-1
)

# ✅ Train both models
print("\n🚀 Training Weighted Averaging Ensemble...")
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# ✅ Weighted Averaging
rf_weight = 0.6  # Weight for RandomForest (higher because it performed better before)
xgb_weight = 0.4  # Weight for XGBoost

# ✅ Model Evaluation
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# ✅ Weighted average predictions
ensemble_preds = (rf_weight * rf_preds) + (xgb_weight * xgb_preds)

mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
r2 = r2_score(y_test, ensemble_preds)

print("\n🔹 Weighted Ensemble Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the models and scaler
model_filename = "weighted_ensemble_nitrogen_model.pkl"
joblib.dump((rf_model, xgb_model, scaler, rf_weight, xgb_weight), model_filename)
print(f"\n✅ Weighted Ensemble Model saved as '{model_filename}'!")

# ✅ Dynamic input prediction
def get_user_input():
    """Ask for dynamic input from the user"""
    values = {}
    for feature in features:
        value = float(input(f"Enter {feature}: "))
        values[feature] = value
    return values

def predict_nitrogen():
    """Predict nitrogen value using user input"""
    rf_model, xgb_model, scaler, rf_weight, xgb_weight = joblib.load(model_filename)

    while True:
        user_input = get_user_input()

        # Scale user input
        df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(df)

        # Predict with both models
        rf_pred = rf_model.predict(scaled_input)[0]
        xgb_pred = xgb_model.predict(scaled_input)[0]

        # Weighted average prediction
        ensemble_pred = (rf_weight * rf_pred) + (xgb_weight * xgb_pred)

        print("\n🔹 User Input:")
        for key, val in user_input.items():
            print(f"{key}: {val}")

        print(f"\n✅ Predicted Nitrogen Value: {ensemble_pred:.2f}")

        # Ask if the user wants to predict again
        again = input("\nDo you want to predict again? (yes/no): ").strip().lower()
        if again != "yes":
            print("\n✅ Exiting the prediction loop. Bye!")
            break

# ✅ Run continuous prediction
predict_nitrogen()
