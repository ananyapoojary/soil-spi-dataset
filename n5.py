import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
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

# ✅ Feature scaling for consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ RandomForest Hyperparameters
rf_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# ✅ XGBoost Hyperparameters
xgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 8,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'random_state': 42,
    'n_jobs': -1
}

# ✅ Initialize models
rf_model = RandomForestRegressor(**rf_params)
xgb_model = XGBRegressor(**xgb_params)

# ✅ Create Ensemble with Stacking
estimators = [
    ('rf', rf_model),
    ('xgb', xgb_model)
]

ensemble_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=150, random_state=42)
)

# ✅ Train the ensemble model
print("\n🚀 Training Ensemble Model...")
ensemble_model.fit(X_train, y_train)

# ✅ Model evaluation
y_pred = ensemble_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Ensemble Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the ensemble model and scaler
model_filename = "ensemble_nitrogen_model.pkl"
joblib.dump((ensemble_model, scaler), model_filename)
print(f"\n✅ Ensemble Model saved as '{model_filename}'!")

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
    ensemble_model, scaler = joblib.load(model_filename)

    while True:
        user_input = get_user_input()
        
        # Scale user input with the same scaler used in training
        df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(df)

        prediction = ensemble_model.predict(scaled_input)[0]

        print("\n🔹 User Input:")
        for key, val in user_input.items():
            print(f"{key}: {val}")

        print(f"\n✅ Predicted Nitrogen Value: {prediction:.2f}")

        # Ask if the user wants to predict again
        again = input("\nDo you want to predict again? (yes/no): ").strip().lower()
        if again != "yes":
            print("\n✅ Exiting the prediction loop. Bye!")
            break

# ✅ Run continuous prediction
predict_nitrogen()
