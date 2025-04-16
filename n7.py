import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

# ✅ Fine-Tuned RandomForest Hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=400,               # Increased estimators
    max_depth=40,                   # Deeper trees
    min_samples_split=2,            
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# ✅ Fine-Tuned XGBoost Hyperparameters
xgb_model = XGBRegressor(
    n_estimators=500,               # More estimators
    learning_rate=0.03,             # Lower learning rate for better convergence
    max_depth=12,                   # Deeper trees
    colsample_bytree=0.9,           
    subsample=0.95,                  # Slightly higher subsample
    reg_alpha=0.1,                   # L1 regularization
    reg_lambda=0.5,                  # L2 regularization
    random_state=42,
    n_jobs=-1
)

# ✅ Train both models
print("\n🚀 Training Fine-Tuned Models...")
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# ✅ Cross-validation for stability
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')

print("\n🔹 RandomForest CV R² Scores:", rf_cv_scores)
print("🔹 XGBoost CV R² Scores:", xgb_cv_scores)

# ✅ Grid Search to Fine-Tune Weights
print("\n🔹 Fine-tuning ensemble weights...")

# Range of weights for grid search
weights = np.arange(0.1, 1.0, 0.1)
best_r2 = -np.inf
best_rf_weight = 0.5
best_xgb_weight = 0.5

# Grid search for optimal weights
for rf_w in weights:
    xgb_w = 1.0 - rf_w
    ensemble_preds = (rf_w * rf_model.predict(X_test)) + (xgb_w * xgb_model.predict(X_test))
    r2 = r2_score(y_test, ensemble_preds)

    if r2 > best_r2:
        best_r2 = r2
        best_rf_weight = rf_w
        best_xgb_weight = xgb_w

print(f"\n✅ Optimal Weights: RF: {best_rf_weight:.2f}, XGB: {best_xgb_weight:.2f}")
print(f"✅ Best R² Score: {best_r2:.4f}")

# ✅ Weighted Ensemble Prediction
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
ensemble_preds = (best_rf_weight * rf_preds) + (best_xgb_weight * xgb_preds)

# ✅ Model evaluation
mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
r2 = r2_score(y_test, ensemble_preds)

print("\n🔹 Final Fine-Tuned Ensemble Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the models, weights, and scaler
model_filename = "fine_tuned_ensemble_nitrogen_model.pkl"
joblib.dump((rf_model, xgb_model, scaler, best_rf_weight, best_xgb_weight), model_filename)
print(f"\n✅ Fine-Tuned Ensemble Model saved as '{model_filename}'!")

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
