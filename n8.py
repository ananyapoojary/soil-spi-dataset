import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
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

# ✅ Model definitions with fine-tuned hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=500,              # Increased estimators
    max_depth=45,                  # Deeper trees
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBRegressor(
    n_estimators=600,              # More estimators
    learning_rate=0.03,            # Reduced learning rate
    max_depth=15,                  # Deeper trees
    colsample_bytree=0.9,
    subsample=0.95,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingRegressor(
    n_estimators=500,              # Additional model for diversity
    learning_rate=0.03,
    max_depth=10,
    subsample=0.95,
    random_state=42
)

# ✅ Train all models
print("\n🚀 Training Stacked Ensemble Models...")
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# ✅ Cross-validation for stability
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2')

print("\n🔹 RandomForest CV R² Scores:", rf_cv_scores)
print("🔹 XGBoost CV R² Scores:", xgb_cv_scores)
print("🔹 Gradient Boosting CV R² Scores:", gb_cv_scores)

# ✅ Stack predictions for the meta-model
rf_preds_train = rf_model.predict(X_train)
xgb_preds_train = xgb_model.predict(X_train)
gb_preds_train = gb_model.predict(X_train)

# ✅ Create a meta-model dataset
meta_train = np.column_stack((rf_preds_train, xgb_preds_train, gb_preds_train))

# ✅ Meta-Model: Linear Regression
meta_model = LinearRegression()
meta_model.fit(meta_train, y_train)

# ✅ Test set predictions
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

# ✅ Stacked ensemble predictions
meta_test = np.column_stack((rf_preds, xgb_preds, gb_preds))
ensemble_preds = meta_model.predict(meta_test)

# ✅ Model Evaluation
mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
r2 = r2_score(y_test, ensemble_preds)

print("\n🔹 Stacked Ensemble Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the models
model_filename = "stacked_ensemble_nitrogen_model.pkl"
joblib.dump((rf_model, xgb_model, gb_model, meta_model, scaler), model_filename)
print(f"\n✅ Stacked Ensemble Model saved as '{model_filename}'!")

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
    rf_model, xgb_model, gb_model, meta_model, scaler = joblib.load(model_filename)

    while True:
        user_input = get_user_input()

        # Scale user input
        df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(df)

        # Predict with all base models
        rf_pred = rf_model.predict(scaled_input)[0]
        xgb_pred = xgb_model.predict(scaled_input)[0]
        gb_pred = gb_model.predict(scaled_input)[0]

        # Stacked ensemble prediction
        ensemble_pred = meta_model.predict(np.array([[rf_pred, xgb_pred, gb_pred]]))[0]

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
