import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
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

# ✅ Feature scaling for improved accuracy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Expanded Hyperparameter tuning range
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500, 600],     # More estimators for stability
    'max_depth': [20, 30, 40, 50],                      # Deeper trees for better fitting
    'min_samples_split': [2, 5, 10, 15],                # Controls overfitting
    'min_samples_leaf': [1, 2, 4, 8],                   # Minimum samples per leaf
    'max_features': ['sqrt', 'log2', None]              # Best-split feature strategy
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# ✅ Randomized Search for faster tuning
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, 
    n_iter=50, cv=5, verbose=1, n_jobs=-1, scoring='r2', random_state=42
)

random_search.fit(X_train, y_train)

# ✅ Best model from RandomizedSearchCV
model = random_search.best_estimator_
print(f"\n✅ Best Parameters: {random_search.best_params_}")

# ✅ Cross-validation evaluation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"\n🔹 Cross-Validation R² Scores: {cv_scores}")
print(f"🔹 Mean CV R² Score: {np.mean(cv_scores):.4f}")

# ✅ Model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Final Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the final optimized model
model_filename = "nitrogen_predictor_ultimate.pkl"
joblib.dump((model, scaler), model_filename)
print(f"\n✅ Final Model saved as '{model_filename}'!")

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
    model, scaler = joblib.load(model_filename)

    while True:
        user_input = get_user_input()
        
        # Scale user input with the same scaler used in training
        df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(df)

        prediction = model.predict(scaled_input)[0]

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
