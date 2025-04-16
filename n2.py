import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ✅ Load dataset
file_path = "your_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)
data = data.dropna()

# ✅ Define features and target
target = 'nitrogen'
features = [
    "Latitude", "Longitude", "clay", "cec", "ocs", "sand", "silt", "ocd",
    "wv1500", "cfvo", "wv0033", "wv0010", "soc", "bdod", "phh2o",
    "Elevation", "Mean Temperature", "Humidity", "Rainfall"
]

X = data[features]
y = data[target]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Hyperparameter tuning
param_grid = {
    'n_estimators': [200, 300, 400],      # Number of trees
    'max_depth': [15, 20, 25],            # Depth of the trees
    'min_samples_split': [2, 5, 10],      # Minimum samples for a split
    'min_samples_leaf': [1, 2, 4]         # Minimum samples per leaf
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='r2')
grid_search.fit(X_train, y_train)

# ✅ Best model from GridSearch
model = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# ✅ Model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Improved Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the tuned model
model_filename = "nitrogen_predictor_optimized.pkl"
joblib.dump(model, model_filename)
print(f"\n✅ Optimized Model saved successfully as '{model_filename}'!")

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
    model = joblib.load(model_filename)
    user_input = get_user_input()
    df = pd.DataFrame([user_input])
    prediction = model.predict(df)[0]
    
    print("\n🔹 User Input:")
    for key, val in user_input.items():
        print(f"{key}: {val}")
    
    print(f"\n✅ Predicted Nitrogen Value: {prediction:.2f}")

# ✅ Run dynamic prediction
predict_nitrogen()
