import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ✅ Load your dataset
# Replace with your actual dataset path
file_path = "merged_output.csv"  
data = pd.read_csv(file_path)

# ✅ Check for missing values and drop them (optional)
data = data.dropna()

# ✅ Define features and target
target = 'nitrogen'
features = [
    "Latitude", "Longitude", "clay", "cec", "ocs", "sand", "silt", "ocd",
    "wv1500", "cfvo", "wv0033", "wv0010", "soc", "bdod", "phh2o",
    "Elevation", "Mean Temperature", "Humidity", "Rainfall"
]

X = data[features]    # Features
y = data[target]       # Target variable

# ✅ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=300,        # More trees for stability
    max_depth=20,             # Prevent overfitting
    random_state=42,
    n_jobs=-1                 # Use all CPU cores
)
model.fit(X_train, y_train)

# ✅ Model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the trained model
model_filename = "nitrogen_predictor.pkl"
joblib.dump(model, model_filename)
print(f"\n✅ Model saved successfully as '{model_filename}'!")

# ✅ Function for single-row prediction
def predict_nitrogen(features_dict):
    """
    Predict nitrogen based on input features.
    :param features_dict: Dictionary with keys matching the feature names
    :return: Predicted nitrogen value
    """
    model = joblib.load(model_filename)  # Load the saved model
    df = pd.DataFrame([features_dict])
    prediction = model.predict(df)[0]
    return prediction

# ✅ Example usage
example_input = {
    "Latitude": 13.5,
    "Longitude": 75.9,
    "clay": 20.5,
    "cec": 25.4,
    "ocs": 10.2,
    "sand": 45.3,
    "silt": 34.1,
    "ocd": 1.8,
    "wv1500": 30.0,
    "cfvo": 12.5,
    "wv0033": 22.8,
    "wv0010": 18.4,
    "soc": 0.9,
    "bdod": 1.35,
    "phh2o": 6.8,
    "Elevation": 250.0,
    "Mean Temperature": 28.5,
    "Humidity": 70.0,
    "Rainfall": 1200.0
}

# ✅ Make a prediction
predicted_nitrogen = predict_nitrogen(example_input)
print(f"\n🔹 Predicted Nitrogen Value: {predicted_nitrogen:.2f}")
