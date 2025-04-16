import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
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

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Hyperparameter tuning with RandomizedSearchCV (faster than GridSearchCV)
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, 
    n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='r2', random_state=42
)

random_search.fit(X_train, y_train)

# ✅ Best model from RandomizedSearchCV
model = random_search.best_estimator_
print(f"\n✅ Best Parameters: {random_search.best_params_}")

# ✅ Model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Final Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the optimized model
model_filename = "nitrogen_predictor_final.pkl"
joblib.dump(model, model_filename)
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
    model = joblib.load(model_filename)

    while True:
        user_input = get_user_input()
        df = pd.DataFrame([user_input])
        prediction = model.predict(df)[0]

        print("\n🔹 User Input:")
        for key, val in user_input.items():
            print(f"{key}: {val}")

        print(f"\n✅ Predicted Nitrogen Value: {prediction:.2f}")

        # Ask if user wants to predict again
        again = input("\nDo you want to predict again? (yes/no): ").strip().lower()
        if again != "yes":
            print("\n✅ Exiting the prediction loop. Bye!")
            break

# ✅ Run continuous prediction
predict_nitrogen()
