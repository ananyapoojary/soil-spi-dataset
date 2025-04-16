import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import xgboost as xgb

# ✅ Load dataset
file_path = "your_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path).dropna()

# ✅ Define features and target
target = 'nitrogen'  # Replace with your target variable
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

# ✅ Fine-tuned base models
rf_model = RandomForestRegressor(
    n_estimators=600,
    max_depth=50,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=12,
    subsample=0.95,
    random_state=42
)

# ✅ XGBoost using `train()` API
print("\n🚀 Training Optimized Stacked Ensemble Models...")

# Convert to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.02,
    "max_depth": 18,
    "colsample_bytree": 0.9,
    "subsample": 0.95,
    "alpha": 0.2,               # L1 regularization
    "lambda": 0.6,              # L2 regularization
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "rmse"
}

# Train XGBoost with early stopping
evals = [(dtrain, "train"), (dtest, "eval")]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=800,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=False
)

# ✅ Train RandomForest and Gradient Boosting models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# ✅ Cross-validation for stability
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='r2')
gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=10, scoring='r2')

print("\n🔹 RandomForest CV R² Scores:", rf_cv_scores)
print("🔹 Gradient Boosting CV R² Scores:", gb_cv_scores)

# ✅ Stack predictions for meta-model
rf_preds_train = rf_model.predict(X_train)
xgb_preds_train = xgb_model.predict(dtrain)
gb_preds_train = gb_model.predict(X_train)

# ✅ Create meta-model dataset
meta_train = np.column_stack((rf_preds_train, xgb_preds_train, gb_preds_train))

# ✅ Use Ridge Regression for meta-model
meta_model = Ridge(alpha=0.1)
meta_model.fit(meta_train, y_train)

# ✅ Test set predictions
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(dtest)
gb_preds = gb_model.predict(X_test)

# ✅ Stacked ensemble predictions
meta_test = np.column_stack((rf_preds, xgb_preds, gb_preds))
ensemble_preds = meta_model.predict(meta_test)

# ✅ Model Evaluation
mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
r2 = r2_score(y_test, ensemble_preds)

print("\n🔹 Optimized Stacked Ensemble Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the models
model_filename = "optimized_stacked_ensemble_nitrogen_model.pkl"
joblib.dump((rf_model, xgb_model, gb_model, meta_model, scaler), model_filename)
print(f"\n✅ Optimized Stacked Ensemble Model saved as '{model_filename}'!")

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

        # Convert to DMatrix for XGBoost
        duser = xgb.DMatrix(scaled_input)

        # Predict with all base models
        rf_pred = rf_model.predict(scaled_input)[0]
        xgb_pred = xgb_model.predict(duser)[0]
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
