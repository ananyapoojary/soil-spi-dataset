import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================
# Step 1: Load the Dataset
# ==============================
# Replace 'your_dataset.csv' with your actual file path
file_path = "b_merged - Sheet1.csv"  
df = pd.read_csv(file_path)

print("\n📊 Dataset Info:")
print(df.info())
print(df.head())

# ==============================
# Step 2: Handle Missing Values
# ==============================
# Identify columns with missing values
missing_cols = df.columns[df.isnull().any()].tolist()
print(f"\n🔎 Columns with missing values: {missing_cols}")

# Split dataset into complete and missing rows
complete_df = df.dropna(subset=missing_cols)
missing_df = df[df.isnull().any(axis=1)]
f
print(f"✅ Complete rows: {len(complete_df)}")
print(f"❌ Missing rows: {len(missing_df)}")

# ==============================
# Step 3: Train ML Model to Fill Missing Values
# ==============================
# Features to use for prediction
features = ['Latitude', 'Longitude', 'clay', 'cec', 'ocs', 'sand', 'silt', 
            'ocd', 'nitrogen', 'wv1500', 'cfvo', 'wv0033', 'wv0010', 
            'soc', 'bdod', 'phh2o']

# Loop through each column with missing values
for col in missing_cols:
    print(f"\n🔹 Predicting missing values for: {col}")
    
    # Separate rows with and without missing values for the current column
    train_df = complete_df.dropna(subset=[col])
    test_df = missing_df[missing_df[col].isnull()]

    if test_df.empty:
        print(f"⚠️ No missing values for {col}. Skipping...")
        continue

    # Prepare data for training
    X_train = train_df[features].drop(col, axis=1)
    y_train = train_df[col]

    X_test = test_df[features].drop(col, axis=1)

    # Train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict missing values
    predictions = model.predict(X_test)

    # Fill missing values with predictions
    df.loc[df[col].isnull(), col] = predictions

    # Evaluate the model
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print(f"✅ {col} - MSE: {mse:.4f}")

# ==============================
# Step 4: Save the Updated Dataset
# ==============================
output_file = "dataset_filled.csv"
df.to_csv(output_file, index=False)
print(f"\n📁 Dataset with missing values filled saved as {output_file}")
