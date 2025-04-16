import pandas as pd

# ✅ Get user input
def get_user_input():
    """Ask for dynamic input from the user"""
    features = [
        "Latitude", "Longitude", "clay", "cec", "ocs", "sand", "silt", "ocd",
        "wv1500", "cfvo", "wv0033", "wv0010", "soc", "bdod", "phh2o",
        "Elevation", "Mean Temperature", "Humidity", "Rainfall", "nitrogen"
    ]
    
    values = {}
    for feature in features:
        values[feature] = float(input(f"Enter {feature}: "))
    
    return pd.DataFrame([values])

# ✅ Calculate P and K using scientific formulas
def estimate_p_k(df):
    df["Phosphorus"] = (
        (df["cec"] * 0.35) + 
        (df["soc"] * 12) - 
        (df["phh2o"] * 2.1) + 
        (df["wv0010"] * 0.25) + 
        (df["Rainfall"] * 0.015)
    )

    df["Potassium"] = (
        (df["cec"] * 0.6) + 
        (df["clay"] * 1.8) - 
        (df["sand"] * 0.3) + 
        (df["wv0010"] * 0.4) - 
        (df["Elevation"] * 0.01)
    )

    print("\n✅ Estimated Phosphorus and Potassium values:")
    print(f"Phosphorus (P): {df['Phosphorus'].iloc[0]:.2f}")
    print(f"Potassium (K): {df['Potassium'].iloc[0]:.2f}")

# ✅ Main script
if __name__ == "__main__":
    print("\n🔹 Enter soil parameters to estimate P & K:")
    user_data = get_user_input()
    estimate_p_k(user_data)
