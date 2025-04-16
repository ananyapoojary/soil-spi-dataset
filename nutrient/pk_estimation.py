import pandas as pd

# ✅ Load dataset
file_path = "your_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path).dropna()

# ✅ Parameters used in formulas
p_formula = (
    (data["cec"] * 0.35) + 
    (data["soc"] * 12) - 
    (data["phh2o"] * 2.1) + 
    (data["wv0010"] * 0.25) + 
    (data["Rainfall"] * 0.015)
)

k_formula = (
    (data["cec"] * 0.6) + 
    (data["clay"] * 1.8) - 
    (data["sand"] * 0.3) + 
    (data["wv0010"] * 0.4) - 
    (data["Elevation"] * 0.01)
)

# ✅ Add predictions to the dataset
data["Phosphorus"] = p_formula
data["Potassium"] = k_formula

# ✅ Save the results
output_file = "soil_with_p_k_estimates.csv"
data.to_csv(output_file, index=False)
print(f"\n✅ Results saved to {output_file}")
