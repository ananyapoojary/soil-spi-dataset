import pandas as pd
import numpy as np

# Load your dataset
file_path = 'merged_output - Sheet1.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# NPK estimation formulas with refined coefficients
def estimate_npk(row):
    # Nitrogen estimation
    N = (
        0.06 * row['soc'] +
        0.03 * row['ocd'] +
        0.12 * row['cec'] -
        0.015 * row['Rainfall'] +
        0.07 * row['clay'] -
        0.03 * row['phh2o'] +
        320  # Bias to fit the range
    )

    # Phosphorus estimation
    P = (
        0.04 * row['cec'] +
        0.05 * row['clay'] -
        0.025 * row['sand'] +
        0.02 * row['soc'] +
        0.03 * row['phh2o'] -
        0.015 * row['Rainfall'] +
        12  # Bias to fit the range
    )

    # Potassium estimation
    K = (
        0.05 * row['cec'] +
        0.07 * row['clay'] -
        0.03 * row['Rainfall'] +
        0.04 * row['phh2o'] +
        150  # Bias to fit the range
    )

    # Clamp values to the valid NPK ranges
    N = np.clip(N, 280, 560)
    P = np.clip(P, 10, 25)
    K = np.clip(K, 110, 280)

    return N, P, K

# Apply the function to each row
df[['N_approx', 'P_approx', 'K_approx']] = df.apply(lambda row: pd.Series(estimate_npk(row)), axis=1)

# Save the result to a new CSV
output_file = 'npk_estimation_clamped.csv'
df.to_csv(output_file, index=False)

print(f"Estimated NPK values saved to {output_file}")
