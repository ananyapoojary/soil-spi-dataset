import pandas as pd
import os
from glob import glob

# ✅ Define the desired column order
column_order = ["Latitude", "Longitude", "Elevation", "Mean Temperature", "Humidity", "Rainfall"]

# ✅ Directory containing the Excel files
input_folder = "/home/ananyapoojari/Downloads/preprocess/a/bfr"   # Replace with your folder path
output_folder = "/home/ananyapoojari/Downloads/preprocess/a/afr"        # Replace with your output folder path
output_file = os.path.join(output_folder, "merged_sorted.xlsx")

# ✅ Get all Excel files
excel_files = glob(os.path.join(input_folder, "*.xlsx"))

# ✅ Initialize empty list to store dataframes
dataframes = []

# ✅ Process each file
for file in excel_files:
    print(f"Processing {file}...")
    
    # Load Excel file
    df = pd.read_excel(file)

    # Ensure consistent column order
    df = df.reindex(columns=column_order)

    # Remove rows with any missing value
    df = df.dropna()

    # Append cleaned dataframe to list
    dataframes.append(df)

# ✅ Merge all dataframes into one
merged_df = pd.concat(dataframes, ignore_index=True)

# ✅ Sort by Latitude and Longitude
merged_df = merged_df.sort_values(by=["Latitude", "Longitude"], ascending=[True, True])

# ✅ Save to a single Excel file
merged_df.to_excel(output_file, index=False)

print(f"\n✅ Merged and sorted Excel file saved at: {output_file}")
