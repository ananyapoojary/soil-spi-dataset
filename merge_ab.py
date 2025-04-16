import pandas as pd

# ✅ File paths
file1 = "/home/ananyapoojari/Downloads/preprocess/a/afr/merged_sorted.xlsx"  # File with Elevation, Temp, Humidity, Rainfall
file2 = "/home/ananyapoojari/Downloads/preprocess/b/afr/merged_sorted.xlsx"  # File with Soil properties
output_file = "/home/ananyapoojari/Downloads/preprocess/m/final.xlsx"  # Merged output file

# ✅ Load both Excel files into DataFrames
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# ✅ Merge based on Latitude and Longitude
merged_df = pd.merge(df1, df2, on=["Latitude", "Longitude"], how="inner")

# ✅ Save the merged dataframe to an Excel file
merged_df.to_excel(output_file, index=False)

print(f"\n✅ Merged Excel file saved at: {output_file}")
