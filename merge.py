import pandas as pd
import gdown

# Google Drive file links (Make sure they are set to "Anyone with the link can view")
file1_link = "https://drive.google.com/file/d/FILE_ID_1/view?usp=sharing"
file2_link = "https://drive.google.com/file/d/FILE_ID_2/view?usp=sharing"

# Extract File IDs from the Google Drive links
file1_id = file1_link.split("/d/")[1].split("/view")[0]
file2_id = file2_link.split("/d/")[1].split("/view")[0]

# Download files from Google Drive
file1_path = "file1.csv"
file2_path = "file2.csv"

gdown.download(f"https://drive.google.com/uc?id={file1_id}", file1_path, quiet=False)
gdown.download(f"https://drive.google.com/uc?id={file2_id}", file2_path, quiet=False)

# Read the CSV files into Pandas DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Perform the merge based on Latitude & Longitude
merged_df = pd.merge(df1, df2, on=["Latitude", "Longitude"], how="inner")

# Save the merged file
merged_df.to_csv("merged_output.csv", index=False)

print("Merge successful! File saved as merged_output.csv")
