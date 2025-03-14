import requests
import csv
import time
import random
from tqdm import tqdm

def get_soil_properties(lat, lon, max_retries=5):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depths=0-5cm&properties=phh2o,soc,bdod,clay,sand,silt,cec,ocd,nitrogen,wv0010,wv0033,wv1500,cfvo,ocs"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = {"Latitude": lat, "Longitude": lon}

                # Extract soil properties dynamically
                layers = data.get("properties", {}).get("layers", [])
                for layer in layers:
                    property_name = layer.get("name", "Unknown")
                    depths = layer.get("depths", [])

                    # Ensure depths and values exist
                    if depths and "values" in depths[0] and "mean" in depths[0]["values"]:
                        results[property_name] = depths[0]["values"]["mean"]
                    else:
                        results[property_name] = "N/A"  # Set "N/A" explicitly if data is missing

                return results

            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)

        except requests.Timeout:
            time.sleep(5)

        except requests.RequestException:
            time.sleep(5)

    return {"Latitude": lat, "Longitude": lon, "Error": "No data available"}

# Example locations
locations = [
    {"lat": 6, "lon": 68},
    {"lat": 9.8, "lon": 77.5},
    {"lat": 6.9, "lon": 68.2},
    {"lat": 6.2, "lon": 69.3},
    {"lat": 7.3, "lon": 69.4},
]

# Fetch data
soil_data = []
for loc in tqdm(locations, desc="Fetching Soil Data", unit="location"):
    soil_data.append(get_soil_properties(**loc))

# Extract all field names dynamically
all_fields = set()
for entry in soil_data:
    all_fields.update(entry.keys())

# Save to CSV
csv_filename = "soil_data7.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

print(f"\n✅ Soil data successfully saved to {csv_filename}")
