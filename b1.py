import requests
import csv
import time
import random
from tqdm import tqdm

# List of specific coordinates
coordinates = [
(8, 79.9), (8, 80), (8, 80.1), (8, 80.2), (8, 80.3), (8, 80.4), (8, 80.5), (8, 80.6), (8, 80.7), (8, 80.8),  
(8, 80.9), (8, 81), (8, 81.1), (8, 81.2), (8, 81.3), (8, 81.4), (8, 81.5), (8, 93.4), (8, 93.5),  
(8.1, 77.5), (8.1, 79.9), (8.1, 80), (8.1, 80.1), (8.1, 80.2), (8.1, 80.3), (8.1, 80.4), (8.1, 80.5), (8.1, 80.6), (8.1, 80.7),  
(8.1, 80.8), (8.1, 80.9), (8.1, 81), (8.1, 81.1), (8.1, 81.2), (8.1, 81.3), (8.1, 81.4), (8.1, 93.5),  
(8.2, 77.3), (8.2, 77.4), (8.2, 77.5), (8.2, 77.6), (8.2, 77.7), (8.2, 79.7), (8.2, 79.9), (8.2, 80), (8.2, 80.1), (8.2, 80.2),  
(8.2, 80.3), (8.2, 80.4), (8.2, 80.5), (8.2, 80.6), (8.2, 80.7), (8.2, 80.8), (8.2, 80.9), (8.2, 81), (8.2, 81.1), (8.2, 81.2),  
(8.2, 81.3), (8.2, 81.4), (8.2, 93.2), (8.2, 93.5)

]

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

                    if depths and "values" in depths[0] and "mean" in depths[0]["values"]:
                        results[property_name] = depths[0]["values"]["mean"]
                    else:
                        results[property_name] = "N/A"  

                return results

            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)

        except requests.Timeout:
            time.sleep(5)

        except requests.RequestException:
            time.sleep(5)

    return {"Latitude": lat, "Longitude": lon, "Error": "No data available"}

# Fetch data
soil_data = []
for lat, lon in tqdm(coordinates, desc="Fetching Soil Data", unit="location"):
    print(f"Fetching data for Latitude: {lat}, Longitude: {lon}")
    soil_data.append(get_soil_properties(lat, lon))

# Extract all field names dynamically
all_fields = set()
for entry in soil_data:
    all_fields.update(entry.keys())

# Save to CSV
csv_filename = "b1.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

print(f"\n✅ Soil data successfully saved to {csv_filename}")
