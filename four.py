import requests
import csv
import time
import random
from tqdm import tqdm  # Progress bar

# Function to fetch multiple soil properties in a single API request
def get_soil_properties(lat, lon, max_retries=5):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depths=0-5cm&properties=phh2o,soc,bdod,clay,sand,silt"
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Fetching soil data for ({lat}, {lon}) - Attempt {attempt + 1}")
            response = requests.get(url, timeout=30)  # Increased timeout

            if response.status_code == 200:
                data = response.json()
                results = {"Latitude": lat, "Longitude": lon}

                # Extract soil properties dynamically
                layers = data.get("properties", {}).get("layers", [])
                for layer in layers:
                    property_name = layer.get("name", "Unknown")
                    depths = layer.get("depths", [])
                    
                    if depths and "values" in depths[0]:
                        results[property_name] = depths[0]["values"].get("mean", "N/A")
                    else:
                        results[property_name] = "N/A"

                print(f"✅ Data retrieved for ({lat}, {lon})\n")
                return results
            
            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f"⚠️ Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            else:
                print(f"❌ API request failed ({lat}, {lon}) with status {response.status_code}: {response.text}")
                return {"Latitude": lat, "Longitude": lon, "Error": f"API error {response.status_code}"}

        except requests.Timeout:
            print(f"⚠️ Timeout error ({lat}, {lon}) - Retrying...")
            time.sleep(5)  # Wait before retrying

        except requests.RequestException as e:
            print(f"⚠️ Network error ({lat}, {lon}): {e}")
            time.sleep(5)  # Short delay before retry

    print(f"❌ Max retries exceeded for ({lat}, {lon})\n")
    return {"Latitude": lat, "Longitude": lon, "Error": "Max retries exceeded"}

# Example locations (valid lat-lon values for India)
locations = [
    {"lat": 8.5, "lon": 77.2},  # Tamil Nadu
    {"lat": 22.3, "lon": 88.4},  # West Bengal
    {"lat": 6, "lon":68}   # Rajasthan
]

# Fetch data for all locations with a progress bar
soil_data = []
for loc in tqdm(locations, desc="Fetching Soil Data", unit="location"):
    soil_data.append(get_soil_properties(**loc))

# **Dynamically extract all field names from data**
all_fields = set(["Latitude", "Longitude", "Error"])  # Default fields
for entry in soil_data:
    all_fields.update(entry.keys())

# Save to CSV
csv_filename = "soil_data1.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

print(f"\n✅ Soil data successfully saved to {csv_filename}")
