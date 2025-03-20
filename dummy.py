import requests
import csv
import time
import random
import itertools
from tqdm import tqdm

# ✅ Define India's latitude and longitude range
LAT_MIN, LAT_MAX = 12.1, 12.2   # Latitude range
LON_MIN, LON_MAX = 76, 76.2     # Longitude range

RESOLUTION = 0.1  # Step size for grid

# ✅ Function to generate floating-point range values
def frange(start, stop, step):
    while start <= stop:
        yield round(start, 2)
        start += step

# ✅ Generate coordinate pairs
coordinates = list(itertools.product(
    [round(lat, 2) for lat in frange(LAT_MIN, LAT_MAX, RESOLUTION)],
    [round(lon, 2) for lon in frange(LON_MIN, LON_MAX, RESOLUTION)]
))

# ✅ Function to check if a coordinate is on land using OpenStreetMap API
def is_land(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    
    headers = {"User-Agent": "SoilDataApp"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Check if the coordinate contains a "country" field (indicating land)
            return "country" in data.get("address", {})
    except requests.RequestException:
        return False

    return False

# ✅ Function to fetch soil properties with retries
def get_soil_properties(lat, lon, max_retries=5):
    url = f"https://rest.isric.org/soilgrids/v2.1/properties/query?lon={lon}&lat={lat}&depths=0-5cm&properties=phh2o,soc,bdod,clay,sand,silt,cec,ocd,nitrogen,wv0010,wv0033,wv1500,cfvo,ocs"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = {"Latitude": lat, "Longitude": lon}

                # ✅ Properly extract soil properties
                properties = data.get("properties", {}).get("layers", [])
                
                for layer in properties:
                    property_name = layer.get("name", "Unknown")
                    
                    for depth in layer.get("depths", []):
                        mean_value = depth.get("values", {}).get("mean", "N/A")
                        results[f"{property_name}"] = mean_value

                return results

            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        except requests.Timeout:
            print(f"Timeout error for {lat}, {lon}. Retrying...")
            time.sleep(5)

        except requests.RequestException as e:
            print(f"Request failed for {lat}, {lon}: {e}")
            time.sleep(5)

    return {"Latitude": lat, "Longitude": lon, "Error": "No data available"}

# ✅ Fetch soil data only for land coordinates
soil_data = []
for lat, lon in tqdm(coordinates, desc="Fetching Soil Data", unit="location"):
    if is_land(lat, lon):  # 🌍 Only fetch soil data for land coordinates
        print(f"🌿 Land detected. Fetching soil data for {lat}, {lon}...")
        soil_data.append(get_soil_properties(lat, lon))
    else:
        print(f"🌊 Ocean detected. Skipping {lat}, {lon}...")

# ✅ Extract all field names dynamically
all_fields = set()
for entry in soil_data:
    all_fields.update(entry.keys())

# ✅ Save to CSV
csv_filename = "dummy.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

print(f"\n✅ Soil data for land coordinates successfully saved to {csv_filename}")
