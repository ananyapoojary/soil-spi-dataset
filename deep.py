import requests
import time
import csv

# Function to fetch soil data with retries
def fetch_soil_data(lat, lon, retries=5, timeout=60):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depths=0-5cm"
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} for {lat}, {lon}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error fetching data for {lat}, {lon}: {e}")
        time.sleep(3 ** attempt)  # Exponential backoff with increased delay
        attempt += 1
    return None  # Return None if all attempts fail

# Function to extract soil properties
def get_soil_properties(lat, lon):
    soil_data = fetch_soil_data(lat, lon)
    results = {"Latitude": lat, "Longitude": lon}

    if soil_data:
        layers = soil_data.get("properties", {}).get("layers", [])
        for layer in layers:
            property_name = layer.get("name", "Unknown")
            depths = layer.get("depths", [])
            if depths:
                results[property_name] = depths[0].get("values", {}).get("mean", "N/A")
            else:
                results[property_name] = "N/A"
    else:
        results["Error"] = "Failed to retrieve data"

    return results

# Sample dataset (Replace with actual list of coordinates)
coordinates = [
    (12.91, 77.59),
    (15.5, 73.83),
    (13.08, 80.27),
    (19.07, 72.87),
    (22.57, 88.36),
]

# Fetch data for all coordinates
soil_data = [get_soil_properties(lat, lon) for lat, lon in coordinates]

# Dynamically determine all field names
all_fieldnames = set()
for entry in soil_data:
    all_fieldnames.update(entry.keys())
all_fieldnames = sorted(all_fieldnames)  # Sorting for consistency

# Save to CSV
csv_filename = "combined_soil_data.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=all_fieldnames)
    writer.writeheader()
    writer.writerows(soil_data)

print(f"Soil data has been saved to {csv_filename}")
