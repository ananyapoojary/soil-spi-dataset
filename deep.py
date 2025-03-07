import requests
import csv

# Function to fetch multiple soil properties in a single API request
def get_soil_properties(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depths=0-5cm"

    response = requests.get(url)
    results = {"Latitude": lat, "Longitude": lon}

    if response.status_code == 200:
        try:
            data = response.json()
            layers = data.get("properties", {}).get("layers", [])
            
            for layer in layers:
                property_name = layer.get("name", "Unknown")
                depths = layer.get("depths", [])
                
                if depths:
                    results[property_name] = depths[0].get("values", {}).get("mean", "N/A")
                else:
                    results[property_name] = "N/A"
        except Exception as e:
            results["Error"] = f"Parsing error: {str(e)}"
    else:
        results["Error"] = f"API request failed with status code {response.status_code}"

    return results

# Example locations
locations = [
    {"lat": 6.5, "lon": -69.2},
    {"lat": 39.1, "lon": -67.3},
    {"lat": 20.5, "lon": 78.9}  # Add more locations if needed
]

# Fetch data for all locations
soil_data = [get_soil_properties(**loc) for loc in locations]

# Dynamically determine all field names from the collected data
all_fieldnames = set()
for entry in soil_data:
    all_fieldnames.update(entry.keys())
all_fieldnames = sorted(all_fieldnames)  # Sorting for consistency

# CSV file path
csv_filename = "soil_data.csv"

# Writing to CSV
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=all_fieldnames)

    writer.writeheader()
    writer.writerows(soil_data)

print(f"Soil data has been saved to {csv_filename}")
