import time
import requests
import pandas as pd

# Define alternative APIs for soil and environmental data
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup"


def fetch_data(url, params, retries=3, backoff_factor=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait_time = backoff_factor ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"API error ({response.status_code}): {response.text}")
                break
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            break
    return None


def get_elevation(lat, lon):
    params = {"locations": f"{lat},{lon}"}
    result = fetch_data(ELEVATION_API, params)
    return result["results"][0]["elevation"] if result else "N/A"


def get_climate_data(lat, lon):
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
        "start": "20240101",
        "end": "20240101"
    }
    result = fetch_data(POWER_API, params)
    if result and "properties" in result:
        data = result["properties"]["parameter"]
        return {
            "Mean Temperature": data.get("T2M", {}).get("20240101", "N/A"),
            "Humidity": data.get("RH2M", {}).get("20240101", "N/A"),
            "Rainfall": data.get("PRECTOTCORR", {}).get("20240101", "N/A")
        }
    return {"Mean Temperature": "N/A", "Humidity": "N/A", "Rainfall": "N/A"}


# Sample coordinates in India
coordinates = [
    (12.91, 77.59),
    (15.5, 73.83),
    (13.08, 80.27),
    (19.07, 72.87),
    (22.57, 88.36)
]

dataset = []
for i, (lat, lon) in enumerate(coordinates, start=1):
    print(f"Processing {i}/{len(coordinates)}: Lat {lat}, Lon {lon}")
    elevation = get_elevation(lat, lon)
    climate_data = get_climate_data(lat, lon)
    dataset.append({
        "Latitude": lat,
        "Longitude": lon,
        "Elevation": elevation,
        **climate_data
    })

# Convert to DataFrame and save as CSV
output_file = "enhanced_india_soil_dataset4.csv"
df = pd.DataFrame(dataset)
df.to_csv(output_file, index=False)
print(f"Dataset saved as {output_file}")
