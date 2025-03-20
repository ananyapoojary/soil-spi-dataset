import time
import requests
import pandas as pd
import itertools

# Define APIs
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup"

# Define India's latitude and longitude range
LAT_MIN, LAT_MAX = 8, 14  # India's latitude range
LON_MIN, LON_MAX = 68, 98 # India's longitude range
RESOLUTION = 0.1  # Step size for grid

# Define floating-point range generator function
def frange(start, stop, step):
    """Generate floating point range values."""
    while start <= stop:
        yield round(start, 2)
        start += step

# Generate coordinate pairs
coordinates = list(itertools.product(
    [round(lat, 2) for lat in frange(LAT_MIN, LAT_MAX, RESOLUTION)],
    [round(lon, 2) for lon in frange(LON_MIN, LON_MAX, RESOLUTION)]
))

def fetch_data(url, params, retries=3, backoff_factor=2):
    """Fetch API data with retries."""
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
    """Fetch elevation data."""
    params = {"locations": f"{lat},{lon}"}
    result = fetch_data(ELEVATION_API, params)
    return result["results"][0]["elevation"] if result else "N/A"

def get_climate_data(lat, lon):
    """Fetch climate data."""
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

# Generate dataset
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

# Save dataset as CSV
output_file = "final part 1.csv"
df = pd.DataFrame(dataset)
df.to_csv(output_file, index=False)
print(f"✅ Dataset saved as {output_file}")
