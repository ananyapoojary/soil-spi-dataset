
import requests
import csv
import time
import random
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define India's latitude and longitude range
LAT_MIN, LAT_MAX = 11.6, 12  # Latitude range
LON_MIN, LON_MAX = 68, 98     # Longitude range
RESOLUTION = 0.1               # Step size for grid

# API rate limit settings
MAX_CALLS_PER_MINUTE = 5
WAIT_TIME = 60 / MAX_CALLS_PER_MINUTE  # Sleep interval between requests
MAX_WORKERS = 5  # Number of parallel threads

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

# Function to fetch soil properties with retry and rate limit handling
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
                        results[property_name] = "N/A"

                return results

            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(1, 5)
                print(f"Rate limit hit. Sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        except requests.Timeout:
            print("Timeout, retrying...")
            time.sleep(5)

        except requests.RequestException as e:
            print(f"Request exception: {e}")
            time.sleep(5)

    return {"Latitude": lat, "Longitude": lon, "Error": "No data available"}

# Parallel processing with rate limiting
def parallel_fetch(coordinates, max_workers=MAX_WORKERS):
    soil_data = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_soil_properties, lat, lon): (lat, lon) for lat, lon in coordinates}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Soil Data"):
            result = future.result()
            soil_data.append(result)

            # Rate limiting mechanism
            elapsed_time = time.time() - start_time
            if len(soil_data) % MAX_CALLS_PER_MINUTE == 0:
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                start_time = time.time()

    return soil_data

# Fetch data in parallel with rate limiting
print("\n🚀 Starting parallel data fetching...")
soil_data = parallel_fetch(coordinates)

# Extract all field names dynamically
all_fields = set()
for entry in soil_data:
    all_fields.update(entry.keys())

# Save to CSV
csv_filename = "11.6-12.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

print(f"\n✅ Soil data successfully saved to {csv_filename}")
