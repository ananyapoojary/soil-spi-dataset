import requests
import csv
import time
import random
import itertools
import signal
import sys
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

# Define India's latitude and longitude range
LAT_MIN, LAT_MAX = 10.1, 10.5   # Latitude range
LON_MIN, LON_MAX = 68, 98      # Longitude range
RESOLUTION = 0.1                # Step size for grid

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

# API rate limit: 2 calls per minute
CALLS = 3
PER_MINUTE = 60

# List to store fetched data
soil_data = []

# Graceful shutdown handler
def save_and_exit(signum, frame):
    """Save the data to CSV and exit gracefully."""
    print("\n🛑 Termination signal received. Saving data before exit...")

    # Extract all field names dynamically
    all_fields = set()
    for entry in soil_data:
        all_fields.update(entry.keys())

    # Save data to CSV
    csv_filename = "soil_data_partial2.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_fields))
        writer.writeheader()
        writer.writerows(soil_data)

    print(f"\n✅ Partial data saved to {csv_filename}")
    sys.exit(0)

# Attach signal handlers
signal.signal(signal.SIGINT, save_and_exit)  # Handle Ctrl + C
signal.signal(signal.SIGTERM, save_and_exit) # Handle termination signals

@sleep_and_retry
@limits(calls=CALLS, period=PER_MINUTE)
def get_soil_properties(lat, lon, max_retries=3):
    """Fetch soil properties with retries."""
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depths=0-5cm&properties=phh2o,soc,bdod,clay,sand,silt,cec,ocd,nitrogen,wv0010,wv0033,wv1500,cfvo,ocs"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)

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
                wait_time = random.uniform(0, 10)  # Sleep randomly between 30-60s
                print(f"Rate limit hit. Sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            elif response.status_code >= 500:  # Server errors
                print(f"Server error ({response.status_code}). Retrying...")
                time.sleep(10)

        except requests.Timeout:
            print("Timeout, retrying...")
            time.sleep(5)

        except requests.RequestException as e:
            print(f"Request exception: {e}")
            time.sleep(5)

    return {"Latitude": lat, "Longitude": lon, "Error": "No data available"}

# Sequential fetching with automatic rate limiting
def fetch_with_rate_limit(coordinates):
    for lat, lon in tqdm(coordinates, desc="Fetching Soil Data"):
        result = get_soil_properties(lat, lon)
        soil_data.append(result)

    return soil_data

# Fetch data with automatic rate limiting
print("\n🚀 Starting rate-limited data fetching...")
soil_data = fetch_with_rate_limit(coordinates)

# Extract all field names dynamically
all_fields = set()
for entry in soil_data:
    all_fields.update(entry.keys())

# Save to CSV
csv_filename = "soil_data_10.1-10.5.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

print(f"\n✅ Soil data successfully saved to {csv_filename}")
