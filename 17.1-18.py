import requests
import csv
import time
import random
import itertools
from tqdm import tqdm

# Define India's latitude and longitude range
LAT_MIN, LAT_MAX = 17, 18    # Latitude range
LON_MIN, LON_MAX = 68, 98      # Longitude range
RESOLUTION = 0.1                # Step size for grid

# API rate limit settings
MAX_CALLS_PER_MINUTE = 5
WAIT_TIME = 60 / MAX_CALLS_PER_MINUTE  # Sleep interval between requests

# Initialize counters
successful_fetches = 0
empty_cells = 0
no_data = 0

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

# Function to fetch soil properties with retries and proper rate limiting
def get_soil_properties(lat, lon, max_retries=5):
    global successful_fetches, empty_cells, no_data

    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depths=0-5cm&properties=phh2o,soc,bdod,clay,sand,silt,cec,ocd,nitrogen,wv0010,wv0033,wv1500,cfvo,ocs"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = {"Latitude": lat, "Longitude": lon}

                # Extract soil properties dynamically
                layers = data.get("properties", {}).get("layers", [])
                empty_count = 0  # Track empty cells for this request

                for layer in layers:
                    property_name = layer.get("name", "Unknown")
                    depths = layer.get("depths", [])

                    if depths and "values" in depths[0] and "mean" in depths[0]["values"]:
                        value = depths[0]["values"]["mean"]
                        results[property_name] = value

                        # Check for empty cells
                        if value == "N/A":
                            empty_count += 1
                    else:
                        results[property_name] = "N/A"
                        empty_count += 1

                # Count successful fetch and empty cells
                successful_fetches += 1
                empty_cells += empty_count
                return results

            elif response.status_code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(1, 5)
                print(f"Rate limit hit. Sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        except requests.Timeout:
            print(f"Timeout on {lat}, {lon}. Retrying... ({attempt + 1}/{max_retries})")
            time.sleep(5)

        except requests.RequestException as e:
            print(f"Request exception: {e}")
            time.sleep(5)

    # Increment "No data available" counter on failure
    no_data += 1
    return {"Latitude": lat, "Longitude": lon, "Error": "No data available"}

# Sequential fetching with rate limiting
def fetch_with_rate_limit(coordinates):
    soil_data = []
    failed_requests = []
    start_time = time.time()

    for i, (lat, lon) in enumerate(tqdm(coordinates, desc="Fetching Soil Data"), start=1):
        result = get_soil_properties(lat, lon)
        soil_data.append(result)

        if "Error" in result:
            failed_requests.append(result)

        # Enforce the rate limit
        if i % MAX_CALLS_PER_MINUTE == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            start_time = time.time()

    return soil_data, failed_requests

# Fetch data with proper rate limiting
print("\n🚀 Starting rate-limited data fetching...")
soil_data, failed_requests = fetch_with_rate_limit(coordinates)

# Extract all field names dynamically
all_fields = set()
for entry in soil_data:
    all_fields.update(entry.keys())

# Save successful data to CSV
csv_filename = "11.1-12_with_stats.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_fields))
    writer.writeheader()
    writer.writerows(soil_data)

# Save failed requests to a separate CSV for reprocessing
if failed_requests:
    failed_filename = "failed_requests.csv"
    with open(failed_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(failed_requests[0].keys()))
        writer.writeheader()
        writer.writerows(failed_requests)
    print(f"\n⚠️ {len(failed_requests)} failed requests saved to {failed_filename} for reprocessing.")

# Display summary stats
print("\n📊 **Summary Stats:**")
print(f"✅ Successful Fetches: {successful_fetches}")
print(f"⚠️ Empty Cells: {empty_cells}")
print(f"❌ No Data Available: {no_data}")
print(f"\n✅ Soil data successfully saved to {csv_filename}")
