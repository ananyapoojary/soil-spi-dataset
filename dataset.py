import requests
import pandas as pd

# Example dataset with lon, lat values
data = {
    'lon': [12.34, 56.78],
    'lat': [98.76, 54.32]
}
df = pd.DataFrame(data)

# Function to get NPK values from SoilGrids API
def get_npk(lon, lat):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=nitrogen,phosphorus,potassium"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        nitrogen = data.get("properties", {}).get("nitrogen", {}).get("mean", None)
        phosphorus = data.get("properties", {}).get("phosphorus", {}).get("mean", None)
        potassium = data.get("properties", {}).get("potassium", {}).get("mean", None)
        return nitrogen, phosphorus, potassium
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NPK data for lon={lon}, lat={lat}: {e}")
        return None, None, None

# Function to get Elevation using Open-Elevation API
def get_elevation(lon, lat):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        elevation_data = response.json()
        return elevation_data['results'][0]['elevation']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elevation data for lon={lon}, lat={lat}: {e}")
        return None

# Function to get Temperature and Humidity using OpenWeatherMap API
def get_weather(lon, lat, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        temperature = weather_data.get('main', {}).get('temp', None)
        humidity = weather_data.get('main', {}).get('humidity', None)
        return temperature, humidity
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data for lon={lon}, lat={lat}: {e}")
        return None, None

# Add your OpenWeatherMap API key here
OPENWEATHERMAP_API_KEY = "your_actual_api_key"  # Replace with a valid API key

# Apply functions to each row in the dataframe
df[['Nitrogen', 'Phosphorus', 'Potassium']] = df.apply(lambda row: pd.Series(get_npk(row['lon'], row['lat'])), axis=1)
df['Elevation'] = df.apply(lambda row: get_elevation(row['lon'], row['lat']), axis=1)
df[['Temperature', 'Humidity']] = df.apply(lambda row: pd.Series(get_weather(row['lon'], row['lat'], OPENWEATHERMAP_API_KEY)), axis=1)

# Save the dataframe to a CSV file
df.to_csv("soil_weather_data.csv", index=False)

# Print the final dataframe
print(df)
