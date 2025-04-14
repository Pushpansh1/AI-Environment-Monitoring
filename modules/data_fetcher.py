import os
import requests
import yaml
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables and configuration
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
WEATHER_ENDPOINT = config['api']['weather_endpoint']
AIR_QUALITY_ENDPOINT = config['api']['air_quality_endpoint']
FORECAST_ENDPOINT = config['api']['forecast_endpoint']
UNITS = config['api']['units']

def fetch_current_weather(lat, lon):
   
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': UNITS
        }
        
        response = requests.get(WEATHER_ENDPOINT, params=params)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

def fetch_air_quality(lat, lon):
   
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY
        }
        
        response = requests.get(AIR_QUALITY_ENDPOINT, params=params)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching air quality data: {e}")
        return None

def fetch_weather_forecast(lat, lon, cnt=40):
   
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': UNITS,
            'cnt': cnt
        }
        
        response = requests.get(FORECAST_ENDPOINT, params=params)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching forecast data: {e}")
        return None

def get_location_by_name(city_name):
    
    try:
        geocoding_endpoint = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': city_name,
            'limit': 1,
            'appid': API_KEY
        }
        
        response = requests.get(geocoding_endpoint, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            return data[0]['lat'], data[0]['lon']
        else:
            logger.warning(f"No location found for city: {city_name}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching location data: {e}")
        return None