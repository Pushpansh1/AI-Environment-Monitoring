import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_aqi_description_epa(aqi):
    
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_weather_data(weather_data):
   
    if not weather_data:
        logger.warning("No weather data to process")
        return None
    
    try:
        processed_data = {
            'temperature': weather_data['main']['temp'],
            'feels_like': weather_data['main']['feels_like'],
            'temp_min': weather_data['main']['temp_min'],
            'temp_max': weather_data['main']['temp_max'],
            'pressure': weather_data['main']['pressure'],
            'humidity': weather_data['main']['humidity'],
            'wind_speed': weather_data['wind']['speed'],
            'wind_direction': weather_data['wind'].get('deg', 0),
            'clouds': weather_data['clouds']['all'],
            'weather_main': weather_data['weather'][0]['main'],
            'weather_description': weather_data['weather'][0]['description'],
            'weather_icon': weather_data['weather'][0]['icon'],
            'location_name': weather_data['name'],
            'country': weather_data['sys']['country'],
            'sunrise': datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(weather_data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add rain and snow data if available
        if 'rain' in weather_data:
            processed_data['rain_1h'] = weather_data['rain'].get('1h', 0)
        else:
            processed_data['rain_1h'] = 0
            
        if 'snow' in weather_data:
            processed_data['snow_1h'] = weather_data['snow'].get('1h', 0)
        else:
            processed_data['snow_1h'] = 0
            
        return processed_data
    
    except KeyError as e:
        logger.error(f"Missing key in weather data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing weather data: {e}")
        return None

def process_air_quality_data(air_quality_data):
    
    if not air_quality_data:
        logger.warning("No air quality data to process")
        return None
    
    try:
        from .aqi_calculator import calculate_aqi
        
        components = air_quality_data['list'][0]['components']
        
        # Convert CO from mg/m³ to μg/m³ for consistency
        co_concentration = components['co'] * 1000
        
        # Prepare pollutant concentrations for AQI calculation
        pollutants = {
            'pm2_5': components['pm2_5'],
            'pm10': components['pm10'],
            'o3': components['o3'],
            'co': co_concentration,
            'so2': components['so2'],
            'no2': components['no2']
        }
        
        # Calculate AQI using EPA formula
        aqi_result = calculate_aqi(pollutants)
        
        if aqi_result is None:
            logger.warning("Failed to calculate AQI using EPA formula, falling back to API value")
            aqi = air_quality_data['list'][0]['main']['aqi']
            aqi_description = {
                1: "Good",
                2: "Fair",
                3: "Moderate",
                4: "Poor",
                5: "Very Poor"
            }.get(aqi, "Unknown")
        else:
            aqi = aqi_result['aqi']
            aqi_description = get_aqi_description_epa(aqi)
        
        processed_data = {
            'aqi': aqi,
            'aqi_description': aqi_description,
            'co': components['co'],  # Carbon monoxide, mg/m3
            'no': components['no'],  # Nitrogen monoxide, μg/m3
            'no2': components['no2'],  # Nitrogen dioxide, μg/m3
            'o3': components['o3'],  # Ozone, μg/m3
            'so2': components['so2'],  # Sulphur dioxide, μg/m3
            'pm2_5': components['pm2_5'],  # Fine particles, μg/m3
            'pm10': components['pm10'],  # Coarse particles, μg/m3
            'nh3': components['nh3'],  # Ammonia, μg/m3
            'timestamp': datetime.fromtimestamp(air_quality_data['list'][0]['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add detailed AQI information if available
        if aqi_result is not None:
            processed_data.update({
                'dominant_pollutant': aqi_result['dominant_pollutant'],
                'pollutant_aqi': aqi_result['pollutant_aqi']
            })
        
        return processed_data
    
    except KeyError as e:
        logger.error(f"Missing key in air quality data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing air quality data: {e}")
        return None

def process_forecast_data(forecast_data):
    
    if not forecast_data:
        logger.warning("No forecast data to process")
        return None
    
    try:
        forecast_list = []
        
        for item in forecast_data['list']:
            forecast_item = {
                'timestamp': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'temp_min': item['main']['temp_min'],
                'temp_max': item['main']['temp_max'],
                'pressure': item['main']['pressure'],
                'humidity': item['main']['humidity'],
                'weather_main': item['weather'][0]['main'],
                'weather_description': item['weather'][0]['description'],
                'weather_icon': item['weather'][0]['icon'],
                'clouds': item['clouds']['all'],
                'wind_speed': item['wind']['speed'],
                'wind_direction': item['wind'].get('deg', 0),
                'probability': item.get('pop', 0) * 100  # Probability of precipitation (%)
            }
            
            # Add rain and snow data if available
            if 'rain' in item:
                forecast_item['rain_3h'] = item['rain'].get('3h', 0)
            else:
                forecast_item['rain_3h'] = 0
                
            if 'snow' in item:
                forecast_item['snow_3h'] = item['snow'].get('3h', 0)
            else:
                forecast_item['snow_3h'] = 0
                
            forecast_list.append(forecast_item)
        
        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(forecast_list)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except KeyError as e:
        logger.error(f"Missing key in forecast data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing forecast data: {e}")
        return None

def normalize_data(df, columns=None):
    
    if df is None or df.empty:
        logger.warning("No data to normalize")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        normalized_df = df.copy()
        
        # If no columns specified, normalize all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        # Normalize each column
        for col in columns:
            if col in normalized_df.columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0
        
        return normalized_df
    
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return df  # Return original DataFrame if normalization fails