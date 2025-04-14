import os
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def load_sample_data(data_type='weather'):
   
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        if data_type == 'weather':
            file_path = os.path.join(data_dir, 'historical_weather.csv')
        elif data_type == 'air_quality':
            file_path = os.path.join(data_dir, 'air_quality_samples.csv')
        elif data_type == 'flood':
            file_path = os.path.join(data_dir, 'flood_data.csv')
        else:
            logger.error(f"Unknown data type: {data_type}")
            return None
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Sample data file not found: {file_path}")
            return generate_sample_data(data_type)
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return generate_sample_data(data_type)

def generate_sample_data(data_type='weather'):
   
    try:
        # Create date range for the past 7 days with hourly data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        if data_type == 'weather':
            # Generate weather data
            np.random.seed(42)  # For reproducibility
            
            # Base temperature with daily and hourly variations
            base_temp = 25  # Base temperature in Celsius
            daily_variation = 5  # Daily temperature variation
            hourly_variation = 3  # Hourly temperature variation
            
            # Generate temperature with daily and hourly patterns
            hour_of_day = np.array([d.hour for d in date_range])
            day_of_year = np.array([d.dayofyear for d in date_range])
            
            # Temperature follows a sinusoidal pattern throughout the day
            hourly_pattern = np.sin(2 * np.pi * hour_of_day / 24) * hourly_variation
            
            # Add some daily variation
            daily_pattern = np.sin(2 * np.pi * day_of_year / 365) * daily_variation
            
            # Combine patterns with some random noise
            temperature = base_temp + hourly_pattern + daily_pattern + np.random.normal(0, 1, len(date_range))
            
            # Generate other weather variables
            humidity = 60 + 20 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 5, len(date_range))
            humidity = np.clip(humidity, 0, 100)  # Clip to valid range
            
            pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, len(date_range))
            
            wind_speed = 5 + 3 * np.sin(2 * np.pi * hour_of_day / 12) + np.random.normal(0, 1, len(date_range))
            wind_speed = np.clip(wind_speed, 0, 20)  # Clip to valid range
            
            # Create DataFrame
            df = pd.DataFrame({
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'clouds': np.random.randint(0, 100, len(date_range)),
                'rain_1h': np.random.exponential(0.5, len(date_range)) * (np.random.random(len(date_range)) > 0.8),
                'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain', 'Drizzle'], len(date_range))
            }, index=date_range)
            
        elif data_type == 'air_quality':
            # Generate air quality data
            np.random.seed(42)  # For reproducibility
            
            # AQI follows daily patterns with higher pollution during rush hours
            hour_of_day = np.array([d.hour for d in date_range])
            
            # Higher AQI during morning and evening rush hours
            rush_hour_pattern = np.maximum(0, 20 - 2 * np.abs(hour_of_day - 8)) + np.maximum(0, 20 - 2 * np.abs(hour_of_day - 18))
            
            # Base AQI with daily pattern and random variation
            aqi = 50 + rush_hour_pattern + np.random.normal(0, 10, len(date_range))
            aqi = np.clip(aqi, 0, 300)  # Clip to valid range
            
            # Generate other air quality components
            co = 500 + 200 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 50, len(date_range))
            no2 = 20 + 10 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 5, len(date_range))
            o3 = 30 + 15 * np.sin(2 * np.pi * (hour_of_day - 12) / 24) + np.random.normal(0, 5, len(date_range))
            pm25 = 15 + 10 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 3, len(date_range))
            
            # Create DataFrame
            df = pd.DataFrame({
                'aqi': aqi.astype(int),
                'co': co,
                'no2': no2,
                'o3': o3,
                'pm2_5': pm25,
                'pm10': pm25 * 1.5 + np.random.normal(0, 5, len(date_range)),
                'so2': 5 + np.random.normal(0, 2, len(date_range))
            }, index=date_range)
            
        elif data_type == 'flood':
            # Generate flood risk data
            np.random.seed(42)  # For reproducibility
            
            # Generate precipitation data
            precipitation = np.random.exponential(2, len(date_range)) * (np.random.random(len(date_range)) > 0.7)
            
            # Generate water level data (higher after precipitation)
            water_level = np.zeros(len(date_range))
            for i in range(1, len(date_range)):
                # Water level depends on previous level and current precipitation
                water_level[i] = max(0, water_level[i-1] * 0.95 + precipitation[i] * 0.5)
            
            # Generate flood risk based on water level and precipitation
            flood_risk = 0.1 + 0.6 * (water_level / water_level.max()) + 0.3 * (precipitation / precipitation.max())
            flood_risk = np.clip(flood_risk, 0, 1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'precipitation': precipitation,
                'water_level': water_level,
                'flood_risk': flood_risk,
                'is_flood': (flood_risk > 0.7).astype(int)
            }, index=date_range)
            
        else:
            logger.error(f"Unknown data type: {data_type}")
            return None
        
        # Save generated data
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        if data_type == 'weather':
            file_path = os.path.join(data_dir, 'historical_weather.csv')
        elif data_type == 'air_quality':
            file_path = os.path.join(data_dir, 'air_quality_samples.csv')
        elif data_type == 'flood':
            file_path = os.path.join(data_dir, 'flood_data.csv')
        
        # Save with timestamp as a column
        df_to_save = df.reset_index()
        df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
        df_to_save.to_csv(file_path, index=False)
        
        return df
    
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return None

def create_time_series_plot(data, columns, title, figsize=(12, 6)):
   
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        for col in columns:
            if col in data.columns:
                ax.plot(data.index, data[col], label=col)
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")
        return None

def create_correlation_heatmap(data, figsize=(10, 8)):
   
    try:
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        
        ax.set_title('Correlation Heatmap')
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return None

def format_weather_icon_url(icon_code):
   
    return f"http://openweathermap.org/img/wn/{icon_code}@2x.png"

def celsius_to_fahrenheit(celsius):
   
    return (celsius * 9/5) + 32

def get_aqi_description(aqi):
   
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"