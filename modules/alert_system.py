import os
import yaml
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class AlertSystem:
   
    
    def __init__(self):
       
        self.thresholds = config['thresholds']
        self.alerts = []
    
    def check_temperature_alert(self, temperature):
       
        try:
            high_threshold = self.thresholds['temperature']['high']
            low_threshold = self.thresholds['temperature']['low']
            
            if temperature >= high_threshold:
                return {
                    'type': 'temperature',
                    'level': 'warning',
                    'message': f'High temperature alert: {temperature}°C exceeds threshold of {high_threshold}°C',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            elif temperature <= low_threshold:
                return {
                    'type': 'temperature',
                    'level': 'warning',
                    'message': f'Low temperature alert: {temperature}°C below threshold of {low_threshold}°C',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking temperature alert: {e}")
            return None
    
    def check_humidity_alert(self, humidity):
       
        try:
            high_threshold = self.thresholds['humidity']['high']
            low_threshold = self.thresholds['humidity']['low']
            
            if humidity >= high_threshold:
                return {
                    'type': 'humidity',
                    'level': 'warning',
                    'message': f'High humidity alert: {humidity}% exceeds threshold of {high_threshold}%',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            elif humidity <= low_threshold:
                return {
                    'type': 'humidity',
                    'level': 'warning',
                    'message': f'Low humidity alert: {humidity}% below threshold of {low_threshold}%',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking humidity alert: {e}")
            return None
    
    def check_air_quality_alert(self, aqi):
       
        try:
            poor_threshold = self.thresholds['air_quality']['poor']
            
            if aqi >= poor_threshold:
                return {
                    'type': 'air_quality',
                    'level': 'danger',
                    'message': f'Poor air quality alert: AQI {aqi} exceeds threshold of {poor_threshold}',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking air quality alert: {e}")
            return None
    
    def check_wind_speed_alert(self, wind_speed):
        
        try:
            high_threshold = self.thresholds['wind_speed']['high']
            
            if wind_speed >= high_threshold:
                return {
                    'type': 'wind_speed',
                    'level': 'warning',
                    'message': f'High wind speed alert: {wind_speed} m/s exceeds threshold of {high_threshold} m/s',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking wind speed alert: {e}")
            return None
    
    def check_precipitation_alert(self, precipitation):
       
        try:
            heavy_threshold = self.thresholds['precipitation']['heavy']
            
            if precipitation >= heavy_threshold:
                return {
                    'type': 'precipitation',
                    'level': 'warning',
                    'message': f'Heavy precipitation alert: {precipitation} mm/h exceeds threshold of {heavy_threshold} mm/h',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking precipitation alert: {e}")
            return None
    
    def check_disaster_risk_alert(self, risk_data):
       
        try:
            if risk_data['risk_level'] == 'High':
                return {
                    'type': 'disaster_risk',
                    'level': 'danger',
                    'message': f'High risk of {risk_data["highest_risk"]} detected with probability {risk_data["highest_probability"]:.2f}',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking disaster risk alert: {e}")
            return None
    
    def generate_alerts(self, weather_data, air_quality_data, risk_data=None):
       
        try:
            # Clear previous alerts
            self.alerts = []
            
            # Check temperature alert
            if 'temperature' in weather_data:
                temp_alert = self.check_temperature_alert(weather_data['temperature'])
                if temp_alert:
                    self.alerts.append(temp_alert)
            
            # Check humidity alert
            if 'humidity' in weather_data:
                humidity_alert = self.check_humidity_alert(weather_data['humidity'])
                if humidity_alert:
                    self.alerts.append(humidity_alert)
            
            # Check wind speed alert
            if 'wind_speed' in weather_data:
                wind_alert = self.check_wind_speed_alert(weather_data['wind_speed'])
                if wind_alert:
                    self.alerts.append(wind_alert)
            
            # Check precipitation alert
            if 'rain_1h' in weather_data:
                precip_alert = self.check_precipitation_alert(weather_data['rain_1h'])
                if precip_alert:
                    self.alerts.append(precip_alert)
            
            # Check air quality alert
            if 'aqi' in air_quality_data:
                aqi_alert = self.check_air_quality_alert(air_quality_data['aqi'])
                if aqi_alert:
                    self.alerts.append(aqi_alert)
            
            # Check disaster risk alert
            if risk_data:
                risk_alert = self.check_disaster_risk_alert(risk_data)
                if risk_alert:
                    self.alerts.append(risk_alert)
            
            return self.alerts
        
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def get_active_alerts(self):
       
        return self.alerts