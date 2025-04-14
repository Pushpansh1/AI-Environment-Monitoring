import os
import numpy as np
import pandas as pd
import logging
import yaml
from datetime import datetime, timedelta
from .model_trainer import load_model
from .aqi_calculator import calculate_aqi
from .feature_engineer import (
    extract_time_features,
    create_lag_features,
    create_rolling_features,
    create_weather_condition_features,
    create_temperature_features,
    create_air_quality_features
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class PredictionEngine:
    
    
    def __init__(self):
        
        self.models = {}
        self.load_models()
    
    def load_models(self):
        
        # Check if models directory exists
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models with None values
        self.models['temperature_forecast'] = None
        self.models['disaster_risk'] = None
        self.models['air_quality_forecast'] = None
        
        # Check if model files exist before loading
        temp_model_path = os.path.join(models_dir, 'temperature_forecast_model.h5')
        disaster_model_path = os.path.join(models_dir, 'disaster_risk_model.pkl')
        air_quality_model_path = os.path.join(models_dir, 'air_quality_forecast_model.pkl')
        
        try:
            # Load temperature forecast model (LSTM) if exists
            if os.path.exists(temp_model_path):
                self.models['temperature_forecast'] = load_model('temperature_forecast_model', model_type='keras')
                logger.info("Temperature forecast model loaded successfully")
            else:
                logger.warning("Temperature forecast model file not found. Using fallback predictions.")
            
            # Load disaster risk model if exists
            if os.path.exists(disaster_model_path):
                self.models['disaster_risk'] = load_model('disaster_risk_model', model_type='sklearn')
                logger.info("Disaster risk model loaded successfully")
            else:
                logger.warning("Disaster risk model file not found. Using fallback predictions.")
            
            # Load air quality prediction model if exists
            if os.path.exists(air_quality_model_path):
                self.models['air_quality_forecast'] = load_model('air_quality_forecast_model', model_type='sklearn')
                logger.info("Air quality forecast model loaded successfully")
            else:
                logger.warning("Air quality forecast model file not found. Using fallback predictions.")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def prepare_data_for_prediction(self, data, model_type):
       
        try:
            if model_type == 'temperature_forecast':
                # For LSTM model
                lookback = config['models']['temperature_forecast']['lookback']
                
                # Extract features
                data = extract_time_features(data)
                data = create_temperature_features(data)
                
                # Select relevant columns for temperature prediction
                relevant_cols = [
                    'temperature', 'humidity', 'pressure', 'wind_speed',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'temp_range', 'temp_change'
                ]
                
                # Filter columns that exist in the data
                available_cols = [col for col in relevant_cols if col in data.columns]
                
                # Prepare data for LSTM (reshape to [samples, time_steps, features])
                X = data[available_cols].values
                X = X.reshape(1, X.shape[0], len(available_cols))
                
                return X
            
            elif model_type == 'disaster_risk':
                # For disaster risk model (Random Forest or Gradient Boosting)
                
                # Extract features
                data = extract_time_features(data)
                data = create_weather_condition_features(data)
                data = create_temperature_features(data)
                
                # Create lag features for relevant columns
                lag_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
                available_lag_cols = [col for col in lag_columns if col in data.columns]
                
                if available_lag_cols:
                    data = create_lag_features(data, available_lag_cols)
                
                # Create rolling features for relevant columns
                if available_lag_cols:
                    data = create_rolling_features(data, available_lag_cols)
                
                # Drop non-numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                
                return numeric_data
                
            elif model_type == 'air_quality_forecast':
                # For air quality prediction model
                
                # Extract features
                data = extract_time_features(data)
                
                # Create air quality specific features if function exists
                try:
                    data = create_air_quality_features(data)
                except Exception as e:
                    logger.warning(f"Could not create air quality features: {e}")
                
                # Create lag features for relevant air quality columns
                air_quality_columns = ['pm2_5', 'pm10', 'o3', 'co', 'so2', 'no2']
                available_aq_cols = [col for col in air_quality_columns if col in data.columns]
                
                if available_aq_cols:
                    data = create_lag_features(data, available_aq_cols)
                    data = create_rolling_features(data, available_aq_cols)
                
                # Include weather features as they affect air quality
                weather_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
                available_weather_cols = [col for col in weather_columns if col in data.columns]
                
                if available_weather_cols:
                    data = create_lag_features(data, available_weather_cols)
                
                # Drop non-numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                
                return numeric_data
            
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {e}")
            return None
    
    def predict_temperature(self, data, forecast_hours=24):
       
        try:
            model = self.models.get('temperature_forecast')
            
            if model is None:
                logger.warning("Temperature forecast model not loaded, using fallback prediction")
                # Fallback prediction: use average of recent temperatures with small random variations
                results = []
                last_timestamp = data.index[-1]
                
                # Get recent average temperature if available
                if 'temperature' in data.columns and not data['temperature'].empty:
                    avg_temp = data['temperature'].mean()
                else:
                    avg_temp = 25.0  # Default fallback value
                
                # Generate simple forecast with small random variations
                for i in range(forecast_hours):
                    next_timestamp = last_timestamp + timedelta(hours=i+1)
                    # Add small random variation (-1 to +1 degrees)
                    variation = np.random.uniform(-1, 1)
                    results.append({
                        'timestamp': next_timestamp,
                        'predicted_temperature': avg_temp + variation
                    })
                
                return pd.DataFrame(results).set_index('timestamp')
            
            # Initialize results
            results = []
            current_data = data.copy()
            
            # Make predictions for each hour in the forecast horizon
            for i in range(forecast_hours):
                # Prepare data for prediction
                X = self.prepare_data_for_prediction(current_data, 'temperature_forecast')
                
                # Make prediction
                pred = model.predict(X)[0][0]
                
                # Create timestamp for this prediction
                last_timestamp = current_data.index[-1]
                next_timestamp = last_timestamp + timedelta(hours=1)
                
                # Store result
                results.append({
                    'timestamp': next_timestamp,
                    'predicted_temperature': pred
                })
                
                # Update current_data for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row['temperature'] = pred
                new_row.name = next_timestamp
                
                current_data = pd.concat([current_data, pd.DataFrame([new_row])])
            
            # Create DataFrame from results
            forecast_df = pd.DataFrame(results)
            forecast_df.set_index('timestamp', inplace=True)
            
            return forecast_df
        
        except Exception as e:
            logger.error(f"Error predicting temperature: {e}")
            return None
    
    def predict_disaster_risk(self, data):
       
        try:
            model = self.models.get('disaster_risk')
            
            if model is None:
                logger.warning("Disaster risk model not loaded, using fallback prediction")
                # Fallback prediction: use moderate risk with slight variations based on weather conditions
                
                # Default moderate risk score
                risk_score = 0.5
                
                # Adjust risk based on weather conditions if available
                if 'temperature' in data.columns and not data['temperature'].empty:
                    # Increase risk for extreme temperatures
                    avg_temp = data['temperature'].mean()
                    if avg_temp > 35 or avg_temp < 0:
                        risk_score += 0.1
                
                if 'humidity' in data.columns and not data['humidity'].empty:
                    # Increase risk for high humidity (flood risk)
                    avg_humidity = data['humidity'].mean()
                    if avg_humidity > 80:
                        risk_score += 0.1
                
                if 'wind_speed' in data.columns and not data['wind_speed'].empty:
                    # Increase risk for high winds
                    avg_wind = data['wind_speed'].mean()
                    if avg_wind > 20:
                        risk_score += 0.1
                
                # Add small random variation
                risk_score += np.random.uniform(-0.05, 0.05)
                # Ensure risk score is between 0 and 1
                risk_score = max(0, min(1, risk_score))
                
                # Map risk score to category
                risk_category = self._map_risk_score_to_category(risk_score)
                
                return {
                    'risk_score': float(risk_score),
                    'risk_category': risk_category,
                    'note': 'Fallback prediction (model not available)'
                }
            
            # Prepare data for prediction
            X = self.prepare_data_for_prediction(data, 'disaster_risk')
            
            if X is None or X.empty:
                logger.error("Failed to prepare data for disaster risk prediction")
                return None
            
            # Make prediction
            risk_prob = model.predict_proba(X)[-1]
            
            # Get class names
            class_names = model.classes_
            
            # Create dictionary of risks
            risks = {class_name: prob for class_name, prob in zip(class_names, risk_prob)}
            
            # Add overall risk level based on threshold
            threshold = config['models']['disaster_risk']['threshold']
            max_risk = max(risks.items(), key=lambda x: x[1])
            
            risks['highest_risk'] = max_risk[0]
            risks['highest_probability'] = max_risk[1]
            risks['risk_level'] = 'High' if max_risk[1] >= threshold else 'Low'
            
            return risks
        
        except Exception as e:
            logger.error(f"Error predicting disaster risk: {e}")
            return None
            
    def predict_air_quality(self, data, forecast_hours=24):
       
        try:
            model = self.load_model('air_quality_forecast', 'sklearn')
            
            if model is None:
                logger.warning("Air quality forecast model not loaded, using fallback prediction")
                # Fallback prediction: use moderate AQI with variations based on time of day
                results = []
                last_timestamp = data.index[-1]
                
                # Default moderate AQI values
                default_values = {
                    'pm2_5': 12.0,  # Moderate level
                    'pm10': 50.0,  # Moderate level
                    'o3': 0.07,    # Moderate level
                    'co': 4.0,     # Good level
                    'so2': 0.03,   # Good level
                    'no2': 0.05    # Good level
                }
                
                # Generate simple forecast with variations
                for i in range(forecast_hours):
                    next_timestamp = last_timestamp + timedelta(hours=i+1)
                    hour_of_day = next_timestamp.hour
                    
                    # Simulate daily patterns (worse air quality during rush hours)
                    time_factor = 1.0
                    if hour_of_day in [7, 8, 9, 16, 17, 18]:  # Rush hours
                        time_factor = 1.2
                    elif hour_of_day in [1, 2, 3, 4]:  # Early morning
                        time_factor = 0.8
                    
                    # Create prediction with variations
                    prediction = {}
                    for pollutant, base_value in default_values.items():
                        # Add random variation (-10% to +10%)
                        variation = np.random.uniform(-0.1, 0.1)
                        prediction[pollutant] = base_value * time_factor * (1 + variation)
                    
                    # Calculate AQI
                    aqi_result = calculate_aqi(prediction)
                    
                    # Handle the case where calculate_aqi returns None or doesn't have expected structure
                    if aqi_result is None or not isinstance(aqi_result, dict):
                        prediction['aqi'] = 50  # Default moderate AQI
                        prediction['dominant_pollutant'] = 'Unknown'
                    else:
                        prediction['aqi'] = aqi_result.get('aqi', 50)
                        prediction['dominant_pollutant'] = aqi_result.get('dominant_pollutant', 'Unknown')
                        
                    prediction['timestamp'] = next_timestamp
                    
                    results.append(prediction)
                
                return pd.DataFrame(results).set_index('timestamp')
            
            # If we have a trained model, use it for prediction
            # Initialize results
            results = []
            current_data = data.copy()
            
            # Make predictions for each hour in the forecast horizon
            for i in range(forecast_hours):
                # Prepare data for prediction
                X = self.prepare_data_for_prediction(current_data, 'air_quality_forecast')
                
                if X is None or X.empty:
                    logger.error("Failed to prepare data for air quality prediction")
                    return None
                
                # Make prediction for each pollutant
                pollutant_predictions = {}
                
                for pollutant in ['pm2_5', 'pm10', 'o3', 'co', 'so2', 'no2']:
                    if f'{pollutant}_model' in self.models:
                        pollutant_model = self.models[f'{pollutant}_model']
                        pred = pollutant_model.predict(X)[-1]
                        pollutant_predictions[pollutant] = pred
                
                # Calculate AQI using EPA standards
                aqi_result = calculate_aqi(pollutant_predictions)
                
                if aqi_result is None:
                    logger.error("Failed to calculate AQI from predictions")
                    continue
                
                # Create timestamp for this prediction
                last_timestamp = current_data.index[-1]
                next_timestamp = last_timestamp + timedelta(hours=1)
                
                # Store result
                results.append({
                    'timestamp': next_timestamp,
                    'predicted_aqi': aqi_result['aqi'],
                    'dominant_pollutant': aqi_result['dominant_pollutant'],
                    'pollutant_predictions': pollutant_predictions
                })
                
                # Update current_data for next iteration
                new_row = current_data.iloc[-1].copy()
                for pollutant, value in pollutant_predictions.items():
                    if pollutant in new_row:
                        new_row[pollutant] = value
                new_row.name = next_timestamp
                
                current_data = pd.concat([current_data, pd.DataFrame([new_row])])
            
            # Create DataFrame from results
            forecast_df = pd.DataFrame([
                {**{'timestamp': r['timestamp'], 'predicted_aqi': r['predicted_aqi'], 'dominant_pollutant': r['dominant_pollutant']}, 
                 **r['pollutant_predictions']} 
                for r in results
            ])
            forecast_df.set_index('timestamp', inplace=True)
            
            return forecast_df
        
        except Exception as e:
            logger.error(f"Error predicting air quality: {e}")
            return None