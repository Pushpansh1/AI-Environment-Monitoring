import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_time_features(df):
   
    if df is None or df.empty:
        logger.warning("No data for time feature extraction")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Extract time features
        result_df['hour'] = result_df.index.hour
        result_df['day'] = result_df.index.day
        result_df['month'] = result_df.index.month
        result_df['year'] = result_df.index.year
        result_df['dayofweek'] = result_df.index.dayofweek
        result_df['is_weekend'] = result_df.index.dayofweek >= 5
        
        # Cyclical encoding for hour (to capture the cyclical nature of time)
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        
        # Cyclical encoding for month
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        
        # Cyclical encoding for day of week
        result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df['dayofweek'] / 7)
        result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df['dayofweek'] / 7)
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error extracting time features: {e}")
        return df  # Return original DataFrame if extraction fails

def create_lag_features(df, columns, lag_periods=[1, 3, 6, 12, 24]):
    
    if df is None or df.empty:
        logger.warning("No data for lag feature creation")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Create lag features for each column and lag period
        for col in columns:
            if col in result_df.columns:
                for lag in lag_periods:
                    result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
        
        # Drop rows with NaN values resulting from lag creation
        result_df = result_df.dropna()
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error creating lag features: {e}")
        return df  # Return original DataFrame if creation fails

def create_rolling_features(df, columns, windows=[3, 6, 12, 24]):
   
    if df is None or df.empty:
        logger.warning("No data for rolling feature creation")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Create rolling window features for each column and window size
        for col in columns:
            if col in result_df.columns:
                for window in windows:
                    rolling = result_df[col].rolling(window=window)
                    result_df[f'{col}_rolling_mean_{window}'] = rolling.mean()
                    result_df[f'{col}_rolling_std_{window}'] = rolling.std()
                    result_df[f'{col}_rolling_min_{window}'] = rolling.min()
                    result_df[f'{col}_rolling_max_{window}'] = rolling.max()
        
        # Drop rows with NaN values resulting from rolling window creation
        result_df = result_df.dropna()
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error creating rolling features: {e}")
        return df  # Return original DataFrame if creation fails

def create_weather_condition_features(df):
    
    if df is None or df.empty or 'weather_main' not in df.columns:
        logger.warning("No data or missing 'weather_main' column for weather condition feature creation")
        return df
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Common weather conditions
        weather_conditions = [
            'Clear', 'Clouds', 'Rain', 'Drizzle', 'Thunderstorm', 
            'Snow', 'Mist', 'Smoke', 'Haze', 'Dust', 'Fog'
        ]
        
        # Create binary features for each weather condition
        for condition in weather_conditions:
            result_df[f'is_{condition.lower()}'] = (result_df['weather_main'] == condition).astype(int)
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error creating weather condition features: {e}")
        return df  # Return original DataFrame if creation fails

def create_temperature_features(df):
   
    if df is None or df.empty:
        logger.warning("No data for temperature feature creation")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Check if required columns exist
        required_columns = ['temperature', 'temp_min', 'temp_max']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for temperature feature creation: {missing_columns}")
            return df
        
        # Temperature range (daily variation)
        result_df['temp_range'] = result_df['temp_max'] - result_df['temp_min']
        
        # Temperature change from previous hour
        result_df['temp_change'] = result_df['temperature'].diff()
        
        # Temperature acceleration (change in temperature change)
        result_df['temp_acceleration'] = result_df['temp_change'].diff()
        
        # Drop rows with NaN values resulting from diff operations
        result_df = result_df.dropna()
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error creating temperature features: {e}")
        return df  # Return original DataFrame if creation fails

def create_air_quality_features(df):
   
    if df is None or df.empty:
        logger.warning("No data for air quality feature creation")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Define air quality pollutant columns
        pollutant_columns = ['pm2_5', 'pm10', 'o3', 'co', 'so2', 'no2']
        available_pollutants = [col for col in pollutant_columns if col in result_df.columns]
        
        if not available_pollutants:
            logger.warning("No air quality pollutant columns found in data")
            return df
        
        # Calculate pollutant ratios (useful for source identification)
        if 'pm2_5' in available_pollutants and 'pm10' in available_pollutants:
            result_df['pm_ratio'] = result_df['pm2_5'] / result_df['pm10']
        
        if 'no2' in available_pollutants and 'o3' in available_pollutants:
            result_df['no2_o3_ratio'] = result_df['no2'] / result_df['o3']
        
        # Calculate total pollutant load
        result_df['total_pollutant_load'] = result_df[available_pollutants].sum(axis=1)
        
        # Calculate dominant pollutant
        for pollutant in available_pollutants:
            result_df[f'{pollutant}_dominance'] = result_df[pollutant] / result_df['total_pollutant_load']
        
        # Create pollutant change rates
        for pollutant in available_pollutants:
            result_df[f'{pollutant}_change'] = result_df[pollutant].diff()
        
        # Create day/night indicator (air quality often varies between day and night)
        if 'hour' in result_df.columns:
            result_df['is_daytime'] = ((result_df['hour'] >= 6) & (result_df['hour'] <= 18)).astype(int)
        
        # Create rush hour indicator (air quality often worse during rush hours)
        if 'hour' in result_df.columns:
            result_df['is_rush_hour'] = (((result_df['hour'] >= 7) & (result_df['hour'] <= 9)) | 
                                        ((result_df['hour'] >= 16) & (result_df['hour'] <= 19))).astype(int)
        
        # Handle NaN values from calculations
        result_df = result_df.fillna(method='bfill').fillna(method='ffill')
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error creating air quality features: {e}")
        return df  # Return original DataFrame if creation fails

def prepare_features_for_model(df, target_col=None, drop_cols=None):
   
    if df is None or df.empty:
        logger.warning("No data for feature preparation")
        return None
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Drop specified columns
        if drop_cols:
            cols_to_drop = [col for col in drop_cols if col in result_df.columns]
            result_df = result_df.drop(columns=cols_to_drop)
        
        # Handle categorical variables
        cat_columns = result_df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_columns:
            # One-hot encode categorical variables
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=True)
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df = result_df.drop(columns=[col])
        
        # Separate features and target if target_col is provided
        if target_col and target_col in result_df.columns:
            X = result_df.drop(columns=[target_col])
            y = result_df[target_col]
            return X, y
        else:
            return result_df
    
    except Exception as e:
        logger.error(f"Error preparing features for model: {e}")
        return df  # Return original DataFrame if preparation fails