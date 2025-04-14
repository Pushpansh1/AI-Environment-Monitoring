import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AQI Breakpoint tables as per EPA standards
BREAKPOINTS = {
    'pm2_5': [
        {'min': 0.0, 'max': 12.0, 'aqi_min': 0, 'aqi_max': 50},
        {'min': 12.1, 'max': 35.4, 'aqi_min': 51, 'aqi_max': 100},
        {'min': 35.5, 'max': 55.4, 'aqi_min': 101, 'aqi_max': 150},
        {'min': 55.5, 'max': 150.4, 'aqi_min': 151, 'aqi_max': 200},
        {'min': 150.5, 'max': 250.4, 'aqi_min': 201, 'aqi_max': 300},
        {'min': 250.5, 'max': 500.4, 'aqi_min': 301, 'aqi_max': 500}
    ],
    'pm10': [
        {'min': 0, 'max': 54, 'aqi_min': 0, 'aqi_max': 50},
        {'min': 55, 'max': 154, 'aqi_min': 51, 'aqi_max': 100},
        {'min': 155, 'max': 254, 'aqi_min': 101, 'aqi_max': 150},
        {'min': 255, 'max': 354, 'aqi_min': 151, 'aqi_max': 200},
        {'min': 355, 'max': 424, 'aqi_min': 201, 'aqi_max': 300},
        {'min': 425, 'max': 604, 'aqi_min': 301, 'aqi_max': 500}
    ],
    'o3': [
        {'min': 0, 'max': 54, 'aqi_min': 0, 'aqi_max': 50},
        {'min': 55, 'max': 70, 'aqi_min': 51, 'aqi_max': 100},
        {'min': 71, 'max': 85, 'aqi_min': 101, 'aqi_max': 150},
        {'min': 86, 'max': 105, 'aqi_min': 151, 'aqi_max': 200},
        {'min': 106, 'max': 200, 'aqi_min': 201, 'aqi_max': 300}
    ],
    'co': [
        {'min': 0.0, 'max': 4.4, 'aqi_min': 0, 'aqi_max': 50},
        {'min': 4.5, 'max': 9.4, 'aqi_min': 51, 'aqi_max': 100},
        {'min': 9.5, 'max': 12.4, 'aqi_min': 101, 'aqi_max': 150},
        {'min': 12.5, 'max': 15.4, 'aqi_min': 151, 'aqi_max': 200},
        {'min': 15.5, 'max': 30.4, 'aqi_min': 201, 'aqi_max': 300},
        {'min': 30.5, 'max': 50.4, 'aqi_min': 301, 'aqi_max': 500}
    ],
    'so2': [
        {'min': 0, 'max': 35, 'aqi_min': 0, 'aqi_max': 50},
        {'min': 36, 'max': 75, 'aqi_min': 51, 'aqi_max': 100},
        {'min': 76, 'max': 185, 'aqi_min': 101, 'aqi_max': 150},
        {'min': 186, 'max': 304, 'aqi_min': 151, 'aqi_max': 200},
        {'min': 305, 'max': 604, 'aqi_min': 201, 'aqi_max': 300},
        {'min': 605, 'max': 1004, 'aqi_min': 301, 'aqi_max': 500}
    ],
    'no2': [
        {'min': 0, 'max': 53, 'aqi_min': 0, 'aqi_max': 50},
        {'min': 54, 'max': 100, 'aqi_min': 51, 'aqi_max': 100},
        {'min': 101, 'max': 360, 'aqi_min': 101, 'aqi_max': 150},
        {'min': 361, 'max': 649, 'aqi_min': 151, 'aqi_max': 200},
        {'min': 650, 'max': 1249, 'aqi_min': 201, 'aqi_max': 300},
        {'min': 1250, 'max': 2049, 'aqi_min': 301, 'aqi_max': 500}
    ]
}

def calculate_pollutant_aqi(concentration, pollutant):
    """
    Calculate AQI for a specific pollutant using EPA's formula.
    
    Args:
        concentration (float): Pollutant concentration
        pollutant (str): Pollutant type ('pm2_5', 'pm10', 'o3', 'co', 'so2', 'no2')
        
    Returns:
        int: Calculated AQI value for the pollutant
    """
    try:
        if pollutant not in BREAKPOINTS:
            logger.error(f"Unknown pollutant type: {pollutant}")
            return None
            
        # Handle negative or zero concentrations
        if concentration is None or concentration < 0:
            return 0
            
        # Find the appropriate breakpoint range
        breakpoint_table = BREAKPOINTS[pollutant]
        selected_range = None
        
        for bpt in breakpoint_table:
            if bpt['min'] <= concentration <= bpt['max']:
                selected_range = bpt
                break
        
        # If concentration exceeds all ranges, handle appropriately
        if selected_range is None:
            if concentration > breakpoint_table[-1]['max']:
               
                return 500
            return 0
        
        # Calculate AQI using EPA's formula
        aqi = ((
            selected_range['aqi_max'] - selected_range['aqi_min']
        ) * (
            concentration - selected_range['min']
        ) / (
            selected_range['max'] - selected_range['min']
        )) + selected_range['aqi_min']
        
        return int(round(aqi))
        
    except Exception as e:
        logger.error(f"Error calculating AQI for {pollutant}: {e}")
        return None

def calculate_aqi(pollutants):
    try:
        aqi_values = {}
        max_aqi = 0
        dominant_pollutant = None
        
        # Calculate AQI for each pollutant
        for pollutant, concentration in pollutants.items():
            if pollutant in BREAKPOINTS:
                aqi = calculate_pollutant_aqi(concentration, pollutant)
                if aqi is not None:
                    aqi_values[pollutant] = aqi
                    if aqi > max_aqi:
                        max_aqi = aqi
                        dominant_pollutant = pollutant
        
        if not aqi_values:
            logger.warning("No valid pollutant data for AQI calculation")
            return None
        
        # Prepare result dictionary
        result = {
            'aqi': max_aqi,
            'dominant_pollutant': dominant_pollutant,
            'pollutant_aqi': aqi_values
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating overall AQI: {e}")
        return None