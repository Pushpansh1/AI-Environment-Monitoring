import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yaml
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from modules.data_fetcher import fetch_current_weather, fetch_air_quality, fetch_weather_forecast, get_location_by_name
from modules.data_processor import process_weather_data, process_air_quality_data, process_forecast_data
from modules.prediction_engine import PredictionEngine
from modules.alert_system import AlertSystem
from modules.utils import (
    load_sample_data, 
    create_time_series_plot, 
    create_correlation_heatmap,
    format_weather_icon_url,
    celsius_to_fahrenheit,
    get_aqi_description
)

# Load environment variables and configuration
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Check if API key is available
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
if not API_KEY:
    st.error("OpenWeatherMap API key not found. Please add it to the .env file.")
    st.stop()

# Initialize prediction engine and alert system
prediction_engine = PredictionEngine()
alert_system = AlertSystem()

# Set page configuration
st.set_page_config(
    page_title="AI Environment Monitoring System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .alert-warning {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
    }
    .alert-danger {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .metric-card {
        background-color: #F5F5F5;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>AI Environment Monitoring System</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")

# Location input
location_method = st.sidebar.radio("Select location by:", ["City Name", "Coordinates"])

if location_method == "City Name":
    city_name = st.sidebar.text_input("City Name", value=config['ui']['default_location']['city'])
    if st.sidebar.button("Get Location"):
        with st.spinner("Fetching location..."):
            location = get_location_by_name(city_name)
            if location:
                lat, lon = location
                st.sidebar.success(f"Location found: {city_name} ({lat:.4f}, {lon:.4f})")
            else:
                st.sidebar.error(f"Location not found for: {city_name}")
                lat, lon = config['ui']['default_location']['lat'], config['ui']['default_location']['lon']
    else:
        # Use default location
        lat, lon = config['ui']['default_location']['lat'], config['ui']['default_location']['lon']
else:
    lat = st.sidebar.number_input("Latitude", value=config['ui']['default_location']['lat'], format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=config['ui']['default_location']['lon'], format="%.4f")

# Units selection
units = st.sidebar.selectbox("Temperature Units", ["Celsius", "Fahrenheit"], index=0)

# Forecast horizon
forecast_hours = st.sidebar.slider("Forecast Horizon (hours)", min_value=12, max_value=120, value=48, step=12)

# Data refresh button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Demo mode toggle
demo_mode = st.sidebar.checkbox("Demo Mode (Use Sample Data)", value=False)

# Main content
# Create tabs for different sections
tabs = st.tabs(["Current Conditions", "Forecast", "Predictions & Alerts", "Data Analysis"])

# Current Conditions Tab
with tabs[0]:
    st.markdown("<h2 class='sub-header'>Current Weather and Air Quality</h2>", unsafe_allow_html=True)
    
    # Fetch and process data
    if demo_mode:
        # Use sample data in demo mode
        weather_data = load_sample_data('weather').iloc[-1].to_dict()
        air_quality_data = load_sample_data('air_quality').iloc[-1].to_dict()
    else:
        # Fetch real-time data
        with st.spinner("Fetching current weather data..."):
            raw_weather = fetch_current_weather(lat, lon)
            weather_data = process_weather_data(raw_weather)
        
        with st.spinner("Fetching air quality data..."):
            raw_air_quality = fetch_air_quality(lat, lon)
            air_quality_data = process_air_quality_data(raw_air_quality)
    
    if weather_data and air_quality_data:
        # Display current conditions in a grid layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if 'weather_icon' in weather_data:
                st.image(format_weather_icon_url(weather_data['weather_icon']), width=100)
            
            temp_value = weather_data['temperature']
            if units == "Fahrenheit":
                temp_value = celsius_to_fahrenheit(temp_value)
                temp_unit = "°F"
            else:
                temp_unit = "°C"
            
            st.markdown(f"<div class='metric-value'>{temp_value:.1f}{temp_unit}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>{weather_data['weather_description'].title()}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Humidity</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{weather_data['humidity']}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Wind Speed</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{weather_data['wind_speed']} m/s</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Direction: {weather_data['wind_direction']}°</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Pressure</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{weather_data['pressure']} hPa</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            aqi_desc, aqi_color = get_aqi_description(air_quality_data['aqi'])
            
            st.markdown(f"<div class='metric-card' style='border-left: 5px solid {aqi_color};'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Air Quality Index</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{air_quality_data['aqi']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>{aqi_desc}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Precipitation (1h)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{weather_data.get('rain_1h', 0):.1f} mm</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional details in expandable section
        with st.expander("Additional Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Weather Details")
                details_df = pd.DataFrame({
                    'Metric': ['Feels Like', 'Min Temperature', 'Max Temperature', 'Clouds', 'Sunrise', 'Sunset'],
                    'Value': [
                        f"{weather_data['feels_like']:.1f}°C",
                        f"{weather_data['temp_min']:.1f}°C",
                        f"{weather_data['temp_max']:.1f}°C",
                        f"{weather_data['clouds']}%",
                        weather_data['sunrise'],
                        weather_data['sunset']
                    ]
                })
                st.table(details_df)
            
            with col2:
                st.subheader("Air Quality Details")
                air_details_df = pd.DataFrame({
                    'Pollutant': ['CO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10'],
                    'Value (μg/m³)': [
                        f"{air_quality_data['co']:.1f}",
                        f"{air_quality_data['no2']:.1f}",
                        f"{air_quality_data['o3']:.1f}",
                        f"{air_quality_data['so2']:.1f}",
                        f"{air_quality_data['pm2_5']:.1f}",
                        f"{air_quality_data['pm10']:.1f}"
                    ]
                })
                st.table(air_details_df)
    else:
        st.error("Failed to fetch current conditions. Please check your API key and try again.")

# Forecast Tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Weather Forecast</h2>", unsafe_allow_html=True)
    
    # Fetch and process forecast data
    if demo_mode:
        # Use sample data in demo mode
        forecast_df = load_sample_data('weather')
    else:
        # Fetch real-time forecast data
        with st.spinner("Fetching forecast data..."):
            raw_forecast = fetch_weather_forecast(lat, lon)
            forecast_df = process_forecast_data(raw_forecast)
    
    if forecast_df is not None and not forecast_df.empty:
        # Temperature forecast chart
        st.subheader("Temperature Forecast")
        
        # Convert temperature if needed
        if units == "Fahrenheit":
            temp_col = 'temperature_f'
            forecast_df[temp_col] = forecast_df['temperature'].apply(celsius_to_fahrenheit)
            temp_unit = "°F"
        else:
            temp_col = 'temperature'
            temp_unit = "°C"
        
        # Create temperature chart with Plotly
        fig = px.line(
            forecast_df, 
            x=forecast_df.index, 
            y=temp_col,
            labels={'x': 'Date', 'y': f'Temperature ({temp_unit})'},
            title=f'Temperature Forecast for the Next {len(forecast_df)} Hours'
        )
        
        # Add feels like temperature
        if units == "Fahrenheit":
            forecast_df['feels_like_f'] = forecast_df['feels_like'].apply(celsius_to_fahrenheit)
            feels_like_col = 'feels_like_f'
        else:
            feels_like_col = 'feels_like'
        
        fig.add_scatter(
            x=forecast_df.index, 
            y=forecast_df[feels_like_col], 
            mode='lines', 
            name=f'Feels Like ({temp_unit})'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=f'Temperature ({temp_unit})',
            legend_title='Metric',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation forecast
        st.subheader("Precipitation Forecast")
        
        # Create precipitation chart with Plotly
        fig = px.bar(
            forecast_df, 
            x=forecast_df.index, 
            y='rain_3h',
            labels={'x': 'Date', 'y': 'Precipitation (mm)'},
            title='Precipitation Forecast for the Next 5 Days'
        )
        
        # Add probability of precipitation
        fig.add_scatter(
            x=forecast_df.index, 
            y=forecast_df['probability'], 
            mode='lines', 
            name='Probability (%)',
            yaxis='y2'
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Precipitation (mm)',
            yaxis2=dict(
                title='Probability (%)',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            legend_title='Metric',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather conditions forecast
        st.subheader("Weather Conditions Forecast")
        
        # Create a table with weather conditions
        forecast_table = forecast_df[['weather_main', 'weather_description', 'humidity', 'wind_speed']].copy()
        forecast_table.index = forecast_table.index.strftime('%Y-%m-%d %H:%M')
        forecast_table.columns = ['Weather', 'Description', 'Humidity (%)', 'Wind Speed (m/s)']
        
        st.dataframe(forecast_table, use_container_width=True)
    else:
        st.error("Failed to fetch forecast data. Please check your API key and try again.")

# Predictions & Alerts Tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>AI Predictions and Alerts</h2>", unsafe_allow_html=True)
    
    # Get historical data for predictions
    if demo_mode:
        # Use sample data in demo mode
        historical_weather = load_sample_data('weather')
        historical_air_quality = load_sample_data('air_quality')
        flood_data = load_sample_data('flood')
    else:
        # In real mode, we would need to fetch historical data
        # For now, use sample data as a placeholder
        historical_weather = load_sample_data('weather')
        historical_air_quality = load_sample_data('air_quality')
        flood_data = load_sample_data('flood')
    
    # Temperature prediction
    st.subheader("Temperature Prediction")
    
    # Make temperature predictions
    with st.spinner("Generating temperature predictions..."):
        # In a real scenario, we would use the prediction_engine
        # For demo, we'll create a simple forecast
        
        last_date = historical_weather.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=forecast_hours, freq='H')
        
        # Create a simple prediction (in a real app, use the prediction engine)
        # This is a placeholder that mimics the prediction engine's output
        base_temp = historical_weather['temperature'].iloc[-1]
        hour_of_day = np.array([d.hour for d in future_dates])
        
        # Temperature follows a sinusoidal pattern throughout the day
        hourly_pattern = np.sin(2 * np.pi * (hour_of_day / 24)) * 3
        
        # Add some daily variation and random noise
        predicted_temps = base_temp + hourly_pattern + np.random.normal(0, 1, len(future_dates))
        
        # Create DataFrame for predictions
        temp_predictions = pd.DataFrame({
            'predicted_temperature': predicted_temps
        }, index=future_dates)
    
    # Display temperature predictions
    if units == "Fahrenheit":
        temp_predictions['predicted_temperature_f'] = temp_predictions['predicted_temperature'].apply(celsius_to_fahrenheit)
        temp_col = 'predicted_temperature_f'
        temp_unit = "°F"
    else:
        temp_col = 'predicted_temperature'
        temp_unit = "°C"
    
    # Create temperature prediction chart with Plotly
    fig = go.Figure()
    
    # Add historical data
    if units == "Fahrenheit":
        historical_temp = historical_weather['temperature'].apply(celsius_to_fahrenheit)
    else:
        historical_temp = historical_weather['temperature']
    
    fig.add_trace(go.Scatter(
        x=historical_weather.index,
        y=historical_temp,
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=temp_predictions.index,
        y=temp_predictions[temp_col],
        mode='lines',
        name='Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence interval (simulated)
    upper_bound = temp_predictions[temp_col] + 2
    lower_bound = temp_predictions[temp_col] - 2
    
    fig.add_trace(go.Scatter(
        x=temp_predictions.index,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=temp_predictions.index,
        y=lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Temperature Prediction for the Next {forecast_hours} Hours',
        xaxis_title='Date',
        yaxis_title=f'Temperature ({temp_unit})',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Disaster risk prediction
    st.subheader("Environmental Risk Assessment")
    
    # In a real scenario, we would use the prediction_engine
    # For demo, we'll create a simple risk assessment
    
    # Generate some sample risk data
    risk_data = {
        'flood': 0.7 if np.mean(flood_data['flood_risk'].iloc[-24:]) > 0.5 else 0.3,
        'storm': 0.6 if np.mean(historical_weather['wind_speed'].iloc[-24:]) > 8 else 0.2,
        'heatwave': 0.8 if np.mean(historical_weather['temperature'].iloc[-24:]) > 30 else 0.1,
        'air_pollution': 0.75 if np.mean(historical_air_quality['aqi'].iloc[-24:]) > 100 else 0.25
    }
    
    # Determine highest risk
    highest_risk = max(risk_data.items(), key=lambda x: x[1])
    risk_data['highest_risk'] = highest_risk[0]
    risk_data['highest_probability'] = highest_risk[1]
    risk_data['risk_level'] = 'High' if highest_risk[1] >= 0.5 else 'Low'
    
    # Display risk assessment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create risk gauge chart
        fig = go.Figure()
        
        for risk_type, probability in risk_data.items():
            if risk_type not in ['highest_risk', 'highest_probability', 'risk_level']:
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': risk_type.capitalize()},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
        
        # Create subplots for multiple gauges
        fig = make_subplots(
            rows=2, 
            cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("Flood Risk", "Storm Risk", "Heatwave Risk", "Air Pollution Risk")
        )
        
        # Add gauges to subplots
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data['flood'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data['storm'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data['heatwave'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data['air_pollution'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display risk summary
        st.markdown(f"### Risk Summary")
        st.markdown(f"**Highest Risk:** {risk_data['highest_risk'].capitalize()}")
        st.markdown(f"**Risk Level:** {risk_data['risk_level']}")
        st.markdown(f"**Probability:** {risk_data['highest_probability'] * 100:.1f}%")
        
        # Generate alerts based on risks and current conditions
        alerts = alert_system.generate_alerts(
            weather_data=weather_data,
            air_quality_data=air_quality_data,
            risk_data=risk_data
        )
        
        # Display alerts
        st.markdown("### Active Alerts")
        
        if alerts:
            for alert in alerts:
                alert_class = "alert-warning" if alert['level'] == 'warning' else "alert-danger"
                st.markdown(f"""
                <div class='alert-box {alert_class}'>
                    <strong>{alert['type'].upper()}:</strong> {alert['message']}
                    <br><small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts at this time.")

# Data Analysis Tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>Data Analysis</h2>", unsafe_allow_html=True)
    
    # Get historical data
    if demo_mode:
        # Use sample data in demo mode
        historical_weather = load_sample_data('weather')
        historical_air_quality = load_sample_data('air_quality')
    else:
        # In real mode, we would need to fetch historical data
        # For now, use sample data as a placeholder
        historical_weather = load_sample_data('weather')
        historical_air_quality = load_sample_data('air_quality')
    
    # Time range selection
    st.subheader("Select Time Range")
    
    # Get min and max dates from data
    min_date = historical_weather.index.min().date()
    max_date = historical_weather.index.max().date()
    
    # Date range picker
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date)
    with col2:
        end_date = st.date_input("End Date", max_date)
    
    # Filter data based on selected date range
    mask = (historical_weather.index.date >= start_date) & (historical_weather.index.date <= end_date)
    filtered_weather = historical_weather[mask]
    
    mask = (historical_air_quality.index.date >= start_date) & (historical_air_quality.index.date <= end_date)
    filtered_air_quality = historical_air_quality[mask]
    
    # Data visualization options
    st.subheader("Data Visualization")
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Time Series", "Correlation Analysis", "Daily Patterns", "Statistical Summary"]
    )
    
    if viz_type == "Time Series":
        # Select data to visualize
        data_type = st.selectbox("Select Data Type", ["Weather", "Air Quality"])
        
        if data_type == "Weather":
            # Select weather variables
            weather_vars = st.multiselect(
                "Select Weather Variables",
                options=filtered_weather.columns.tolist(),
                default=['temperature', 'humidity', 'pressure']
            )
            
            if weather_vars:
                # Create time series plot
                fig = px.line(
                    filtered_weather, 
                    x=filtered_weather.index, 
                    y=weather_vars,
                    labels={'x': 'Date', 'value': 'Value'},
                    title=f'Weather Variables Time Series ({start_date} to {end_date})'
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend_title='Variable',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one weather variable.")
        
        elif data_type == "Air Quality":
            # Select air quality variables
            air_vars = st.multiselect(
                "Select Air Quality Variables",
                options=filtered_air_quality.columns.tolist(),
                default=['aqi', 'pm2_5', 'o3']
            )
            
            if air_vars:
                # Create time series plot
                fig = px.line(
                    filtered_air_quality, 
                    x=filtered_air_quality.index, 
                    y=air_vars,
                    labels={'x': 'Date', 'value': 'Value'},
                    title=f'Air Quality Variables Time Series ({start_date} to {end_date})'
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend_title='Variable',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one air quality variable.")
    
    elif viz_type == "Correlation Analysis":
        # Select data for correlation analysis
        data_type = st.selectbox("Select Data Type for Correlation", ["Weather", "Air Quality", "Combined"])
        
        if data_type == "Weather":
            # Create correlation heatmap for weather data
            corr_matrix = filtered_weather.select_dtypes(include=[np.number]).corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title=f'Weather Variables Correlation Matrix'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot for selected variables
            st.subheader("Scatter Plot")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X Variable", filtered_weather.select_dtypes(include=[np.number]).columns.tolist())
            with col2:
                y_var = st.selectbox("Y Variable", filtered_weather.select_dtypes(include=[np.number]).columns.tolist(), index=1)
            
            fig = px.scatter(
                filtered_weather,
                x=x_var,
                y=y_var,
                trendline="ols",
                title=f'Scatter Plot: {x_var} vs {y_var}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "Air Quality":
            # Create correlation heatmap for air quality data
            corr_matrix = filtered_air_quality.select_dtypes(include=[np.number]).corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title=f'Air Quality Variables Correlation Matrix'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot for selected variables
            st.subheader("Scatter Plot")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X Variable", filtered_air_quality.select_dtypes(include=[np.number]).columns.tolist())
            with col2:
                y_var = st.selectbox("Y Variable", filtered_air_quality.select_dtypes(include=[np.number]).columns.tolist(), index=1)
            
            fig = px.scatter(
                filtered_air_quality,
                x=x_var,
                y=y_var,
                trendline="ols",
                title=f'Scatter Plot: {x_var} vs {y_var}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "Combined":
            # Combine weather and air quality data
            # First, resample both datasets to hourly frequency to ensure alignment
            weather_hourly = filtered_weather.resample('H').mean()
            air_quality_hourly = filtered_air_quality.resample('H').mean()
            
            # Merge the datasets
            combined_data = pd.merge(
                weather_hourly,
                air_quality_hourly,
                left_index=True,
                right_index=True,
                how='inner',
                suffixes=('_weather', '_air')
            )
            
            if not combined_data.empty:
                # Create correlation heatmap for combined data
                corr_matrix = combined_data.select_dtypes(include=[np.number]).corr()
                
                # Select top correlations for readability
                top_corr = pd.DataFrame()
                for col in corr_matrix.columns:
                    # Get top 5 correlations for each column
                    top_corr_col = corr_matrix[col].sort_values(ascending=False).head(5)
                    top_corr = pd.concat([top_corr, top_corr_col])
                
                # Remove duplicates
                top_corr = top_corr.drop_duplicates()
                
                # Create a more focused correlation matrix
                st.subheader("Top Weather-Air Quality Correlations")
                
                # Display top correlations as a table
                st.dataframe(top_corr, use_container_width=True)
                
                # Scatter plot for selected variables
                st.subheader("Cross-Domain Scatter Plot")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Weather Variable", filtered_weather.select_dtypes(include=[np.number]).columns.tolist())
                with col2:
                    y_var = st.selectbox("Air Quality Variable", filtered_air_quality.select_dtypes(include=[np.number]).columns.tolist())
                
                # Create a temporary dataframe for the scatter plot
                scatter_df = pd.DataFrame({
                    'weather_var': weather_hourly[x_var],
                    'air_var': air_quality_hourly[y_var]
                })
                
                # Remove NaN values
                scatter_df = scatter_df.dropna()
                
                if not scatter_df.empty:
                    fig = px.scatter(
                        scatter_df,
                        x='weather_var',
                        y='air_var',
                        trendline="ols",
                        labels={'weather_var': x_var, 'air_var': y_var},
                        title=f'Scatter Plot: {x_var} vs {y_var}'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough overlapping data points for the selected variables.")
            else:
                st.warning("No overlapping data between weather and air quality datasets for the selected time range.")
    
    elif viz_type == "Daily Patterns":
        # Select data type
        data_type = st.selectbox("Select Data Type for Daily Patterns", ["Weather", "Air Quality"])
        
        if data_type == "Weather":
            # Select weather variable
            weather_var = st.selectbox(
                "Select Weather Variable",
                options=filtered_weather.select_dtypes(include=[np.number]).columns.tolist(),
                index=0
            )
            
            # Extract hour from datetime index
            filtered_weather['hour'] = filtered_weather.index.hour
            
            # Group by hour and calculate statistics
            hourly_stats = filtered_weather.groupby('hour')[weather_var].agg(['mean', 'min', 'max', 'std'])
            
            # Create hourly pattern plot
            fig = go.Figure()
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['mean'],
                mode='lines+markers',
                name='Mean',
                line=dict(color='blue', width=2)
            ))
            
            # Add min and max range
            fig.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['max'],
                mode='lines',
                name='Max',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['min'],
                mode='lines',
                name='Min',
                line=dict(color='green', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Daily Pattern of {weather_var}',
                xaxis_title='Hour of Day',
                yaxis_title=weather_var,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(24)),
                    ticktext=[f'{h:02d}:00' for h in range(24)]
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "Air Quality":
            # Select air quality variable
            air_var = st.selectbox(
                "Select Air Quality Variable",
                options=filtered_air_quality.select_dtypes(include=[np.number]).columns.tolist(),
                index=0
            )
            
            # Extract hour from datetime index
            filtered_air_quality['hour'] = filtered_air_quality.index.hour
            
            # Group by hour and calculate statistics
            hourly_stats = filtered_air_quality.groupby('hour')[air_var].agg(['mean', 'min', 'max', 'std'])
            
            # Create hourly pattern plot
            fig = go.Figure()
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['mean'],
                mode='lines+markers',
                name='Mean',
                line=dict(color='blue', width=2)
            ))
            
            # Add min and max range
            fig.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['max'],
                mode='lines',
                name='Max',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['min'],
                mode='lines',
                name='Min',
                line=dict(color='green', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Daily Pattern of {air_var}',
                xaxis_title='Hour of Day',
                yaxis_title=air_var,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(24)),
                    ticktext=[f'{h:02d}:00' for h in range(24)]
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Statistical Summary":
        # Select data type
        data_type = st.selectbox("Select Data Type for Summary", ["Weather", "Air Quality"])
        
        if data_type == "Weather":
            # Display statistical summary of weather data
            st.subheader("Weather Data Statistical Summary")
            
            # Select only numeric columns
            numeric_weather = filtered_weather.select_dtypes(include=[np.number])
            
            # Calculate statistics
            stats_df = numeric_weather.describe().T
            
            # Add additional statistics
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['cv'] = stats_df['std'] / stats_df['mean'] * 100  # Coefficient of variation
            
            # Format the table
            formatted_stats = stats_df.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                '25%': '{:.2f}',
                '50%': '{:.2f}',
                '75%': '{:.2f}',
                'max': '{:.2f}',
                'range': '{:.2f}',
                'cv': '{:.2f}%'
            })
            
            st.dataframe(formatted_stats, use_container_width=True)
            
            # Display distribution plots
            st.subheader("Distribution Plots")
            
            # Select variable for distribution plot
            dist_var = st.selectbox(
                "Select Variable for Distribution",
                options=numeric_weather.columns.tolist(),
                index=0
            )
            
            # Create histogram with KDE
            fig = px.histogram(
                numeric_weather,
                x=dist_var,
                nbins=30,
                marginal="box",
                title=f'Distribution of {dist_var}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "Air Quality":
            # Display statistical summary of air quality data
            st.subheader("Air Quality Data Statistical Summary")
            
            # Select only numeric columns
            numeric_air = filtered_air_quality.select_dtypes(include=[np.number])
            
            # Calculate statistics
            stats_df = numeric_air.describe().T
            
            # Add additional statistics
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['cv'] = stats_df['std'] / stats_df['mean'] * 100  # Coefficient of variation
            
            # Format the table
            formatted_stats = stats_df.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                '25%': '{:.2f}',
                '50%': '{:.2f}',
                '75%': '{:.2f}',
                'max': '{:.2f}',
                'range': '{:.2f}',
                'cv': '{:.2f}%'
            })
            
            st.dataframe(formatted_stats, use_container_width=True)
            
            # Display distribution plots
            st.subheader("Distribution Plots")
            
            # Select variable for distribution plot
            dist_var = st.selectbox(
                "Select Variable for Distribution",
                options=numeric_air.columns.tolist(),
                index=0
            )
            
            # Create histogram with KDE
            fig = px.histogram(
                numeric_air,
                x=dist_var,
                nbins=30,
                marginal="box",
                title=f'Distribution of {dist_var}'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### AI Environment Monitoring System")
st.markdown("Developed for environmental monitoring and prediction using AI techniques.")
st.markdown("© 2025 AI Environment Monitoring System")