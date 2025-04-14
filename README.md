# AI-Based Environment Monitoring and Prediction System

## Overview

This project is an AI-based system for monitoring environmental conditions and predicting potential environmental hazards. It uses real-time weather and air quality data, along with machine learning models to provide forecasts and risk assessments.

## Features

- **Real-time Environmental Monitoring**: Fetches and displays current weather and air quality data.
- **Weather Forecasting**: Shows weather forecasts for the next several days.
- **AI-Powered Predictions**: Uses machine learning models to predict temperature trends and environmental risks.
- **Alert System**: Generates alerts based on environmental conditions and risk assessments.
- **Data Analysis**: Provides tools for analyzing historical environmental data.

## System Architecture

The system consists of the following components:

1. **Data Fetcher**: Retrieves data from external APIs (OpenWeatherMap).
2. **Data Processor**: Processes and transforms raw data into a usable format.
3. **Feature Engineer**: Extracts and creates features for machine learning models.
4. **Model Trainer**: Trains machine learning models for predictions.
5. **Prediction Engine**: Makes predictions using trained models.
6. **Alert System**: Generates alerts based on thresholds and predictions.
7. **Web Interface**: Streamlit-based user interface for interacting with the system.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GouravSinghThakur/AI_Environment_Monitoring_System.git
cd AI_Environment_Monitoring_System
```
2. Install dependencies: