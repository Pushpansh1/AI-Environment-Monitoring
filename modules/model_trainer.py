import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the models directory path
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def train_random_forest(X_train, y_train, params=None):
   
    try:
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Use provided parameters if available, otherwise use defaults
        model_params = params if params else default_params
        
        # Create and train the model
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        
        logger.info("Random Forest model trained successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        return None

def train_gradient_boosting(X_train, y_train, params=None):
  
    try:
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Use provided parameters if available, otherwise use defaults
        model_params = params if params else default_params
        
        # Create and train the model
        model = GradientBoostingRegressor(**model_params)
        model.fit(X_train, y_train)
        
        logger.info("Gradient Boosting model trained successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training Gradient Boosting model: {e}")
        return None

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
  
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

def prepare_lstm_data(data, target_col, lookback=24):
   
    X, y = [], []
    
    for i in range(len(data) - lookback):
        X.append(data.iloc[i:(i + lookback)].values)
        y.append(data.iloc[i + lookback][target_col])
    
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, input_shape, epochs=50, batch_size=32, validation_split=0.2):
  
    try:
        # Create the model
        model = create_lstm_model(input_shape)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("LSTM model trained successfully")
        return model, history
    
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return None, None

def evaluate_model(model, X_test, y_test, model_type='sklearn'):
   
    try:
        if model_type == 'sklearn':
            # Make predictions
            y_pred = model.predict(X_test)
        elif model_type == 'keras':
            # Make predictions
            y_pred = model.predict(X_test).flatten()
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create metrics dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def save_model(model, model_name, model_type):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    if model_type == 'keras':
        model.save(os.path.join(models_dir, f'{model_name}.h5'))
    elif model_type == 'sklearn':
        joblib.dump(model, os.path.join(models_dir, f'{model_name}.pkl'))
    else:
        raise ValueError(f'Unsupported model type: {model_type}')


def load_model(model_name, model_type):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    if model_type == 'keras':
        from keras.models import load_model
        return load_model(os.path.join(models_dir, f'{model_name}.h5'))
    elif model_type == 'sklearn':
        return joblib.load(os.path.join(models_dir, f'{model_name}.pkl'))
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

def tune_hyperparameters(X_train, y_train, model_type='random_forest', param_grid=None):
   
    try:
        # Default parameter grids
        default_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        # Use provided parameter grid if available, otherwise use defaults
        grid = param_grid if param_grid else default_param_grids.get(model_type)
        
        if not grid:
            logger.error(f"Unknown model type for hyperparameter tuning: {model_type}")
            return None, None
        
        # Create the model
        if model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None, None
        
        # Create the grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters for {model_type}: {best_params}")
        return best_model, best_params
    
    except Exception as e:
        logger.error(f"Error tuning hyperparameters: {e}")
        return None, None