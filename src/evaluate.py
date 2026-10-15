"""Model evaluation metrics for WalmartCast.

This module contains evaluation metric functions used across notebooks and tests.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics: MAE, MSE, RMSE, R².
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary with 'MAE', 'MSE', 'RMSE', 'R2'
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }


def baseline_prediction(y_train, X_test_stores):
    """
    Create baseline predictions using per-store mean from training data.
    
    Args:
        y_train: Training target values (pandas Series with Store index)
        X_test_stores: Test Store column (pandas Series)
    
    Returns:
        numpy array: Baseline predictions
    """
    # This should receive a DataFrame with Store and Weekly_Sales
    # and return predictions based on store means
    raise NotImplementedError("Use the notebook's baseline implementation")


def format_metrics(metrics, precision=2):
    """
    Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary with metric values
        precision: Number of decimal places
    
    Returns:
        dict: Formatted metrics
    """
    formatted = {}
    for key, value in metrics.items():
        if key == 'R2':
            formatted[key] = round(value, 4)
        else:
            formatted[key] = round(value, precision)
    return formatted
