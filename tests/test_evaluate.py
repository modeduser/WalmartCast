"""Tests for evaluation metrics in src/evaluate.py"""

import sys
sys.path.insert(0, r'c:\Users\yashk\OneDrive\Documents\Projects\WalmartCast')

import numpy as np
from src.evaluate import calculate_metrics, format_metrics


def test_calculate_metrics_perfect_prediction():
    """Test that perfect predictions yield MAE=0, MSE=0, RMSE=0, R2=1.0."""
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['MAE'] == 0.0
    assert metrics['MSE'] == 0.0
    assert metrics['RMSE'] == 0.0
    assert metrics['R2'] == 1.0


def test_calculate_metrics_known_values():
    """Test calculate_metrics with known values."""
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    
    # Expected MAE = (|10-12| + |20-18| + |30-33|) / 3 = (2 + 2 + 3) / 3 = 7/3 = 2.333...
    expected_mae = 7.0 / 3.0
    
    # Residuals: -2, 2, -3
    # MSE = (4 + 4 + 9) / 3 = 17/3 = 5.666...
    expected_mse = 17.0 / 3.0
    expected_rmse = np.sqrt(expected_mse)
    
    # R² = 1 - SS_res / SS_tot
    # y_mean = 20
    # SS_tot = (10-20)² + (20-20)² + (30-20)² = 100 + 0 + 100 = 200
    # SS_res = (-2)² + (2)² + (-3)² = 4 + 4 + 9 = 17
    # R² = 1 - 17/200 = 0.915
    expected_r2 = 1.0 - 17.0 / 200.0
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert abs(metrics['MAE'] - expected_mae) < 1e-6
    assert abs(metrics['MSE'] - expected_mse) < 1e-6
    assert abs(metrics['RMSE'] - expected_rmse) < 1e-6
    assert abs(metrics['R2'] - expected_r2) < 1e-6


def test_calculate_metrics_returns_all_keys():
    """Test that calculate_metrics returns dict with all required keys."""
    metrics = calculate_metrics([1], [1])
    
    assert 'MAE' in metrics
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics


def test_format_metrics_rounds_r2_to_4dp():
    """Test that format_metrics rounds R2 to 4 decimal places."""
    metrics = {'R2': 0.96581234}
    formatted = format_metrics(metrics)
    
    assert formatted['R2'] == 0.9658


def test_format_metrics_rounds_others_to_2dp():
    """Test that format_metrics rounds other metrics to 2 decimal places."""
    metrics = {'MAE': 1234.5678}
    formatted = format_metrics(metrics)
    
    assert formatted['MAE'] == 1234.57
