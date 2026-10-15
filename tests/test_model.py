"""Tests for saved model artifacts in models/"""

import sys
sys.path.insert(0, r'c:\Users\yashk\OneDrive\Documents\Projects\WalmartCast')

import os
import json
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = r'c:\Users\yashk\OneDrive\Documents\Projects\WalmartCast'


def test_model_file_exists():
    """Test that best_model.joblib exists on disk."""
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')
    assert os.path.exists(model_path), f"Model file not found at {model_path}"


def test_model_loads():
    """Test that the saved model loads and has a predict method."""
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')
    model = joblib.load(model_path)
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_model_predicts_finite_positive():
    """Test that the model predicts a finite positive value for a sample row."""
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')
    model = joblib.load(model_path)
    
    # Create one row with specified values
    row_df = pd.DataFrame({
        'Store': [1],
        'Holiday_Flag': [0],
        'Temperature': [42.31],
        'Fuel_Price': [2.572],
        'CPI': [211.096],
        'Unemployment': [8.106],
        'Year': [2010],
        'Month': [2],
        'WeekOfYear': [5],
        'Quarter': [1]
    })
    
    prediction = model.predict(row_df)
    
    assert np.isfinite(prediction[0]), f"Prediction is not finite: {prediction[0]}"
    assert prediction[0] > 0, f"Prediction should be positive, got {prediction[0]}"


def test_feature_info_exists_and_has_required_keys():
    """Test that feature_info.json exists and has all required keys."""
    info_path = os.path.join(PROJECT_ROOT, 'models', 'feature_info.json')
    assert os.path.exists(info_path), f"feature_info.json not found at {info_path}"
    
    with open(info_path, 'r') as f:
        feature_info = json.load(f)
    
    required_keys = ['best_model_name', 'feature_columns', 'training_date_range', 
                     'sklearn_version', 'selection_rule']
    for key in required_keys:
        assert key in feature_info, f"Missing required key: {key}"


def test_feature_info_model_name():
    """Test that feature_info.json reports 'Linear Regression' as best model."""
    info_path = os.path.join(PROJECT_ROOT, 'models', 'feature_info.json')
    
    with open(info_path, 'r') as f:
        feature_info = json.load(f)
    
    assert feature_info['best_model_name'] == 'Linear Regression', \
        f"Expected 'Linear Regression', got '{feature_info['best_model_name']}'"
