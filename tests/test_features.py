"""Tests for feature engineering functions in src/features.py"""

import sys
sys.path.insert(0, r'c:\Users\yashk\OneDrive\Documents\Projects\WalmartCast')

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.features import get_feature_columns, prepare_features, make_pipeline


def test_get_feature_columns_returns_expected():
    """Test that get_feature_columns returns exactly the 10 expected columns."""
    expected = {'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 
                'CPI', 'Unemployment', 'Year', 'Month', 'WeekOfYear', 'Quarter'}
    actual = set(get_feature_columns())
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_prepare_features_adds_time_columns():
    """Test that prepare_features adds Year, Month, WeekOfYear, Quarter."""
    df = pd.DataFrame({
        'Store': [1, 2],
        'Date': pd.to_datetime(['2010-02-05', '2010-03-12']),
        'Holiday_Flag': [0, 0],
        'Temperature': [42.31, 38.51],
        'Fuel_Price': [2.572, 2.548],
        'CPI': [211.096, 211.242],
        'Unemployment': [8.106, 8.106]
    })
    
    result = prepare_features(df)
    
    assert 'Year' in result.columns
    assert 'Month' in result.columns
    assert 'WeekOfYear' in result.columns
    assert 'Quarter' in result.columns


def test_prepare_features_week_number():
    """Test that prepare_features correctly computes time features for 2010-02-05."""
    df = pd.DataFrame({
        'Store': [1],
        'Date': pd.to_datetime(['2010-02-05']),
        'Holiday_Flag': [0],
        'Temperature': [42.31],
        'Fuel_Price': [2.572],
        'CPI': [211.096],
        'Unemployment': [8.106]
    })
    
    result = prepare_features(df)
    
    assert result['WeekOfYear'].iloc[0] == 5, f"Expected WeekOfYear=5, got {result['WeekOfYear'].iloc[0]}"
    assert result['Month'].iloc[0] == 2, f"Expected Month=2, got {result['Month'].iloc[0]}"
    assert result['Year'].iloc[0] == 2010, f"Expected Year=2010, got {result['Year'].iloc[0]}"


def test_make_pipeline_returns_pipeline():
    """Test that make_pipeline returns a sklearn Pipeline for each model type."""
    linear_pipe = make_pipeline('linear')
    tree_pipe = make_pipeline('tree')
    forest_pipe = make_pipeline('forest')
    
    assert isinstance(linear_pipe, Pipeline)
    assert isinstance(tree_pipe, Pipeline)
    assert isinstance(forest_pipe, Pipeline)


def test_pipeline_fit_predict():
    """Test that a pipeline can fit and predict on minimal data."""
    # Create 20 rows of synthetic data
    np.random.seed(42)
    n_rows = 20
    df = pd.DataFrame({
        'Store': np.random.randint(1, 5, n_rows),
        'Holiday_Flag': np.random.randint(0, 2, n_rows),
        'Temperature': np.random.uniform(30, 80, n_rows),
        'Fuel_Price': np.random.uniform(2, 4, n_rows),
        'CPI': np.random.uniform(200, 220, n_rows),
        'Unemployment': np.random.uniform(5, 10, n_rows),
        'Year': [2010] * n_rows,
        'Month': np.random.randint(1, 13, n_rows),
        'WeekOfYear': np.random.randint(1, 53, n_rows),
        'Quarter': np.random.randint(1, 5, n_rows)
    })
    y = np.random.uniform(10000, 50000, n_rows)
    
    pipeline = make_pipeline('linear')
    pipeline.fit(df, y)
    predictions = pipeline.predict(df)
    
    assert predictions.shape == (n_rows,), f"Expected shape ({n_rows},), got {predictions.shape}"
