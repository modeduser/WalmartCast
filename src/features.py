"""Feature engineering and pipeline builders for WalmartCast models.

This module contains reusable feature transformation logic that is used by both
the training notebooks and the Streamlit app to ensure consistency.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def prepare_features(df):
    """
    Prepare features for modeling.
    
    Args:
        df: DataFrame with at least Store, Date, Holiday_Flag, Temperature,
            Fuel_Price, CPI, Unemployment columns
    
    Returns:
        DataFrame with time features added if not already present
    """
    df = df.copy()
    
    # Add time features if they don't exist
    if 'Year' not in df.columns:
        df['Year'] = df['Date'].dt.year
    if 'Month' not in df.columns:
        df['Month'] = df['Date'].dt.month
    if 'WeekOfYear' not in df.columns:
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    if 'Quarter' not in df.columns:
        df['Quarter'] = df['Date'].dt.quarter
    
    return df


def get_feature_columns():
    """
    Returns the list of feature columns used for modeling.
    
    Returns:
        list: Feature column names
    """
    return ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 
            'CPI', 'Unemployment', 'Year', 'Month', 'WeekOfYear', 'Quarter']


def make_pipeline(model_name, random_state=42):
    """
    Create a scikit-learn pipeline for the specified model.
    
    Args:
        model_name: One of 'linear', 'tree', or 'forest'
        random_state: Random seed for reproducibility
    
    Returns:
        Pipeline object ready for fitting
    """
    if model_name == 'linear':
        # Linear Regression: one-hot encode Store AND WeekOfYear, scale others
        categorical_features = ['Store', 'WeekOfYear']
        numeric_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 
                           'CPI', 'Unemployment', 'Year', 'Month', 'Quarter']
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
    elif model_name == 'tree':
        # Decision Tree: one-hot encode Store only, others pass through
        categorical_features = ['Store']
        numeric_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 
                           'CPI', 'Unemployment', 'Year', 'Month', 'WeekOfYear', 'Quarter']
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', DecisionTreeRegressor(
                max_depth=None,
                min_samples_leaf=3,
                random_state=random_state
            ))
        ])
        
    elif model_name == 'forest':
        # Random Forest: one-hot encode Store only, others pass through
        categorical_features = ['Store']
        numeric_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 
                           'CPI', 'Unemployment', 'Year', 'Month', 'WeekOfYear', 'Quarter']
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'linear', 'tree', or 'forest'.")
    
    return pipeline


def get_model_name(model_name):
    """
    Returns the display name for a model.
    
    Args:
        model_name: Short name ('linear', 'tree', 'forest')
    
    Returns:
        str: Display name
    """
    names = {
        'linear': 'Linear Regression',
        'tree': 'Decision Tree',
        'forest': 'Random Forest'
    }
    return names.get(model_name, model_name)
