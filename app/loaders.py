"""Cached data loaders for WalmartCast app."""
import os
import sys
import json
import pandas as pd
import streamlit as st

# Ensure project root is on path so src/ is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DATA_DIR = os.path.join(_ROOT, 'data')
_MODEL_DIR = os.path.join(_ROOT, 'models')


@st.cache_resource
def load_model():
    """Load the best saved model. Shows friendly error if missing."""
    import joblib
    path = os.path.join(_MODEL_DIR, 'best_model.joblib')
    if not os.path.exists(path):
        st.error('Model file not found. Run notebook 02_modeling.ipynb (Phase 3) to create it.')
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_cleaned_data():
    """Load cleaned.csv. Shows friendly error if missing."""
    path = os.path.join(_DATA_DIR, 'cleaned.csv')
    if not os.path.exists(path):
        st.error('cleaned.csv not found. Run notebook 01_eda_cleaning.ipynb to create it.')
        st.stop()
    df = pd.read_csv(path, parse_dates=['Date'])
    return df


@st.cache_data
def load_metrics():
    """Load models/metrics.json as a DataFrame."""
    path = os.path.join(_MODEL_DIR, 'metrics.json')
    if not os.path.exists(path):
        st.error('metrics.json not found. Run notebook 02_modeling.ipynb to create it.')
        st.stop()
    with open(path) as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def load_test_predictions():
    """Load models/test_predictions.csv."""
    path = os.path.join(_MODEL_DIR, 'test_predictions.csv')
    if not os.path.exists(path):
        st.error('test_predictions.csv not found. Run notebook 02_modeling.ipynb to create it.')
        st.stop()
    df = pd.read_csv(path, parse_dates=['Date'])
    return df


@st.cache_data
def load_feature_info():
    """Load models/feature_info.json."""
    path = os.path.join(_MODEL_DIR, 'feature_info.json')
    if not os.path.exists(path):
        st.error('feature_info.json not found. Run notebook 02_modeling.ipynb to create it.')
        st.stop()
    with open(path) as f:
        return json.load(f)
