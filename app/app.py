"""
Streamlit App — Walmart Weekly Sales Predictor
Page 1: Predict weekly sales for a given store and date.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import date

# --- Page config ---
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

# --- Load model and data ---
@st.cache_resource
def load_model():
    """Load the saved model pipeline. Cached so it only loads once."""
    return joblib.load('models/best_model.joblib')

@st.cache_data
def load_data():
    """Load the cleaned dataset for comparison with actual values."""
    return pd.read_csv('data/cleaned.csv', parse_dates=['Date'])

@st.cache_data
def load_feature_info():
    """Load feature info (column names, model type)."""
    with open('models/feature_info.json', 'r') as f:
        return json.load(f)

# Try loading everything; show error if files are missing
try:
    model = load_model()
    df = load_data()
    feature_info = load_feature_info()
except FileNotFoundError as e:
    st.error(f"❌ Could not load required file: {e}")
    st.info("Make sure you've run the modeling notebook (02_modeling.ipynb) first.")
    st.stop()

# --- Header ---
st.title("🛒 Walmart Weekly Sales Predictor")
st.markdown(f"**Model:** {feature_info['best_model']} | "
            f"**Data range:** {df['Date'].min().date()} to {df['Date'].max().date()}")

st.warning("⚠️ **Limitation:** This model was trained on data from 2010–2012 only. "
           "Tree-based models cannot extrapolate beyond the training data range, "
           "and even Linear Regression may not be accurate outside this period. "
           "The date picker is limited to the dataset's date range.")

# --- Input form ---
st.header("Enter Prediction Inputs")

col1, col2 = st.columns(2)

with col1:
    # Store selector
    store = st.selectbox("Store Number", options=list(range(1, 46)),
                         help="Walmart store number (1–45)")
    
    # Date picker — limited to dataset range
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    selected_date = st.date_input("Week Date", value=date(2012, 6, 1),
                                  min_value=min_date, max_value=max_date,
                                  help="Pick a date within Feb 2010 – Oct 2012")
    
    # Holiday flag
    holiday_flag = st.selectbox("Holiday Week?", options=[0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                help="1 if this is a holiday week (Super Bowl, Labor Day, Thanksgiving, Christmas)")

with col2:
    # External features with sensible defaults
    temperature = st.number_input("Temperature (°F)", min_value=-10.0,
                                  max_value=120.0, value=60.0, step=1.0,
                                  help="Regional temperature in Fahrenheit")
    fuel_price = st.number_input("Fuel Price ($)", min_value=1.0,
                                 max_value=6.0, value=3.5, step=0.01,
                                 help="Regional fuel price per gallon")
    cpi = st.number_input("CPI", min_value=100.0, max_value=250.0,
                          value=200.0, step=0.1,
                          help="Consumer Price Index")
    unemployment = st.number_input("Unemployment Rate (%)", min_value=0.0,
                                   max_value=20.0, value=7.5, step=0.1,
                                   help="Regional unemployment rate")

# --- Derive time features from the selected date ---
year = selected_date.year
month = selected_date.month
week_of_year = selected_date.isocalendar()[1]
quarter = (month - 1) // 3 + 1

# --- Predict button ---
if st.button("🔮 Predict Weekly Sales", type="primary"):
    # Build the input DataFrame matching the feature columns
    input_data = pd.DataFrame([{
        'Store': store,
        'Holiday_Flag': holiday_flag,
        'Temperature': temperature,
        'Fuel_Price': fuel_price,
        'CPI': cpi,
        'Unemployment': unemployment,
        'Year': year,
        'Month': month,
        'WeekOfYear': week_of_year,
        'Quarter': quarter
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.success(f"### 💰 Predicted Weekly Sales: **${prediction:,.2f}**")
    
    # Check if this store+date exists in the actual data
    actual_row = df[(df['Store'] == store) &
                    (df['Date'].dt.date == selected_date)]
    
    if len(actual_row) > 0:
        actual_sales = actual_row['Weekly_Sales'].values[0]
        error = actual_sales - prediction
        pct_error = abs(error) / actual_sales * 100
        
        st.info(f"**Actual sales for this store-week:** ${actual_sales:,.2f}\n\n"
                f"**Error:** ${error:,.2f} ({pct_error:.1f}%)")
    else:
        st.info("ℹ️ No actual data exists for this exact store + date combination.")
    
    # Show derived features
    with st.expander("View derived features"):
        st.write(f"**Year:** {year} | **Month:** {month} | "
                 f"**Week of Year:** {week_of_year} | **Quarter:** {quarter}")
        st.dataframe(input_data)
