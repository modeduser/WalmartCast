"""WalmartCast Predict page."""
import os
import sys

# Ensure project root and app dir are on path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_APP_DIR = os.path.join(_ROOT, 'app')
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from ui import load_css, page_header, chart_layout
from loaders import load_cleaned_data, load_model
from src.features import get_feature_columns

st.set_page_config(page_title='Predict — WalmartCast', page_icon='🔮', layout='wide')
load_css()
page_header('Predict Sales', 'Enter store conditions to forecast weekly sales')

# Load data and model
cleaned_df = load_cleaned_data()
model = load_model()

# Two-column layout
col_left, col_right = st.columns([1, 1.2], gap='large')

# LEFT COLUMN: INPUTS
with col_left:
    with st.container(border=True):
        st.markdown('#### Inputs')
        
        # Store selection
        store = st.selectbox('Store', range(1, 46), index=0)
        
        # Get all unique dates for this store, sorted
        week_dates = sorted(cleaned_df[cleaned_df['Store'] == store]['Date'].unique())
        
        # Date range from dataset
        min_date = cleaned_df['Date'].min()
        max_date = cleaned_df['Date'].max()
        
        # Date input
        selected_date = st.date_input(
            'Week',
            value=min_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        # Convert selected_date to Timestamp and find nearest available date
        selected_timestamp = pd.Timestamp(selected_date)
        
        # Find nearest date in week_dates
        nearest_date = min(week_dates, key=lambda d: abs((d - selected_timestamp).days))
        
        st.caption(f"Nearest available week: {nearest_date.strftime('%Y-%m-%d')}")
        
        # Prefill all other inputs from cleaned.csv for this store+date
        matching_rows = cleaned_df[(cleaned_df['Store'] == store) & (cleaned_df['Date'] == nearest_date)]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            
            holiday = st.checkbox('Holiday week', value=bool(row['Holiday_Flag']))
            temp = st.number_input('Temperature (°F)', value=float(row['Temperature']), step=1.0, format='%.1f')
            fuel = st.number_input('Fuel Price ($/gal)', value=float(row['Fuel_Price']), step=0.01, format='%.2f')
            cpi = st.number_input('CPI', value=float(row['CPI']), step=0.1, format='%.2f')
            unemp = st.number_input('Unemployment (%)', value=float(row['Unemployment']), step=0.1, format='%.2f')
        else:
            # Fallback defaults if no matching row found
            holiday = st.checkbox('Holiday week', value=False)
            temp = st.number_input('Temperature (°F)', value=60.0, step=1.0, format='%.1f')
            fuel = st.number_input('Fuel Price ($/gal)', value=3.00, step=0.01, format='%.2f')
            cpi = st.number_input('CPI', value=200.0, step=0.1, format='%.2f')
            unemp = st.number_input('Unemployment (%)', value=8.0, step=0.1, format='%.2f')
        
        predict_btn = st.button('Predict Sales', type='primary', use_container_width=True)

# RIGHT COLUMN: RESULTS
with col_right:
    # Initialize session state for prediction persistence
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'pred_store' not in st.session_state:
        st.session_state['pred_store'] = None
    if 'pred_date' not in st.session_state:
        st.session_state['pred_date'] = None
    
    # Clear prediction if store or date changed
    if st.session_state['pred_store'] != store or st.session_state['pred_date'] != nearest_date:
        st.session_state['prediction'] = None
    
    # If predict button clicked, make prediction
    if predict_btn:
        # Build input DataFrame for the model
        input_date = pd.Timestamp(nearest_date)
        input_df = pd.DataFrame([{
            'Store': store,
            'Holiday_Flag': int(holiday),
            'Temperature': temp,
            'Fuel_Price': fuel,
            'CPI': cpi,
            'Unemployment': unemp,
            'Year': input_date.year,
            'Month': input_date.month,
            'WeekOfYear': int(input_date.isocalendar()[1]),
            'Quarter': input_date.quarter
        }])
        
        # Get feature columns and prepare input
        feature_cols = get_feature_columns()
        X = input_df[feature_cols]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Store in session state
        st.session_state['prediction'] = prediction
        st.session_state['pred_store'] = store
        st.session_state['pred_date'] = nearest_date
    
    # Display results
    if st.session_state['prediction'] is None:
        st.info('Set the inputs on the left and click Predict Sales.')
    else:
        prediction = st.session_state['prediction']
        
        with st.container(border=True):
            # Calculate store average
            store_avg = cleaned_df[cleaned_df['Store'] == store]['Weekly_Sales'].mean()
            
            # Display prediction metric
            st.metric(
                'Predicted Weekly Sales',
                f'${prediction:,.0f}',
                delta=f'vs store avg: ${prediction - store_avg:+,.0f}'
            )
            
            # Check if this week exists in data
            actual_rows = cleaned_df[(cleaned_df['Store'] == store) & (cleaned_df['Date'] == nearest_date)]
            
            if len(actual_rows) > 0:
                actual_sales = actual_rows.iloc[0]['Weekly_Sales']
                st.metric(
                    'Actual Sales (from data)',
                    f'${actual_sales:,.0f}',
                    delta=f'Difference: ${prediction - actual_sales:+,.0f}'
                )
                st.caption('⚠️ This week was part of the model\'s training data. See Model Results for honest held-out accuracy.')
            
            # Line chart: store history + predicted point
            store_hist = cleaned_df[cleaned_df['Store'] == store].sort_values('Date')
            
            fig = go.Figure()
            
            # Historical sales line
            fig.add_trace(go.Scatter(
                x=store_hist['Date'],
                y=store_hist['Weekly_Sales'],
                mode='lines',
                name='Historical',
                line=dict(color='#2563EB', width=2)
            ))
            
            # Predicted point
            input_date = pd.Timestamp(nearest_date)
            fig.add_trace(go.Scatter(
                x=[input_date],
                y=[prediction],
                mode='markers',
                name='Prediction',
                marker=dict(color='#F59E0B', size=12, symbol='star')
            ))
            
            fig.update_layout(
                title=f'Store {store} Weekly Sales History',
                xaxis_title='Week',
                yaxis_title='Weekly Sales ($)'
            )
            
            chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption('Model predictions are based on 2010–2012 training data and may not reflect current conditions.')
