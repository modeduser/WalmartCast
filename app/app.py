"""WalmartCast Home page."""
import os
import sys

# Ensure project root on path so src/ and app/ are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from ui import load_css, page_header, kpi_row, callout, chart_layout
from loaders import load_cleaned_data, load_metrics, load_feature_info

st.set_page_config(page_title='WalmartCast', page_icon='📈', layout='wide')
load_css()

# Load data
df = load_cleaned_data()
metrics_df = load_metrics()
feature_info = load_feature_info()

# Page header
page_header('WalmartCast', 'Sales forecasting for Walmart stores using historical data')

# KPI Row
stores_count = df['Store'].nunique()
records_count = len(df)
date_min = df['Date'].min().strftime('%b %Y')
date_max = df['Date'].max().strftime('%b %Y')
date_range = f"{date_min} – {date_max}"

# Best model info from metrics
best_model_name = feature_info['best_model_name']
best_r2 = metrics_df.loc[metrics_df['Model'] == best_model_name, 'R2'].values[0]
best_model_display = f"{best_model_name} | R² {best_r2:.4f}"

kpi_row([
    {'label': 'Stores', 'value': stores_count},
    {'label': 'Weekly Records', 'value': f"{records_count:,}"},
    {'label': 'Date Range', 'value': date_range},
    {'label': 'Best Model', 'value': best_model_display}
])

st.markdown('---')

# Chart 1: Total weekly sales over time
st.markdown('### Total Weekly Sales Over Time')
with st.container(border=True):
    # Aggregate sales by date
    sales_by_date = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    # Get holiday dates
    holiday_dates = df[df['Holiday_Flag'] == 1]['Date'].unique()
    
    # Create line chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=sales_by_date['Date'],
        y=sales_by_date['Weekly_Sales'],
        mode='lines',
        line=dict(color='#2563EB', width=2),
        name='Weekly Sales'
    ))
    
    # Add holiday markers
    for holiday_date in holiday_dates:
        fig1.add_vline(
            x=holiday_date,
            line_dash='dot',
            line_color='#F59E0B',
            opacity=0.5
        )
    
    fig1.update_layout(
        title='Total Weekly Sales Across All Stores',
        xaxis_title='Week',
        yaxis_title='Weekly Sales ($)',
        showlegend=False
    )
    fig1 = chart_layout(fig1)
    st.plotly_chart(fig1, use_container_width=True)

st.markdown('---')

# Chart 2: Average sales by month
st.markdown('### Average Weekly Sales by Month')
with st.container(border=True):
    # Aggregate sales by month
    sales_by_month = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
    
    # Month names for x-axis labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create bar chart
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=sales_by_month['Month'],
        y=sales_by_month['Weekly_Sales'],
        marker_color='#2563EB',
        name='Average Sales'
    ))
    
    fig2.update_layout(
        title='Average Weekly Sales by Month',
        xaxis_title='Month',
        yaxis_title='Weekly Sales ($)',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=month_names
        ),
        showlegend=False
    )
    fig2 = chart_layout(fig2)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('---')

# How it works section
st.markdown('### How It Works')
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown('#### 🔍 Clean & Explore')
        st.markdown('Load Walmart.csv, verify data quality, add time features, explore sales patterns')

with col2:
    with st.container(border=True):
        st.markdown('#### 🤖 Train 3 Models')
        st.markdown('Linear Regression, Decision Tree, Random Forest — trained on past weeks, tested on future weeks')

with col3:
    with st.container(border=True):
        st.markdown('#### 📊 Predict & Compare')
        st.markdown('Choose a store and week; the best model predicts sales; compare against actuals')

st.markdown('---')

# Callout with limitations
callout('info', 'Limitations: weekly store-level predictions only. Data covers 2010–2012. Models cannot extrapolate beyond this period. Temperature, fuel, CPI and unemployment have weak individual correlations with sales.')
