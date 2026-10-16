"""Model Results page for WalmartCast Streamlit app."""
import os
import sys

# Add project root and app dir to path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_APP_DIR = os.path.join(_ROOT, 'app')
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly import express as px

from ui import load_css, page_header, chart_layout
from loaders import load_metrics, load_test_predictions, load_feature_info


# Page config
st.set_page_config(page_title='Model Results — WalmartCast', page_icon='📊', layout='wide')
load_css()

# Load data
metrics_df = load_metrics()
preds_df = load_test_predictions()
fi = load_feature_info()

# Determine best model by mean RMSE across Standard and Holiday-season
eval_filter = metrics_df['Evaluation'].isin(['Standard', 'Holiday-season'])
filtered_metrics = metrics_df[eval_filter].copy()
mean_rmse = filtered_metrics.groupby('Model')['RMSE'].mean()
best_model_name = mean_rmse.idxmin()
std_r2 = metrics_df[(metrics_df['Model'] == best_model_name) & (metrics_df['Evaluation'] == 'Standard')]['R2'].values[0]

# Header with best model badge
col1, col2 = st.columns([3, 1])
with col1:
    page_header('Model Results', 'Evaluation across three test strategies')
with col2:
    st.markdown(
        f'<div style="background:#DBEAFE;color:#1D4ED8;padding:0.4rem 0.8rem;border-radius:8px;'
        f'font-weight:600;text-align:center;margin-top:1.5rem">🏆 Best: {best_model_name}</div>',
        unsafe_allow_html=True
    )

# One-sentence summary
hol_r2 = metrics_df[(metrics_df['Model'] == best_model_name) & (metrics_df['Evaluation'] == 'Holiday-season')]['R2'].values[0]
max_std_r2 = metrics_df[metrics_df['Evaluation'] == 'Standard']['R2'].max()
selection_text = "the best" if std_r2 == max_std_r2 else "selected by mean RMSE"
st.markdown(
    f'**{best_model_name}** achieves R² = {std_r2:.4f} on the standard test and {hol_r2:.4f} on the '
    f'holiday-season test — {selection_text} across both evaluations.'
)

# Leaderboard table
st.subheader('Leaderboard')
leaderboard_data = filtered_metrics.copy()
pivot_data = leaderboard_data.pivot(index='Model', columns='Evaluation', values=['R2', 'RMSE']).reset_index()
pivot_data.columns = ['Model', 'Holiday-season R²', 'Standard R²', 'Holiday-season RMSE', 'Standard RMSE']
pivot_data = pivot_data[['Model', 'Standard R²', 'Standard RMSE', 'Holiday-season R²', 'Holiday-season RMSE']]

# Highlight best model
def highlight_best(row):
    if row['Model'] == best_model_name:
        return ['background-color: #DBEAFE'] * len(row)
    return [''] * len(row)

st.dataframe(
    pivot_data.style.apply(highlight_best, axis=1).format({
        'Standard R²': '{:.4f}',
        'Holiday-season R²': '{:.4f}',
        'Standard RMSE': '{:,.0f}',
        'Holiday-season RMSE': '{:,.0f}'
    }),
    use_container_width=True,
    hide_index=True
)

# Expander explaining metrics
with st.expander('What do MAE, MSE, RMSE and R² mean?'):
    st.markdown('''
    - **MAE (Mean Absolute Error)**: Average absolute prediction error in dollars — easy to explain to stakeholders.
    - **MSE (Mean Squared Error)**: Squared error; penalizes large misses more than MAE.
    - **RMSE (Root Mean Squared Error)**: Square root of MSE — same units as sales, easier to interpret than MSE.
    - **R² (R-squared)**: Proportion of sales variation explained by the model (1.0 = perfect, 0 = predicts the mean).
    ''')

# Four tabs
tab1, tab2, tab3, tab4 = st.tabs(['Compare models', 'Actual vs predicted', 'What drives sales', 'Lessons'])

with tab1:
    st.subheader('Compare models')
    col_left, col_right = st.columns(2)
    
    with col_left:
        # RMSE comparison
        fig_rmse = px.bar(
            filtered_metrics,
            x='Model',
            y='RMSE',
            color='Evaluation',
            barmode='group',
            title='RMSE by Model and Evaluation',
            color_discrete_sequence=['#2563EB', '#F59E0B']
        )
        fig_rmse = chart_layout(fig_rmse)
        fig_rmse.update_yaxes(title='RMSE ($)', tickformat='$,.0f')
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col_right:
        # R² comparison
        fig_r2 = px.bar(
            filtered_metrics,
            x='Model',
            y='R2',
            color='Evaluation',
            barmode='group',
            title='R² by Model and Evaluation',
            color_discrete_sequence=['#2563EB', '#F59E0B']
        )
        fig_r2 = chart_layout(fig_r2)
        fig_r2.update_yaxes(title='R² (proportion)', tickformat='.4f')
        st.plotly_chart(fig_r2, use_container_width=True)

with tab2:
    st.subheader('Actual vs predicted')
    
    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        store_sel = st.selectbox('Store', sorted(preds_df['Store'].unique()))
    with col_f2:
        eval_sel = st.selectbox('Evaluation', ['Standard', 'Holiday-season'])
    
    # Filter and plot
    filtered_preds = preds_df[(preds_df['Store'] == store_sel) & (preds_df['Evaluation'] == eval_sel)].copy()
    filtered_preds = filtered_preds.sort_values('Date')
    
    fig_preds = go.Figure()
    fig_preds.add_trace(go.Scatter(
        x=filtered_preds['Date'],
        y=filtered_preds['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#2563EB', width=3)
    ))
    fig_preds.add_trace(go.Scatter(
        x=filtered_preds['Date'],
        y=filtered_preds['Pred_Linear'],
        mode='lines',
        name='Linear Regression',
        line=dict(dash='dot', width=2)
    ))
    fig_preds.add_trace(go.Scatter(
        x=filtered_preds['Date'],
        y=filtered_preds['Pred_Tree'],
        mode='lines',
        name='Decision Tree',
        line=dict(dash='dash', width=2)
    ))
    fig_preds.add_trace(go.Scatter(
        x=filtered_preds['Date'],
        y=filtered_preds['Pred_Forest'],
        mode='lines',
        name='Random Forest',
        line=dict(dash='dashdot', width=2)
    ))
    
    fig_preds.update_layout(
        title=f'Store {store_sel} — {eval_sel} Test',
        xaxis_title='Date',
        yaxis_title='Weekly Sales ($)'
    )
    fig_preds = chart_layout(fig_preds)
    st.plotly_chart(fig_preds, use_container_width=True)
    
    st.caption('Predictions shown are from the held-out test period — the model did not see these weeks during training.')

with tab3:
    st.subheader('What drives sales')
    
    img_path = os.path.join(_ROOT, 'report', 'figures', 'model_feature_importance.png')
    if os.path.exists(img_path):
        st.image(img_path, caption='Random Forest feature importance (aggregated by original feature)', use_container_width=True)
    else:
        st.info('Feature importance figure not found. Run notebook 02_modeling.ipynb to generate it.')
    
    st.caption(
        'Store identity (one-hot encoded) is the strongest predictor — each store has a distinct sales level. '
        'WeekOfYear captures annual seasonality. Holiday_Flag, Temperature and economic variables add smaller '
        'but meaningful signals.'
    )

with tab4:
    st.subheader('Lessons')
    
    img_path = os.path.join(_ROOT, 'report', 'figures', 'diag_rf_depth_sweep.png')
    if os.path.exists(img_path):
        st.image(img_path, caption='Random Forest R² on Standard test vs max tree depth', use_container_width=True)
    else:
        st.info('Depth sweep figure not found. Run notebook 02_modeling.ipynb to generate it.')
    
    st.markdown('''
**Why the holiday-season test matters.** A model evaluated only on ordinary weeks can look excellent (R² > 0.96) while badly missing Thanksgiving and Christmas spikes. Training on just one prior holiday season limits how well any model can learn these patterns.

**Tree depth and one-hot encoding.** With one-hot Store columns, a shallow tree runs out of depth before learning store-level patterns — R² collapses below 0.5 at depth 4. Removing the depth limit allows the tree to first isolate each store, then fit its sales pattern.

**What the models learned.** Store identity dominates: each store has a stable sales level that the model learns. WeekOfYear captures the November–December surge. Economic variables (Temperature, CPI, Fuel, Unemployment) add small marginal signal.
''')
