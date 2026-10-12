"""
Streamlit Page 2 — Model Comparison
Shows the metrics table, RMSE/R² bar charts, and feature importance.
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Model Comparison")
st.markdown("Comparing 3 regression models across two evaluation splits.")

# --- Load metrics ---
try:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    metrics_df = pd.DataFrame(metrics)
except FileNotFoundError:
    st.error("❌ Could not find models/metrics.json. Run the modeling notebook first.")
    st.stop()

# --- Metrics table ---
st.header("Results Table")
display_df = metrics_df.copy()
display_df['MAE'] = display_df['MAE'].apply(lambda x: f'${x:,.0f}')
display_df['MSE'] = display_df['MSE'].apply(lambda x: f'{x:,.0f}')
display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f'${x:,.0f}')
display_df['R2'] = display_df['R2'].apply(lambda x: f'{x:.4f}')
st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Bar charts ---
st.header("Visual Comparison")

col1, col2 = st.columns(2)

with col1:
    # RMSE bar chart
    fig_rmse = px.bar(metrics_df, x='Model', y='RMSE', color='Split',
                      barmode='group', title='RMSE by Model and Split',
                      labels={'RMSE': 'RMSE ($)'})
    fig_rmse.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    # R² bar chart
    fig_r2 = px.bar(metrics_df, x='Model', y='R2', color='Split',
                    barmode='group', title='R² by Model and Split',
                    labels={'R2': 'R² Score'})
    fig_r2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_r2, use_container_width=True)

# --- Feature importance ---
st.header("Random Forest Feature Importance")
st.markdown("Shows which features the Random Forest model considers most important. "
            "Store dummies dominate because *which store* matters most for sales.")

# Display the saved feature importance figure
fig_path = 'report/figures/model_feature_importance.png'
if os.path.exists(fig_path):
    st.image(fig_path, use_container_width=True)
else:
    st.warning("Feature importance figure not found. Run the modeling notebook first.")

# --- Explanation ---
st.header("Analysis")
st.markdown("""
### Why does Linear Regression win?

When we **one-hot encode** the Store column, Linear Regression gets 45 separate coefficients — one per store. 
This lets it learn each store's baseline sales level precisely. Combined with Month and WeekOfYear features, 
LR captures the two main patterns: **which store** and **what time of year**.

The Decision Tree and Random Forest models were configured with limited depth (`max_depth=8` and `max_depth=12`), 
which restricts how finely they can split across 45 stores AND time features simultaneously. With hyperparameter 
tuning (deeper trees, more estimators), they could potentially compete.

### Time-based vs Random split

The time-based split is the **fairer evaluation** because it simulates real forecasting: train on the past, 
predict the future. The random split is "easier" because neighbouring weeks from the same store can leak 
into both train and test sets. However, in this case, LR actually performed slightly better on the time-based 
split — likely because the test period (Apr–Oct 2012) doesn't include the volatile holiday season.
""")
