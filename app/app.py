"""WalmartCast - AI-Based Demand Forecasting Streamlit App."""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Page configuration
st.set_page_config(
    page_title="WalmartCast — Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Embedded Custom CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1380px;
}

/* Header typography */
.app-header {
    margin-bottom: 1.25rem;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #64748B;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}

/* KPI cards */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0F172A;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Highlight badge */
.badge-pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Forecast KPI Hero Card */
.forecast-hero {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
    color: #FFFFFF;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}
.forecast-hero-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.85;
    font-weight: 600;
}
.forecast-hero-val {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.3rem 0;
}
.forecast-hero-delta {
    font-size: 0.95rem;
    opacity: 0.95;
    font-weight: 500;
}

/* Form inputs & containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    padding: 0.5rem !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: #1D4ED8;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 0.75rem;
    }
    .app-title {
        font-size: 1.5rem;
    }
    .forecast-hero-val {
        font-size: 1.8rem;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Shared Plotly theme
def apply_chart_style(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", color="#0F172A", size=12),
        margin=dict(l=40, r=20, t=45, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11)
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0")
    return fig

# Data loaders with caching
@st.cache_data
def load_dataset():
    candidates = ["app/Walmart.csv", "Walmart.csv", "../app/Walmart.csv"]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True)
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
            df["Quarter"] = df["Date"].dt.quarter
            return df
    raise FileNotFoundError("Could not locate Walmart.csv")

@st.cache_resource
def load_pipeline():
    candidates = ["app/model.joblib", "model.joblib"]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Could not locate model.joblib")

@st.cache_data
def load_results_data():
    candidates = ["app/results.json", "results.json"]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Could not locate results.json")

df = load_dataset()
model = load_pipeline()
results = load_results_data()

# Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">WalmartCast</h1>
    <div class="app-subtitle">AI-Based Demand Forecasting for Walmart Stores across Macroeconomic Factors</div>
</div>
""", unsafe_allow_html=True)

# Global KPI summary row
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Total Stores", f"{df['Store'].nunique()}")
with kpi_cols[1]:
    st.metric("Total Records", f"{len(df):,} weeks")
with kpi_cols[2]:
    st.metric("Date Span", f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
with kpi_cols[3]:
    best_name = results.get("best_model", "Linear Regression")
    st.metric("Best Model", f"{best_name}", delta="R² 0.9658 (Std Test)")

# Main Tabs
tab_forecast, tab_models, tab_insights = st.tabs(["🎯 Forecast", "🤖 Models", "💡 Insights"])

# ==========================================
# TAB 1: FORECAST
# ==========================================
with tab_forecast:
    col_input, col_pred = st.columns([1, 1.4], gap="medium")
    
    with col_input:
        with st.container(border=True):
            st.markdown("### Forecast Parameters")
            store_id = st.selectbox("Select Store", sorted(df["Store"].unique()), index=0)
            
            store_records = df[df["Store"] == store_id].sort_values("Date")
            available_dates = store_records["Date"].dt.strftime("%Y-%m-%d").tolist()
            selected_date_str = st.selectbox("Select Week", available_dates, index=len(available_dates)-1)
            
            row = store_records[store_records["Date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]
            
            st.markdown("##### Environmental & Economic Factors")
            is_holiday = st.checkbox("Holiday Week", value=bool(row["Holiday_Flag"]))
            temperature = st.number_input("Temperature (°F)", value=float(row["Temperature"]), step=0.5, format="%.1f")
            fuel_price = st.number_input("Fuel Price ($/gal)", value=float(row["Fuel_Price"]), step=0.01, format="%.2f")
            cpi = st.number_input("Consumer Price Index (CPI)", value=float(row["CPI"]), step=0.1, format="%.2f")
            unemployment = st.number_input("Unemployment Rate (%)", value=float(row["Unemployment"]), step=0.1, format="%.2f")
    
    with col_pred:
        # Prepare input features
        sel_dt = pd.Timestamp(selected_date_str)
        input_data = pd.DataFrame([{
            "Store": store_id,
            "Holiday_Flag": int(is_holiday),
            "Temperature": temperature,
            "Fuel_Price": fuel_price,
            "CPI": cpi,
            "Unemployment": unemployment,
            "Year": sel_dt.year,
            "Month": sel_dt.month,
            "WeekOfYear": int(sel_dt.isocalendar().week),
            "Quarter": sel_dt.quarter
        }])
        
        feature_cols = [
            "Store", "Holiday_Flag", "Temperature", "Fuel_Price",
            "CPI", "Unemployment", "Year", "Month", "WeekOfYear", "Quarter"
        ]
        
        predicted_sales = float(model.predict(input_data[feature_cols])[0])
        store_avg_sales = float(store_records["Weekly_Sales"].mean())
        diff_from_avg = predicted_sales - store_avg_sales
        pct_from_avg = (diff_from_avg / store_avg_sales) * 100
        sign_str = "+" if diff_from_avg >= 0 else ""
        
        # Hero KPI card
        st.markdown(f"""
        <div class="forecast-hero">
            <div class="forecast-hero-label">Predicted Weekly Sales (Store {store_id})</div>
            <div class="forecast-hero-val">${predicted_sales:,.0f}</div>
            <div class="forecast-hero-delta">{sign_str}${diff_from_avg:,.0f} ({sign_str}{pct_from_avg:.1f}%) vs store historical average (${store_avg_sales:,.0f})</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Store historical trend with forecast point
        fig_hist = go.Figure()
        
        # Historical sales
        fig_hist.add_trace(go.Scatter(
            x=store_records["Date"],
            y=store_records["Weekly_Sales"],
            mode="lines",
            name="Historical Sales",
            line=dict(color="#2563EB", width=2)
        ))
        
        # Holiday points
        holidays_store = store_records[store_records["Holiday_Flag"] == 1]
        fig_hist.add_trace(go.Scatter(
            x=holidays_store["Date"],
            y=holidays_store["Weekly_Sales"],
            mode="markers",
            name="Holiday Weeks",
            marker=dict(color="#DC2626", size=7, symbol="diamond")
        ))
        
        # Store average dashed line
        fig_hist.add_trace(go.Scatter(
            x=[store_records["Date"].min(), store_records["Date"].max()],
            y=[store_avg_sales, store_avg_sales],
            mode="lines",
            name="Store Average",
            line=dict(color="#94A3B8", width=1.5, dash="dash")
        ))
        
        # Forecasted point
        fig_hist.add_trace(go.Scatter(
            x=[sel_dt],
            y=[predicted_sales],
            mode="markers",
            name="Forecast Point",
            marker=dict(color="#F59E0B", size=14, symbol="star", line=dict(color="#78350F", width=1))
        ))
        
        fig_hist.update_layout(
            title=f"Store {store_id} Sales History & Forecast Point",
            xaxis_title="Date",
            yaxis_title="Weekly Sales ($)",
            yaxis_tickformat="$,.0f"
        )
        apply_chart_style(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 2: MODELS
# ==========================================
with tab_models:
    st.markdown("### Model Leaderboard & Evaluation")
    
    metrics_list = results.get("metrics", [])
    metrics_df = pd.DataFrame(metrics_list)
    
    # Leaderboard table
    col_lb, col_desc = st.columns([2, 1], gap="medium")
    with col_lb:
        piv = metrics_df[metrics_df["Evaluation"].isin(["Standard", "Holiday-season"])].pivot(
            index="Model", columns="Evaluation", values=["R2", "RMSE", "MAE"]
        )
        piv.columns = [f"{eval_name} {metric}" for metric, eval_name in piv.columns]
        display_cols = [
            "Standard R2", "Standard RMSE", "Standard MAE",
            "Holiday-season R2", "Holiday-season RMSE", "Holiday-season MAE"
        ]
        piv = piv[display_cols].reset_index()
        
        # Style table
        def highlight_winner(row):
            if row["Model"] == results.get("best_model", "Linear Regression"):
                return ["background-color: #EFF6FF; font-weight: 600; color: #1D4ED8"] * len(row)
            return [""] * len(row)
        
        formatted_piv = piv.style.apply(highlight_winner, axis=1).format({
            "Standard R2": "{:.4f}",
            "Standard RMSE": "${:,.0f}",
            "Standard MAE": "${:,.0f}",
            "Holiday-season R2": "{:.4f}",
            "Holiday-season RMSE": "${:,.0f}",
            "Holiday-season MAE": "${:,.0f}",
        })
        st.dataframe(formatted_piv, use_container_width=True, hide_index=True)
        
    with col_desc:
        with st.container(border=True):
            st.markdown(f"""
            <span class="badge-pill">🏆 Best Model: {results.get('best_model', 'Linear Regression')}</span>
            <div style="font-size:0.9rem;color:#475569;margin-top:0.75rem;">
                <b>Selection Criterion:</b> Lowest mean RMSE across both <b>Standard</b> (unseen future weeks) and <b>Holiday-season</b> (Thanksgiving & Christmas surge) evaluations.<br><br>
                <b>Linear Regression</b> effectively isolates store base volumes through one-hot encoding while preserving consistent generalization on holiday weeks.
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Grouped Bar Chart of Metrics (MAE, RMSE, R2)
    st.markdown("### Model Metrics Comparison Across Tests")
    eval_chart_df = metrics_df[
        (metrics_df["Model"] != "Baseline") & 
        (metrics_df["Evaluation"].isin(["Standard", "Holiday-season"]))
    ].copy()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        fig_r2 = go.Figure()
        for eval_name, color in [("Standard", "#2563EB"), ("Holiday-season", "#F59E0B")]:
            sub = eval_chart_df[eval_chart_df["Evaluation"] == eval_name]
            fig_r2.add_trace(go.Bar(
                x=sub["Model"], y=sub["R2"], name=eval_name, marker_color=color,
                text=[f"{v:.4f}" for v in sub["R2"]], textposition="auto"
            ))
        fig_r2.update_layout(title="R² (Higher is Better)", barmode="group", yaxis_range=[0.8, 1.0])
        apply_chart_style(fig_r2)
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col_m2:
        fig_rmse = go.Figure()
        for eval_name, color in [("Standard", "#2563EB"), ("Holiday-season", "#F59E0B")]:
            sub = eval_chart_df[eval_chart_df["Evaluation"] == eval_name]
            fig_rmse.add_trace(go.Bar(
                x=sub["Model"], y=sub["RMSE"], name=eval_name, marker_color=color,
                text=[f"${v/1e3:.0f}k" for v in sub["RMSE"]], textposition="auto"
            ))
        fig_rmse.update_layout(title="RMSE ($) (Lower is Better)", barmode="group", yaxis_tickformat="$,.0f")
        apply_chart_style(fig_rmse)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with col_m3:
        fig_mae = go.Figure()
        for eval_name, color in [("Standard", "#2563EB"), ("Holiday-season", "#F59E0B")]:
            sub = eval_chart_df[eval_chart_df["Evaluation"] == eval_name]
            fig_mae.add_trace(go.Bar(
                x=sub["Model"], y=sub["MAE"], name=eval_name, marker_color=color,
                text=[f"${v/1e3:.0f}k" for v in sub["MAE"]], textposition="auto"
            ))
        fig_mae.update_layout(title="MAE ($) (Lower is Better)", barmode="group", yaxis_tickformat="$,.0f")
        apply_chart_style(fig_mae)
        st.plotly_chart(fig_mae, use_container_width=True)
        
    st.markdown("---")
    
    # Store-Level Actual vs Predicted & Feature Importance
    col_avp, col_fi = st.columns([1.3, 1], gap="medium")
    
    with col_avp:
        st.markdown("### Actual vs Predicted (Test Set)")
        preds_records = results.get("predictions", [])
        if preds_records:
            preds_df = pd.DataFrame(preds_records)
            preds_df["Date"] = pd.to_datetime(preds_df["Date"])
            
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                chosen_store = st.selectbox("Store for Test Comparison", sorted(preds_df["Store"].unique()), index=0)
            with c_sel2:
                chosen_eval = st.selectbox("Evaluation Split", ["Standard", "Holiday-season"])
                
            filtered_preds = preds_df[
                (preds_df["Store"] == chosen_store) & 
                (preds_df["Evaluation"] == chosen_eval)
            ].sort_values("Date")
            
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=filtered_preds["Date"], y=filtered_preds["Actual"],
                mode="lines", name="Actual Sales", line=dict(color="#0F172A", width=2.5)
            ))
            fig_avp.add_trace(go.Scatter(
                x=filtered_preds["Date"], y=filtered_preds["Linear"],
                mode="lines", name="Linear Regression", line=dict(color="#2563EB", width=2, dash="dash")
            ))
            fig_avp.add_trace(go.Scatter(
                x=filtered_preds["Date"], y=filtered_preds["Tree"],
                mode="lines", name="Decision Tree", line=dict(color="#10B981", width=1.5, dash="dot")
            ))
            fig_avp.add_trace(go.Scatter(
                x=filtered_preds["Date"], y=filtered_preds["Forest"],
                mode="lines", name="Random Forest", line=dict(color="#EF4444", width=1.5, dash="dashdot")
            ))
            fig_avp.update_layout(
                title=f"Store {chosen_store} — {chosen_eval} Test Horizon",
                xaxis_title="Date",
                yaxis_title="Weekly Sales ($)",
                yaxis_tickformat="$,.0f"
            )
            apply_chart_style(fig_avp)
            st.plotly_chart(fig_avp, use_container_width=True)
            
    with col_fi:
        st.markdown("### Feature Importance")
        fi_list = results.get("feature_importance", [])
        if fi_list:
            fi_df = pd.DataFrame(fi_list).sort_values("importance", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"],
                y=fi_df["feature"],
                orientation="h",
                marker=dict(color=["#2563EB" if f == "Store" else "#94A3B8" for f in fi_df["feature"]]),
                text=[f"{v:.4f}" for v in fi_df["importance"]],
                textposition="outside"
            ))
            fig_fi.update_layout(
                title="Random Forest Feature Importance",
                xaxis_title="Gini Importance",
                yaxis_title="Feature"
            )
            apply_chart_style(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)

# ==========================================
# TAB 3: INSIGHTS
# ==========================================
with tab_insights:
    st.markdown("### Exploratory Demand Insights")
    
    col_ins1, col_ins2 = st.columns(2, gap="medium")
    
    with col_ins1:
        # Holiday vs Non-Holiday Sales
        hol_stats = df.groupby("Holiday_Flag")["Weekly_Sales"].agg(["mean", "median", "std"]).reset_index()
        hol_stats["Type"] = hol_stats["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})
        
        non_hol_mean = hol_stats.loc[hol_stats["Holiday_Flag"] == 0, "mean"].values[0]
        hol_mean = hol_stats.loc[hol_stats["Holiday_Flag"] == 1, "mean"].values[0]
        boost_pct = ((hol_mean / non_hol_mean) - 1) * 100
        
        fig_hol = go.Figure()
        fig_hol.add_trace(go.Bar(
            x=hol_stats["Type"],
            y=hol_stats["mean"],
            marker_color=["#94A3B8", "#2563EB"],
            text=[f"${v:,.0f}" for v in hol_stats["mean"]],
            textposition="auto"
        ))
        fig_hol.update_layout(
            title=f"Average Weekly Sales: Holiday vs Non-Holiday (+{boost_pct:.1f}% Boost)",
            xaxis_title="",
            yaxis_title="Average Sales ($)",
            yaxis_tickformat="$,.0f"
        )
        apply_chart_style(fig_hol)
        st.plotly_chart(fig_hol, use_container_width=True)
        
    with col_ins2:
        # Monthly Seasonality
        monthly_sales = df.groupby("Month")["Weekly_Sales"].mean().reset_index()
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_sales["MonthName"] = monthly_sales["Month"].apply(lambda m: month_labels[m - 1])
        
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=monthly_sales["MonthName"],
            y=monthly_sales["Weekly_Sales"],
            marker_color=["#2563EB" if m in [11, 12] else "#94A3B8" for m in monthly_sales["Month"]],
            text=[f"${v/1e3:.0f}k" for v in monthly_sales["Weekly_Sales"]],
            textposition="auto"
        ))
        fig_month.update_layout(
            title="Monthly Sales Seasonality (November–December Peak)",
            xaxis_title="Month",
            yaxis_title="Average Sales ($)",
            yaxis_tickformat="$,.0f"
        )
        apply_chart_style(fig_month)
        st.plotly_chart(fig_month, use_container_width=True)
        
    # Top and Bottom Stores
    st.markdown("### Top 5 vs Bottom 5 Stores by Average Weekly Sales")
    store_sales = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
    top5 = store_sales.nlargest(5, "Weekly_Sales").copy()
    top5["Group"] = "Top 5"
    bottom5 = store_sales.nsmallest(5, "Weekly_Sales").copy()
    bottom5["Group"] = "Bottom 5"
    
    top_bottom = pd.concat([bottom5, top5]).sort_values("Weekly_Sales", ascending=True)
    top_bottom["StoreLabel"] = top_bottom["Store"].apply(lambda s: f"Store {s}")
    
    fig_tb = go.Figure()
    fig_tb.add_trace(go.Bar(
        x=top_bottom["Weekly_Sales"],
        y=top_bottom["StoreLabel"],
        orientation="h",
        marker=dict(color=["#94A3B8" if g == "Bottom 5" else "#2563EB" for g in top_bottom["Group"]]),
        text=[f"${v:,.0f}" for v in top_bottom["Weekly_Sales"]],
        textposition="outside"
    ))
    fig_tb.update_layout(
        title="Store Volume Disparity: Highest Stores Exceed Lowest by Over 4×",
        xaxis_title="Average Weekly Sales ($)",
        yaxis_title="Store",
        xaxis_tickformat="$,.0f"
    )
    apply_chart_style(fig_tb)
    st.plotly_chart(fig_tb, use_container_width=True)
