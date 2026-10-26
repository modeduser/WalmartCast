"""WalmartCast - AI-Based Demand Forecasting Streamlit App.

Clean, high-performance UI for store-level demand forecasting
and macroeconomic scenario simulation (Light Mode).
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WalmartCast — Demand Forecasting",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# LIGHT MODE DESIGN SYSTEM & CUSTOM CSS
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base reset & typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    color: #0F172A;
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Minimal Header */
.header-wrapper {
    padding-bottom: 1rem;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid #E2E8F0;
}
.header-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0F172A;
    margin: 0;
    line-height: 1.2;
}
.header-subtitle {
    font-size: 0.88rem;
    color: #64748B;
    margin: 4px 0 0 0;
    font-weight: 400;
}

/* Global KPI Cards Row */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 1.25rem;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover {
    border-color: #BFDBFE;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 6px;
}
.kpi-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Modern Pill Tabs (Light Mode) */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #F1F5F9;
    padding: 5px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    margin-bottom: 18px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 20px;
    color: #64748B;
    font-weight: 600;
    font-size: 0.9rem;
    border: none !important;
    background: transparent;
    transition: all 0.18s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0F172A;
}
.stTabs [aria-selected="true"] {
    background-color: #0071DC !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 8px rgba(0, 113, 220, 0.28);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Form Container Card */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1rem 1.15rem !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
}

/* Section titles inside containers */
.control-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.6rem;
}
.section-subdivider {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-top: 0.8rem;
    margin-bottom: 0.4rem;
    padding-top: 0.5rem;
    border-top: 1px solid #F1F5F9;
}

/* Time Pills */
.time-meta-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-top: 0.9rem;
}
.time-pill {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 5px 6px;
    text-align: center;
    font-size: 0.7rem;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* Forecast Executive Hero Card */
.forecast-hero-card {
    background: linear-gradient(135deg, #0071DC 0%, #1D4ED8 100%);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0, 113, 220, 0.25);
    color: #FFFFFF;
}
.hero-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #DBEAFE;
}
.hero-tag-dot {
    width: 6px;
    height: 6px;
    background: #93C5FD;
    border-radius: 50%;
}
.hero-date-badge {
    font-size: 0.75rem;
    color: #FFFFFF;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.18);
    padding: 3px 9px;
    border-radius: 9999px;
}
.hero-value-row {
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
    margin: 0.2rem 0 0.8rem 0;
}
.hero-val {
    font-size: 2.25rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    line-height: 1;
}
.hero-delta-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
}
.delta-pos {
    background: rgba(16, 185, 129, 0.25);
    color: #ECFDF5;
    border: 1px solid rgba(16, 185, 129, 0.45);
}
.delta-neg {
    background: rgba(244, 63, 94, 0.25);
    color: #FFF1F2;
    border: 1px solid rgba(244, 63, 94, 0.45);
}
.hero-benchmarks {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    padding-top: 0.75rem;
}
.hero-bench-item {
    display: flex;
    flex-direction: column;
}
.hero-bench-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #DBEAFE;
    font-weight: 600;
}
.hero-bench-val {
    font-size: 1.02rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 1px;
}
.hero-bench-sub {
    font-size: 0.7rem;
    color: #BFDBFE;
}

/* Leaderboard Highlight Card (Light Mode) */
.leaderboard-winner-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    height: 100%;
}
.winner-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 0.6rem;
}
.winner-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
}

/* Mobile adjustments */
@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .hero-benchmarks {
        grid-template-columns: 1fr;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# UNIFIED PLOTLY LIGHT THEME
# -----------------------------------------------------------------------------
def apply_chart_style(fig, height=360, show_legend=True):
    """Applies a clean, modern light theme with zero collision between title and legend."""
    existing_title = fig.layout.title.text if fig.layout.title else None
    
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color="#475569", size=11),
        title=dict(
            text=existing_title,
            font=dict(size=13, color="#0F172A", family="Plus Jakarta Sans, sans-serif", weight=700),
            x=0.01,
            y=0.96,
            xanchor="left",
            yanchor="top"
        ) if existing_title else None,
        margin=dict(l=45, r=20, t=55 if existing_title else 25, b=45 if show_legend else 30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(0,0,0,0)"
        ) if show_legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#FFFFFF",
            bordercolor="#334155"
        ),
        height=height
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B", size=10),
        title_font=dict(color="#334155", size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B", size=10),
        title_font=dict(color="#334155", size=11)
    )
    return fig


# -----------------------------------------------------------------------------
# DATA LOADERS (CACHED)
# -----------------------------------------------------------------------------
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
best_name = results.get("best_model", "Linear Regression")

# -----------------------------------------------------------------------------
# MINIMAL EXECUTIVE HEADER (LIGHT MODE)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-wrapper">
    <h1 class="header-title">Walmart Store Demand Forecaster</h1>
    <p class="header-subtitle">
        Simulate macroeconomic conditions and forecast weekly store revenue using machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GLOBAL METRIC CARDS ROW (LIGHT MODE)
# -----------------------------------------------------------------------------
date_span_str = f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Total Stores</div>
        <div class="kpi-val">{df['Store'].nunique()}</div>
        <div class="kpi-sub">Across 45 US retail markets</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Historical Horizon</div>
        <div class="kpi-val">{len(df):,}</div>
        <div class="kpi-sub">{date_span_str} (143 wks)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Evaluation R²</div>
        <div class="kpi-val">96.58%</div>
        <div class="kpi-sub">Standard Future Horizon</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Benchmark RMSE</div>
        <div class="kpi-val">$98,689</div>
        <div class="kpi-sub">Unseen test weeks error</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# APPLICATION TABS
# -----------------------------------------------------------------------------
tab_forecast, tab_models, tab_insights = st.tabs([
    "Forecast & Simulator",
    "Model Benchmark",
    "Store Insights"
])


# =============================================================================
# TAB 1: FORECAST & SIMULATOR
# =============================================================================
with tab_forecast:
    col_input, col_pred = st.columns([1, 1.25], gap="large")
    
    # --- Left: Interactive Controls ---
    with col_input:
        with st.container(border=True):
            st.markdown('<div class="control-header">Forecast Parameters</div>', unsafe_allow_html=True)
            
            # Row 1: Store & Date Dropdowns
            c_store, c_week = st.columns(2)
            with c_store:
                store_id = st.selectbox("Store Location", sorted(df["Store"].unique()), index=0)
            with c_week:
                store_records = df[df["Store"] == store_id].sort_values("Date")
                available_dates = store_records["Date"].dt.strftime("%Y-%m-%d").tolist()
                selected_date_str = st.selectbox("Target Week", available_dates, index=len(available_dates)-1)
            
            hist_row = store_records[store_records["Date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]
            
            # Macroeconomic Drivers Inputs (2x2 Grid)
            st.markdown('<div class="section-subdivider">Macro & Environmental Drivers</div>', unsafe_allow_html=True)
            
            is_holiday = st.checkbox(
                "Holiday Surge Week (Thanksgiving, Christmas, Super Bowl, Labor Day)",
                value=bool(hist_row["Holiday_Flag"])
            )
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                temperature = st.number_input(
                    "Temperature (°F)",
                    value=float(hist_row["Temperature"]),
                    step=1.0,
                    format="%.1f"
                )
                fuel_price = st.number_input(
                    "Fuel Price ($/gal)",
                    value=float(hist_row["Fuel_Price"]),
                    step=0.05,
                    format="%.2f"
                )
            with col_d2:
                cpi = st.number_input(
                    "Consumer Price Index (CPI)",
                    value=float(hist_row["CPI"]),
                    step=0.5,
                    format="%.2f"
                )
                unemployment = st.number_input(
                    "Unemployment Rate (%)",
                    value=float(hist_row["Unemployment"]),
                    step=0.1,
                    format="%.2f"
                )
            
            # Calendar Feature Breakdown
            sel_dt = pd.Timestamp(selected_date_str)
            st.markdown(f"""
            <div class="time-meta-grid">
                <div class="time-pill">Year: {sel_dt.year}</div>
                <div class="time-pill">Month: {sel_dt.strftime('%b')}</div>
                <div class="time-pill">Week: {int(sel_dt.isocalendar().week)}</div>
                <div class="time-pill">Quarter: Q{sel_dt.quarter}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # --- Right: Forecast Results & Trend Chart ---
    with col_pred:
        # Prepare input features
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
        actual_sales = float(hist_row["Weekly_Sales"])
        
        diff_from_avg = predicted_sales - store_avg_sales
        pct_from_avg = (diff_from_avg / store_avg_sales) * 100
        
        diff_from_act = predicted_sales - actual_sales
        pct_from_act = (diff_from_act / actual_sales) * 100
        
        delta_class = "delta-pos" if diff_from_avg >= 0 else "delta-neg"
        delta_sign = "+" if diff_from_avg >= 0 else "-"
        
        # Executive Hero Card
        st.markdown(f"""
        <div class="forecast-hero-card">
            <div class="hero-top">
                <div class="hero-tag">
                    <span class="hero-tag-dot"></span> STORE {store_id} FORECAST
                </div>
                <div class="hero-date-badge">Week of {sel_dt.strftime('%B %d, %Y')}</div>
            </div>
            <div class="hero-value-row">
                <div class="hero-val">${predicted_sales:,.0f}</div>
                <div class="hero-delta-pill {delta_class}">
                    {delta_sign}${abs(diff_from_avg):,.0f} ({pct_from_avg:+.1f}%) vs Store Avg
                </div>
            </div>
            <div class="hero-benchmarks">
                <div class="hero-bench-item">
                    <span class="hero-bench-label">Historical Benchmark</span>
                    <span class="hero-bench-val">${actual_sales:,.0f}</span>
                    <span class="hero-bench-sub">{pct_from_act:+.1f}% vs actual week</span>
                </div>
                <div class="hero-bench-item">
                    <span class="hero-bench-label">Store Lifetime Mean</span>
                    <span class="hero-bench-val">${store_avg_sales:,.0f}</span>
                    <span class="hero-bench-sub">Across 143 recorded weeks</span>
                </div>
                <div class="hero-bench-item">
                    <span class="hero-bench-label">Active Model Engine</span>
                    <span class="hero-bench-val">{best_name}</span>
                    <span class="hero-bench-sub">R² 0.9658 test precision</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Historical Trend & Forecast Point Chart
        fig_hist = go.Figure()
        
        # Historical sales trajectory
        fig_hist.add_trace(go.Scatter(
            x=store_records["Date"],
            y=store_records["Weekly_Sales"],
            mode="lines",
            name="Historical Sales",
            line=dict(color="#0071DC", width=2),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        # Holiday points callout
        holidays_store = store_records[store_records["Holiday_Flag"] == 1]
        fig_hist.add_trace(go.Scatter(
            x=holidays_store["Date"],
            y=holidays_store["Weekly_Sales"],
            mode="markers",
            name="Holiday Surges",
            marker=dict(color="#E11D48", size=7, symbol="diamond"),
            hovertemplate="<b>Holiday Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        # Lifetime Store Average dashed line
        fig_hist.add_trace(go.Scatter(
            x=[store_records["Date"].min(), store_records["Date"].max()],
            y=[store_avg_sales, store_avg_sales],
            mode="lines",
            name=f"Store Avg (${store_avg_sales/1e3:.0f}k)",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
            hoverinfo="skip"
        ))
        
        # Active Forecast Point
        fig_hist.add_trace(go.Scatter(
            x=[sel_dt],
            y=[predicted_sales],
            mode="markers",
            name="Simulated Forecast",
            marker=dict(
                color="#F59E0B",
                size=14,
                symbol="star",
                line=dict(color="#78350F", width=1.5)
            ),
            hovertemplate="<b>Forecast Point:</b> %{x|%Y-%m-%d}<br><b>Prediction:</b> $%{y:,.0f}<extra></extra>"
        ))
        
        fig_hist.update_layout(
            title=f"Store {store_id} Sales History & Simulated Forecast",
            xaxis_title="Date",
            yaxis_title="Weekly Revenue ($)",
            yaxis_tickformat="$,.0f"
        )
        apply_chart_style(fig_hist, height=360)
        st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB 2: MODELS & BENCHMARK
# =============================================================================
with tab_models:
    st.markdown("### Model Benchmark & Generalization Evaluation")
    
    metrics_list = results.get("metrics", [])
    metrics_df = pd.DataFrame(metrics_list)
    
    # Leaderboard Table & Explanation
    col_lb, col_desc = st.columns([1.8, 1.2], gap="large")
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
        
        # Clean formatted leaderboard
        st.dataframe(
            piv.style.format({
                "Standard R2": "{:.4f}",
                "Standard RMSE": "${:,.0f}",
                "Standard MAE": "${:,.0f}",
                "Holiday-season R2": "{:.4f}",
                "Holiday-season RMSE": "${:,.0f}",
                "Holiday-season MAE": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True
        )
        
    with col_desc:
        st.markdown(f"""
        <div class="leaderboard-winner-card">
            <div class="winner-pill">Production Choice: {best_name}</div>
            <div class="winner-desc">
                <strong>Why Linear Regression Won:</strong><br>
                Achieved the <strong>lowest overall RMSE</strong> across both future unseen weeks (<strong>$98,689</strong>) and the holiday season surge (<strong>$188,501</strong>).<br><br>
                While Random Forest had slightly higher R² on the standard test, Linear Regression generalized significantly better to volatile holiday peaks without overfitting store baselines.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Comparison Bar Charts
    st.markdown("#### Regression Metrics Comparison Across Evaluation Splits")
    eval_chart_df = metrics_df[
        (metrics_df["Model"] != "Baseline") & 
        (metrics_df["Evaluation"].isin(["Standard", "Holiday-season"]))
    ].copy()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        fig_r2 = go.Figure()
        for eval_name, color in [("Standard", "#0071DC"), ("Holiday-season", "#F59E0B")]:
            sub = eval_chart_df[eval_chart_df["Evaluation"] == eval_name]
            fig_r2.add_trace(go.Bar(
                x=sub["Model"], y=sub["R2"], name=eval_name, marker_color=color,
                text=[f"{v:.4f}" for v in sub["R2"]], textposition="auto"
            ))
        fig_r2.update_layout(title="R² Score (Higher is Better)", barmode="group", yaxis_range=[0.8, 1.0])
        apply_chart_style(fig_r2, height=310)
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col_m2:
        fig_rmse = go.Figure()
        for eval_name, color in [("Standard", "#0071DC"), ("Holiday-season", "#F59E0B")]:
            sub = eval_chart_df[eval_chart_df["Evaluation"] == eval_name]
            fig_rmse.add_trace(go.Bar(
                x=sub["Model"], y=sub["RMSE"], name=eval_name, marker_color=color,
                text=[f"${v/1e3:.0f}k" for v in sub["RMSE"]], textposition="auto"
            ))
        fig_rmse.update_layout(title="RMSE Error (Lower is Better)", barmode="group", yaxis_tickformat="$,.0f")
        apply_chart_style(fig_rmse, height=310)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with col_m3:
        fig_mae = go.Figure()
        for eval_name, color in [("Standard", "#0071DC"), ("Holiday-season", "#F59E0B")]:
            sub = eval_chart_df[eval_chart_df["Evaluation"] == eval_name]
            fig_mae.add_trace(go.Bar(
                x=sub["Model"], y=sub["MAE"], name=eval_name, marker_color=color,
                text=[f"${v/1e3:.0f}k" for v in sub["MAE"]], textposition="auto"
            ))
        fig_mae.update_layout(title="MAE Error (Lower is Better)", barmode="group", yaxis_tickformat="$,.0f")
        apply_chart_style(fig_mae, height=310)
        st.plotly_chart(fig_mae, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Store-Level Actual vs Predicted & Feature Importance
    col_avp, col_fi = st.columns([1.35, 1], gap="large")
    
    with col_avp:
        st.markdown("#### Test Set Trajectory: Actual vs Predicted")
        preds_records = results.get("predictions", [])
        if preds_records:
            preds_df = pd.DataFrame(preds_records)
            preds_df["Date"] = pd.to_datetime(preds_df["Date"])
            
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                chosen_store = st.selectbox("Store for Horizon Plot", sorted(preds_df["Store"].unique()), index=0)
            with c_sel2:
                chosen_eval = st.selectbox("Test Split Horizon", ["Standard", "Holiday-season"])
                
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
                mode="lines", name="Linear Regression", line=dict(color="#0071DC", width=2, dash="dash")
            ))
            fig_avp.add_trace(go.Scatter(
                x=filtered_preds["Date"], y=filtered_preds["Tree"],
                mode="lines", name="Decision Tree", line=dict(color="#10B981", width=1.5, dash="dot")
            ))
            fig_avp.add_trace(go.Scatter(
                x=filtered_preds["Date"], y=filtered_preds["Forest"],
                mode="lines", name="Random Forest", line=dict(color="#E11D48", width=1.5, dash="dashdot")
            ))
            fig_avp.update_layout(
                title=f"Store {chosen_store} — {chosen_eval} Test Split",
                xaxis_title="Date",
                yaxis_title="Weekly Sales ($)",
                yaxis_tickformat="$,.0f"
            )
            apply_chart_style(fig_avp, height=350)
            st.plotly_chart(fig_avp, use_container_width=True)
            
    with col_fi:
        st.markdown("#### Feature Importance")
        fi_list = results.get("feature_importance", [])
        if fi_list:
            fi_df = pd.DataFrame(fi_list).sort_values("importance", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"],
                y=fi_df["feature"],
                orientation="h",
                marker=dict(color=["#0071DC" if f == "Store" else "#CBD5E1" for f in fi_df["feature"]]),
                text=[f"{v:.4f}" for v in fi_df["importance"]],
                textposition="outside"
            ))
            fig_fi.update_layout(
                title="Random Forest Gini Feature Importance",
                xaxis_title="Relative Importance",
                yaxis_title=""
            )
            apply_chart_style(fig_fi, height=395, show_legend=False)
            st.plotly_chart(fig_fi, use_container_width=True)


# =============================================================================
# TAB 3: INSIGHTS & EXPLORATION
# =============================================================================
with tab_insights:
    st.markdown("### Exploratory Retail & Demand Drivers")
    
    col_ins1, col_ins2 = st.columns(2, gap="large")
    
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
            marker_color=["#CBD5E1", "#0071DC"],
            text=[f"${v:,.0f}" for v in hol_stats["mean"]],
            textposition="auto"
        ))
        fig_hol.update_layout(
            title=f"Holiday vs Non-Holiday Average Weekly Sales (+{boost_pct:.1f}% Surge)",
            xaxis_title="",
            yaxis_title="Average Sales ($)",
            yaxis_tickformat="$,.0f"
        )
        apply_chart_style(fig_hol, height=330, show_legend=False)
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
            marker_color=["#0071DC" if m in [11, 12] else "#CBD5E1" for m in monthly_sales["Month"]],
            text=[f"${v/1e3:.0f}k" for v in monthly_sales["Weekly_Sales"]],
            textposition="auto"
        ))
        fig_month.update_layout(
            title="Monthly Sales Seasonality (Q4 Holiday Peak Dominance)",
            xaxis_title="Month",
            yaxis_title="Average Sales ($)",
            yaxis_tickformat="$,.0f"
        )
        apply_chart_style(fig_month, height=330, show_legend=False)
        st.plotly_chart(fig_month, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Store Volume Disparity: Top 5 vs Bottom 5
    st.markdown("#### Store Volume Distribution: Highest vs Lowest Performing Stores")
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
        marker=dict(color=["#CBD5E1" if g == "Bottom 5" else "#0071DC" for g in top_bottom["Group"]]),
        text=[f"${v:,.0f}" for v in top_bottom["Weekly_Sales"]],
        textposition="outside"
    ))
    fig_tb.update_layout(
        title="Store Volume Disparity: Top Store Weekly Sales Exceed Lowest by Over 4×",
        xaxis_title="Average Weekly Sales ($)",
        yaxis_title="Store",
        xaxis_tickformat="$,.0f"
    )
    apply_chart_style(fig_tb, height=360, show_legend=False)
    st.plotly_chart(fig_tb, use_container_width=True)
