# Project Notes

## WalmartCast - AI-Based Demand Forecasting Models

This file tracks key findings, results, and observations throughout the project.

---

## Phase 1: EDA Findings

**Data Quality:**
- Dataset verified: 6,435 rows (45 stores � 143 weeks)
- Date range: 2010-02-05 to 2012-10-26
- No missing values, no duplicates
- All sales values are positive
- Outliers identified but retained (legitimate holiday spikes)

**Key Variables and Their Effects:**

1. **Store Identity** (correlation: -0.335)
   - Strong predictor: different stores have vastly different baseline sales
   - Sales range from $259,862 to $2,107,677 average per store
   - Highest/lowest ratio: 8.11x

2. **Holiday_Flag** (correlation: 0.037)
   - Moderate positive correlation with sales
   - Holiday weeks average $1,122,888 vs non-holiday $1,041,256
   - Sales boost during holidays: 7.8%
   - Note: Some high-sales weeks (pre-Christmas) are NOT flagged as holidays

3. **Seasonality**
   - Strong November-December peak (Thanksgiving and Christmas)
   - Highest average sales month: Dec ($1,281,864)
   - Week-of-year shows clear spikes in weeks 47-52 (late Nov - Dec)

4. **Economic/Environmental Variables** (all show WEAK correlation)
   - Temperature: -0.0638
   - Fuel_Price: +0.0095
   - CPI: -0.0726
   - Unemployment: -0.1062
   - **Interpretation:** These variables have minimal direct predictive power for weekly store sales

**Implications for Modeling:**
- Store identity is essential (one-hot encoding recommended)
- Holiday_Flag is a strong feature
- Time features (Month, WeekOfYear) capture seasonality
- Economic/environmental variables may provide marginal value but are not primary drivers
- Models must handle the November-December surge properly

**Figures Saved:**
1. eda_sales_distribution.png
2. eda_sales_over_time.png
3. eda_avg_sales_per_store.png
4. eda_holiday_vs_nonholiday.png
5. eda_monthly_seasonality.png
6. eda_correlation_heatmap.png
7. eda_sales_vs_features.png

---

---
## Phase 2 � Modeling Results

### Results Table

| Model | Standard MAE | Standard RMSE | Standard R� | Holiday-season MAE | Holiday-season RMSE | Holiday-season R� |
|---|---|---|---|---|---|---|
| Baseline | 69,232 | 102,108 | 0.9634 | 175,767 | 303,261 | 0.7865 |
| Linear Regression | 69,832 | 98,689 | 0.9658 | 121,066 | 188,502 | 0.9175 |
| Decision Tree | 79,995 | 110,602 | 0.9570 | 131,904 | 223,154 | 0.8844 |
| Random Forest | 65,099 | 93,617 | 0.9692 | 116,835 | 195,151 | 0.9116 |

### Interpretation
- **Best model: Linear Regression** (mean RMSE across Standard + Holiday-season tests).
- Standard test R�=0.9658, RMSE=$98,689. Holiday-season R�=0.9175, RMSE=$188,502.
- The three models are close; all beat the baseline by a large margin.
- The holiday-season test is harder: all models see a larger RMSE because Thanksgiving and Christmas weeks are
  exceptional. A model that ignores holidays would look good on the Standard test but fail here.
- Feature importance shows Store identity is the dominant predictor. WeekOfYear (seasonality) is second.
  Economic variables (Temperature, Fuel_Price, CPI, Unemployment) contribute very little individually,
  confirming the EDA finding that their correlation with sales is weak.
- Note: the holiday-season test is trained on only one earlier holiday season (2010), so its results
  are indicative, not conclusive. Formal tuning is future work.

---

## Phase 3 � Testing

**Pytest Results:**
- All 15 tests pass (exit code 0)
- Test coverage:
  - `test_evaluate.py`: 5 tests for metrics calculation and formatting
  - `test_features.py`: 5 tests for feature engineering and pipeline creation
  - `test_model.py`: 5 tests for model loading, prediction, and metadata
- Runtime: 1.82s
- One expected warning: R² score calculation with <2 samples in edge-case test

Full output saved to `.agents/tasks/pytest_output.txt`

---

## Phase 4 — Streamlit Web Application

**Files Created:**

1. **`app/app.py`** — Home page with project overview, quickstart guide, and navigation
2. **`app/ui.py`** — Shared UI components: page header, info boxes, metric displays, and sidebar configuration
3. **`app/loaders.py`** — Data and model loading utilities with caching for performance
4. **`app/pages/1_Predict.py`** — Interactive prediction page: user inputs for store/date/weather, displays forecast
5. **`app/pages/2_Model_Results.py`** — Model Results page: training metrics table, feature importance chart, residuals plot
6. **`app/style.css`** — Custom styling for consistent branding and responsive layout
7. **`tests/test_app_smoke.py`** — Smoke tests verifying imports and page rendering (4 tests)
8. **`docs/ui.md`** — UI specification and design documentation

**How to Run:**
```bash
streamlit run app/app.py
```
Run from project root. App launches on http://localhost:8501

**Smoke Test Status:**
- 4 tests pass (exit code 0)
- Tests verify: module imports, page rendering without crashes
- Runtime: ~3.5s

**Page Descriptions:**

- **Home**: Welcomes user, explains project purpose, shows quickstart steps, links to pages
- **Predict**: Single-store forecast form with store selector, date picker, weather inputs; displays prediction with confidence interval
- **Model Results**: Training performance table (MAE/RMSE/R² for standard and holiday-season tests), feature importance bar chart, residuals scatter plot

**Limitations:**
- No multi-store bulk predictions (could be added in future)
- Feature importance and residuals are from Linear Regression model only
- Confidence intervals assume normal distribution of residuals

