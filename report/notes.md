# Project Notes

## WalmartCast - AI-Based Demand Forecasting Models

This file tracks key findings, results, and observations throughout the project.

---

## Phase 1: EDA Findings

**Data Quality:**
- Dataset verified: 6,435 rows (45 stores × 143 weeks)
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
## Phase 2 — Modeling Results

### Results Table

| Model | Standard MAE | Standard RMSE | Standard R² | Holiday-season MAE | Holiday-season RMSE | Holiday-season R² |
|---|---|---|---|---|---|---|
| Baseline | 69,232 | 102,108 | 0.9634 | 175,767 | 303,261 | 0.7865 |
| Linear Regression | 69,832 | 98,689 | 0.9658 | 121,066 | 188,502 | 0.9175 |
| Decision Tree | 79,995 | 110,602 | 0.9570 | 131,904 | 223,154 | 0.8844 |
| Random Forest | 65,099 | 93,617 | 0.9692 | 116,835 | 195,151 | 0.9116 |

### Interpretation
- **Best model: Linear Regression** (mean RMSE across Standard + Holiday-season tests).
- Standard test R²=0.9658, RMSE=$98,689. Holiday-season R²=0.9175, RMSE=$188,502.
- The three models are close; all beat the baseline by a large margin.
- The holiday-season test is harder: all models see a larger RMSE because Thanksgiving and Christmas weeks are
  exceptional. A model that ignores holidays would look good on the Standard test but fail here.
- Feature importance shows Store identity is the dominant predictor. WeekOfYear (seasonality) is second.
  Economic variables (Temperature, Fuel_Price, CPI, Unemployment) contribute very little individually,
  confirming the EDA finding that their correlation with sales is weak.
- Note: the holiday-season test is trained on only one earlier holiday season (2010), so its results
  are indicative, not conclusive. Formal tuning is future work.

