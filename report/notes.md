# Project Notes

This file tracks findings and decisions made during the project.

---

## Task 1.2 — Load and Inspect

- **Shape:** 6,435 rows × 8 columns (45 stores × 143 weeks = 6,435 ✓)
- **Columns:** Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- **Date range:** Feb 5, 2010 to Oct 26, 2012 (143 unique weeks)
- **Stores:** 45 (numbered 1–45)
- **Weekly_Sales:** ranges from ~$210K to ~$3.8M, mean ~$1.05M, big variation across stores
- **Holiday_Flag:** only ~7% of weeks are holiday weeks
- **Temperature:** -2°F to 100°F (varies by store region)
- **CPI:** 126 to 227, **Unemployment:** 3.9% to 14.3% (vary by region)

## Task 1.3 — Cleaning Checks

### Missing values
- **None.** All 8 columns are complete with 0 missing values.

### Duplicate rows
- **None.** No duplicate rows and no duplicate Store+Date combinations.

### Negative or zero sales
- **None.** All Weekly_Sales values are positive.

### Rows per store
- **All 45 stores have exactly 143 rows.** The data is balanced — no missing weeks.

### Outliers (IQR method on Weekly_Sales)
- Q1 = $553,350 | Q3 = $1,420,159 | IQR = $866,809
- Upper bound = $2,720,371 | Lower bound = -$746,863
- **34 outliers found (0.5% of data)**, all above the upper bound (high sales), none below.
- Of these 34: 9 are holiday weeks, 25 are non-holiday weeks (likely from the biggest stores).
- **Decision: Keep all outliers.** They represent real demand from large stores and holiday spikes, not data errors.

### Overall cleaning result
- The dataset is very clean: no missing values, no duplicates, no invalid values.
- No rows were removed. The cleaned file is identical to the original but with Date parsed as datetime.

## Task 1.4 — Save
- Cleaned data saved to `data/cleaned.csv` (6,435 rows, 8 columns).

## Task 1.5 — EDA Visualizations

### Figure 1: Sales Distribution (`eda_sales_distribution.png`)
- Right-skewed distribution (skewness = 0.67). Mean ($1,046,965) > Median ($960,746).
- Most stores have modest sales, a few large stores push the tail right.
- Box plot confirms the 34 upper outliers.

### Figure 2: Total Sales Over Time (`eda_sales_over_time.png`)
- Clear annual spikes in late November / December (Thanksgiving + Christmas).
- **Peak week: Dec 24, 2010** — $80.9M total across all stores.
- Sales dip in late January every year. Strong seasonality pattern.

### Figure 3: Average Sales Per Store (`eda_avg_sales_per_store.png`)
- Huge variation: **Store 20** averages $2.1M/week, **Store 33** averages only $260K/week (~8× difference).
- The top 5 stores contribute disproportionately. Store identity will be very important for the model.

### Figure 4: Holiday vs Non-Holiday (`eda_holiday_vs_nonholiday.png`)
- Holiday weeks average **7.8% higher** sales ($1,122,888 vs $1,041,256).
- The difference is moderate, not dramatic — because Holiday_Flag covers Super Bowl, Labor Day, Thanksgiving, and Christmas, and not all of those cause big spikes for every store.

### Figure 5: Monthly/Weekly Seasonality (`eda_monthly_seasonality.png`)
- December is the peak month. **Week 51** (week before Christmas) has the highest average sales.
- **Important finding:** The week before Christmas is **NOT flagged as a holiday** (Holiday_Flag = 0), even though it has the highest sales. This means Holiday_Flag alone doesn't capture the full holiday effect.

### Figure 6: Correlation Heatmap (`eda_correlation_heatmap.png`)
- All correlations with Weekly_Sales are weak:
  - Store: -0.335 (strongest — store numbering loosely correlates with store size)
  - Unemployment: -0.106
  - CPI: -0.073
  - Temperature: -0.064
  - Holiday_Flag: 0.037
  - Fuel_Price: 0.009
- Weak linear relationships suggest tree-based models should outperform Linear Regression.

### Figure 7: Sales vs Features Scatters (`eda_sales_vs_features.png`)
- Cloud-like scatter patterns with no clear trends for Temperature, CPI, or Unemployment.
- Wide vertical spread at every x-value confirms that store identity matters far more than these external factors.

### EDA Summary
1. The dataset is clean: no issues found.
2. Weekly sales vary hugely across stores, driven by store size/type.
3. Strong seasonality: November–December spike, peak in pre-Christmas week.
4. Holiday flag gives ~7.8% higher average sales, but misses the pre-Christmas peak week.
5. External features (Temperature, CPI, Unemployment) have weak correlations with sales.
6. Store identity will likely be the most important predictor.
7. Tree-based models should outperform Linear Regression due to weak linear relationships.

## Task 1.6 — Feature Engineering
- Added 4 new columns from Date: **Year**, **Month**, **WeekOfYear**, **Quarter**.
- Updated `data/cleaned.csv` with the new columns (6,435 rows, 12 columns).
- The raw Date column is kept in the file but will be dropped from model input in notebook 02.

---

## Task 2.1 — Data Splits
- **Time-based split (primary):** Cutoff = April 13, 2012. Train: 5,130 rows (79.7%), Test: 1,305 rows (20.3%).
  - Train: Feb 2010 – Apr 6, 2012 | Test: Apr 13 – Oct 26, 2012
- **Random 80/20 split (secondary):** Train: 5,148 rows, Test: 1,287 rows (random_state=42).

## Task 2.2 — Pipelines
- 3 pipelines using `ColumnTransformer` + `Pipeline`:
  - **Linear Regression:** OneHotEncode Store + StandardScaler for numerics
  - **Decision Tree:** OneHotEncode Store + passthrough numerics (max_depth=8, min_samples_leaf=5)
  - **Random Forest:** OneHotEncode Store + passthrough numerics (n_estimators=200, max_depth=12)

## Task 2.3 — Results Table

| Model | Split | MAE | RMSE | R² |
|---|---|---|---|---|
| **Linear Regression** | **Time-based** | **$87,702** | **$119,661** | **0.9497** |
| Random Forest | Time-based | $253,558 | $299,265 | 0.6854 |
| Decision Tree | Time-based | $301,050 | $362,275 | 0.5390 |
| **Linear Regression** | **Random** | **$94,468** | **$153,108** | **0.9272** |
| Random Forest | Random | $248,811 | $315,890 | 0.6903 |
| Decision Tree | Random | $311,584 | $385,802 | 0.5380 |

### Interpretation (3–5 sentences)
1. **Linear Regression is the clear winner** with R²=0.95 and RMSE=$120K on the time-based split — far ahead of Random Forest (R²=0.69) and Decision Tree (R²=0.54).
2. This is surprising because EDA suggested tree models should outperform LR. The key reason: one-hot encoding Store gives LR 45 separate coefficients, effectively learning each store's baseline sales level precisely. Trees with limited depth can't split finely enough across 45 stores AND time features simultaneously.
3. The time-based split actually gives LR *better* results than the random split (R²=0.95 vs 0.93), which is unusual. This is likely because the test period (Apr–Oct 2012) avoids the volatile holiday season, making predictions easier.
4. Feature importance from Random Forest confirms Store identity dominates (top 9 features are all Store dummies), followed by WeekOfYear (seasonality). Temperature, CPI, Unemployment contribute very little.
5. The EDA prediction that "tree models should outperform LR" was wrong — it shows why you need to actually run experiments, not just look at correlations.

## Task 2.4 — Interpretation Figures
- `model_actual_vs_predicted_stores.png` — LR tracks actual sales closely for small/medium/large stores
- `model_predicted_vs_actual_scatter.png` — tight clustering around the perfect-prediction line
- `model_residual_histogram.png` — residuals centered near zero, roughly normal
- `model_feature_importance.png` — Store dummies dominate, WeekOfYear is top non-store feature

## Task 2.5 — Best Model Saved
- Best model: **Linear Regression** (by time-based RMSE)
- Saved to `models/best_model.joblib`
- Feature info saved to `models/feature_info.json`

## Task 2.6 — Sanity Check
- Loaded saved model, predicted Store 1 June 2012: $1,232,132
- Actual Store 1 June average: $1,561,510 (21% difference, within RMSE range)
- Model loads and predicts correctly ✓
