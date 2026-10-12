# AI-Based Demand Forecasting Models — Walmart Weekly Sales

**Course:** AIML Lab (CCE3131)  
**Scope:** Weekly store-level sales prediction only. No product-level demand or inventory planning.

---

## 1. Introduction

Accurate demand forecasting helps retail businesses optimize inventory, staffing, and supply chain decisions. This project builds and evaluates machine learning models to predict weekly sales for individual Walmart stores, using historical data from February 2010 to October 2012.

The goal is to compare three regression models — Linear Regression, Decision Tree, and Random Forest — and deploy the best one in an interactive web application.

## 2. Dataset

The dataset comes from Kaggle and contains **6,435 rows** (45 stores × 143 weeks) with 8 columns:

| Column | Description | Role |
|---|---|---|
| Store | Store number (1–45) | Feature (categorical) |
| Date | Week date (dd-mm-yyyy) | Source of time features |
| Weekly_Sales | Sales for that store-week ($) | **Target** |
| Holiday_Flag | 1 if holiday week, else 0 | Feature |
| Temperature | Regional temperature (°F) | Feature |
| Fuel_Price | Regional fuel price ($) | Feature |
| CPI | Consumer Price Index | Feature |
| Unemployment | Regional unemployment rate (%) | Feature |

**Data quality:** The dataset is very clean — no missing values, no duplicates, no negative sales, and all 45 stores have exactly 143 weeks of data.

## 3. Methodology

### 3.1 Feature Engineering
Four time-based features were derived from the Date column:
- **Year** (2010–2012) — captures long-term trends
- **Month** (1–12) — captures monthly seasonality
- **WeekOfYear** (1–52) — captures weekly seasonality (more precise than month)
- **Quarter** (1–4) — captures seasonal patterns

### 3.2 Data Splitting
Two evaluation strategies were used:

1. **Time-based split (primary):** Train on data before April 13, 2012 (80%), test on data from April–October 2012 (20%). This simulates real forecasting — training on the past and predicting the future.

2. **Random 80/20 split (secondary):** Reported for comparison. The random split is optimistic because neighbouring weeks from the same store can appear in both train and test sets.

### 3.3 Models
Three models were built using scikit-learn Pipelines to prevent data leakage:

1. **Linear Regression** — with StandardScaler for numeric features
2. **Decision Tree Regressor** — max_depth=8, min_samples_leaf=5
3. **Random Forest Regressor** — n_estimators=200, max_depth=12

All models used OneHotEncoder for the Store column (45 binary features) and `random_state=42` for reproducibility.

### 3.4 Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (coefficient of determination)

## 4. EDA Findings

Key findings from exploratory data analysis:

1. **Store variation is huge:** Store 20 averages $2.1M/week while Store 33 averages only $260K/week (~8× difference). Store identity will be the most important predictor.

2. **Strong seasonality:** November–December show clear sales spikes, with Week 51 (week before Christmas) having the highest average sales. Interestingly, this peak week is NOT flagged as a holiday in the data.

3. **Holiday effect is moderate:** Holiday weeks average ~7.8% higher sales ($1.12M vs $1.04M).

4. **Weak external features:** Temperature (r=-0.064), CPI (r=-0.073), and Unemployment (r=-0.106) have very weak correlations with weekly sales.

5. **34 outliers (0.5%):** All on the high side, a mix of holiday spikes and large-store normal sales. Kept in the data as they represent real demand.

*(See figures in report/figures/: eda_sales_distribution.png, eda_sales_over_time.png, eda_avg_sales_per_store.png, eda_holiday_vs_nonholiday.png, eda_monthly_seasonality.png, eda_correlation_heatmap.png, eda_sales_vs_features.png)*

## 5. Models and Evaluation

### 5.1 Results Table

| Model | Split | MAE | RMSE | R² |
|---|---|---|---|---|
| **Linear Regression** | **Time-based** | **$87,702** | **$119,661** | **0.9497** |
| Random Forest | Time-based | $253,558 | $299,265 | 0.6854 |
| Decision Tree | Time-based | $301,050 | $362,275 | 0.5390 |
| **Linear Regression** | **Random** | **$94,468** | **$153,108** | **0.9272** |
| Random Forest | Random | $248,811 | $315,890 | 0.6903 |
| Decision Tree | Random | $311,584 | $385,802 | 0.5380 |

### 5.2 Best Model
**Linear Regression** is the best model by both RMSE and R² on the time-based split.

## 6. Results and Interpretation

**Linear Regression wins by a large margin.** This was surprising because EDA suggested weak linear relationships between features and sales. The key reason: one-hot encoding Store gives LR 45 separate coefficients — effectively learning each store's baseline level. Combined with Month and WeekOfYear, LR captures the two dominant patterns: *which store* and *what time of year*.

**Why do trees underperform?** The Decision Tree (max_depth=8) and Random Forest (max_depth=12) can't split finely enough across 45 stores AND time features simultaneously with their limited depth. With hyperparameter tuning (deeper trees), they could potentially improve.

**Time-based vs Random split:** For Linear Regression, the time-based split gives slightly better results (R²=0.95 vs 0.93). This is because the test period (Apr–Oct 2012) avoids the volatile holiday season. For tree models, the splits performed similarly.

**Feature importance (Random Forest):** Store dummies dominate the top 9 positions. The highest non-store feature is WeekOfYear (seasonality). Temperature, CPI, and Unemployment barely register.

**Residuals:** Roughly centered around zero and approximately normally distributed, with a slight right tail from weeks where actual sales spiked unexpectedly.

*(See figures: model_actual_vs_predicted_stores.png, model_predicted_vs_actual_scatter.png, model_residual_histogram.png, model_feature_importance.png)*

## 7. Application

A two-page Streamlit web app was built:

1. **Predict Page:** Users enter a store number, date, and external features (temperature, fuel price, CPI, unemployment). The app derives Year/Month/WeekOfYear/Quarter from the date, loads the saved model, and displays the predicted weekly sales. If the store+date exists in the actual data, it also shows the actual value and error.

2. **Model Comparison Page:** Displays the full metrics table, interactive RMSE and R² bar charts (Plotly), the feature importance chart, and an analysis explaining why Linear Regression outperforms.

**Limitation note (displayed in app):** The data covers only 2010–2012. Tree models cannot extrapolate, and predictions outside this period may be inaccurate. The date picker is limited to the dataset's date range.

## 8. Limitations

1. **Data covers only 2010–2012.** The model cannot predict future sales accurately because economic conditions, store layouts, and consumer behavior change over time.
2. **No product-level data.** The model predicts total store sales, not individual product demand.
3. **Limited features.** The dataset lacks store size, promotion data, and local demographics that could improve predictions.
4. **Tree models not tuned.** Decision Tree and Random Forest used fixed hyperparameters. Tuning could improve their performance.
5. **No lag features.** Previous week's sales (a strong predictor) were not used because the plan specified fixed features.

## 9. Conclusion

Linear Regression achieved R² = 0.95 on the time-based split, explaining 95% of the variance in weekly store sales. The model's success is driven by one-hot encoding of store identity and time features (Month, WeekOfYear). External features contribute minimally.

The project demonstrates that sometimes simpler models outperform complex ones, especially when the key driver (store identity) can be captured through proper feature encoding.

## 10. Future Scope

- Hyperparameter tuning with TimeSeriesSplit cross-validation
- Additional models (e.g., Gradient Boosting, XGBoost)
- Lag features (previous week's sales for the same store)
- Explore page in the app with store/date filters and trend plots
- Deployment on Streamlit Community Cloud
