# Slide Outline — Walmart Weekly Sales Forecasting

## Slide 1: Title
- AI-Based Demand Forecasting Models — Walmart Weekly Sales
- Course: AIML Lab (CCE3131)
- Student name, date

## Slide 2: Problem & Dataset
- Problem: Predict weekly sales per Walmart store
- Dataset: 45 stores, 143 weeks (Feb 2010 – Oct 2012), 6,435 rows
- Columns: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- Scope: Store-level weekly sales only

## Slide 3: EDA Highlights
- Store variation: $260K – $2.1M average (8× difference)
- Strong Nov/Dec seasonality (peak = Week 51, pre-Christmas)
- External features (Temperature, CPI, Unemployment) have weak correlations
- Figures: sales distribution, sales over time, store comparison

## Slide 4: Method
- Features: Store (one-hot), Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, WeekOfYear, Quarter
- 3 models: Linear Regression, Decision Tree, Random Forest
- Evaluation: Time-based split (primary) + Random split (comparison)
- Pipelines to prevent data leakage

## Slide 5: Results Table
- Full metrics table (3 models × 2 splits × 4 metrics)
- Linear Regression wins: R² = 0.95, RMSE = $120K
- Why LR wins: one-hot Store gives 45 baseline coefficients
- Figures: predicted vs actual scatter, residual histogram

## Slide 6: App Demo
- Streamlit web app with two pages
- Predict page: enter store/date/features → get prediction
- Model Comparison page: metrics table, charts, feature importance
- Screenshot or live demo

## Slide 7: Limitations & Future Work
- Data only covers 2010–2012
- No product-level data, no promotion data
- Tree models not tuned (fixed hyperparameters)
- Future: tuning, lag features, Gradient Boosting, deployment
