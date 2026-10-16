# WalmartCast

**AI-Based Demand Forecasting Models for Walmart Weekly Sales**

A mini AIML lab project that predicts weekly sales of Walmart stores using historical data and multiple regression models.

## Project Overview

WalmartCast trains and compares three regression models (Linear Regression, Decision Tree, Random Forest) on Walmart store sales data from 2010-2012. The best model is deployed in a simple Streamlit web app that allows users to predict sales for any store-week combination.

## Dataset

- **Source:** Kaggle Walmart Store Sales Dataset
- **Period:** February 2010 - October 2012
- **Records:** 6,435 weekly records (45 stores × 143 weeks)
- **Target:** Weekly_Sales
- **Features:** Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, and time-based features derived from Date

## Quick Start

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd WalmartCast
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Verify data file exists:**
   ```bash
   # Ensure data/Walmart.csv is present
   ```

### Run Notebooks

```bash
# Activate your virtual environment first, then start Jupyter:
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_eda_cleaning.ipynb  — EDA and data cleaning
# 2. notebooks/02_modeling.ipynb       — Feature engineering, models, evaluation
```

### Run Tests

```bash
pytest tests/
```

### Run Web App

```bash
streamlit run app/app.py
```

## Project Structure

```
WalmartCast/
├── data/               # Walmart.csv (raw), cleaned.csv (processed)
├── src/                # features.py, evaluate.py — shared logic
├── notebooks/          # 01_eda_cleaning.ipynb, 02_modeling.ipynb
├── models/             # best_model.joblib, metrics.json, feature_info.json, test_predictions.csv
├── app/                # Streamlit app: app.py, ui.py, loaders.py, style.css
│   └── pages/          # 1_Predict.py, 2_Model_Results.py
├── tests/              # Unit and smoke tests (pytest)
├── report/             # notes.md, draft_report.md, slide_outline.md
│   └── figures/        # 13 saved charts
└── requirements.txt
```

## Models & Results

Three models were trained and evaluated on two time-based tests (Standard and Holiday-season):

| Model | Standard R² | Standard RMSE | Holiday-season R² | Holiday-season RMSE |
|---|---|---|---|---|
| Baseline | 0.9634 | $102,108 | 0.7865 | $303,261 |
| Linear Regression | 0.9658 | $98,689 | 0.9175 | $188,502 |
| Decision Tree | 0.9570 | $110,602 | 0.8844 | $223,154 |
| Random Forest | 0.9692 | $93,617 | 0.9116 | $195,151 |

**Linear Regression** is the best model with the lowest mean RMSE across both evaluations ($143,596). All models significantly beat the baseline, though the holiday-season test (Thanksgiving and Christmas weeks) is substantially harder than the standard test, revealing challenges that ordinary weeks hide.

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Web App:** Streamlit
- **Testing:** pytest

## Author

B.Tech AIML Lab Project (CCE3131)

## License

Educational project - for academic use only.
