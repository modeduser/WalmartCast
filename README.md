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
# Start Jupyter
jupyter notebook

# Run in order:
# 1. notebooks/01_eda_cleaning.ipynb
# 2. notebooks/02_modeling.ipynb
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
├── data/               # Walmart.csv, cleaned.csv
├── src/                # Reusable feature and evaluation code
├── notebooks/          # EDA and modeling notebooks
├── models/             # Trained models and metrics
├── app/                # Streamlit web application
├── tests/              # Unit tests
├── report/             # Findings, figures, documentation
└── requirements.txt    # Python dependencies
```

## Models & Results

Results will be added after Phase 2 (modeling) is complete.

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Web App:** Streamlit
- **Testing:** pytest

## Author

B.Tech AIML Lab Project (CCE3131)

## License

Educational project - for academic use only.
