# Walmart Weekly Sales Forecasting

AI-based demand forecasting models for Walmart store-level weekly sales, built as a mini project for AIML Lab (CCE3131).

## Project Summary

This project predicts **weekly sales per Walmart store** using regression models, then wraps the best model in a small **Streamlit web app**. The dataset covers 45 Walmart stores with weekly data from February 2010 to October 2012 (6,435 rows).

**Models evaluated:** Linear Regression, Decision Tree, Random Forest  
**Best model:** Linear Regression (R² = 0.95, RMSE = $119,661 on time-based test split)  
**Scope:** Weekly store-level sales only. No product-level demand or inventory planning.

## Folder Structure

```
WalmartCast/
├── PROJECT_PLAN.md          # Full project plan and agent rules
├── README.md                # This file
├── requirements.txt         # Python dependencies (pinned versions)
├── .gitignore
├── data/
│   ├── Walmart.csv          # Raw dataset (do not modify)
│   └── cleaned.csv          # Cleaned + feature-engineered data
├── notebooks/
│   ├── 01_eda_cleaning.ipynb   # EDA, cleaning, feature engineering
│   └── 02_modeling.ipynb       # Model training, evaluation, interpretation
├── models/
│   ├── best_model.joblib    # Saved best model pipeline
│   ├── metrics.json         # All model metrics (3 models × 2 splits)
│   └── feature_info.json    # Feature column info
├── app/
│   ├── app.py               # Streamlit app — Predict page
│   └── pages/
│       └── 2_Model_Comparison.py  # Streamlit — Model Comparison page
└── report/
    ├── notes.md             # Running project notes and findings
    └── figures/             # All saved EDA and model figures
```

## Setup

### Prerequisites
- Python 3.10 or later

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/modeduser/WalmartCast.git
cd WalmartCast

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Notebooks

```bash
# Register the kernel (first time only)
python -m ipykernel install --user --name=walmartcast --display-name="Python (WalmartCast)"

# Open Jupyter and run notebooks in order:
# 1. notebooks/01_eda_cleaning.ipynb
# 2. notebooks/02_modeling.ipynb
```

Both notebooks should run top-to-bottom without errors.

## Running the Streamlit App

```bash
streamlit run app/app.py
```

The app has two pages:
1. **Predict** — Enter store number, date, and features to get a sales prediction
2. **Model Comparison** — View metrics table, RMSE/R² charts, and feature importance

## Key Results

| Model | Time-based RMSE | Time-based R² |
|---|---|---|
| **Linear Regression** | **$119,661** | **0.9497** |
| Random Forest | $299,265 | 0.6854 |
| Decision Tree | $362,275 | 0.5390 |

## Dataset

Source: [Kaggle Walmart Sales Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

| Column | Description |
|---|---|
| Store | Store number (1–45) |
| Date | Week date (dd-mm-yyyy) |
| Weekly_Sales | Sales for that store-week (target) |
| Holiday_Flag | 1 if holiday week |
| Temperature | Regional temperature (°F) |
| Fuel_Price | Regional fuel price |
| CPI | Consumer Price Index |
| Unemployment | Regional unemployment rate |
