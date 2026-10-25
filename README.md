# WalmartCast

Demand forecasting for 45 Walmart stores using historical weekly sales, holiday indicators, and macroeconomic factors (Temperature, Fuel Price, CPI, Unemployment).

## Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Choose **GitHub** tab and paste the repository URL: `https://github.com/modeduser/WalmartCast`.
3. Open `WalmartCast.ipynb`.
4. Run all cells (`Runtime > Run all`). No extra package installations are required.

## Run the Streamlit App Locally

```bash
# 1. Clone repository
git clone https://github.com/modeduser/WalmartCast.git
cd WalmartCast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch interactive app
streamlit run app/app.py
```

## Project Structure

```text
WalmartCast/
  WalmartCast.ipynb    # Complete end-to-end data cleaning, EDA, and model pipeline
  requirements.txt     # Python dependencies
  README.md            # Quick start guide
  .gitignore           # Ignored files and caches
  app/
    app.py             # Streamlit web application
    Walmart.csv        # Historical dataset
    model.joblib       # Trained production pipeline (Linear Regression)
    results.json       # Model evaluation metrics, test predictions & feature importance
```
