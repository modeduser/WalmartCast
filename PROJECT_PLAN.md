# PROJECT_PLAN.md — AI-Based Demand Forecasting Models (Walmart Weekly Sales)

> **Instruction to the AI agent:** Read this entire file before doing anything in this repo.
> Re-read the "Agent rules" and the current day's section at the start of every new session.

---

## 1. Context

- **What:** AIML Lab (CCE3131) mini project. Predict **weekly sales per Walmart store** using regression models, then wrap the result in a small Streamlit web app.
- **Owner:** a B.Tech student who is new to this workflow. The owner must be able to **explain every line** in a viva.
- **Target:** a ~90% finished, feedback-ready draft in 3 days. The teacher will review it, then the owner has more time to revise. Do not over-engineer.
- **Dataset:** Kaggle Walmart sales dataset, file at `data/Walmart.csv` (6,435 rows, 45 stores, weekly data from Feb 2010 to Oct 2012).

| Column | Meaning | Role |
|---|---|---|
| Store | Store number (1–45) | Feature (categorical) |
| Date | Week date, format `dd-mm-yyyy` | Source of time features |
| Weekly_Sales | Sales for that store-week | **Target** |
| Holiday_Flag | 1 if holiday week, else 0 | Feature |
| Temperature | Regional temperature (°F) | Feature |
| Fuel_Price | Regional fuel price | Feature |
| CPI | Consumer Price Index | Feature |
| Unemployment | Regional unemployment rate | Feature |

**Scope limit (state this in the report):** weekly store-level sales only. No product-level demand, no inventory planning.

---

## 2. Agent rules (follow always)

1. **Explain as you go.** After writing code, give a short plain-English explanation of what it does and why, suitable for a student.
2. **Small steps.** Do one task at a time from the current day's checklist. Stop and wait for the owner's confirmation before starting the next big task.
3. **Don't silently change the plan.** If you think the plan should change, say so and ask first.
4. **Never delete or overwrite** the raw data file or existing notebooks without asking. Write cleaned data to `data/cleaned.csv`.
5. **Reproducibility:** use `random_state=42` everywhere. Pin versions in `requirements.txt`.
6. **No data leakage.** Fit scalers/encoders on training data only (use scikit-learn `Pipeline`).
7. **Comment code** with short comments in simple language. Prefer clear code over clever code.
8. **Save every figure** to `report/figures/` with a descriptive file name (e.g. `eda_monthly_seasonality.png`).
9. **Keep a running notes file** `report/notes.md`: after each result, append 1–3 plain sentences describing what was found and the key numbers. This becomes the report.
10. **Be honest about results.** If a model does poorly, report it and explain why. Do not tune until numbers look good without saying so.
11. **Do not add extra dependencies or frameworks** (no React, Flask, FastAPI, databases) unless the owner asks.
12. Give a git commit message suggestion after each completed task.

---

## 3. Key technical decisions (already made)

- **Primary evaluation = time-based split.** Train on earlier weeks, test on the latest weeks (about the last 20% of dates, roughly Jan–Oct 2012; decide the exact cutoff date from the data and print it).
- **Secondary evaluation = random 80/20 split**, reported only for comparison. Explain in the report that the random split is optimistic because neighbouring weeks of the same store land in both train and test.
- **Models (exactly three):** Linear Regression, Decision Tree Regressor, Random Forest Regressor.
- **Metrics:** MAE, MSE, RMSE, R² (on the test set).
- **No tuning in the draft.** Use fixed defaults: Decision Tree `max_depth=8`, `min_samples_leaf=5`; Random Forest `n_estimators=200`, `max_depth=12`, `n_jobs=-1`. Tuning is a post-feedback task.
- **Features:** Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, WeekOfYear, Quarter. Drop the raw `Date` from the model input.
- **Front end:** Streamlit, two pages in the draft (Predict, Model comparison).
- **App limitation:** data covers only 2010–2012 and tree models cannot extrapolate. The date picker must be limited to the dataset's date range, with a visible note.

---

## 4. Repository structure

```
walmart-demand-forecasting/
├── PROJECT_PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── Walmart.csv
│   └── cleaned.csv
├── notebooks/
│   ├── 01_eda_cleaning.ipynb
│   └── 02_modeling.ipynb
├── models/
│   ├── best_model.joblib
│   └── metrics.json
├── app/
│   ├── app.py
│   └── pages/
│       └── 2_Model_Comparison.py
└── report/
    ├── notes.md
    └── figures/
```

`requirements.txt` (pin the versions actually installed): pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, streamlit, plotly, jupyter/ipykernel.

---

## 5. Day-by-day plan

### Day 1 — Data and understanding (notebook `01_eda_cleaning.ipynb`)

- [ ] **1.1 Setup:** create the virtual environment, install requirements, create the folder structure, `git init`, first commit.
- [ ] **1.2 Load:** read `data/Walmart.csv`; show shape, dtypes, `head()`; parse `Date` with `dayfirst=True`.
- [ ] **1.3 Cleaning:** check missing values, duplicates, invalid values (e.g. negative sales), date range, and rows per store (should be 143 each). Check outliers with the IQR method **but do not remove holiday spikes** — they are real. Log every decision in `report/notes.md`.
- [ ] **1.4 Save** `data/cleaned.csv`.
- [ ] **1.5 EDA graphs** (each saved to `report/figures/`, each followed by a markdown cell with 1–2 observations):
  1. Distribution of `Weekly_Sales`
  2. Total sales over time (all stores)
  3. Average sales per store (all 45, sorted)
  4. Holiday vs non-holiday sales
  5. Monthly / week-of-year seasonality (note the November–December effect and that the week before Christmas is not flagged as a holiday)
  6. Correlation heatmap of numeric columns
  7. Sales vs Temperature / CPI / Unemployment scatter (shows how weak the relationships are)
- [ ] **1.5b** Write a summary of EDA findings in `report/notes.md`.
- [ ] **1.6 Feature engineering:** add Year, Month, WeekOfYear, Quarter. Save the engineered dataframe as `data/cleaned.csv` (or a clearly named second file).

**Day 1 acceptance:** the notebook runs top to bottom with no errors; 7 figures are saved; notes are written; work is committed.

### Day 2 — Models (notebook `02_modeling.ipynb`)

- [ ] **2.1 Split:** build the time-based split (print the cutoff date and train/test sizes) and the random split.
- [ ] **2.2 Pipelines:** one `Pipeline` per model, using a `ColumnTransformer`: one-hot encode `Store`; scale numeric columns for Linear Regression only; pass numeric columns through for trees.
- [ ] **2.3 Train and evaluate** all three models on both splits. Build a results table (rows = model × split, columns = MAE, MSE, RMSE, R²). Save to `models/metrics.json` and show in the notebook.
- [ ] **2.4 Interpretation:**
  - Actual vs predicted line plot for 3 stores (test period)
  - Predicted vs actual scatter for the best model
  - Residual histogram
  - Random Forest feature importance bar chart
- [ ] **2.5 Choose the best model** by **time-based** RMSE and R². Save the fitted pipeline to `models/best_model.joblib`. Also save the list of feature names/order used.
- [ ] **2.6 Sanity check:** load the saved model in a fresh cell and predict for one example row.
- [ ] **2.7** Update `report/notes.md` with the results table and 3–5 sentences of interpretation, including an honest explanation of the gap between random-split and time-split scores.

**Day 2 acceptance:** the results table exists for 3 models × 2 splits; the best model is saved and reloads; figures are saved; work is committed.

### Day 3 — App and draft documents

- [ ] **3.1 `app/app.py` (Predict page):**
  - Inputs: Store (1–45), week (date picker limited to the dataset's date range), Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment.
  - Derive Year, Month, WeekOfYear, Quarter from the date.
  - Load `models/best_model.joblib` with `@st.cache_resource`.
  - Show the predicted weekly sales. If the chosen store + week exists in the data, also show the actual value and the error.
  - Show a short limitation note.
  - Handle errors gracefully (missing model file, bad input).
- [ ] **3.2 `app/pages/2_Model_Comparison.py`:** read `models/metrics.json`; show the metrics table, a bar chart of RMSE and R² per model, and the feature importance chart (Plotly).
- [ ] **3.3 Test the app:** `streamlit run app/app.py`; try at least 5 different inputs; fix errors.
- [ ] **3.4 `README.md`:** project summary, folder structure, how to set up the environment, how to run the notebooks and the app.
- [ ] **3.5 Draft report** (rough prose is fine, every heading present): Introduction, Dataset, Methodology, EDA findings, Models and evaluation, Results and interpretation, Application screenshots, Limitations, Conclusion, Future scope. Build it from `report/notes.md` and `report/figures/`.
- [ ] **3.6 Slide outline** (6–8 slides): problem, dataset, method, EDA highlights, results table, app demo, limitations/future work.
- [ ] **3.7 Questions for the teacher** (write into `report/notes.md`): is the time-based split acceptable as primary evaluation? Should extra models be added? Should the front end be evaluated? How formal/long should the report be? Is tuning required?

**Day 3 acceptance:** the app runs and predicts; the README is usable by a stranger; the draft report and slide outline exist; everything is committed and pushed to GitHub.

---

## 6. After teacher feedback (not for the draft)

- Hyperparameter tuning with `TimeSeriesSplit` (small grid).
- Extra model (e.g. Gradient Boosting) if the teacher approves.
- Lag features (previous week's sales for the same store) and a re-run of the evaluation.
- Explore page in the app (store/date filters, trend plot).
- Deploy on Streamlit Community Cloud.
- Final report formatting and full presentation.

---

## 7. Definition of "90% done"

- [ ] Notebooks run end to end without errors
- [ ] 3 models × 4 metrics × 2 splits are reported
- [ ] Best model saved and used by the app
- [ ] App runs with `streamlit run app/app.py`
- [ ] README, draft report, slide outline exist
- [ ] Code is on GitHub with a sensible commit history
- [ ] Owner can explain: why a time-based split, what the pipeline does, what the metrics mean, what the feature importance shows
