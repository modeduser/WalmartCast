# WALMARTCAST_PLAN.md — AI-Based Demand Forecasting Models (Walmart Weekly Sales)

> **Instruction to the AI agent:** This is a brand-new project in an empty folder. Read this whole file before doing anything, and re-read the "Agent rules" and the current phase at the start of every new session.
> Work through the phases **in order**. After each phase: summarize what you did and the key numbers, commit, and **stop and wait for my "go"**.
> This is a **mini project for a university AIML lab**. Keep everything simple and explainable. Do not add anything not listed here.

---

## 1. Project in plain words

**WalmartCast** predicts the **weekly sales of a Walmart store** from historical data. A user picks a store and a week (and conditions such as holiday week, temperature, fuel price, CPI and unemployment) and the app shows the predicted sales. Behind it, three regression models (**Linear Regression, Decision Tree, Random Forest**) are trained on older weeks, tested on newer weeks they have not seen, compared with **MAE, MSE, RMSE and R²**, and the best one is used in a small web app.

- **Owner:** a B.Tech student (AIML Lab CCE3131). I must be able to **explain every line** in a viva. Explain things in simple language; comment code simply.
- **Dataset:** the Kaggle Walmart store sales dataset, which I will place at `data/Walmart.csv`.
- **Scope (state this in the report):** weekly, store-level sales only. No product-level demand and no inventory planning.

### Dataset facts (VERIFY each one and report any mismatch; do not assume)
| Column | Meaning | Role |
|---|---|---|
| Store | Store number (1–45) | Feature |
| Date | Week date, format `dd-mm-yyyy` (Fridays) | Source of time features |
| Weekly_Sales | Sales for that store-week | **Target** |
| Holiday_Flag | 1 if holiday week, else 0 | Feature |
| Temperature | Regional temperature (°F) | Feature |
| Fuel_Price | Regional fuel price | Feature |
| CPI | Consumer Price Index | Feature |
| Unemployment | Regional unemployment rate | Feature |

Expected: 6,435 rows = 45 stores × 143 weeks, from 2010-02-05 to 2012-10-26, no missing values. Four holiday weeks per year are flagged (Super Bowl, Labor Day, Thanksgiving, Christmas). The data ends in October 2012, so **Thanksgiving/Christmas 2012 are not in the data**.

### Synopsis objectives → where they are done
| Objective | Phase |
|---|---|
| 1. Understand and clean the data | 1 |
| 2. Explore sales patterns across stores and time | 1 |
| 3. Turn Date into time-based features | 2 |
| 4. Study the effect of holiday, economic and environmental variables | 1 (EDA) and 2 (feature importance) |
| 5. Train and compare multiple regression models | 2 |
| 6. Evaluate with standard regression metrics | 2 |
| 7. Select and interpret the best model on unseen data | 2–3 |

---

## 2. Agent rules (always)

1. Explain each step in plain language; one task at a time; stop at the end of every phase.
2. **No data leakage:** fit encoders and scalers on training data only, by using scikit-learn `Pipeline`s.
3. `random_state=42` everywhere. Pin package versions in `requirements.txt`.
4. Never modify `data/Walmart.csv`; write cleaned data to `data/cleaned.csv`.
5. Put reusable logic in `src/` and import it from the notebooks **and** the app, so training and the app use identical feature code.
6. Save every figure to `report/figures/` with a descriptive name. After each result, append 1–3 plain sentences to `report/notes.md` (findings and key numbers).
7. **Be honest.** Report whatever the numbers show. Do not tune against the test periods beyond what this plan specifies.
8. Notebooks must be run with "Restart kernel and Run All" and **saved with their outputs visible**.
9. Dependencies only: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, streamlit, joblib, pytest, ipykernel. No other frameworks, no databases.
10. Check the OS and Python version first (need Python 3.10+). If something is missing, tell me how to install it and wait.
11. Suggest a git commit message after each task.

---

## 3. Phase 0 — Setup

- [ ] 0.1 Confirm `data/Walmart.csv` exists (if the file has another name, ask me before renaming).
- [ ] 0.2 Create a virtual environment `.venv` and activate it. Install the packages in rule 9, then write `requirements.txt` with the exact installed versions.
- [ ] 0.3 Create this structure:

```
walmartcast/
├── WALMARTCAST_PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml
├── data/            Walmart.csv, cleaned.csv
├── src/             __init__.py, features.py, evaluate.py
├── notebooks/       01_eda_cleaning.ipynb, 02_modeling.ipynb
├── models/          best_model.joblib, metrics.json, feature_info.json, test_predictions.csv
├── app/             app.py, ui.py, loaders.py, style.css, pages/
├── tests/
└── report/          notes.md, figures/
```

- [ ] 0.4 `.gitignore`: `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`. Keep `data/` and `models/` tracked.
- [ ] 0.5 `git init`, first commit, no remote yet. Create `report/notes.md` with the heading "Project Notes".

**Acceptance:** folders exist, the environment works (`python -c "import sklearn, streamlit"`), first commit made.

---

## 4. Phase 1 — Data cleaning and EDA (`notebooks/01_eda_cleaning.ipynb`)

- [ ] 1.1 Load the CSV, parse `Date` with `dayfirst=True`. Show shape, dtypes, `head()`, `describe()`, date range, number of stores, weeks per store. Compare with the expected facts above.
- [ ] 1.2 Cleaning checks, each followed by a markdown cell explaining the check and the finding: missing values, duplicates, zero/negative sales, equal weeks per store, IQR outliers in `Weekly_Sales` (**report only; do not remove**, because holiday spikes are real data).
- [ ] 1.3 Save `data/cleaned.csv` after adding the time features `Year, Month, WeekOfYear (ISO), Quarter`.
- [ ] 1.4 EDA graphs (each saved to `report/figures/` and followed by 1–2 sentences of observation):
  1. Distribution of `Weekly_Sales`
  2. Total sales over time (all stores), with the holiday weeks marked
  3. Average sales per store (all 45, sorted)
  4. Holiday vs non-holiday average sales
  5. Seasonality: average sales by month and by week of year (the November–December peak; note the week before Christmas is **not** flagged as a holiday but is usually very high)
  6. Correlation heatmap of the numeric columns
  7. Sales vs Temperature, Fuel_Price, CPI, Unemployment (scatter; shows how weak these relationships are)
- [ ] 1.5 Write an "EDA findings" summary in `notes.md` that answers objective 4 in plain words (which variables seem to matter and which don't).

**Acceptance:** notebook runs top to bottom with outputs saved and no errors; 7 figures saved; `cleaned.csv` exists; commit.

---

## 5. Phase 2 — Features, models, evaluation (`src/`, `notebooks/02_modeling.ipynb`)

### 5.1 Features
`Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, WeekOfYear, Quarter`. The raw `Date` is not a model input.

### 5.2 Two evaluations, both time-based (train on the past, test on the future)
| Evaluation | Train: Date < | Test window | Why |
|---|---|---|---|
| **Standard test** | 2012-04-13 | 2012-04-13 to end of data | Ordinary weeks, the latest data |
| **Holiday-season test** | 2011-10-28 | 2011-10-28 ≤ Date < 2012-02-01 | Thanksgiving and Christmas, the hardest weeks |

Print the exact cutoff dates and the train/test row counts. Also run a **random 80/20 split** as a secondary comparison and interpret it from the numbers only. Add a note: *the holiday-season test is trained on only one earlier holiday season (2010), so its results are indicative, not conclusive.*

Why two tests (explain in the notebook): the data ends in October 2012, so the standard test contains no Thanksgiving or Christmas weeks. A model that ignores holidays can look excellent there.

### 5.3 Models (`src/features.py` exposes `make_pipeline(name)` for `"linear"`, `"tree"`, `"forest"`)
- **Linear Regression:** one-hot encode `Store` **and** `WeekOfYear` (`handle_unknown="ignore"`); `StandardScaler` on the other numeric columns.
- **Decision Tree:** one-hot encode `Store`; numeric columns pass through; `max_depth=None, min_samples_leaf=3, random_state=42`.
- **Random Forest:** one-hot encode `Store`; numeric columns pass through; `n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1`.

Add a one-line **baseline** (not a model family): predict each store's mean `Weekly_Sales` from the training data. Every real model should beat it.

Add a markdown note: *these settings are sensible defaults picked after a short diagnostic (§5.5), not formal hyperparameter tuning. Formal tuning is future work.*

### 5.4 Results
- [ ] One results table: baseline + 3 models × (Standard, Holiday-season, Random split) × (MAE, MSE, RMSE, R²). Show it in the notebook and save to `models/metrics.json` (list of rows: `Model, Evaluation, MAE, MSE, RMSE, R2`).
- [ ] **Best model** = lowest mean RMSE across the Standard and Holiday-season tests. If the top two are within 3% of each other, call it an explicit tie and prefer the one that does better in the holiday-season test.
- [ ] Save `models/test_predictions.csv` with columns `Evaluation, Store, Date, Actual, Linear, Tree, Forest` for both time-based tests (the app uses it).

**Sanity targets** (from an independent run with exactly these settings; yours should land within about **±0.02 R²** — if not, stop and investigate before continuing):

| Model | Standard test R² | Holiday-season test R² |
|---|---|---|
| Linear Regression | 0.966 | 0.918 |
| Decision Tree | 0.956 | 0.885 |
| Random Forest | 0.969 | 0.911 |

Expect the three models to be close; do not "force" a winner.

### 5.5 Diagnostics and interpretation figures
- [ ] **Optional lesson (one figure):** Random Forest with one-hot `Store` on the Standard test for `max_depth in [4, 8, 12, 16, 20, None]`; plot R² vs depth. Expected: R² rises from about 0.36 (depth 4) to about 0.95 (no limit). Explain in plain language: with one-hot stores, each split can isolate only one store, so shallow trees run out of depth before learning store-level sales. Save as `report/figures/diag_rf_depth_sweep.png`.
- [ ] Actual vs predicted line plots for 3 stores (small, medium, large) over the Standard test **and** over the Holiday-season test.
- [ ] Total weekly sales (sum over stores) actual vs predicted for each model across the Holiday-season test (shows how each model handles Thanksgiving and the pre-Christmas week).
- [ ] Predicted-vs-actual scatter and a residual histogram for the best model.
- [ ] **Feature importance** of the Random Forest, **aggregated by original feature** (sum all `Store_*` one-hot columns into one "Store" bar and show the other features individually). Save as `report/figures/model_feature_importance.png`. Explain what it says about objective 4.

### 5.6 Notes
Update `notes.md` with the results table and 3–6 plain sentences of interpretation: which model won and by how much, how holiday weeks went, whether Temperature/Fuel_Price/CPI/Unemployment help.

**Acceptance:** both notebooks run end to end with saved outputs; results table exists; every model beats the baseline; sanity targets are met (or differences are explained); figures saved; commit.

---

## 6. Phase 3 — Saved artifacts and tests

- [ ] 3.1 Refit the **best** pipeline on **all** 6,435 rows and save `models/best_model.joblib`. State in notes and in the app that the reported metrics come from the held-out tests, not from this refit.
- [ ] 3.2 `models/feature_info.json`: best model name, feature list, training date range, `sklearn_version`, selection rule.
- [ ] 3.3 In a fresh cell, reload the model and predict one row; assert the output is a finite positive number.
- [ ] 3.4 `tests/`: (a) feature builder outputs the expected columns and week numbers; (b) metrics functions on a tiny hand-made example; (c) the saved model loads and predicts.
- [ ] 3.5 Run `pytest tests/`.

**Acceptance:** `pytest` passes; artifacts exist; commit. **Stop here so I can review the modeling before the app is built.**

---

## 7. Phase 4 — The web app (Streamlit), simple but polished

Goal: a clean, modern look with very little code. Three pages, one consistent style. All numbers shown come from `metrics.json` and the data files, never hard-coded.

### 7.1 Style foundation
- `.streamlit/config.toml`: `base = "light"`, `primaryColor = "#2563EB"`, `backgroundColor = "#F8FAFC"`, `secondaryBackgroundColor = "#EEF2F7"`, `textColor = "#0F172A"`, `font = "sans serif"`; `[client] toolbarMode = "minimal"`.
- `app/style.css` (loaded once from `app/ui.py`): Inter via the Google Fonts stylesheet with a system fallback; page padding a little tighter and content max width about 1200px; rounded (12px) bordered cards with a very soft shadow; `st.metric` values larger and bold; primary buttons filled blue with 10px radius; headings with a clear size hierarchy; hide the default footer.
- `app/ui.py` helpers: `load_css()`, `page_header(title, subtitle)`, `kpi_row(items)` (cards built from `st.container(border=True)` + `st.metric`), `callout(kind, text)`, and `chart_layout(fig)` that applies one shared Plotly look: white background, light grid, font Inter, blue `#2563EB` as the main series colour and amber `#F59E0B` for highlights or the second series, horizontal legend at the top, thousands separators in hover text, axis titles with units.
- `app/loaders.py`: cached loaders (`st.cache_resource` for the model, `st.cache_data` for CSV/JSON). If a file is missing show a friendly message (for example "Run notebook 02 to create the model"), never a stack trace.
- Each page starts with `sys.path` handling so that `src/` and `app/` can be imported when running `streamlit run app/app.py` from the project root. Check the installed Streamlit version and use only APIs it supports; fix any deprecation warnings.

### 7.2 Pages
**Home (`app/app.py`)**
1. `st.set_page_config(page_title="WalmartCast", page_icon="📈", layout="wide")`.
2. Header: "WalmartCast" and the subtitle "Predict weekly Walmart store sales from historical data".
3. KPI row (4 cards): stores (45), weekly records (6,435), date range, **best model with its Standard-test R²**.
4. Two short charts in cards: total weekly sales over time with holiday weeks highlighted, and average sales by month.
5. "How it works" in three short cards: *Clean & explore → Train 3 models → Predict & compare*.
6. A callout with honest limitations: weekly store-level only, 2010–2012 data, models cannot predict outside this period, economic variables add little.

**Predict (`app/pages/1_Predict.py`)**
- Left card (inputs): **Store** (1–45), **Week** date picker limited to the dataset range (snap to the nearest week in the data and show which week is used), Holiday week toggle, Temperature, Fuel price, CPI, Unemployment. **Prefill** the inputs with that store's values for the chosen week from the data, so the user can tweak them for what-if exploration. A primary **Predict sales** button.
- Right card (result): the predicted sales as a large number; a delta vs that store's average weekly sales (with an up/down colour); a line chart of the store's weekly sales history with the predicted point highlighted in amber.
- If the chosen week exists in the data, also show the actual sales and the difference, with a small note: *this week was part of the model's training data, so see the Model Results page for honest held-out accuracy.*
- A short limitation caption under the chart.

**Model Results (`app/pages/2_Model_Results.py`)**
- Header with a "Best model" chip and a one-sentence summary generated from `metrics.json`.
- Leaderboard table (baseline + 3 models; R² and RMSE for both time-based tests) with the best model highlighted; a small expander "What do MAE, MSE, RMSE and R² mean?" with one plain sentence each.
- Tabs: **Compare models** (grouped bar charts of RMSE and R² per model per evaluation); **Actual vs predicted** (pick a store and one of the two evaluations; chart from `test_predictions.csv` with one line per model); **What drives sales** (the aggregated feature-importance chart plus a plain caption); **Lessons** (the depth-sweep figure if it exists, and three sentences on why the holiday-season test matters).
- All explanatory text with numbers is generated from the files.

### 7.3 Quality checks
- Open every page, click every control, and confirm there are no exceptions or warnings in the console.
- If the IDE has a browser/preview tool, take screenshots of each page at about **1440px** and **390px** width and fix anything that overflows, overlaps or looks cramped (at most 3 refinement rounds). Otherwise run `streamlit run app/app.py` and tell me what to screenshot. Save final screenshots in `docs/ui/`.
- Accessibility basics: readable contrast, no information by colour alone (use labels), chart titles/captions in plain language.
- `tests/test_app_smoke.py` using `streamlit.testing.v1.AppTest`: Home loads; Predict returns a prediction for store 1; Model Results loads.

**Acceptance:** `streamlit run app/app.py` (from the project root) runs all three pages cleanly; smoke tests pass; screenshots saved; commit.

---

## 8. Phase 5 — Documentation

- [ ] `README.md`: project summary, results table (from `metrics.json`), setup steps, how to run the notebooks, tests and app, folder map.
- [ ] `report/draft_report.md`, following the synopsis headings: Introduction; Dataset Description; Problem Statement; Motivation and Objectives; Proposed Methodology; Models and Evaluation; Results; Interpretation (actual vs predicted, feature importance); Application; Limitations; Conclusion; Future Scope. Insert the saved figures. Include a short subsection "Lessons learned": (1) a test window without holiday weeks hides weaknesses, (2) tree depth limits can cripple trees on one-hot features, (3) how seasonality is encoded matters as much as the model family.
- [ ] `report/slide_outline.md`: 8–10 slides (problem, dataset, workflow, EDA highlights, models, results table, holiday-season test, app demo, limitations and future work).
- [ ] Teacher questions in `notes.md`: (1) Is a time-based evaluation (plus a holiday-season test) acceptable as the primary evaluation, with the random split as a comparison? (2) Should hyperparameter tuning be added? (3) Should the front end be part of the evaluated deliverable? (4) How formal and long should the report be?
- Limitations to state: only two holiday seasons in the data; weekly store-level only; models trained on 2010–2012 only; model settings are sensible defaults, not formally tuned.
- Future scope: hyperparameter tuning with `TimeSeriesSplit`, lag features, gradient boosting, a larger daily dataset (for example Rossmann Store Sales), deployment.

**Acceptance:** all numbers in README/report come from `metrics.json`; no placeholder text; commit.

---

## 9. Phase 6 — Final verification

1. Create a fresh virtual environment, install `requirements.txt`, run both notebooks with "Restart and Run All", run `pytest tests/`, run the app.
2. Confirm both notebooks have saved outputs and no error cells.
3. Confirm the results table matches across `metrics.json`, the notebook, README and report.
4. Show `git log --oneline` and `git status` (clean). If I give you a GitHub URL, add it as `origin` and push; do not create the remote yourself.

---

## 10. Out of scope (do NOT do)

Formal hyperparameter tuning, lag/rolling features, XGBoost or other model families, deep learning, other datasets, databases, authentication, deployment, React/FastAPI or any non-Streamlit front end.

## 11. Reporting style

At the end of each phase give: (1) files changed, (2) key numbers (and screenshots in Phase 4), (3) anything that disagreed with this plan or looked odd, (4) a suggested commit message.
