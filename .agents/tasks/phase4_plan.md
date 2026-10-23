# Implementation Plan: Phase 4 — Streamlit Web App

## Context

This plan implements Phase 4 of WALMARTCAST_PLAN.md: building a clean, modern Streamlit web app with three pages (Home, Predict, Model Results). The app displays sales predictions and model results using the trained Linear Regression model (best_model.joblib) and existing data artifacts (metrics.json, test_predictions.csv, feature_info.json).

**Key facts verified:**
- Streamlit 1.66.0 is installed (supports `st.container(border=True)` — added in v1.29)
- Best model: Linear Regression, R² = 0.9658 (Standard), 0.9175 (Holiday-season)
- Feature columns: Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, WeekOfYear, Quarter
- Date range: 2010-02-05 to 2012-10-26 (6,435 rows)
- Figures available: 13 saved figures including feature importance, depth sweep, actual vs predicted
- Tests exist in tests/ using pytest; pattern: sys.path.insert(0, project_root)
- app/pages/ directory exists but is empty (only .gitkeep)

## Implementation Plan

- [ ] 1. Create app/style.css with Inter font, card styling, color palette, and Streamlit element customization.
      CSS rules: Google Font import for Inter, html/body/h1-h3 font stack with Inter and system fallback, .main content max-width 1200px with centered layout and reduced padding, .stMetric value font-size 2rem and font-weight 600, .stButton > button primary style with #2563EB background and 10px border-radius, custom classes .card (white background, border-radius 12px, box-shadow soft, padding 1.5rem), footer visibility hidden.
      Files: app/style.css
      Verify: File exists; CSS rules match spec above.

- [ ] 2. Create app/ui.py with helper functions for consistent UI components.
      Functions:
      - `load_css()`: reads app/style.css and injects via st.markdown with unsafe_allow_html=True
      - `page_header(title: str, subtitle: str)`: renders st.title(title) and st.markdown(subtitle) with custom gray color
      - `kpi_row(items: list[dict])`: creates st.columns(len(items)), each with st.container(border=True) containing st.metric(label, value, delta) from item dict keys
      - `callout(kind: str, text: str)`: renders st.info/warning/error based on kind parameter
      - `chart_layout(fig)`: applies Plotly layout updates (paper_bgcolor white, plot_bgcolor white, font family Inter, grid color light gray, primary color #2563EB, legend orientation h at top, hovermode x unified, separatethousands True); returns updated fig
      Files: app/ui.py
      Verify: `python -c "import sys; sys.path.insert(0, r'c:\Users\yashk\OneDrive\Documents\Projects\WalmartCast'); from app.ui import load_css, page_header, kpi_row, callout, chart_layout"` runs without error.

- [ ] 3. Create app/loaders.py with cached data and model loaders.
      Functions (all with appropriate @st.cache_resource or @st.cache_data):
      - `load_model()`: joblib.load('models/best_model.joblib'), returns pipeline; on error shows st.error friendly message and returns None
      - `load_metrics()`: pd.read_json('models/metrics.json'), returns DataFrame
      - `load_feature_info()`: json.load('models/feature_info.json'), returns dict
      - `load_test_predictions()`: pd.read_csv('models/test_predictions.csv'), parse Date column, returns DataFrame
      - `load_cleaned_data()`: pd.read_csv('data/cleaned.csv', parse_dates=['Date'], dayfirst=True), returns DataFrame
      All paths relative to project root. Use try-except blocks with friendly st.error messages like "Run notebook 02 to create the model" instead of raising exceptions.
      Files: app/loaders.py
      Verify: `python -c "import sys; sys.path.insert(0, r'c:\Users\yashk\OneDrive\Documents\Projects\WalmartCast'); import streamlit; from app.loaders import load_model, load_metrics, load_feature_info, load_test_predictions, load_cleaned_data"` runs without error (may show Streamlit warnings, but no import errors).

- [ ] 4. Create app/app.py (Home page) with project overview, KPIs, and summary charts.
      Structure:
      - sys.path setup: `if project_root not in sys.path: sys.path.insert(0, project_root)` where project_root is Path(__file__).parent.parent
      - st.set_page_config(page_title="WalmartCast", page_icon="📈", layout="wide")
      - Call ui.load_css()
      - ui.page_header("WalmartCast", "Predict weekly Walmart store sales from historical data")
      - KPI row with 4 metrics: Stores (45), Weekly Records (6,435), Date Range (from feature_info.json training_date_range), Best Model (from feature_info.json best_model_name + metrics.json Standard R² for that model, formatted as "Linear Regression (R² 0.97)")
      - Two columns with charts: (left) Total weekly sales over time line chart using cleaned.csv grouped by Date with holiday weeks highlighted in amber, (right) Average sales by month bar chart
      - Three columns explaining workflow: "Clean & explore" / "Train 3 models" / "Predict & compare"
      - Callout box listing limitations: weekly store-level only, 2010-2012 data, models trained on historical period, economic variables add little predictive power
      All data from loaders, no hard-coded values. Charts styled with ui.chart_layout.
      Files: app/app.py
      Verify: `streamlit run app/app.py` from project root; page loads without errors; 4 KPI cards display; 2 charts render; no hard-coded numbers visible in source.

- [ ] 5. Create app/pages/1_Predict.py (Predict page) with input form and prediction display.
      Structure:
      - sys.path setup same as app.py
      - st.set_page_config(page_title="Predict Sales", page_icon="🔮", layout="wide")
      - Call ui.load_css() and ui.page_header
      - Two-column layout: left col = inputs, right col = results
      - Left column inputs (inside st.container(border=True)):
        * Store number selectbox (1-45)
        * Week date_input with min_value 2010-02-05, max_value 2012-10-26 (snap to nearest Friday in cleaned.csv data)
        * Holiday_Flag checkbox (default False)
        * Temperature number_input (default from cleaned.csv for that store-week if available)
        * Fuel_Price number_input (default from cleaned.csv)
        * CPI number_input (default from cleaned.csv)
        * Unemployment number_input (default from cleaned.csv)
        * Predict button (primary style)
        * Prefill logic: on store/date change, look up that store-week in cleaned.csv and prefill all feature inputs
      - Right column results (displayed after button click):
        * Large st.metric showing predicted Weekly_Sales with delta vs that store's average from cleaned.csv
        * Line chart of that store's historical weekly sales from cleaned.csv with predicted point highlighted in amber
        * If the selected week exists in cleaned.csv, show actual sales and difference with caption "this week was in training data, see Model Results page for held-out accuracy"
        * Limitation caption below chart
      - Prediction: create DataFrame with feature columns from inputs, derive Year/Month/WeekOfYear/Quarter from date, call model.predict()
      Files: app/pages/1_Predict.py
      Verify: `streamlit run app/app.py`, navigate to Predict page; select store 1, pick date 2012-06-01, inputs prefill; click Predict; prediction displays with chart; no exceptions in console.

- [ ] 6. Create app/pages/2_Model_Results.py (Model Results page) with leaderboard, comparisons, and interpretation tabs.
      Structure:
      - sys.path setup same as app.py
      - st.set_page_config(page_title="Model Results", page_icon="📊", layout="wide")
      - Call ui.load_css() and ui.page_header
      - Header with "Best Model" badge (styled with st.markdown custom HTML/CSS) showing Linear Regression and one-sentence summary generated from metrics.json: "Linear Regression achieved R² of 0.9658 on standard test and 0.9175 on holiday-season test"
      - Leaderboard table: filter metrics.json to Baseline + 3 models, show columns Model | Standard RMSE | Standard R² | Holiday RMSE | Holiday R² with best model row highlighted (st.dataframe with styling)
      - Expander "What do MAE, MSE, RMSE and R² mean?" with 4 plain sentences
      - Four tabs (st.tabs): "Compare Models", "Actual vs Predicted", "What Drives Sales", "Lessons"
      - Tab 1 "Compare Models": grouped bar charts (Plotly) showing RMSE and R² by model for Standard and Holiday-season evaluations from metrics.json, styled with ui.chart_layout
      - Tab 2 "Actual vs Predicted": store selectbox (1-45) and evaluation radio (Standard/Holiday-season); line chart from test_predictions.csv showing Actual vs Linear/Tree/Forest predictions for that store-evaluation, styled with ui.chart_layout
      - Tab 3 "What Drives Sales": display model_feature_importance.png from report/figures/ with st.image; caption generated from feature_info.json explaining that Store is most important, followed by time features, while economic variables contribute less
      - Tab 4 "Lessons": display diag_rf_depth_sweep.png if exists; three markdown sentences explaining why holiday-season test matters (the standard test has no holiday weeks since data ends October 2012) and what depth sweep teaches (shallow trees cannot learn store patterns with one-hot encoding)
      All text/numbers generated from data files, no hard-coded values.
      Files: app/pages/2_Model_Results.py
      Verify: `streamlit run app/app.py`, navigate to Model Results page; leaderboard displays with Linear Regression highlighted; all 4 tabs load and show content; figures display; no exceptions.

- [ ] 7. Create tests/test_app_smoke.py using streamlit.testing.v1.AppTest for smoke tests.
      Test functions:
      - `test_home_page_loads()`: AppTest.from_file("app/app.py").run(); assert not at.exception
      - `test_predict_page_loads()`: AppTest.from_file("app/pages/1_Predict.py").run(); assert not at.exception
      - `test_predict_returns_prediction()`: load Predict page, set store selectbox to 1, click predict button, assert prediction metric appears in output
      - `test_model_results_loads()`: AppTest.from_file("app/pages/2_Model_Results.py").run(); assert not at.exception
      Add sys.path.insert(0, project_root) at top of test file. All tests should pass if data artifacts exist.
      Files: tests/test_app_smoke.py
      Verify: `pytest tests/test_app_smoke.py -v` from project root; all tests pass.

- [ ] 8. Update README.md Models & Results section with instructions to run the Streamlit app.
      Add after "Run Tests" section:
      ```markdown
      ### Run Web App
      
      ```bash
      streamlit run app/app.py
      ```
      
      The app has three pages:
      - **Home**: Overview, KPIs, and summary charts
      - **Predict**: Interactive sales prediction for any store-week
      - **Model Results**: Model comparison, accuracy metrics, and feature importance
      ```
      Replace placeholder "Results will be added after Phase 2" with actual results table formatted from metrics.json showing Model, Standard R², Holiday R², and mean RMSE.
      Files: README.md
      Verify: README.md contains Streamlit run instructions and results table with actual numbers from metrics.json.

- [ ] 9. Create docs/ui directory and verify app functionality across all pages.
      Steps:
      - Run `streamlit run app/app.py` from project root
      - Open Home page, verify 4 KPIs display with correct numbers, 2 charts render
      - Navigate to Predict page, select store 1, date 2012-06-01, verify inputs prefill, click Predict, verify prediction displays with chart
      - Navigate to Model Results page, verify leaderboard shows 4 models, all 4 tabs load with content
      - Check browser console (F12) for any JavaScript errors or Streamlit warnings
      - Take screenshots at 1440px width: home.png, predict.png, model_results.png and save to docs/ui/
      - Verify no layout overflow or cramped spacing
      Files: docs/ui/home.png, docs/ui/predict.png, docs/ui/model_results.png (screenshots)
      Verify: All three pages load cleanly; `pytest tests/test_app_smoke.py` passes; screenshots saved; commit with message "Complete Phase 4: Streamlit web app".

## Notes

- **Streamlit 1.66.0 compatibility**: Version 1.66.0 supports st.container(border=True) (added in 1.29), st.tabs, st.columns, st.metric with delta, date_input with min/max, and all standard widgets used in this plan. No deprecated APIs used.
- **sys.path pattern**: All app files use `sys.path.insert(0, str(Path(__file__).parent.parent))` to enable imports from src/ when running `streamlit run app/app.py` from project root.
- **CSS injection**: style.css is loaded once via st.markdown in ui.load_css() and applies to all pages due to Streamlit's session state.
- **Data loading**: loaders.py uses @st.cache_data for CSV/JSON (mutable data) and @st.cache_resource for model (immutable object). All loaders handle missing files gracefully with st.error instead of raising exceptions.
- **Plotly chart styling**: ui.chart_layout applies consistent styling (Inter font, blue #2563EB primary color, amber #F59E0B highlights, white background, light grid, horizontal legend, thousand separators in hover).
- **Prefill logic**: Predict page looks up store-week in cleaned.csv and prefills all numeric inputs so users can tweak for what-if scenarios.
- **Verification**: Each step includes a concrete verification command. Final verification runs all smoke tests and manual UI checks before commit.
