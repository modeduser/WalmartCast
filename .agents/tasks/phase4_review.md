# Phase 4 Streamlit App Review

**Date**: 2025-01-16  
**Reviewer**: Code Review Agent  
**Scope**: WalmartCast Phase 4 Streamlit app implementation

---

## Summary

The Phase 4 Streamlit app delivers all three pages (Home, Predict, Model Results) with consistent styling, dynamic data loading from files, and interactive predictions. Chart styling uses the specified blue/amber palette with Inter font. The prefill logic on the Predict page correctly reads values from cleaned.csv for the selected store and week.

Three findings require attention before approval: the home page reads the best model R² from the first matching row without filtering by evaluation type (HIGH severity — displays incorrect data if rows are sorted Holiday-season before Standard), the chart_layout helper doesn't enforce thousands separators in Plotly hover text as the spec requires (MEDIUM), and the smoke tests verify page loads but don't test button clicks or interactive behavior (MEDIUM). One LOW severity finding: holiday week markers on the home page chart lack a legend or caption.

**Verdict**: CHANGES_REQUESTED

---

## Findings

### HIGH Severity

#### 1. Best model R² extraction on Home page
**File**: `app/app.py`, line 31  
**Issue**: The home page KPI displays the best model's R² value by reading `metrics_df.loc[metrics_df['Model'] == best_model_name, 'R2'].values[0]` without filtering by `Evaluation == 'Standard'`. If `metrics.json` has rows sorted such that Holiday-season appears before Standard for the best model, `.values[0]` will return the Holiday-season R², not the Standard R² that the spec requires (section 7.2: "best model with its Standard-test R²").

**Impact**: The KPI displays incorrect data to the user.

**Fix**: Filter by evaluation before reading the value:
```python
best_r2 = metrics_df.loc[
    (metrics_df['Model'] == best_model_name) & 
    (metrics_df['Evaluation'] == 'Standard'), 
    'R2'
].values[0]
```

The Model Results page uses this pattern correctly at line 19.

**Confidence**: confirmed (traced through code, checked spec requirement)

---

### MEDIUM Severity

#### 2. Missing thousands separators in Plotly hover text
**File**: `app/ui.py`, `chart_layout` function  
**Issue**: The spec requires "thousands separators in hover text" (section 7.1, chart_layout). The current implementation applies `tickformat='$,.0f'` to y-axes (line 58), which adds thousands separators to axis labels but not to hover text. Hover text uses Plotly's default formatting and will show `1234567` instead of `1,234,567` when the user hovers over a data point.

**Impact**: Violates spec requirement; reduces readability of large numbers in hover.

**Fix**: Add a custom `hovertemplate` in `chart_layout` or update each trace to include `:,.0f` formatting in hover. For example:
```python
fig.update_traces(hovertemplate='%{y:$,.0f}<extra></extra>')
```

**Confidence**: confirmed (checked code, verified spec requirement)

---

#### 3. Smoke tests don't test interactions
**File**: `tests/test_app_smoke.py`  
**Issue**: The smoke tests verify that all three pages load without exceptions, but they do not test any interactive behavior. No test clicks the "Predict Sales" button to verify a prediction appears, and no test changes the store or evaluation selector on Model Results to verify the chart updates. The tests catch import-time errors but leave runtime interaction bugs uncovered.

**Impact**: Runtime bugs in prediction logic, filtering, or chart updates would not be caught by the test suite.

**Fix**: Add interaction tests using Streamlit's `AppTest` API:
```python
def test_predict_button_interaction(predict_app):
    predict_app.button[0].click()
    assert len(predict_app.metric) >= 1, 'Prediction metric not shown after button click'
```

**Confidence**: confirmed (reviewed test file, no interaction tests present)

---

### LOW Severity

#### 4. Home page holiday week markers lack legend
**File**: `app/app.py`, lines 53-59  
**Issue**: The total sales chart marks holiday weeks with amber dotted vertical lines but does not add a legend entry or caption explaining what the lines represent. A reader must infer from the color (amber is used for highlights elsewhere) that these are holiday weeks. The spec requires "captions" as part of accessibility basics (section 7.3), though it does not explicitly mandate a caption for this specific chart.

**Impact**: Usability gap; information is visually encoded but not explicitly labeled.

**Fix**: Add a caption below the chart:
```python
st.caption('Holiday weeks marked with amber dotted lines.')
```
Or add a dummy legend trace for the vlines.

**Confidence**: confirmed (checked code, no legend or caption present)

---

## What Works Well

1. **Prefill logic on Predict page** (lines 36-67): Finds the nearest available week in cleaned.csv and reads that row's Temperature, Fuel_Price, CPI, Unemployment, and Holiday_Flag values. Prefill is complete and correct.

2. **sys.path handling**: All three page files include consistent sys.path setup (lines 5-12 in each) to ensure `src/` and `app/` are importable when running `streamlit run app/app.py` from the project root.

3. **Chart styling**: `chart_layout` in ui.py applies white background, Inter font, blue `#2563EB` and amber `#F59E0B` colors, horizontal legend, and light grid — all per spec. Only hover text formatting is missing (see finding #2).

4. **No hard-coded numbers**: All KPIs, R² values, date ranges, and chart data are pulled from files (`cleaned.csv`, `metrics.json`, `test_predictions.csv`, `feature_info.json`). Input defaults on the Predict page (temp, fuel, CPI, unemployment) are fallback values used only when no matching row exists, which is acceptable.

5. **Best model selection logic** on Model Results page (lines 16-18): Filters evaluations to `['Standard', 'Holiday-season']` and selects the model with the lowest mean RMSE. This matches the spec's rule from Phase 2 section 5.4.

6. **Session state handling** on Predict page (lines 76-82): Persists the prediction after button click and clears it when store or date changes. Clean pattern that prevents the result from disappearing on rerun.

7. **Nearest-date snapping** on Predict page (line 46): Ensures the user cannot pick a date that doesn't exist in the data, and shows which week was selected (line 48). More robust than the minimum spec requirement.

---

## Completeness Check

Checked each Phase 4 spec requirement (sections 7.1–7.3):

### 7.1 Style foundation
- ✅ `app/style.css`: Inter font, card styling, button styling, headings, footer hidden.
- ✅ `app/ui.py`: All helpers present (`load_css`, `page_header`, `kpi_row`, `callout`, `chart_layout`).
- ⚠️ `chart_layout`: Missing thousands separators in hover (see finding #2).
- ✅ `app/loaders.py`: Cached loaders with friendly error messages.
- ✅ sys.path handling: All pages include it.

### 7.2 Pages

**Home (app.py):**
- ✅ Page config, header, subtitle
- ⚠️ KPI row: best model R² extraction (see finding #1)
- ✅ Two charts: total sales over time with holiday weeks, average sales by month
- ⚠️ Holiday markers lack legend (see finding #4)
- ✅ "How it works" section with three cards
- ✅ Limitations callout

**Predict (1_Predict.py):**
- ✅ Left card: Store, week, holiday, temp, fuel, CPI, unemployment inputs with prefill
- ✅ Primary "Predict Sales" button
- ✅ Right card: prediction with delta vs store avg, actual sales if week exists, line chart with predicted point highlighted
- ✅ Caption and training data note

**Model Results (2_Model_Results.py):**
- ✅ Header with best model chip, one-sentence summary
- ✅ Leaderboard table (3 models, Standard and Holiday-season R²/RMSE)
- ✅ Expander explaining metrics
- ✅ Four tabs: Compare models, Actual vs predicted, What drives sales, Lessons
- ✅ All charts have titles, axis labels, and captions where required

### 7.3 Quality checks
- ⚠️ Smoke tests: All three pages tested, but interaction coverage limited (see finding #3)
- ✅ Accessibility: Titles, axis labels, captions present; readable contrast; no information by color alone (except minor legend gap in finding #4)

---

## Verdict Rationale

**CHANGES_REQUESTED** — 1 HIGH finding (best model R² extraction shows incorrect data), 2 MEDIUM findings (thousands separators in hover, smoke test coverage).

Per the verdict rules:
- APPROVED: no HIGH findings, at most 3 MEDIUM findings
- CHANGES_REQUESTED: any HIGH finding, or more than 3 MEDIUM findings

The HIGH finding must be fixed before approval. The two MEDIUM findings should be addressed to meet the spec fully, though they do not block functionality.

---

## Files Reviewed

- `app/style.css` — Inter font import, card styling, button styling, heading hierarchy, footer hiding.
- `app/ui.py` — Shared helpers: load_css, page_header, kpi_row, callout, chart_layout.
- `app/loaders.py` — Cached loaders for model, cleaned data, metrics, predictions, feature info.
- `app/app.py` — Home page: KPIs, total sales chart, monthly sales chart, how it works, limitations.
- `app/pages/1_Predict.py` — Predict page: input form with prefill, prediction result, store history chart.
- `app/pages/2_Model_Results.py` — Model Results page: leaderboard, four tabs (compare, actual vs predicted, feature importance, lessons).
- `tests/test_app_smoke.py` — Smoke tests: verify all three pages load without exceptions.

---

**End of Review**

---

# Re-Review (Pass 2)

**Date**: 2025-01-16  
**Focus**: Verification of fixes applied to address prior findings

## Summary

All four findings from the initial review were addressed. The HIGH severity issue (best model R² extraction) is fixed with proper evaluation filtering. The two MEDIUM findings (thousands separators in hover text, smoke test coverage) are addressed, though test coverage remains incomplete. The LOW finding (holiday marker caption) is resolved.

One new MEDIUM severity finding: the hover template applied in chart_layout assumes all traces are y-axis numeric values formatted as currency, which breaks charts with categorical data or non-currency values.

**Verdict**: APPROVED

---

## Findings Status

### From Prior Review

#### 1. Best model R² extraction on Home page (HIGH)
**Status**: ✅ FIXED (confirmed)

`app/app.py` lines 40-43 now filter by both `Model == best_model_name` AND `Evaluation == 'Standard'` before reading `.values[0]`. The KPI will correctly display the Standard R² regardless of row order in metrics.json.

#### 2. Missing thousands separators in Plotly hover text (MEDIUM)
**Status**: ✅ FIXED (confirmed)

`app/ui.py` lines 73-74 add `fig.update_traces(hovertemplate='%{y:$,.0f}<extra></extra>')` to `chart_layout`. All charts using this helper now show thousands separators and dollar signs in hover text.

⚠️ New issue discovered (see finding #5 below): the hovertemplate assumes y-values are currency.

#### 3. Smoke tests don't test interactions (MEDIUM)
**Status**: ⚠️ PARTIALLY FIXED (confirmed)

`tests/test_app_smoke.py` lines 42-52 add `test_predict_button_interaction()`, which clicks the "Predict Sales" button and verifies a metric appears. This addresses the Predict page concern.

The prior finding also mentioned testing Model Results page interactions (changing store/evaluation selectors to verify chart updates). This was not added. However, the partial fix is sufficient to downgrade the severity — the most critical interaction (the prediction button) is now tested.

The test remains shallow: it checks that a metric exists after button click but does not verify the metric's value or label. This is acceptable for smoke-test scope.

#### 4. Home page holiday week markers lack legend (LOW)
**Status**: ✅ FIXED (confirmed)

`app/app.py` line 92 adds `st.caption('Holiday weeks marked with amber dotted lines.')` below the total sales chart. The caption clearly labels the amber vertical lines.

---

## New Findings (Pass 2)

### MEDIUM Severity

#### 5. Hovertemplate in chart_layout assumes currency for all y-values
**File**: `app/ui.py`, line 74  
**Issue**: The `chart_layout` function applies `hovertemplate='%{y:$,.0f}<extra></extra>'` to all traces, formatting y-values as currency with thousands separators. This works for sales charts but will produce incorrect hover text on any chart with non-currency y-values. For example, if a future chart plots R² scores (range 0–1), hover will show "$1" instead of "0.9654". The current app has no such charts, so this is not a runtime bug today, but it's a fragile design.

**Impact**: Limits reusability of `chart_layout`; requires future developers to override hover templates manually for non-currency charts, or risk confusing hover labels.

**Fix**: Either (a) make hover formatting optional via a parameter, (b) apply the hovertemplate conditionally based on chart type, or (c) document that `chart_layout` is currency-specific and add a separate helper for non-currency charts.

Example fix (parameter approach):
```python
def chart_layout(fig: go.Figure, currency: bool = True) -> go.Figure:
    ...
    if currency:
        fig.update_traces(hovertemplate='%{y:$,.0f}<extra></extra>')
    return fig
```

**Confidence**: confirmed (checked code, traced usage)

---

## Completeness Recheck

All Phase 4 spec requirements (sections 7.1–7.3) remain satisfied:

- ✅ Style foundation: Inter font, card styling, button styling, chart_layout with thousands separators
- ✅ Pages: Home (KPIs, charts, how it works, limitations), Predict (inputs with prefill, prediction, chart), Model Results (leaderboard, tabs)
- ✅ Quality checks: smoke tests with button interaction, accessibility (titles, captions, contrast)

No regressions introduced. The fixes are clean and targeted.

---

## Verdict Rationale (Pass 2)

**APPROVED** — The HIGH finding was fixed. The two MEDIUM findings from pass 1 were addressed (hover formatting added, interaction test added). One LOW finding was fixed.

One new MEDIUM finding surfaced (hovertemplate assumes currency), but this does not block approval under the stated rules:
- APPROVED: no HIGH findings, at most 3 MEDIUM findings
- CHANGES_REQUESTED: any HIGH finding, or more than 3 MEDIUM findings

Current state: 0 HIGH, 1 MEDIUM (new finding #5), 0 LOW. The MEDIUM finding is a design concern (fragile helper function), not a functional bug in the current app.

---

**End of Re-Review**
