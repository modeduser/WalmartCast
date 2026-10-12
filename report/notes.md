# Project Notes

This file tracks findings and decisions made during the project.

---

## Task 1.2 — Load and Inspect

- **Shape:** 6,435 rows × 8 columns (45 stores × 143 weeks = 6,435 ✓)
- **Columns:** Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- **Date range:** Feb 5, 2010 to Oct 26, 2012 (143 unique weeks)
- **Stores:** 45 (numbered 1–45)
- **Weekly_Sales:** ranges from ~$210K to ~$3.8M, mean ~$1.05M, big variation across stores
- **Holiday_Flag:** only ~7% of weeks are holiday weeks
- **Temperature:** -2°F to 100°F (varies by store region)
- **CPI:** 126 to 227, **Unemployment:** 3.9% to 14.3% (vary by region)

## Task 1.3 — Cleaning Checks

### Missing values
- **None.** All 8 columns are complete with 0 missing values.

### Duplicate rows
- **None.** No duplicate rows and no duplicate Store+Date combinations.

### Negative or zero sales
- **None.** All Weekly_Sales values are positive.

### Rows per store
- **All 45 stores have exactly 143 rows.** The data is balanced — no missing weeks.

### Outliers (IQR method on Weekly_Sales)
- Q1 = $553,350 | Q3 = $1,420,159 | IQR = $866,809
- Upper bound = $2,720,371 | Lower bound = -$746,863
- **34 outliers found (0.5% of data)**, all above the upper bound (high sales), none below.
- Of these 34: 9 are holiday weeks, 25 are non-holiday weeks (likely from the biggest stores).
- **Decision: Keep all outliers.** They represent real demand from large stores and holiday spikes, not data errors.

### Overall cleaning result
- The dataset is very clean: no missing values, no duplicates, no invalid values.
- No rows were removed. The cleaned file is identical to the original but with Date parsed as datetime.
- Saved to `data/cleaned.csv`.

## Task 1.4 — Save
- Cleaned data saved to `data/cleaned.csv` (6,435 rows, 8 columns).
