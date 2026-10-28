# Weyland-Yutani Mining Output Dashboard (Python / Streamlit)

Task #5, part II: a web dashboard that reads the daily mine-output **generator spreadsheet**
live from Google Sheets and analyses it.

- **Generator spreadsheet (part I):** https://docs.google.com/spreadsheets/d/1Eo5jdZx9pzSfgGoJA_qfhPfO9xB-Fw4a/edit
- **Stack:** Streamlit · pandas · NumPy · SciPy · Plotly · matplotlib · ReportLab

## Features

- **Live data** from the `Data` sheet via the public Google Visualization endpoint (no API key).
  *Refresh data* button and optional auto-refresh (15 / 30 / 60 s).
- **Statistics** for every mine and the total: mean, standard deviation, median, Q1/Q3, IQR, min/max.
- **Anomaly tests** with adjustable parameters:
  - IQR rule (k × IQR fences)
  - Z-score (|z| threshold)
  - Distance from moving average (window, ±% threshold, centred or trailing)
  - Grubbs' test (iterative, two-sided; significance α; max outliers; critical value from `scipy.stats.t`)
  - Optional day-of-week adjustment and an "at least N tests agree" filter
- **Charts** (Plotly): line, bar or stacked; spikes ▲ / drops ▼ highlighted; polynomial trendlines of degree 1–4
  (`numpy.polyfit`, R² shown); per-mine small multiples; optional shading of generator events.
- **PDF report** (ReportLab + matplotlib): cover with executive summary, statistics tables, all charts,
  method and parameters, anomaly overview and a separate section for each anomaly episode
  (facts, zoomed chart, day-by-day table, matching generator event).

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install pytest && pytest -q`

## Deploy (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with GitHub.
2. *Create app* → *Deploy a public app from GitHub* → this repository, branch `main`, main file `app.py`.

## Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `core/data.py` | Google Sheets loader |
| `core/analysis.py` | Statistics, anomaly tests, episodes, polynomial trend |
| `core/charts.py` | Plotly charts |
| `core/report.py` | PDF report |
| `tests/` | Unit tests |
