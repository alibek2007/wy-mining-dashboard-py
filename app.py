"""Weyland-Yutani mine output analytics — Streamlit dashboard.

Reads the generator Google Sheet live, computes statistics and anomaly tests,
draws interactive charts and builds a PDF report."""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from core.analysis import TEST_LABELS, TESTS, Params, analyse, polyfit
from core.charts import color_of, main_chart, mini_chart
from core.data import DEFAULT_SHEET_ID, load_dataset, load_events, parse_sheet_id
from core.report import build_report

st.set_page_config(page_title="Weyland-Yutani Mining Analytics", page_icon="⛏️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: Inter, 'Segoe UI', sans-serif; }
  .block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px; }
  .wy-bar { background:#111827; color:#f9fafb; border-bottom:3px solid #f59e0b; border-radius:10px;
            padding:14px 20px; display:flex; align-items:center; gap:14px; margin-bottom:1rem; }
  .wy-bar .name { font-weight:700; letter-spacing:.14em; font-size:15px; }
  .wy-bar .sub { color:#9ca3af; font-size:12.5px; margin-top:2px; }
  .wy-bar .status { margin-left:auto; font-size:12.5px; color:#d1d5db; }
  .wy-bar .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e;
                 box-shadow:0 0 0 3px rgba(34,197,94,.2); margin-right:8px; }
  .kpis { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:14px; margin-bottom:1rem; }
  .kpi { background:#fff; border:1px solid #e5e7eb; border-left:3px solid #f59e0b; border-radius:10px;
         padding:12px 16px; box-shadow:0 1px 2px rgba(16,24,40,.05); }
  .kpi .label { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#6b7280; font-weight:600; }
  .kpi .value { font-size:26px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; color:#111827; }
  .kpi .value small { font-size:13px; color:#6b7280; font-weight:500; margin-left:3px; }
  .kpi .hint { font-size:12px; color:#4b5563; margin-top:2px; }
  @media (max-width: 1100px) { .kpis { grid-template-columns:repeat(2, minmax(0,1fr)); } }
  .legend-key { font-size:12.5px; color:#4b5563; }
  .legend-key b.up { color:#c2410c; } .legend-key b.down { color:#b91c1c; }
  section[data-testid="stSidebar"] h3 { font-size:12px; text-transform:uppercase; letter-spacing:.1em; color:#6b7280; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def fetch(sheet_id: str, refresh_token: int):
    """refresh_token changes on 'Refresh' / auto-refresh to bypass the cache."""
    return load_dataset(sheet_id), load_events(sheet_id), datetime.now()


ss = st.session_state
ss.setdefault("refresh_token", 0)
ss.setdefault("sheet_id", st.query_params.get("sheet", DEFAULT_SHEET_ID))

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Data source")
    sheet_in = st.text_input("Google Sheet URL or ID", ss.sheet_id)
    if parse_sheet_id(sheet_in) != ss.sheet_id:
        ss.sheet_id = parse_sheet_id(sheet_in) or DEFAULT_SHEET_ID
    st.markdown(f"[Open generator spreadsheet ↗](https://docs.google.com/spreadsheets/d/{ss.sheet_id}/edit)")
    c1, c2 = st.columns(2)
    if c1.button("↻ Refresh data", use_container_width=True):
        ss.refresh_token += 1
    auto = c2.selectbox("Auto-refresh", ["Off", "15 s", "30 s", "60 s"], label_visibility="collapsed")

    st.markdown("### Chart")
    chart_type = st.segmented_control("Chart type", ["Line", "Bar", "Stacked"], default="Line") or "Line"
    trend = st.selectbox("Trendline", [0, 1, 2, 3, 4], index=1,
                         format_func=lambda d: "None" if d == 0 else f"Polynomial · degree {d}")
    show_events = st.checkbox("Shade generator events", value=False)

    st.markdown("### Anomaly detection")
    p = Params()
    p.deseasonalize = st.checkbox("Adjust for day-of-week pattern", True,
                                  help="Divide by an estimated weekday factor before testing, so regular low Sundays are not reported as drops.")
    p.min_tests = st.select_slider("Flag when at least N tests agree", [1, 2, 3, 4], value=1)
    with st.expander("IQR rule", expanded=True):
        p.iqr_on = st.toggle("Enabled", True, key="iqr_on")
        p.iqr_k = st.slider("k × IQR", 0.5, 4.0, 1.5, 0.1, disabled=not p.iqr_on)
    with st.expander("Z-score", expanded=True):
        p.z_on = st.toggle("Enabled", True, key="z_on")
        p.z_threshold = st.slider("|z| threshold", 1.0, 5.0, 2.5, 0.1, disabled=not p.z_on)
    with st.expander("Distance from moving average", expanded=True):
        p.ma_on = st.toggle("Enabled", True, key="ma_on")
        p.ma_window = st.slider("Window, days", 3, 30, 7, disabled=not p.ma_on)
        p.ma_pct = st.slider("Threshold, ± %", 5, 80, 30, disabled=not p.ma_on) / 100
        p.ma_mode = st.radio("Window type", ["centered", "trailing"], horizontal=True, disabled=not p.ma_on,
                             format_func=lambda m: "Centred" if m == "centered" else "Trailing")
    with st.expander("Grubbs' test", expanded=True):
        p.grubbs_on = st.toggle("Enabled", True, key="g_on")
        p.grubbs_alpha = st.select_slider("Significance α", [0.01, 0.05, 0.10], value=0.05, disabled=not p.grubbs_on)
        p.grubbs_max = st.slider("Max outliers (iterations)", 1, 30, 10, disabled=not p.grubbs_on)

run_every = {"Off": None, "15 s": 15, "30 s": 30, "60 s": 60}[auto]


# ------------------------------------------------------------------ main (re-runs on auto-refresh)
@st.fragment(run_every=run_every)
def dashboard():
    token = ss.refresh_token if run_every is None else (ss.refresh_token, int(time.time() // run_every))
    try:
        with st.spinner("Loading from Google Sheets…"):
            df, events, fetched = fetch(ss.sheet_id, hash(token))
    except Exception as exc:  # noqa: BLE001 - shown to the user
        st.error(f"Could not load the spreadsheet: {exc}. Make sure it is shared as “Anyone with the link” and has a sheet named “Data”.")
        return

    a = analyse(df, p, events, fetched)
    st.markdown(f"""
      <div class="wy-bar">
        <svg viewBox="0 0 32 32" width="34" height="34"><rect width="32" height="32" rx="6" fill="#1f2937"/>
        <path d="M6 9l4 14 6-10 6 10 4-14" stroke="#f59e0b" stroke-width="3" fill="none" stroke-linejoin="round"/></svg>
        <div><div class="name">WEYLAND-YUTANI CORP.</div><div class="sub">Off-world Mining Division · Output Analytics</div></div>
        <div class="status"><span class="dot"></span>Updated {fetched:%H:%M:%S} · {len(a.dates)} days · {len(a.mines)} mines</div>
      </div>""", unsafe_allow_html=True)

    # KPIs
    t = a.total.stats
    fitted, _ = polyfit(a.total.values, 1)
    spikes = sum(e.direction > 0 for e in a.episodes)
    kpis = [
        ("Mean daily output", f"{t['mean']:,.0f}", "t", f"{len(a.mines)} mines · {t['n']} days"),
        ("Standard deviation", f"{t['std']:,.0f}", "t", f"CV {t['cv']:.1%}"),
        ("Median", f"{t['median']:,.0f}", "t", f"Range {t['min']:,.0f} – {t['max']:,.0f} t"),
        ("Interquartile range", f"{t['iqr']:,.0f}", "t", f"Q1 {t['q1']:,.0f} · Q3 {t['q3']:,.0f}"),
        ("Anomaly episodes", str(len(a.episodes)), "", f"{spikes} spikes · {len(a.episodes) - spikes} drops · trend {(fitted[-1] - fitted[0]) / fitted[0]:+.1%}"),
    ]
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="label">{lbl}</div><div class="value">{v}<small>{u}</small></div><div class="hint">{h}</div></div>'
        for lbl, v, u, h in kpis) + "</div>", unsafe_allow_html=True)

    # Main chart
    with st.container(border=True):
        h1, h2, h3 = st.columns([3, 2, 2])
        h1.subheader("Daily output")
        focus = h2.selectbox("Series", ["Total", "All mines", *[m.name for m in a.mines]],
                             disabled=chart_type == "Stacked", label_visibility="collapsed")
        h3.markdown('<div class="legend-key"><b class="up">▲</b> spike &nbsp; <b class="down">▼</b> drop &nbsp; ┅ trendline</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(main_chart(a, focus, chart_type.lower(), trend, show_events), use_container_width=True)
        if chart_type == "Stacked":
            st.caption("Stacked by mine; triangles mark anomalies in the total, red/orange segments are anomalous days of a mine.")

    # Statistics
    with st.container(border=True):
        st.subheader("Descriptive statistics")
        st.caption("Daily output in tonnes. Anomaly columns count flagged days per test with the current parameters.")
        enabled = p.enabled()
        rows = []
        for r in a.all_series:
            s = r.stats
            rows.append({"Mine": r.name, "Mean": s["mean"], "Std dev": s["std"], "Median": s["median"], "Q1": s["q1"],
                         "Q3": s["q3"], "IQR": s["iqr"], "Min": s["min"], "Max": s["max"],
                         **{TEST_LABELS[x]: r.count(x) for x in enabled}, "Episodes": len(r.episodes)})
        num = st.column_config.NumberColumn(format="%.1f")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: num for c in ["Mean", "Std dev", "Median", "Q1", "Q3", "IQR", "Min", "Max"]})

    # Per-mine small multiples
    with st.container(border=True):
        st.subheader("Per-mine view")
        cols = st.columns(2)
        for i, m in enumerate(a.mines):
            with cols[i % 2]:
                st.markdown(f"<span style='color:{color_of(a, m.name)}'>■</span> **{m.name}** "
                            f"<span style='color:#6b7280;font-size:12.5px'>· mean {m.stats['mean']:,.0f} t · sd {m.stats['std']:,.0f} · "
                            f"{len(m.points)} flagged days</span>", unsafe_allow_html=True)
                st.plotly_chart(mini_chart(a, m, trend, chart_type.lower()), use_container_width=True, key=f"mini{i}")

    # Anomalies
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        c1.subheader("Detected anomalies")
        kind = c2.segmented_control("Show", ["All", "Spikes", "Drops"], default="All", label_visibility="collapsed") or "All"
        eps = [e for e in a.episodes if kind == "All" or (e.direction > 0) == (kind == "Spikes")]
        st.caption("Consecutive flagged days (gap ≤ 2 days, same direction) form one episode. Deviation is measured against the series median.")
        if eps:
            st.dataframe(pd.DataFrame([{
                "#": e.id, "Series": e.series, "Type": ("▲ " if e.direction > 0 else "▼ ") + e.kind,
                "Start": e.start.date(), "End": e.end.date(), "Days": e.days, "Peak date": e.peak_date.date(),
                "Peak value": e.peak_value, "vs median": (e.peak_value / a.series(e.series).stats["median"] - 1) * 100,
                "Tests": ", ".join(TEST_LABELS[x] for x in e.tests),
                "Generator event": ", ".join(ev.name for ev in e.events) or "—",
            } for e in eps]), hide_index=True, use_container_width=True,
                column_config={"Peak value": st.column_config.NumberColumn(format="%.1f"),
                               "vs median": st.column_config.NumberColumn(format="%+.1f%%")})
        else:
            st.info("No anomalies with the current parameters.")

    # PDF report
    with st.container(border=True):
        st.subheader("PDF report")
        st.caption("Cover and summary, statistics tables, all charts, method and parameters, and a separate section for every anomaly.")
        b1, b2 = st.columns([1, 3])
        signature = (repr(p), chart_type, trend, fetched)  # a stale report is not offered after changes
        if b1.button("📄 Generate PDF report", type="primary", use_container_width=True):
            with st.spinner("Building report…"):
                ss.pdf = build_report(a, chart_type.lower(), trend)
                ss.pdf_sig = signature
                ss.pdf_name = f"WY-mine-output-report-{datetime.now():%Y-%m-%d}.pdf"
        if ss.get("pdf") and ss.get("pdf_sig") == signature:
            b2.download_button("⬇ Download report", ss.pdf, file_name=ss.pdf_name, mime="application/pdf")

    st.caption("Data is read live from the generator spreadsheet (Google Sheets, “Data” sheet). All statistics are computed with pandas / NumPy / SciPy.")


dashboard()
