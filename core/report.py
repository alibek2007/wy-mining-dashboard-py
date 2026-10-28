"""PDF report: ReportLab layout with matplotlib charts (no browser needed, so it
works on Streamlit Community Cloud)."""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, NextPageTemplate, PageBreak,  # noqa: E402
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, BaseDocTemplate, Frame)

from .analysis import TEST_LABELS, Analysis, Episode, SeriesResult, polyfit  # noqa: E402
from .charts import DROP_COLOR, SPIKE_COLOR, TOTAL_COLOR, color_of  # noqa: E402

PAGE_W, PAGE_H = A4
M = 16 * mm
CW = PAGE_W - 2 * M
INK, INK2, MUTED = colors.HexColor("#111827"), colors.HexColor("#4b5563"), colors.HexColor("#6b7280")
ACCENT, LINE, SOFT = colors.HexColor("#f59e0b"), colors.HexColor("#e5e7eb"), colors.HexColor("#f3f4f6")
SPIKE, DROP = colors.HexColor(SPIKE_COLOR), colors.HexColor(DROP_COLOR)

S = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=6, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK2, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK2),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10, textColor=INK, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK2, leftIndent=10,
                             bulletIndent=0, spaceAfter=3),
}


def n1(v, d=1):
    return f"{v:,.{d}f}"


def pct(v, d=1):
    return f"{v * 100:+.{d}f}%"


# ------------------------------------------------------------------ matplotlib charts
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#d1d5db", "axes.labelcolor": "#4b5563",
    "xtick.color": "#5b6472", "ytick.color": "#5b6472", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#eef0f3", "grid.linewidth": 0.8, "legend.frameon": False,
})


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _markers(ax, r: SeriesResult, y=None, label=True):
    for d, c, mk, name in [(1, SPIKE_COLOR, "^", "spike"), (-1, DROP_COLOR, "v", "drop")]:
        pts = r.points[r.points["dir"] == d]
        if len(pts):
            yy = pts["value"] if y is None else y.loc[pts.index]
            ax.scatter(pts.index, yy, marker=mk, s=42, color=c, edgecolor="white", linewidth=0.8, zorder=5,
                       label=f"{name}s" if label else None)


def _events(ax, a: Analysis):
    for e in a.events:
        ax.axvspan(e.start - pd.Timedelta(hours=12), e.end + pd.Timedelta(hours=12),
                   color="#c2410c" if e.magnitude >= 0 else "#1e40af", alpha=0.06, lw=0)
        ax.text(e.start, 1.0, e.name, transform=ax.get_xaxis_transform(), fontsize=6.5, color="#5b6472", va="bottom")


def _axes(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Output, t/day")
    ax.margins(x=0.01)


def chart_series(a: Analysis, rs: list[SeriesResult], kind: str, trend: int, title: str, h: float = 3.0) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, h))
    multi = len(rs) > 1
    for r in rs:
        c = color_of(a, r.name)
        if kind == "bar":
            cols = [(SPIKE_COLOR if r.points.at[d, "dir"] > 0 else DROP_COLOR) if d in r.points.index else c for d in r.values.index]
            ax.bar(r.values.index, r.values, width=0.85, color=cols, label=r.name, alpha=0.9)
        else:
            ax.plot(r.values.index, r.values, color=c, lw=1.4, label=r.name)
        _markers(ax, r, label=not multi)
        if trend:
            fitted, r2 = polyfit(r.values, trend)
            ax.plot(r.values.index, fitted, ls="--", lw=1.4, color=c if multi else TOTAL_COLOR,
                    label=None if multi else f"trend deg {trend} (R² {r2:.2f})")
    _events(ax, a)
    _axes(ax)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color="#111827", pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=6, fontsize=7.5)
    return _png(fig)


def chart_stacked(a: Analysis, trend: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    bottom = pd.Series(0.0, index=a.dates)
    for m in a.mines:
        ax.bar(m.values.index, m.values, bottom=bottom, width=0.85, color=color_of(a, m.name), label=m.name)
        bottom += m.values
    _markers(ax, a.total, y=a.total.values)
    if trend:
        fitted, r2 = polyfit(a.total.values, trend)
        ax.plot(a.dates, fitted, ls="--", lw=1.4, color=TOTAL_COLOR, label=f"total trend deg {trend} (R² {r2:.2f})")
    _events(ax, a)
    _axes(ax)
    ax.set_title("Output stacked by mine (triangles: anomalies in the total)", loc="left", fontsize=9.5, fontweight="bold", pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=8, fontsize=7.5)
    return _png(fig)


def chart_zoom(a: Analysis, r: SeriesResult, e: Episode) -> io.BytesIO:
    lo, hi = e.start - pd.Timedelta(days=14), e.end + pd.Timedelta(days=14)
    s = r.values.loc[lo:hi]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axvspan(e.start - pd.Timedelta(hours=12), e.end + pd.Timedelta(hours=12), color=SPIKE_COLOR if e.direction > 0 else DROP_COLOR, alpha=0.1, lw=0)
    ax.plot(s.index, s, color=color_of(a, r.name), lw=1.6, marker="o", ms=2.5)
    ax.axhline(r.stats["median"], color="#6b7280", ls=":", lw=1)
    ax.text(s.index[0], r.stats["median"], " median", fontsize=7, color="#6b7280", va="bottom")
    pts = e.flagged
    ax.scatter(pts.index, pts["value"], marker="^" if e.direction > 0 else "v", s=46,
               color=SPIKE_COLOR if e.direction > 0 else DROP_COLOR, edgecolor="white", zorder=5)
    _axes(ax)
    return _png(fig)


# ------------------------------------------------------------------ tables
def _table(data, col_widths=None, right_from=1, bold_last=False, font=8.3, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    st = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ALIGN", (right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), SOFT), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font - 0.6),
               ("TEXTCOLOR", (0, 0), (-1, 0), INK2)]
    if bold_last:
        st += [("FONT", (0, -1), (-1, -1), "Helvetica-Bold", font), ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f9fafb"))]
    t.setStyle(TableStyle(st))
    return t


def _img(buf, height_mm):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(buf).getSize()
    buf.seek(0)
    return Image(buf, width=CW, height=CW * h / w)


# ------------------------------------------------------------------ report
def build_report(a: Analysis, view_type: str = "line", trend: int = 1) -> bytes:
    out = io.BytesIO()
    date_range = f"{a.dates[0]:%Y-%m-%d} to {a.dates[-1]:%Y-%m-%d}"
    generated = datetime.now()

    def cover(c, _doc):
        c.saveState()
        c.setFillColor(INK); c.rect(0, PAGE_H - 92 * mm, PAGE_W, 92 * mm, fill=1, stroke=0)
        c.setFillColor(ACCENT); c.rect(0, PAGE_H - 93.6 * mm, PAGE_W, 1.6 * mm, fill=1, stroke=0)
        c.setStrokeColor(ACCENT); c.setLineWidth(4); c.setLineJoin(1)
        x0, y0 = M, PAGE_H - 22 * mm
        pts = [(0, 0), (5, -17), (12, -5), (19, -17), (24, 0)]
        p = c.beginPath(); p.moveTo(x0 + pts[0][0] * mm, y0 + pts[0][1] * mm)
        for px, py in pts[1:]:
            p.lineTo(x0 + px * mm, y0 + py * mm)
        c.drawPath(p, stroke=1, fill=0)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 11); c.drawString(M + 30 * mm, PAGE_H - 29 * mm, "WEYLAND-YUTANI CORP.")
        c.setFillColor(colors.HexColor("#9ca3af")); c.setFont("Helvetica", 8.5); c.drawString(M + 30 * mm, PAGE_H - 34.5 * mm, "Off-world Mining Division")
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 24); c.drawString(M, PAGE_H - 62 * mm, "Mine Output Analysis Report")
        c.setFillColor(colors.HexColor("#d1d5db")); c.setFont("Helvetica", 11)
        c.drawString(M, PAGE_H - 71 * mm, f"Daily extraction, descriptive statistics and anomaly review  |  {date_range}")
        c.setFillColor(colors.HexColor("#9ca3af")); c.setFont("Helvetica", 8.5)
        c.drawString(M, PAGE_H - 80 * mm, f"Generated {generated:%d/%m/%Y %H:%M}  |  Data fetched {a.fetched_at:%d/%m/%Y %H:%M}")
        c.restoreState()

    def later(c, doc):
        c.saveState()
        c.setFont("Helvetica-Bold", 7.5); c.setFillColor(INK); c.drawString(M, PAGE_H - 10 * mm, "WEYLAND-YUTANI CORP.")
        w = c.stringWidth("WEYLAND-YUTANI CORP.", "Helvetica-Bold", 7.5)
        c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
        c.drawString(M + w + 4 * mm, PAGE_H - 10 * mm, f"Mine Output Analysis Report  |  {date_range}")
        c.setStrokeColor(ACCENT); c.setLineWidth(1.4); c.line(M, PAGE_H - 12.5 * mm, PAGE_W - M, PAGE_H - 12.5 * mm)
        c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(M, 11 * mm, PAGE_W - M, 11 * mm)
        c.drawString(M, 7 * mm, "Internal use only - generated automatically from the output generator spreadsheet")
        c.drawRightString(PAGE_W - M, 7 * mm, f"Page {doc.page}")
        c.restoreState()

    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=20 * mm, bottomMargin=16 * mm,
                          title="Mine Output Analysis Report", author="Weyland-Yutani Mining Analytics")
    doc.addPageTemplates([
        PageTemplate("cover", [Frame(M, 16 * mm, CW, PAGE_H - 16 * mm - 100 * mm, id="c")], onPage=cover),
        PageTemplate("body", [Frame(M, 16 * mm, CW, PAGE_H - 36 * mm, id="b")], onPage=later),
    ])

    story = [NextPageTemplate("body")]
    t = a.total.stats
    spikes = sum(e.direction > 0 for e in a.episodes)
    enabled = a.params.enabled()

    # KPI boxes
    def kpi(label, value, hint):
        lbl = ParagraphStyle("kl", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED)
        val = ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=INK)
        hnt = ParagraphStyle("kh", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=INK2)
        return Table([[Paragraph(label.upper(), lbl)], [Paragraph(value, val)], [Paragraph(hint, hnt)]],
                     colWidths=[(CW - 9 * mm) / 4],
                     style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
                                       ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    kpis = [kpi("Mean daily output", f"{n1(t['mean'], 0)} t", f"{len(a.mines)} mines, {t['n']} days"),
            kpi("Standard deviation", f"{n1(t['std'], 0)} t", f"CV {t['cv'] * 100:.1f}%"),
            kpi("Median / IQR", f"{n1(t['median'], 0)} t", f"IQR {n1(t['iqr'], 0)} t"),
            kpi("Anomaly episodes", str(len(a.episodes)), f"{spikes} spikes, {len(a.episodes) - spikes} drops")]
    story += [Table([kpis], colWidths=[(CW - 9 * mm) / 4 + 3 * mm] * 4, style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)])),
              Spacer(1, 8 * mm)]

    story.append(Paragraph("Executive summary", S["h2"]))
    best = max(a.mines, key=lambda m: m.stats["mean"])
    volatile = max(a.mines, key=lambda m: m.stats["cv"])
    drops = [(e, e.peak_value / a.series(e.series).stats["median"] - 1) for e in a.episodes if e.direction < 0 and e.series != "Total"]
    worst = min(drops, key=lambda x: x[1]) if drops else None
    bullets = [
        f"Total output over the period: <b>{n1(t['sum'], 0)} t</b> across {len(a.mines)} mines ({n1(t['mean'], 0)} t/day on average, median {n1(t['median'], 0)} t/day).",
        f"Highest-producing mine: <b>{best.name}</b> with {n1(best.stats['mean'], 0)} t/day on average. Most volatile: <b>{volatile.name}</b> (coefficient of variation {volatile.stats['cv'] * 100:.1f}%).",
        f"<b>{len(a.episodes)} anomaly episodes</b> detected ({spikes} spikes, {len(a.episodes) - spikes} drops) using {', '.join(TEST_LABELS[x] for x in enabled)}"
        + (f", requiring at least {a.params.min_tests} tests to agree." if a.params.min_tests > 1 else "."),
        (f"Deepest drop: <b>{worst[0].series}</b> on {worst[0].peak_date:%Y-%m-%d}, {pct(worst[1])} versus its median"
         + (f" (coincides with generator event \"{worst[0].events[0].name}\")." if worst[0].events else ".")) if worst else "No drops detected with the current parameters.",
        "Tests ran on day-of-week adjusted values, so the regular weekly cycle (e.g. reduced Sunday output) is not reported as anomalous."
        if a.params.deseasonalize else "Tests ran on raw daily values (no day-of-week adjustment).",
    ]
    story += [Paragraph(b, S["bullet"], bulletText="•") for b in bullets]
    story += [Spacer(1, 4 * mm), Paragraph("Contents", S["h2"])]
    story += [Paragraph(f"{i}.&nbsp;&nbsp;{s}", S["body"]) for i, s in enumerate(
        ["Descriptive statistics", "Charts", "Anomaly detection method", "Anomaly overview", "Anomaly details (one section per episode)"], 1)]

    # 1. statistics
    story += [PageBreak(), Paragraph("1. Descriptive statistics", S["h1"]),
              Paragraph("Daily output in tonnes per mine and for the total of all mines. Standard deviation is the sample standard deviation; quartiles use linear interpolation (as QUARTILE.INC in Sheets).", S["body"])]
    data = [["Mine", "Mean", "Std dev", "Median", "Q1", "Q3", "IQR", "Min", "Max", "Total"]]
    for r in a.all_series:
        s = r.stats
        data.append([r.name, *[n1(s[k]) for k in ["mean", "std", "median", "q1", "q3", "iqr", "min", "max"]], n1(s["sum"], 0)])
    tbl = _table(data, [32 * mm] + [(CW - 32 * mm) / 9] * 9, bold_last=True)
    for i, r in enumerate(a.all_series, 1):
        tbl.setStyle(TableStyle([("LINEBEFORE", (0, i), (0, i), 3, colors.HexColor(color_of(a, r.name)))]))
    story += [tbl, Spacer(1, 6 * mm), Paragraph("Anomalies flagged per test (days)", S["h2"])]
    data = [["Mine", *[TEST_LABELS[x] for x in enabled], "Flagged days", "Episodes", "Spikes", "Drops"]]
    for r in a.all_series:
        data.append([r.name, *[str(r.count(x)) for x in enabled], str(len(r.points)), str(len(r.episodes)),
                     str(sum(e.direction > 0 for e in r.episodes)), str(sum(e.direction < 0 for e in r.episodes))])
    story.append(_table(data, bold_last=True))

    # 2. charts
    kind = "bar" if view_type == "bar" else "line"
    story += [PageBreak(), Paragraph("2. Charts", S["h1"]),
              Paragraph(f"Spikes are marked with an upward orange triangle, drops with a downward red triangle (bar charts colour the anomalous bar). "
                        f"{f'Dashed lines are least-squares polynomial trendlines of degree {trend}.' if trend else 'No trendline selected.'} "
                        "Shaded bands show the events configured in the generator.", S["body"]),
              _img(chart_series(a, [a.total], kind, trend, f"Total output ({kind})"), 0), Spacer(1, 4 * mm),
              _img(chart_stacked(a, trend), 0), Spacer(1, 4 * mm),
              _img(chart_series(a, a.mines, "line", 0, "All mines compared"), 0)]
    for m in a.mines:
        story += [CondPageBreak(70 * mm), _img(chart_series(a, [m], kind, trend,
                  f"{m.name}  |  mean {n1(m.stats['mean'], 0)} t, sd {n1(m.stats['std'], 0)} t, {len(m.points)} flagged days", h=2.6), 0), Spacer(1, 3 * mm)]

    # 3. method
    p = a.params
    story += [PageBreak(), Paragraph("3. Anomaly detection method", S["h1"]), Paragraph(
        "Each series (every mine and the total) is tested independently. "
        + ("Values are first divided by an estimated day-of-week factor (median ratio of each weekday to its centred 7-day mean), so the regular weekly cycle is not mistaken for anomalies. "
           if p.deseasonalize else "Values are tested as-is. ")
        + f"A day is flagged when at least {p.min_tests} of the enabled tests flag it. Consecutive flagged days in the same direction, with gaps of at most 2 days, form one anomaly episode.", S["body"])]
    rows = [["Test", "Status", "Parameters", "Rule"],
            ["IQR rule", "On" if p.iqr_on else "Off", f"k = {p.iqr_k}", Paragraph("Outlier if value &lt; Q1 - k x IQR or value &gt; Q3 + k x IQR", S["cell"])],
            ["Z-score", "On" if p.z_on else "Off", f"|z| > {p.z_threshold}", Paragraph("z = (value - mean) / standard deviation", S["cell"])],
            ["Moving-avg distance", "On" if p.ma_on else "Off", f"{p.ma_mode}, {p.ma_window} d, ±{p.ma_pct * 100:.0f}%", Paragraph("Outlier if |value - MA| / MA exceeds the threshold", S["cell"])],
            ["Grubbs' test", "On" if p.grubbs_on else "Off", f"alpha {p.grubbs_alpha}, max {p.grubbs_max}", Paragraph("Iterative two-sided test: remove the most extreme value while G &gt; G critical (Student t)", S["cell"])]]
    story.append(_table(rows, [36 * mm, 15 * mm, 42 * mm, CW - 93 * mm], right_from=99))
    if a.events:
        story += [Spacer(1, 5 * mm), Paragraph("Generator events (reference)", S["h2"]),
                  Paragraph("Events configured in the generator spreadsheet. They are not used by the tests; they let detected anomalies be checked against the known ground truth.", S["body"]),
                  _table([["Event", "Start", "End", "Duration", "Magnitude"]] + [[e.name, f"{e.start:%Y-%m-%d}", f"{e.end:%Y-%m-%d}", f"{e.duration} d", pct(e.magnitude, 0)] for e in a.events], right_from=3)]

    # 4. overview
    story += [CondPageBreak(60 * mm), Spacer(1, 4 * mm), Paragraph("4. Anomaly overview", S["h1"])]
    if a.episodes:
        short = {"iqr": "IQR", "z": "Z", "ma": "MA", "grubbs": "Grubbs"}
        rows = [["#", "Series", "Type", "Start", "End", "Days", "Peak", "vs median", "Tests", "Generator event"]]
        for e in a.episodes:
            med = a.series(e.series).stats["median"]
            rows.append([str(e.id), e.series, e.kind, f"{e.start:%Y-%m-%d}", f"{e.end:%Y-%m-%d}", str(e.days), n1(e.peak_value),
                         pct(e.peak_value / med - 1), ", ".join(short[x] for x in e.tests), Paragraph(", ".join(ev.name for ev in e.events) or "-", S["cell"])])
        tb = _table(rows, [8 * mm, 27 * mm, 13 * mm, 19 * mm, 19 * mm, 10 * mm, 17 * mm, 16 * mm, 24 * mm, CW - 153 * mm], right_from=99, font=7.6)
        for i, e in enumerate(a.episodes, 1):
            tb.setStyle(TableStyle([("TEXTCOLOR", (2, i), (2, i), SPIKE if e.direction > 0 else DROP), ("FONT", (2, i), (2, i), "Helvetica-Bold", 7.6)]))
        story.append(tb)
    else:
        story.append(Paragraph("No anomalies were detected with the current parameters.", S["body"]))

    # 5. details — one section per episode
    if a.episodes:
        story += [PageBreak(), Paragraph("5. Anomaly details", S["h1"])]
        for e in a.episodes:
            r = a.series(e.series)
            med = r.stats["median"]
            badge = Table([[e.kind.upper()]], colWidths=[16 * mm], rowHeights=[5.5 * mm], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), SPIKE if e.direction > 0 else DROP), ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONT", (0, 0), (-1, -1), "Helvetica-Bold", 7.5), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            title = Paragraph(f"<b>5.{e.id}&nbsp;&nbsp;{e.series}: {e.start:%Y-%m-%d}" + (f" to {e.end:%Y-%m-%d}" if e.days > 1 else "") + "</b>",
                              ParagraphStyle("t", parent=S["h2"], spaceBefore=0, spaceAfter=0))
            head = Table([[badge, title]], colWidths=[19 * mm, CW - 19 * mm], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            facts = [
                ["Duration", f"{e.days} day{'s' if e.days > 1 else ''} ({len(e.flagged)} flagged)", "Peak date", f"{e.peak_date:%Y-%m-%d}"],
                ["Peak value", f"{n1(e.peak_value)} t", "Series median", f"{n1(med)} t"],
                ["Deviation vs median", pct(e.peak_value / med - 1), "Peak z-score", f"{(e.peak_value - r.stats['mean']) / r.stats['std']:.2f}"],
                ["Tests triggered", Paragraph(", ".join(TEST_LABELS[x] for x in e.tests), S["cell"]), "Generator event",
                 Paragraph(", ".join(f"{ev.name} ({pct(ev.magnitude, 0)})" for ev in e.events) or "none - unexplained", S["cell"])],
            ]
            ft = Table(facts, colWidths=[34 * mm, 55 * mm, 30 * mm, CW - 119 * mm], style=TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
                ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 8.5), ("FONT", (3, 0), (3, -1), "Helvetica-Bold", 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            days = [["Date", "Day", "Output, t", "vs median", "Flagged by"]]
            for i in range(e.start_i, e.end_i + 1):
                d = a.dates[i]
                v = r.values.iloc[i]
                tests = e.flagged.at[d, "tests"] if d in e.flagged.index else None
                days.append([f"{d:%Y-%m-%d}", f"{d:%a}", n1(v), pct(v / med - 1), ", ".join(TEST_LABELS[x] for x in tests) if tests else "not flagged (gap)"])
            story.append(KeepTogether([head, Spacer(1, 3 * mm), ft, Spacer(1, 3 * mm), _img(chart_zoom(a, r, e), 0), Spacer(1, 3 * mm),
                                       _table(days, [24 * mm, 14 * mm, 22 * mm, 20 * mm, CW - 80 * mm], right_from=99, font=7.8), Spacer(1, 8 * mm)]))

    doc.build(story)
    return out.getvalue()
