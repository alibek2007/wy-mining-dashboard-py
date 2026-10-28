"""Interactive Plotly charts for the dashboard."""
from __future__ import annotations

import plotly.graph_objects as go

from .analysis import Analysis, SeriesResult, polyfit

# Categorical slots in fixed order (validated palette) + extras for mines 9-10.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#6250d6", "#8c5a2b", "#3d9fb8", "#7a7a74"]
TOTAL_COLOR = "#1f2a37"
SPIKE_COLOR = "#c2410c"
DROP_COLOR = "#b91c1c"
FONT = "Inter, Segoe UI, sans-serif"


def color_of(a: Analysis, name: str) -> str:
    if name == "Total":
        return TOTAL_COLOR
    return SERIES_COLORS[[m.name for m in a.mines].index(name) % len(SERIES_COLORS)]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{alpha})"


def _markers(fig, r: SeriesResult, show_legend: bool, y_override=None):
    for direction, color, symbol, label in [(1, SPIKE_COLOR, "triangle-up", "spikes"), (-1, DROP_COLOR, "triangle-down", "drops")]:
        pts = r.points[r.points["dir"] == direction]
        if pts.empty:
            continue
        y = pts["value"] if y_override is None else y_override.loc[pts.index]
        fig.add_trace(go.Scatter(
            x=pts.index, y=y, mode="markers", name=f"{r.name} — {label}",
            legendgroup=label, showlegend=show_legend,
            marker=dict(symbol=symbol, size=11, color=color, line=dict(color="white", width=1.2)),
            customdata=[", ".join(t) for t in pts["tests"]],
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.1f} t<br>tests: %{customdata}<extra>" + f"{r.name} {label[:-1]}</extra>",
        ))


def _trend(fig, r: SeriesResult, degree: int, color: str):
    fitted, r2 = polyfit(r.values, degree)
    fig.add_trace(go.Scatter(
        x=r.values.index, y=fitted, mode="lines", name=f"{r.name} — trend (deg {degree}, R² {r2:.2f})",
        line=dict(color=color, width=2, dash="dash"), hoverinfo="skip",
    ))


def main_chart(a: Analysis, focus: str, chart_type: str, trend: int, show_events: bool, height: int = 460) -> go.Figure:
    """focus: 'Total', 'All mines' or a mine name. chart_type: line | bar | stacked."""
    fig = go.Figure()
    if chart_type == "stacked":
        for m in a.mines:
            c = color_of(a, m.name)
            flagged = m.points.index
            colors = [(SPIKE_COLOR if m.points.at[d, "dir"] > 0 else DROP_COLOR) if d in flagged else c for d in m.values.index]
            fig.add_trace(go.Bar(x=m.values.index, y=m.values, name=m.name, marker=dict(color=colors, line=dict(width=0)),
                                 hovertemplate="%{y:,.1f} t<extra>" + m.name + "</extra>"))
        fig.update_layout(barmode="stack", bargap=0.08)
        _markers(fig, a.total, True)
        if trend:
            _trend(fig, a.total, trend, TOTAL_COLOR)
    else:
        series = [a.total] if focus == "Total" else a.mines if focus == "All mines" else [a.series(focus)]
        multi = len(series) > 1
        for r in series:
            c = color_of(a, r.name)
            if chart_type == "line":
                fig.add_trace(go.Scatter(x=r.values.index, y=r.values, mode="lines", name=r.name,
                                         line=dict(color=c, width=2, shape="spline", smoothing=0.4),
                                         hovertemplate="%{y:,.1f} t<extra>" + r.name + "</extra>"))
            else:
                flagged = r.points.index
                colors = [(SPIKE_COLOR if r.points.at[d, "dir"] > 0 else DROP_COLOR) if d in flagged else _rgba(c, 0.85) for d in r.values.index]
                fig.add_trace(go.Bar(x=r.values.index, y=r.values, name=r.name, marker=dict(color=colors, line=dict(width=0)),
                                     hovertemplate="%{y:,.1f} t<extra>" + r.name + "</extra>"))
            _markers(fig, r, show_legend=not multi)
            if trend:
                _trend(fig, r, trend, c if multi else TOTAL_COLOR)
        if chart_type == "bar":
            fig.update_layout(barmode="group", bargap=0.1)

    if show_events:
        for e in a.events:
            fig.add_vrect(x0=e.start, x1=e.end, fillcolor="#c2410c" if e.magnitude >= 0 else "#1e40af",
                          opacity=0.07, line_width=0, annotation_text=e.name, annotation_position="top left",
                          annotation_font=dict(size=10, color="#5b6472"))
    style(fig, height)
    return fig


def mini_chart(a: Analysis, r: SeriesResult, trend: int, chart_type: str) -> go.Figure:
    fig = main_chart(a, r.name, "bar" if chart_type == "bar" else "line", trend, False, height=200)
    fig.update_layout(showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
    fig.update_yaxes(title=None, nticks=4)
    return fig


def zoom_chart(a: Analysis, r: SeriesResult, start, end, pad_days: int = 14) -> go.Figure:
    import pandas as pd
    lo, hi = start - pd.Timedelta(days=pad_days), end + pd.Timedelta(days=pad_days)
    s = r.values.loc[lo:hi]
    fig = go.Figure(go.Scatter(x=s.index, y=s, mode="lines+markers", line=dict(color=color_of(a, r.name), width=2),
                               marker=dict(size=4), name=r.name))
    fig.add_vrect(x0=start - pd.Timedelta(hours=12), x1=end + pd.Timedelta(hours=12), fillcolor="#b91c1c", opacity=0.08, line_width=0)
    fig.add_hline(y=r.stats["median"], line=dict(color="#6b7280", dash="dot", width=1), annotation_text="median")
    style(fig, 260)
    fig.update_layout(showlegend=False)
    return fig


def style(fig: go.Figure, height: int):
    fig.update_layout(
        height=height, template="plotly_white", font=dict(family=FONT, size=12, color="#374151"),
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, tickformat="%d %b", linecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.07)", title="Output, t/day", tickformat=",", zeroline=False)
