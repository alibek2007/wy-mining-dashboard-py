"""Descriptive statistics, anomaly tests, episode grouping and polynomial trends."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sps

from .data import Event

TESTS = ["iqr", "z", "ma", "grubbs"]
TEST_LABELS = {"iqr": "IQR rule", "z": "Z-score", "ma": "Moving-avg distance", "grubbs": "Grubbs' test"}


@dataclass
class Params:
    deseasonalize: bool = True
    min_tests: int = 1
    iqr_on: bool = True
    iqr_k: float = 1.5
    z_on: bool = True
    z_threshold: float = 2.5
    ma_on: bool = True
    ma_window: int = 7
    ma_pct: float = 0.30
    ma_mode: str = "centered"          # or "trailing"
    grubbs_on: bool = True
    grubbs_alpha: float = 0.05
    grubbs_max: int = 10

    def enabled(self) -> list[str]:
        return [t for t in TESTS if getattr(self, f"{t}_on")]


# ---------------------------------------------------------------- statistics
def describe(s: pd.Series) -> dict:
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])   # linear interpolation = QUARTILE.INC
    return {
        "n": int(s.size), "mean": s.mean(), "std": s.std(ddof=1), "median": med,
        "q1": q1, "q3": q3, "iqr": q3 - q1, "min": s.min(), "max": s.max(),
        "sum": s.sum(), "cv": s.std(ddof=1) / s.mean(),
    }


def moving_average(x: np.ndarray, window: int, mode: str) -> np.ndarray:
    """Trailing: mean of the previous `window` days (today excluded).
    Centred: mean of the window around the day, the day itself excluded."""
    s = pd.Series(x)
    w = max(2, int(window))
    if mode == "trailing":
        return s.shift(1).rolling(w, min_periods=w).mean().to_numpy()
    half = w // 2
    total = s.rolling(2 * half + 1, center=True, min_periods=2 * half + 1).sum()
    return ((total - s) / (2 * half)).to_numpy()


def weekday_factors(s: pd.Series) -> np.ndarray:
    """Multiplicative day-of-week factors: median ratio of each value to its
    centred 7-day mean, normalised to average 1."""
    ma = s.rolling(7, center=True).mean()
    ratio = (s / ma).dropna()
    f = ratio.groupby(ratio.index.dayofweek).median().reindex(range(7), fill_value=1.0).to_numpy()
    return f / f.mean()


# ---------------------------------------------------------------- tests
# Each test returns an array of direction (+1 spike, -1 drop, 0 normal).
def iqr_test(x: np.ndarray, k: float) -> np.ndarray:
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    return np.where(x > q3 + k * iqr, 1, np.where(x < q1 - k * iqr, -1, 0))


def z_test(x: np.ndarray, threshold: float) -> np.ndarray:
    z = (x - x.mean()) / (x.std(ddof=1) or 1)
    return np.where(np.abs(z) > threshold, np.sign(z), 0).astype(int)


def ma_test(x: np.ndarray, window: int, pct: float, mode: str) -> np.ndarray:
    ma = moving_average(x, window, mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (x - ma) / ma
    d = np.nan_to_num(d)
    return np.where(np.abs(d) > pct, np.sign(d), 0).astype(int)


def grubbs_critical(n: int, alpha: float) -> float:
    t = sps.t.ppf(1 - alpha / (2 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))


def grubbs_test(x: np.ndarray, alpha: float, max_outliers: int) -> np.ndarray:
    """Iterative two-sided Grubbs: remove the most extreme value while G > G_crit."""
    out = np.zeros(x.size, dtype=int)
    idx = np.arange(x.size)
    for _ in range(max_outliers):
        if idx.size <= 6:
            break
        sub = x[idx]
        m, sd = sub.mean(), sub.std(ddof=1)
        if sd == 0:
            break
        g = np.abs(sub - m) / sd
        k = int(np.argmax(g))
        if g[k] <= grubbs_critical(idx.size, alpha):
            break
        out[idx[k]] = int(np.sign(sub[k] - m))
        idx = np.delete(idx, k)
    return out


# ---------------------------------------------------------------- per series
@dataclass
class Episode:
    series: str
    direction: int
    start: datetime
    end: datetime
    start_i: int
    end_i: int
    peak_date: datetime
    peak_value: float
    flagged: pd.DataFrame
    tests: list[str]
    events: list[Event] = field(default_factory=list)
    id: int = 0

    @property
    def days(self) -> int:
        return self.end_i - self.start_i + 1

    @property
    def kind(self) -> str:
        return "Spike" if self.direction > 0 else "Drop"


@dataclass
class SeriesResult:
    name: str
    values: pd.Series
    stats: dict
    flags: pd.DataFrame           # one int column per enabled test
    points: pd.DataFrame          # flagged days: value, adjusted, dir, tests
    episodes: list[Episode]

    def count(self, test: str) -> int:
        return int((self.flags[test] != 0).sum()) if test in self.flags else 0


def analyse_series(name: str, s: pd.Series, p: Params, events: list[Event], max_gap: int = 2) -> SeriesResult:
    x = s.to_numpy(dtype=float)
    work = x
    if p.deseasonalize:
        f = weekday_factors(s)
        work = x / f[s.index.dayofweek]

    flags = {}
    if p.iqr_on:
        flags["iqr"] = iqr_test(work, p.iqr_k)
    if p.z_on:
        flags["z"] = z_test(work, p.z_threshold)
    if p.ma_on:
        flags["ma"] = ma_test(work, p.ma_window, p.ma_pct, p.ma_mode)
    if p.grubbs_on:
        flags["grubbs"] = grubbs_test(work, p.grubbs_alpha, p.grubbs_max)
    flags_df = pd.DataFrame(flags, index=s.index)

    hit_count = (flags_df != 0).sum(axis=1) if len(flags_df.columns) else pd.Series(0, index=s.index)
    mask = hit_count >= max(1, p.min_tests)
    dir_sum = flags_df.sum(axis=1) if len(flags_df.columns) else pd.Series(0, index=s.index)
    points = pd.DataFrame({
        "i": np.arange(len(s))[mask.to_numpy()],
        "value": s[mask],
        "adjusted": work[mask.to_numpy()],
        "dir": np.where(dir_sum[mask] >= 0, 1, -1),
        "tests": [[t for t in flags_df.columns if flags_df.at[d, t] != 0] for d in s.index[mask]],
    })

    episodes = _group(name, points, s, max_gap, events)
    return SeriesResult(name, s, describe(s), flags_df, points, episodes)


def _group(name, points: pd.DataFrame, s: pd.Series, max_gap: int, events: list[Event]) -> list[Episode]:
    groups: list[list] = []
    for date, row in points.iterrows():
        if groups and row["dir"] == groups[-1][-1][1]["dir"] and row["i"] - groups[-1][-1][1]["i"] <= max_gap + 1:
            groups[-1].append((date, row))
        else:
            groups.append([(date, row)])
    out = []
    for g in groups:
        df = pd.DataFrame([r for _, r in g], index=[d for d, _ in g])
        direction = int(df["dir"].iloc[0])
        peak = df["adjusted"].idxmax() if direction > 0 else df["adjusted"].idxmin()
        start, end = df.index[0], df.index[-1]
        tests = sorted({t for ts in df["tests"] for t in ts}, key=TESTS.index)
        ev = [e for e in events if e.start <= end + pd.Timedelta(days=1) and e.end >= start - pd.Timedelta(days=1)]
        out.append(Episode(name, direction, start, end, int(df["i"].iloc[0]), int(df["i"].iloc[-1]),
                           peak, float(s.loc[peak]), df, tests, ev))
    return out


@dataclass
class Analysis:
    dates: pd.DatetimeIndex
    mines: list[SeriesResult]
    total: SeriesResult
    params: Params
    events: list[Event]
    fetched_at: datetime
    episodes: list[Episode]

    @property
    def all_series(self) -> list[SeriesResult]:
        return [*self.mines, self.total]

    def series(self, name: str) -> SeriesResult:
        return next(s for s in self.all_series if s.name == name)


def analyse(df: pd.DataFrame, p: Params, events: list[Event], fetched_at: datetime) -> Analysis:
    mines = [analyse_series(c, df[c], p, events) for c in df.columns]
    total = analyse_series("Total", df.sum(axis=1), p, events)
    eps = sorted((e for r in [*mines, total] for e in r.episodes),
                 key=lambda e: (e.start, e.series == "Total"))
    for i, e in enumerate(eps, 1):
        e.id = i
    return Analysis(df.index, mines, total, p, events, fetched_at, eps)


def polyfit(y: pd.Series | np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Least-squares polynomial trend; x is scaled to [-1, 1] for stability.
    Returns fitted values and R²."""
    y = np.asarray(y, dtype=float)
    x = np.linspace(-1, 1, y.size)
    fitted = np.polyval(np.polyfit(x, y, degree), x)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - fitted) ** 2).sum() / ss_tot if ss_tot else 1.0
    return fitted, float(r2)
