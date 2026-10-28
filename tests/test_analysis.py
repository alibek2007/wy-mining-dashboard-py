import numpy as np
import pandas as pd

from core.analysis import Params, analyse_series, grubbs_critical, grubbs_test, iqr_test, polyfit


def test_grubbs_critical_matches_tables():
    assert abs(grubbs_critical(10, 0.05) - 2.290) < 2e-3
    assert abs(grubbs_critical(100, 0.05) - 3.384) < 3e-3


def test_outlier_tests_find_spike_and_drop():
    x = 100 + 3 * np.sin(np.arange(60))
    x[20], x[40] = 160, 40
    g = grubbs_test(x, 0.05, 5)
    assert g[20] == 1 and g[40] == -1 and np.count_nonzero(g) == 2
    i = iqr_test(x, 1.5)
    assert i[20] == 1 and i[40] == -1


def test_polyfit_exact_quadratic():
    y = 3 + 2 * np.arange(50) - 0.1 * np.arange(50) ** 2
    assert abs(polyfit(y, 2)[1] - 1) < 1e-9


def test_episode_grouping():
    idx = pd.date_range("2026-01-01", periods=90)
    s = pd.Series(1000 + 20 * np.sin(np.arange(90)), index=idx)
    s.iloc[30:35] = 400
    r = analyse_series("M", s, Params(deseasonalize=False), [])
    drops = [e for e in r.episodes if e.direction < 0]
    assert len(drops) == 1 and drops[0].start == idx[30] and drops[0].days == 5
