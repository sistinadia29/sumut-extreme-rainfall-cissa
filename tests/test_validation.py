"""
Uji metrik validasi dengan data buatan yang jawabannya diketahui.
Jalankan dari root proyek:  python tests/test_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain.validation import (best_day_shift, by_decade, daily_metrics,  # noqa: E402
                                   matched, monthly_metrics, wet_quantiles)


def make_pair(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1985-01-01", "2004-12-31", freq="D")
    rain = rng.gamma(0.5, 20, len(idx)) * (rng.random(len(idx)) < 0.5)
    obs = pd.Series(rain, idx)
    # estimasi = observasi digeser 1 hari lebih awal, ekstrem diredam 30%, sedikit derau
    est = pd.Series(np.minimum(rain, np.quantile(rain, 0.95)) * 0.7 + rain * 0.0, idx).shift(-1)
    est = est + np.abs(rng.normal(0, 0.2, len(idx)))
    return obs, est


def test_shift_detected():
    obs, est = make_pair()
    s, rs = best_day_shift(obs, est)
    assert s == -1 and rs[-1] > rs[0]


def test_metrics_and_quantiles():
    obs, est = make_pair()
    d = matched(obs, est, -1)
    m = daily_metrics(d)
    assert m["r"] > 0.8 and m["bias_pct"] < 0
    assert 0 <= m["pod"] <= 1 and 0 <= m["far"] <= 1
    q = wet_quantiles(d).set_index("quantile")
    assert q.loc["P99", "ratio_est_obs"] < 0.8          # ekstrem diremehkan
    mm = monthly_metrics(d)
    assert mm["n_months"] > 200 and mm["r_monthly"] > 0.9
    dec = by_decade(d)
    assert set(dec["decade"]) == {1980, 1990, 2000}


if __name__ == "__main__":
    test_shift_detected()
    test_metrics_and_quantiles()
    print("Semua uji validasi lolos.")
