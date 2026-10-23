"""
Uji CiSSA dengan deret buatan: tren + siklus 5 tahun + siklus tahunan + derau.
Jalankan dari root proyek:  python tests/test_cissa.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain.cissa import cissa, group_components, select_window, w_correlation  # noqa: E402


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_cissa_recovers_known_signals():
    rng = np.random.default_rng(42)
    t = np.arange(540)                       # 45 tahun data bulanan
    trend = 0.02 * t
    enso = 2.0 * np.sin(2 * np.pi * t / 60)  # siklus 5 tahun
    season = 3.0 * np.sin(2 * np.pi * t / 12)
    x = 100 + trend + enso + season + rng.normal(0, 1.0, t.size)

    freqs, comps, power = cissa(x, L=168)
    # 1) Rekonstruksi persis: jumlah semua komponen = deret asli
    assert np.allclose(comps.sum(axis=0), x, atol=1e-8)
    assert abs(power.sum() - 1) < 1e-9

    g = group_components(freqs, comps, trend_min_period=96, band=(18, 96))
    assert np.allclose(sum(g.values()), x, atol=1e-8)
    # 2) Setiap kelompok menangkap sinyal yang benar
    r_trend = corr(g["trend"], trend)
    r_enso = corr(g["interannual"], enso)
    r_season = corr(g["seasonal"], season)
    print(f"korelasi tren {r_trend:.3f} | antartahunan {r_enso:.3f} | musiman {r_season:.3f}")
    assert r_trend > 0.98
    assert r_enso > 0.95
    assert r_season > 0.98


def test_grouping_frequencies():
    freqs = np.arange(85) / 168
    comps = np.eye(85)
    g = group_components(freqs, comps)
    # k=0,1 tren; k=2..9 antartahunan; k=14,28,...,84 musiman
    assert np.flatnonzero(g["trend"]).tolist() == [0, 1]
    assert np.flatnonzero(g["interannual"]).tolist() == list(range(2, 10))
    assert np.flatnonzero(g["seasonal"]).tolist() == [14, 28, 42, 56, 70, 84]


def test_w_correlation_and_selection():
    t = np.arange(540)
    a = np.sin(2 * np.pi * t / 12)
    b = np.sin(2 * np.pi * t / 60)
    assert abs(w_correlation(a, a, 168) - 1) < 1e-12
    assert abs(w_correlation(a, b, 168)) < 0.02
    rng = np.random.default_rng(0)
    x = 100 + 0.02 * t + 2 * b + 3 * a + rng.normal(0, 1, t.size)
    best, rows = select_window(x, range(96, 265, 12))
    assert best in range(96, 265, 12) and len(rows) == 15
    assert all(r["max_wcorr"] < 0.05 for r in rows)


if __name__ == "__main__":
    test_grouping_frequencies()
    test_cissa_recovers_known_signals()
    test_w_correlation_and_selection()
    print("Semua uji CiSSA lolos.")
