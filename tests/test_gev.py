"""
Uji GEV non-stasioner dengan data buatan yang parameternya diketahui.
Jalankan dari root proyek:  python tests/test_gev.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain.gev import bootstrap, fit, lrt, return_level, simulate  # noqa: E402


def test_return_level_matches_scipy():
    for xi in (-0.2, 0.0, 0.15):
        rl = return_level(50, 10, xi, 20)
        ref = stats.genextreme.ppf(1 - 1 / 20, c=-xi, loc=50, scale=10)
        assert abs(rl - ref) < 1e-6


def test_recovers_parameters():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 3000)
    X = np.column_stack([np.ones_like(x), x])
    y = simulate(60 + 8 * x, 12, 0.1, rng)
    f = fit(y, X)
    assert abs(f["beta"][0] - 60) < 1.5 and abs(f["beta"][1] - 8) < 1.0
    assert abs(f["sigma"] - 12) < 1.0 and abs(f["xi"] - 0.1) < 0.05


def test_lrt_detects_and_rejects():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 45)
    X = np.column_stack([np.ones_like(x), x])
    y_eff = simulate(60 + 10 * x, 8, 0.1, rng)
    assert lrt(fit(y_eff), fit(y_eff, X))[1] < 0.01          # efek kuat terdeteksi
    false_pos = 0
    for s in range(100):
        r = np.random.default_rng(100 + s)
        y0 = simulate(np.full(45, 60.0), 8, 0.1, r)
        false_pos += lrt(fit(y0), fit(y0, X))[1] < 0.05
    assert false_pos <= 12                                    # ~5% salah positif


def test_bootstrap_runs():
    rng = np.random.default_rng(3)
    y = simulate(np.full(45, 60.0), 8, 0.1, rng)
    X = np.ones((45, 1))
    f = fit(y, X)
    b = bootstrap(y, X, f, lambda g: return_level(g["beta"][0], g["sigma"], g["xi"], 20), n_boot=50)
    rl = return_level(f["beta"][0], f["sigma"], f["xi"], 20)
    assert len(b) >= 45 and np.percentile(b, 2.5) < rl < np.percentile(b, 97.5)


if __name__ == "__main__":
    test_return_level_matches_scipy()
    test_recovers_parameters()
    test_lrt_detects_and_rejects()
    test_bootstrap_runs()
    print("Semua uji GEV lolos.")
