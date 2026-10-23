"""
Uji modul statistik (surrogate acak fase, korelasi lag, FDR).
Jalankan dari root proyek:  python tests/test_stats.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain.stats import fdr_bh, lag_correlations, phase_randomize, surrogate_lag_test  # noqa: E402


def test_phase_randomize_keeps_spectrum():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=540))          # deret sangat berautokorelasi
    s = phase_randomize(x, rng)
    px = np.abs(np.fft.rfft(x - x.mean()))[1:]
    ps = np.abs(np.fft.rfft(s - s.mean()))[1:]
    assert np.allclose(px, ps, rtol=1e-8)
    assert abs(np.corrcoef(x, s)[0, 1]) < 0.9      # fasenya benar-benar berubah


def test_lag_direction():
    t = np.arange(540)
    y = np.sin(2 * np.pi * t / 48)
    x = np.roll(y, 3)                             # x tertinggal 3 bulan dari y
    r = lag_correlations(x, y, 6)
    assert int(np.argmax(r)) == 3


def test_surrogate_detects_real_and_rejects_fake():
    rng = np.random.default_rng(1)
    t = np.arange(540)
    enso = np.convolve(rng.normal(size=600), np.ones(24) / 24, mode="same")[:540]
    rain = -0.8 * np.roll(enso, 2) + 0.3 * np.convolve(rng.normal(size=600), np.ones(24) / 24, "same")[:540]
    real = surrogate_lag_test(rain, enso, 6, n_surr=500, seed=0)
    assert real["r"] < 0 and real["p_surrogate"] < 0.01 and real["lag_months"] in (1, 2, 3)
    unrelated = np.convolve(rng.normal(size=600), np.ones(24) / 24, mode="same")[:540]
    fake = surrogate_lag_test(unrelated, enso, 6, n_surr=500, seed=0)
    assert fake["p_surrogate"] > 0.05


def test_fdr():
    q = fdr_bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)


if __name__ == "__main__":
    test_phase_randomize_keeps_spectrum()
    test_lag_direction()
    test_surrogate_detects_real_and_rejects_fake()
    test_fdr()
    print("Semua uji statistik lolos.")
