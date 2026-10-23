"""
Uji modul statistik (surrogate acak fase, korelasi lag, FDR).
Jalankan dari root proyek:  python tests/test_stats.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain.stats import (commonality, fdr_bh, lag_correlations, partial_corr,  # noqa: E402
                              partial_lag_test, phase_randomize, shift, surrogate_lag_test)


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


def test_partial_correlation_separates_confounder():
    # Hujan hanya bergantung pada ENSO; IOD berkorelasi dengan ENSO tapi tidak memengaruhi hujan
    rng = np.random.default_rng(5)
    smooth = lambda: np.convolve(rng.normal(size=600), np.ones(18) / 18, mode="same")[:540]
    enso = smooth()
    iod = 0.7 * enso + 0.5 * smooth()
    rain = -1.0 * enso + 0.4 * smooth()
    assert abs(np.corrcoef(rain, iod)[0, 1]) > 0.5           # korelasi biasa menipu
    assert abs(partial_corr(rain, iod, enso)) < 0.2          # parsial membongkarnya
    p_enso = partial_lag_test(rain, enso, iod, 0, 3, n_surr=300, seed=1)
    p_iod = partial_lag_test(rain, iod, enso, 0, 3, n_surr=300, seed=1)
    assert p_enso["p_surrogate"] < 0.01 and p_iod["p_surrogate"] > 0.05
    com = commonality(rain, enso, iod, 0, 0)
    assert com["unique_1"] > 0.3 and com["unique_2"] < 0.03
    assert abs(com["unique_1"] + com["unique_2"] + com["shared"] - com["r2_full"]) < 1e-9
    assert np.isnan(shift(np.arange(5.0), 2)[:2]).all() and shift(np.arange(5.0), 2)[2] == 0


if __name__ == "__main__":
    test_phase_randomize_keeps_spectrum()
    test_lag_direction()
    test_surrogate_detects_real_and_rejects_fake()
    test_fdr()
    test_partial_correlation_separates_confounder()
    print("Semua uji statistik lolos.")
