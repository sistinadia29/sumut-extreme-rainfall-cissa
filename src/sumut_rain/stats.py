"""
Statistik untuk korelasi deret waktu yang berautokorelasi kuat (misalnya komponen CiSSA).

Uji signifikansi memakai surrogate acak fase (Ebisuzaki 1997, J. Climate 10, 2147-2153):
fase Fourier satu deret diacak sambil spektrum dayanya dipertahankan, lalu korelasi
dihitung ulang ribuan kali. Statistik uji = |r| maksimum di semua lag yang dicoba,
sehingga p-value sudah memperhitungkan bahwa kita memilih lag terbaik.
"""

import numpy as np


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate dengan spektrum daya sama dan fase acak."""
    x = np.asarray(x, float)
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0, 2 * np.pi, len(X))
    Y = np.abs(X) * np.exp(1j * phases)
    Y[0] = 0.0
    if n % 2 == 0:                      # frekuensi Nyquist harus real
        Y[-1] = np.abs(X[-1]) * rng.choice([-1.0, 1.0])
    return np.fft.irfft(Y, n) + x.mean()


def lag_correlations(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """r antara x(t) dan y(t - lag), lag = 0..max_lag (y mendahului x)."""
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        a = x[lag:]
        b = y[:len(y) - lag] if lag else y
        out[lag] = np.corrcoef(a, b)[0, 1]
    return out


def surrogate_lag_test(x: np.ndarray, y: np.ndarray, max_lag: int = 6,
                       n_surr: int = 1000, seed: int = 0) -> dict:
    """Korelasi lag terbaik antara x dan y, dengan p-value surrogate acak fase pada x."""
    r = lag_correlations(x, y, max_lag)
    best = int(np.nanargmax(np.abs(r)))
    obs = abs(r[best])
    rng = np.random.default_rng(seed)
    null = np.empty(n_surr)
    for i in range(n_surr):
        null[i] = np.nanmax(np.abs(lag_correlations(phase_randomize(x, rng), y, max_lag)))
    p = (1 + np.sum(null >= obs)) / (n_surr + 1)
    return {"lag_months": best, "r": float(r[best]), "p_surrogate": float(p),
            "r95_null": float(np.quantile(null, 0.95))}


def fdr_bh(p: np.ndarray) -> np.ndarray:
    """q-value Benjamini-Hochberg untuk banyak uji sekaligus."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out
