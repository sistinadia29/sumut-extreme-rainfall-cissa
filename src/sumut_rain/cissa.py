"""
Circulant Singular Spectrum Analysis (CiSSA), numpy murni.

Mengikuti Bogalo, Poncela & Senra (2021), "Circulant singular spectrum analysis:
A new automated procedure for signal extraction", Signal Processing 179, 107824.

Inti metode: pada matriks autokovarians sirkulan, vektor eigennya adalah vektor
Fourier u_k, sehingga setiap komponen elementer berasosiasi dengan satu frekuensi
w_k = (k-1)/L yang diketahui. Komponen ke-k diperoleh dengan memproyeksikan
matriks trajektori ke u_k dan pasangan konjugatnya u_{L+2-k}, lalu dirata-ratakan
diagonalnya (Hankelisasi). Jumlah semua komponen = deret asli (persis).

Ujung deret diperpanjang ke depan dan ke belakang dengan model AR (H titik)
untuk mengurangi efek tepi, lalu dipotong kembali.
"""

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------- perpanjangan AR
def _ar_fit(z: np.ndarray, pmax: int) -> np.ndarray:
    """Koefisien AR(p) dengan kuadrat terkecil, p dipilih dengan AIC."""
    n = len(z)
    best = None
    for p in range(1, pmax + 1):
        X = np.column_stack([z[p - 1 - i:n - 1 - i] for i in range(p)])
        y = z[p:]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        aic = len(y) * np.log(resid.var()) + 2 * p
        if best is None or aic < best[0]:
            best = (aic, coef)
    return best[1]


def _ar_forecast(z: np.ndarray, coef: np.ndarray, h: int) -> np.ndarray:
    hist = list(z[-len(coef):])
    out = []
    for _ in range(h):
        nxt = float(np.dot(coef, hist[::-1][:len(coef)]))
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def extend_series(x: np.ndarray, h: int, pmax: int | None = None) -> np.ndarray:
    """Perpanjang deret h titik di kedua ujung dengan model AR (maju & mundur)."""
    x = np.asarray(x, dtype=float)
    pmax = pmax or max(1, min(24, len(x) // 4))
    mu = x.mean()
    z = x - mu
    fwd = _ar_forecast(z, _ar_fit(z, pmax), h)
    zb = z[::-1]
    bwd = _ar_forecast(zb, _ar_fit(zb, pmax), h)[::-1]
    return np.concatenate([bwd, z, fwd]) + mu


# ---------------------------------------------------------------- CiSSA
def _hankel_average(Y: np.ndarray) -> np.ndarray:
    """Rata-rata anti-diagonal matriks L x N menjadi deret panjang L+N-1."""
    L, N = Y.shape
    idx = np.add.outer(np.arange(L), np.arange(N))
    s = np.bincount(idx.ravel(), weights=Y.ravel(), minlength=L + N - 1)
    c = np.bincount(idx.ravel(), minlength=L + N - 1)
    return s / c


def cissa(x: np.ndarray, L: int, h: int | None = None, extend: bool = True):
    """Dekomposisi CiSSA.

    Mengembalikan:
        freqs : (M,) frekuensi w_k = (k-1)/L dalam siklus per satuan waktu, M = floor(L/2)+1
        comps : (M, T) komponen elementer; comps.sum(axis=0) == x
        power : (M,) energi relatif tiap frekuensi (estimasi spektrum, untuk diagnosis)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if not 2 <= L <= T // 2:
        raise ValueError(f"L harus antara 2 dan T/2 (T={T}), didapat L={L}")
    h = L if h is None else h
    xe = extend_series(x, h) if (extend and h > 0) else x
    h = h if (extend and h > 0) else 0

    X = np.lib.stride_tricks.sliding_window_view(xe, L).T      # (L, N)
    F = np.fft.fft(X, axis=0)
    M = L // 2 + 1
    comps = np.empty((M, len(xe)))
    power = np.empty(M)
    for k in range(M):
        mask = np.zeros(L, dtype=bool)
        mask[k] = True
        mask[(L - k) % L] = True
        Xk = np.fft.ifft(F * mask[:, None], axis=0).real
        comps[k] = _hankel_average(Xk)
        power[k] = (np.abs(F[mask]) ** 2).sum()
    freqs = np.arange(M) / L
    return freqs, comps[:, h:h + T], power / power.sum()


def group_components(freqs: np.ndarray, comps: np.ndarray, *, period_seasonal: float = 12,
                     trend_min_period: float = 96, band: tuple = (18, 96)) -> dict:
    """Kelompokkan komponen elementer berdasarkan periode (dalam satuan waktu deret).

    trend       : periode > trend_min_period (termasuk frekuensi nol)
    interannual : band[0] <= periode <= band[1]   (pita ENSO/IOD)
    seasonal    : harmonik dari period_seasonal (12, 6, 4, 3, 2.4, 2 bulan)
    residual    : sisanya
    """
    with np.errstate(divide="ignore"):
        period = np.where(freqs > 0, 1 / freqs, np.inf)
    # Harmonik musiman j/period_seasonal diambil dari frekuensi Fourier TERDEKAT.
    # Jika L kelipatan period_seasonal, frekuensinya tepat sama (tanpa kebocoran).
    harmonic = np.zeros_like(freqs, dtype=bool)
    j = 1
    while j / period_seasonal <= freqs.max() + 1e-12:
        harmonic[np.argmin(np.abs(freqs - j / period_seasonal))] = True
        j += 1
    groups = {
        "trend": period > trend_min_period,
        "interannual": (period >= band[0]) & (period <= band[1]) & ~harmonic,
        "seasonal": harmonic,
    }
    used = np.zeros_like(freqs, dtype=bool)
    for m in groups.values():
        used |= m
    groups["residual"] = ~used
    return {name: comps[m].sum(axis=0) for name, m in groups.items()}


# ---------------------------------------------------------------- pemilihan L otomatis
def w_correlation(a: np.ndarray, b: np.ndarray, L: int) -> float:
    """w-correlation antara dua deret hasil rekonstruksi (Golyandina dkk. 2001).
    Bobot w_i = jumlah kemunculan titik ke-i dalam matriks trajektori:
    w_i = min(i+1, L*, T-i), dengan L* = min(L, T-L+1). Nilai mendekati 0 = terpisah baik."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    T = len(a)
    Ls = min(L, T - L + 1)
    i = np.arange(T)
    w = np.minimum(np.minimum(i + 1, Ls), T - i)
    na = np.sqrt(np.sum(w * a * a))
    nb = np.sqrt(np.sum(w * b * b))
    if na == 0 or nb == 0:
        return np.nan
    return float(np.sum(w * a * b) / (na * nb))


def select_window(x: np.ndarray, candidates, *, signal_groups=("trend", "interannual", "seasonal"),
                  **group_kwargs):
    """Jalankan CiSSA berulang untuk setiap L kandidat, lalu pilih L dengan
    keterpisahan terbaik: w-correlation absolut maksimum antar kelompok sinyal
    sekecil mungkin. Mengembalikan (L_terpilih, daftar baris hasil per L)."""
    rows = []
    for L in candidates:
        freqs, comps, _ = cissa(x, L)
        g = group_components(freqs, comps, **group_kwargs)
        row = {"L": int(L)}
        for a, b in combinations(signal_groups, 2):
            row[f"wcorr_{a}_{b}"] = round(abs(w_correlation(g[a], g[b], L)), 4)
        vals = [v for k, v in row.items() if k.startswith("wcorr_")]
        row["max_wcorr"] = np.inf if any(np.isnan(vals)) else max(vals)
        rows.append(row)
    best = min(rows, key=lambda r: (r["max_wcorr"], r["L"]))["L"]
    return best, rows
