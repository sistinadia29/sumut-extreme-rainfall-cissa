"""
Indeks ekstrem curah hujan ETCCDI, dihitung dengan numpy murni supaya mudah diuji.

Semua fungsi menerima array curah hujan harian `pr` berbentuk (waktu, ...) dalam mm/hari.
Sel yang seluruhnya NaN (laut) tetap NaN di keluaran.

Definisi (Zhang et al. 2011, ETCCDI):
    PRCPTOT : total hujan tahunan dari hari basah (pr >= 1 mm)            [mm]
    RX1DAY  : hujan harian maksimum dalam setahun                          [mm]
    RX5DAY  : akumulasi 5 hari berturut-turut maksimum dalam setahun       [mm]
    R20MM   : jumlah hari dengan hujan >= 20 mm                            [hari]
    R95P    : total hujan dari hari dengan pr > persentil-95 hari basah    [mm]
              (persentil dihitung pada periode acuan, per sel grid)
    CDD     : jumlah hari kering (pr < 1 mm) berturut-turut terpanjang     [hari]
"""

import numpy as np

INDEX_META = {
    "prcptot": ("Annual total wet-day precipitation", "mm"),
    "rx1day": ("Maximum 1-day precipitation", "mm"),
    "rx5day": ("Maximum consecutive 5-day precipitation", "mm"),
    "r20mm": ("Number of very heavy precipitation days (>= 20 mm)", "days"),
    "r95p": ("Precipitation from very wet days (> 95th percentile)", "mm"),
    "cdd": ("Maximum length of dry spell (< 1 mm)", "days"),
}


def max_consecutive(flags: np.ndarray) -> np.ndarray:
    """Panjang deret True berturut-turut terpanjang di sepanjang sumbu 0."""
    flags = np.asarray(flags, dtype=bool)
    run = np.zeros(flags.shape[1:], dtype=np.int32)
    best = np.zeros_like(run)
    for f in flags:
        run = (run + 1) * f
        np.maximum(best, run, out=best)
    return best


def wet_day_percentile(pr: np.ndarray, q: float = 95, wet: float = 1.0,
                       min_wet_days: int = 20) -> np.ndarray:
    """Persentil ke-q dari hari basah (pr >= wet) untuk setiap sel grid."""
    flat = pr.reshape(pr.shape[0], -1)
    out = np.full(flat.shape[1], np.nan, dtype=np.float32)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        w = col[np.isfinite(col) & (col >= wet)]
        if w.size >= min_wet_days:
            out[j] = np.percentile(w, q)
    return out.reshape(pr.shape[1:])


def annual_indices(pr: np.ndarray, years: np.ndarray, *, wet: float = 1.0,
                   heavy: float = 20.0, thr95: np.ndarray | None = None):
    """Hitung semua indeks per tahun.

    Mengembalikan (tahun_unik, dict nama -> array (n_tahun, ...)).
    RX5DAY memakai jendela 5 hari yang berakhir di tahun tersebut, sehingga
    jendela yang melewati pergantian tahun ikut ke tahun hari terakhirnya.
    """
    pr = np.asarray(pr, dtype=np.float32)
    years = np.asarray(years)
    uniq = np.unique(years)
    valid = np.isfinite(pr).all(axis=0)
    shape = (len(uniq),) + pr.shape[1:]
    out = {k: np.full(shape, np.nan, dtype=np.float32) for k in INDEX_META}
    if thr95 is None:
        out.pop("r95p")

    idx = np.arange(len(years))
    for i, y in enumerate(uniq):
        sel = idx[years == y]
        start, end = sel[0], sel[-1] + 1
        p = pr[start:end]
        p0 = np.nan_to_num(p)

        out["prcptot"][i] = np.where(p0 >= wet, p0, 0).sum(axis=0)
        out["rx1day"][i] = p0.max(axis=0)
        out["r20mm"][i] = (p0 >= heavy).sum(axis=0)
        out["cdd"][i] = max_consecutive(p0 < wet)

        # Jendela 5 hari: sertakan 4 hari terakhir tahun sebelumnya bila ada
        lead = min(4, start)
        ext = np.nan_to_num(pr[start - lead:end])
        n = ext.shape[0] - 4
        if n > 0:
            s5 = sum(ext[k:k + n] for k in range(5))
            out["rx5day"][i] = s5.max(axis=0)

        if thr95 is not None:
            out["r95p"][i] = np.where((p0 >= wet) & (p0 > thr95), p0, 0).sum(axis=0)

    for k in out:
        out[k][:, ~valid] = np.nan
    return uniq, out
