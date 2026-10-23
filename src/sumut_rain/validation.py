"""
Metrik validasi curah hujan harian: estimasi (CHIRPS) vs observasi (stasiun).
Semua fungsi menerima pandas Series yang diindeks tanggal (mm/hari).
"""

import numpy as np
import pandas as pd


def best_day_shift(obs: pd.Series, est: pd.Series, shifts=(-1, 0, 1)) -> tuple[int, dict]:
    """Pergeseran hari s yang memaksimalkan korelasi obs(t) dengan est(t + s).
    Menangani perbedaan definisi 'hari' (misalnya SYNOP 00 UTC vs hari kalender)."""
    rs = {}
    for s in shifts:
        e = est.shift(-s)                    # e(t) = est(t + s)
        d = pd.concat([obs, e], axis=1, keys=["o", "e"]).dropna()
        rs[s] = float(d["o"].corr(d["e"])) if len(d) > 30 else np.nan
    best = max(rs, key=lambda k: -np.inf if np.isnan(rs[k]) else rs[k])
    return best, rs


def matched(obs: pd.Series, est: pd.Series, shift: int = 0) -> pd.DataFrame:
    e = est.shift(-shift)
    return pd.concat([obs, e], axis=1, keys=["obs", "est"]).dropna()


def daily_metrics(d: pd.DataFrame, wet: float = 1.0) -> dict:
    o, e = d["obs"].values, d["est"].values
    hit = np.sum((o >= wet) & (e >= wet))
    miss = np.sum((o >= wet) & (e < wet))
    false = np.sum((o < wet) & (e >= wet))
    return {
        "n_days": len(d),
        "r": float(np.corrcoef(o, e)[0, 1]),
        "spearman": float(pd.Series(o).corr(pd.Series(e), method="spearman")),
        "bias_pct": float(100 * (e.mean() - o.mean()) / o.mean()),
        "rmse_mm": float(np.sqrt(np.mean((e - o) ** 2))),
        "mae_mm": float(np.mean(np.abs(e - o))),
        "pod": float(hit / (hit + miss)) if hit + miss else np.nan,
        "far": float(false / (hit + false)) if hit + false else np.nan,
        "csi": float(hit / (hit + miss + false)) if hit + miss + false else np.nan,
        "wetday_freq_obs": float(np.mean(o >= wet)),
        "wetday_freq_est": float(np.mean(e >= wet)),
    }


def monthly_metrics(d: pd.DataFrame, min_frac: float = 0.9) -> dict:
    """Total bulanan hanya untuk bulan dengan >= min_frac hari berpasangan."""
    g = d.groupby(d.index.to_period("M"))
    n = g.size()
    days = pd.Series([p.days_in_month for p in n.index], index=n.index)
    ok = n[n >= min_frac * days].index
    m = g.sum().loc[ok]
    if len(m) < 6:
        return {"n_months": len(m), "r_monthly": np.nan, "bias_monthly_pct": np.nan}
    return {"n_months": len(m),
            "r_monthly": float(m["obs"].corr(m["est"])),
            "bias_monthly_pct": float(100 * (m["est"].mean() - m["obs"].mean()) / m["obs"].mean())}


def wet_quantiles(d: pd.DataFrame, qs=(0.5, 0.9, 0.95, 0.99), wet: float = 1.0) -> pd.DataFrame:
    """Kuantil hujan pada hari basah, masing-masing dari distribusinya sendiri."""
    o = d.loc[d["obs"] >= wet, "obs"]
    e = d.loc[d["est"] >= wet, "est"]
    rows = []
    for q in qs:
        qo, qe = float(o.quantile(q)), float(e.quantile(q))
        rows.append({"quantile": f"P{int(q * 100)}", "obs_mm": qo, "est_mm": qe, "ratio_est_obs": qe / qo})
    rows.append({"quantile": "max", "obs_mm": float(o.max()), "est_mm": float(e.max()),
                 "ratio_est_obs": float(e.max() / o.max())})
    return pd.DataFrame(rows)


def by_decade(d: pd.DataFrame, min_days: int = 300, wet: float = 1.0) -> pd.DataFrame:
    rows = []
    for dec, g in d.groupby((d.index.year // 10) * 10):
        if len(g) < min_days:
            continue
        o = g.loc[g["obs"] >= wet, "obs"]
        e = g.loc[g["est"] >= wet, "est"]
        rows.append({"decade": int(dec), "n_days": len(g),
                     "r": float(g["obs"].corr(g["est"])),
                     "bias_pct": float(100 * (g["est"].mean() - g["obs"].mean()) / g["obs"].mean()),
                     "p95_ratio_est_obs": float(e.quantile(0.95) / o.quantile(0.95)),
                     "p99_ratio_est_obs": float(e.quantile(0.99) / o.quantile(0.99))})
    return pd.DataFrame(rows)
