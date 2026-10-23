"""
Minggu 4: dekomposisi CiSSA dan uji tren Mann-Kendall.

Langkah:
 1. Deret bulanan rata-rata Sumut dari CHIRPS harian (disimpan ke cache CSV):
      prcp_mm   : total hujan bulanan
      rx1day_mm : hujan harian maksimum dalam bulan tersebut
 2. CiSSA (L dari config) -> kelompok tren, antartahunan (ENSO/IOD), musiman, residu
 3. Porsi varians tiap kelompok + korelasi komponen antartahunan dengan ONI dan DMI
    (lag 0..max_lag bulan, dengan ukuran sampel efektif untuk autokorelasi)
 4. Mann-Kendall termodifikasi (Hamed & Rao 1998) + Sen's slope pada 6 indeks
    tahunan, periode penuh dan periode sensitivitas (default 1991-2025)

Keluaran:
    data/processed/monthly_regional.csv
    data/processed/cissa_components_<var>.csv
    outputs/tables/cissa_variance.csv, cissa_climate_correlation.csv, mann_kendall.csv
    outputs/figures/cissa_<var>.png

Contoh:
    python scripts/analyze_cissa.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymannkendall as mk  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.cissa import cissa, group_components, select_window  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ab"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
VAR_LABEL = {"prcp_mm": "Monthly rainfall (mm)", "rx1day_mm": "Monthly max 1-day rainfall (mm)"}


# ------------------------------------------------------------------ 1. deret bulanan
def monthly_regional(cfg: dict) -> pd.DataFrame:
    proc = project_path(cfg["paths"]["processed"])
    cache = proc / "monthly_regional.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    import geopandas as gpd
    import regionmask
    import xarray as xr

    files = sorted(project_path(cfg["paths"]["raw_chirps"]).glob("chirps_sumut_*.nc"))
    pr = xr.open_mfdataset(files, combine="by_coords")[cfg["chirps"]["variable"]].astype("float32").load()
    prov = gpd.read_file(project_path(cfg["paths"]["external"]) / "sumut_province.geojson")
    inside = regionmask.mask_geopandas(prov, pr["longitude"], pr["latitude"]).notnull()
    pr = pr.where(inside)
    w = np.cos(np.deg2rad(pr["latitude"]))

    total = pr.resample(time="MS").sum(min_count=1).weighted(w).mean(("latitude", "longitude"))
    mx = pr.resample(time="MS").max().weighted(w).mean(("latitude", "longitude"))
    df = pd.DataFrame({"date": total["time"].values,
                       "prcp_mm": total.values.round(2), "rx1day_mm": mx.values.round(2)})
    df.to_csv(cache, index=False)
    print(f"Deret bulanan dibuat: {len(df)} bulan -> {cache.name}")
    return df


# ------------------------------------------------------------------ 3. korelasi
def lagged_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> list[dict]:
    """Korelasi x(t) dengan y(t - lag): indeks iklim mendahului hujan.
    Nilai p memakai ukuran sampel efektif Bretherton et al. (1999)."""
    rows = []
    for lag in range(max_lag + 1):
        a = x[lag:]
        b = y[:len(y) - lag] if lag else y
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        r = np.corrcoef(a, b)[0, 1]
        r1a = np.corrcoef(a[1:], a[:-1])[0, 1]
        r1b = np.corrcoef(b[1:], b[:-1])[0, 1]
        n_eff = max(3.0, len(a) * (1 - r1a * r1b) / (1 + r1a * r1b))
        t = r * np.sqrt((n_eff - 2) / max(1e-12, 1 - r ** 2))
        p = 2 * stats.t.sf(abs(t), df=n_eff - 2)
        rows.append({"lag_months": lag, "r": round(r, 3), "n_eff": round(n_eff, 1), "p_value": round(p, 4)})
    return rows


# ------------------------------------------------------------------ 4. Mann-Kendall
def mann_kendall_table(regional: pd.DataFrame, start_sens: int) -> pd.DataFrame:
    rows = []
    for period, df in [("full", regional), (f"{start_sens}+", regional[regional["year"] >= start_sens])]:
        y0, y1 = int(df["year"].min()), int(df["year"].max())
        for col in [c for c in regional.columns if c != "year"]:
            res = mk.hamed_rao_modification_test(df[col].values)
            rows.append({"index": col, "period": f"{y0}-{y1}", "subset": period,
                         "trend": res.trend, "p_value": round(res.p, 4),
                         "sen_slope_per_decade": round(10 * res.slope, 2),
                         "mean": round(df[col].mean(), 2),
                         "change_pct_per_decade": round(100 * 10 * res.slope / df[col].mean(), 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ gambar
def zscore(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a)


def plot_decomposition(dates, x, g, oni, dmi, var, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax = axes[0]
    ax.plot(dates, x, color=GRAY, linewidth=0.8, label="Monthly series")
    ax.plot(dates, g["trend"], color=BLUE, linewidth=2, label="CiSSA low-frequency component (> 8 yr)")
    ax.set_title(f"{VAR_LABEL[var]} and its long-term component", loc="left", fontsize=10, color=INK)
    for a, ref, name, phase in [(axes[1], oni, "ONI", "El Niño"), (axes[2], dmi, "DMI", "positive IOD")]:
        a.plot(dates, zscore(g["interannual"]), color=BLUE, linewidth=1.6,
               label="CiSSA interannual component (1.5-8 yr)")
        a.plot(dates, -zscore(ref), color=ORANGE, linewidth=1.6, label=f"{name}, sign reversed")
        a.axhline(0, color=GRID, linewidth=1)
        a.set_title(f"Interannual component vs {name} (both standardised; {name} reversed so "
                    f"{phase} points down)", loc="left", fontsize=10, color=INK)
        a.set_ylabel("z-score", color=MUTED, fontsize=9)
    axes[0].set_ylabel("mm", color=MUTED, fontsize=9)
    for a in axes:
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("CiSSA decomposition, North Sumatra (CHIRPS 1981-2025)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selection(sel: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for var, color in (("prcp_mm", BLUE), ("rx1day_mm", ORANGE)):
        d = sel[sel["variable"] == var].sort_values("L")
        if d.empty:
            continue
        ax.plot(d["L"], d["max_wcorr"], color=color, linewidth=1.8, marker="o", markersize=5,
                label=VAR_LABEL[var])
        pick = d[d["selected"]]
        ax.scatter(pick["L"], pick["max_wcorr"], s=120, facecolors="none", edgecolors=color,
                   linewidths=2, zorder=3)
    ax.set_xlabel("Window length L (months)", color=MUTED, fontsize=9)
    ax.set_ylabel("Max |w-correlation| between groups", color=MUTED, fontsize=9)
    ax.set_title("Automatic window selection: lower = better separability (circle = selected L)",
                 loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    cfg = load_config()
    cc = cfg["cissa"]
    proc = project_path(cfg["paths"]["processed"])
    tables = project_path("outputs/tables")
    figs = project_path(cfg["paths"]["figures"])

    monthly = monthly_regional(cfg)
    clim = pd.read_csv(project_path(cfg["paths"]["external"]) / "climate_indices_monthly.csv",
                       parse_dates=["date"])
    df = monthly.merge(clim, on="date", how="left")
    dates = df["date"].values

    gkw = {"trend_min_period": cc["trend_min_period_months"],
           "band": tuple(cc["interannual_band_months"])}
    var_rows, corr_rows, trend_rows, sel_rows, chosen = [], [], [], [], {}
    for var in ("prcp_mm", "rx1day_mm"):
        x = df[var].values.astype(float)

        # --- pilih L: otomatis (CiSSA diulang untuk tiap kandidat) atau tetap dari config
        if str(cc["window_months"]).lower() == "auto":
            lo, hi, step = cc["window_search"]
            hi = min(hi, len(x) // 2)
            # Tahap 1 (kasar): kelipatan 12
            L_coarse, rows = select_window(x, range(lo, hi + 1, step), **gkw)
            for r in rows:
                r["stage"] = "coarse"
            # Tahap 2 (halus): L_coarse +/- radius, langkah 1
            radius = int(cc.get("window_refine_radius", 0))
            if radius > 0:
                fine = [L for L in range(max(lo, L_coarse - radius), min(hi, L_coarse + radius) + 1)
                        if L != L_coarse]
                _, rows_fine = select_window(x, fine, **gkw)
                for r in rows_fine:
                    r["stage"] = "fine"
                rows += rows_fine
            L = min(rows, key=lambda r: (r["max_wcorr"], r["L"]))["L"]
            best_score = min(r["max_wcorr"] for r in rows)
            for r in rows:
                # Info saja (TIDAK dipakai memilih L, supaya tidak memilih demi hasil yang diharapkan)
                fq, cp, _ = cissa(x, r["L"])
                inter = group_components(fq, cp, **gkw)["interannual"]
                r_oni = max(lagged_corr(inter, df["oni"].values, cc["max_lag_months"]),
                            key=lambda d: abs(d["r"]))["r"]
                sel_rows.append({"variable": var, **r, "r_oni_info_only": r_oni,
                                 "near_best": r["max_wcorr"] <= 1.1 * best_score,
                                 "coarse_best": r["L"] == L_coarse,
                                 "selected": r["L"] == L})
        else:
            L = int(cc["window_months"])
        chosen[var] = L

        freqs, comps, _ = cissa(x, L=L)
        g = group_components(freqs, comps, **gkw)
        out = pd.DataFrame({"date": df["date"], "original": x, **{k: v.round(3) for k, v in g.items()}})
        out.to_csv(proc / f"cissa_components_{var.replace('_mm', '')}.csv", index=False)

        total_var = np.var(x)
        for name, comp in g.items():
            var_rows.append({"variable": var, "component": name,
                             "variance_share_pct": round(100 * np.var(comp) / total_var, 1)})
        # Kemiringan OLS komponen tren (lebih stabil daripada selisih titik awal-akhir,
        # karena komponen > 8 tahun juga memuat variasi dekadal)
        months = np.arange(len(x))
        slope = np.polyfit(months, g["trend"], 1)[0] * 120
        trend_rows.append({"variable": var, "L": L, "trend_slope_per_decade": round(slope, 2),
                           "mean": round(float(np.mean(x)), 2),
                           "change_pct_per_decade": round(100 * slope / np.mean(x), 2)})

        for ref in ("oni", "dmi"):
            for row in lagged_corr(g["interannual"], df[ref].values, cc["max_lag_months"]):
                corr_rows.append({"variable": var, "climate_index": ref.upper(), **row})

        plot_decomposition(dates, x, g, df["oni"].values, df["dmi"].values, var,
                           figs / f"cissa_{var.replace('_mm', '')}.png")

    var_df = pd.DataFrame(var_rows)
    corr_df = pd.DataFrame(corr_rows)
    if sel_rows:
        sel_df = pd.DataFrame(sel_rows)
        sel_df.to_csv(tables / "cissa_L_selection.csv", index=False)
        plot_selection(sel_df, figs / "cissa_L_selection.png")
    trend_df = pd.DataFrame(trend_rows)
    var_df.to_csv(tables / "cissa_variance.csv", index=False)
    corr_df.to_csv(tables / "cissa_climate_correlation.csv", index=False)
    trend_df.to_csv(tables / "cissa_trend_slope.csv", index=False)

    regional = pd.read_csv(proc / "indices_regional_annual.csv")
    mk_df = mann_kendall_table(regional, cc["sensitivity_start_year"])
    mk_df.to_csv(tables / "mann_kendall.csv", index=False)

    pd.set_option("display.width", 160)
    if sel_rows:
        print("\n=== Pemilihan L otomatis (w-correlation maksimum antar kelompok; kecil = terpisah baik) ===")
        print(sel_df.pivot(index="L", columns="variable", values="max_wcorr").to_string())
        for var, L in chosen.items():
            near = sel_df[(sel_df["variable"] == var) & sel_df["near_best"]]["L"].tolist()
            coarse = int(sel_df[(sel_df["variable"] == var) & sel_df["coarse_best"]]["L"].iloc[0])
            print(f"{var}: tahap kasar = {coarse} bulan -> tahap halus = {L} bulan ({L / 12:.1f} tahun); "
                  f"L dalam 10% dari skor terbaik: {near}")
    else:
        print(f"\nL tetap dari config: {chosen}")
    print("\n=== Porsi varians komponen CiSSA (%) ===")
    print(var_df.pivot(index="component", columns="variable", values="variance_share_pct").to_string())
    print("\n=== Kemiringan komponen tren CiSSA (OLS) ===")
    print(trend_df.to_string(index=False))
    print("\n=== Korelasi komponen antartahunan dengan indeks iklim (lag terkuat) ===")
    best = corr_df.loc[corr_df.groupby(["variable", "climate_index"])["r"].apply(lambda s: s.abs().idxmax())]
    print(best.to_string(index=False))
    print("\n=== Mann-Kendall (Hamed-Rao) + Sen's slope, indeks tahunan ===")
    print(mk_df[["index", "period", "trend", "p_value", "sen_slope_per_decade",
                 "change_pct_per_decade"]].to_string(index=False))
    print(f"\nTersimpan: {tables}, {figs}, {proc}")


if __name__ == "__main__":
    main()
