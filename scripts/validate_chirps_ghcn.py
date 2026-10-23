"""
Validasi CHIRPS terhadap stasiun GHCN-Daily di Sumatera Utara (hari-berpasangan).

Dijalankan SETELAH find_ghcn_stations.py.

Untuk setiap stasiun:
 1. Ambil deret CHIRPS dari sel darat terdekat (sel laut dilewati)
 2. Cari pergeseran hari terbaik (-1, 0, +1): laporan SYNOP mencatat 24 jam yang
    berakhir 00 UTC (07.00 WIB), sehingga 'hari' stasiun dan CHIRPS bisa bergeser satu hari
 3. Bandingkan hanya pada hari yang datanya ada di keduanya:
      - harian: r, Spearman, bias, RMSE, MAE, deteksi hari hujan (POD, FAR, CSI pada 1 mm)
      - bulanan: total bulan dengan >= 90% hari tersedia
      - kuantil hari basah: P50, P90, P95, P99, maksimum (apakah ekstrem diremehkan?)
      - per dekade: apakah bias CHIRPS di 1980-an berbeda?

Keluaran (ditahan .gitignore sampai naskah dikirim):
    outputs/tables/validation_summary.csv
    outputs/tables/validation_quantiles.csv
    outputs/tables/validation_decadal.csv
    outputs/figures/validation_quantiles.png

Contoh:
    python scripts/validate_chirps_ghcn.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.validation import (best_day_shift, by_decade, daily_metrics,  # noqa: E402
                                   matched, monthly_metrics, wet_quantiles)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def nearest_land_cell(pr: xr.DataArray, land: np.ndarray, lat: float, lon: float):
    lats, lons = pr["latitude"].values, pr["longitude"].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    d = (lat2d - lat) ** 2 + ((lon2d - lon) * np.cos(np.deg2rad(lat))) ** 2
    d[~land] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    km = float(np.sqrt(d[i, j]) * 111.2)
    return i, j, km


def main() -> None:
    cfg = load_config()
    gdir = project_path(f"{cfg['paths']['external']}/ghcn")
    tables = project_path("outputs/tables")
    figs = project_path(cfg["paths"]["figures"])
    wet = cfg["indices"]["wet_day_mm"]

    stations = pd.read_csv(tables / "ghcn_stations.csv")
    files = sorted(project_path(cfg["paths"]["raw_chirps"]).glob("chirps_sumut_*.nc"))
    pr = xr.open_mfdataset(files, combine="by_coords")[cfg["chirps"]["variable"]]
    land = pr.isel(time=0).notnull().values
    print(f"Stasiun: {len(stations)}  |  CHIRPS: {len(files)} tahun\n")

    summary, quant, decadal, qplot = [], [], [], {}
    for _, st in stations.iterrows():
        f = gdir / f"{st['id']}_prcp.csv"
        if not f.exists():
            continue
        obs = pd.read_csv(f, parse_dates=["date"]).set_index("date")["prcp_mm"]
        i, j, km = nearest_land_cell(pr, land, st["lat"], st["lon"])
        est = pr.isel(latitude=i, longitude=j).load().to_series()
        est.index = pd.to_datetime(est.index).normalize()
        obs = obs[(obs.index >= est.index.min()) & (obs.index <= est.index.max())]

        shift, rs = best_day_shift(obs, est)
        d = matched(obs, est, shift)
        name = str(st["name"]).strip()
        m = daily_metrics(d, wet)
        summary.append({"station": name, "id": st["id"], "distance_km": round(km, 1),
                        "day_shift": shift, **{f"r_shift_{k}": round(v, 3) for k, v in rs.items()},
                        **m, **monthly_metrics(d)})
        q = wet_quantiles(d, wet=wet)
        q.insert(0, "station", name)
        quant.append(q)
        dec = by_decade(d, wet=wet)
        dec.insert(0, "station", name)
        decadal.append(dec)
        qplot[name] = d
        print(f"  {name:<20} jarak sel {km:4.1f} km | geser hari {shift:+d} | "
              f"{m['n_days']} hari berpasangan | r = {m['r']:.2f}")

    sm = pd.DataFrame(summary).round(3)
    qt = pd.concat(quant).round(2)
    dc = pd.concat(decadal).round(3) if decadal else pd.DataFrame()
    sm.to_csv(tables / "validation_summary.csv", index=False)
    qt.to_csv(tables / "validation_quantiles.csv", index=False)
    dc.to_csv(tables / "validation_decadal.csv", index=False)
    plot_qq(qplot, wet, figs / "validation_quantiles.png")

    pd.set_option("display.width", 180)
    print("\n=== Ringkasan harian & bulanan (CHIRPS vs stasiun) ===")
    print(sm[["station", "day_shift", "n_days", "r", "spearman", "bias_pct", "rmse_mm", "pod", "far",
              "csi", "wetday_freq_obs", "wetday_freq_est", "n_months", "r_monthly",
              "bias_monthly_pct"]].to_string(index=False))
    print("\n=== Kuantil hari basah (rasio CHIRPS/stasiun < 1 = CHIRPS meremehkan) ===")
    print(qt.pivot(index="station", columns="quantile", values="ratio_est_obs")
          [["P50", "P90", "P95", "P99", "max"]].to_string())
    if not dc.empty:
        print("\n=== Per dekade (dekade dengan >= 300 hari berpasangan) ===")
        print(dc.to_string(index=False))
    print(f"\nTersimpan di {tables} dan {figs}")


def plot_qq(pairs: dict, wet: float, out: Path) -> None:
    probs = np.linspace(0.01, 0.995, 200)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    vmax = 0
    for (name, d), color in zip(pairs.items(), COLORS):
        o = np.quantile(d.loc[d["obs"] >= wet, "obs"], probs)
        e = np.quantile(d.loc[d["est"] >= wet, "est"], probs)
        ax.plot(o, e, color=color, linewidth=1.8, label=name)
        vmax = max(vmax, o.max(), e.max())
    ax.plot([0, vmax], [0, vmax], color=MUTED, linewidth=1, linestyle="--", label="1:1")
    ax.set_xlabel("Station wet-day rainfall quantile (mm/day)", color=MUTED, fontsize=9)
    ax.set_ylabel("CHIRPS wet-day rainfall quantile (mm/day)", color=MUTED, fontsize=9)
    ax.set_title("Quantile-quantile: CHIRPS vs GHCN-Daily stations\n(below the dashed line = CHIRPS underestimates)",
                 loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
