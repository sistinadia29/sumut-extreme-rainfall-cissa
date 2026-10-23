"""
Minggu 2: hitung indeks ekstrem curah hujan tahunan (ETCCDI) untuk Sumatera Utara.

Masukan : data/raw/chirps/chirps_sumut_*.nc, data/external/sumut_*.geojson
Keluaran:
    data/processed/extreme_indices_grid.nc        indeks per tahun per sel grid (hanya di dalam Sumut)
    data/processed/indices_regional_annual.csv    rata-rata provinsi per tahun (bobot cos lintang)
    data/processed/indices_kabupaten_annual.csv   rata-rata per kabupaten/kota per tahun
    outputs/figures/indices_regional_timeseries.png

Contoh:
    python scripts/compute_indices.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import regionmask  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.indices import INDEX_META, annual_indices, wet_day_percentile  # noqa: E402

LINE_COLOR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load_chirps(cfg: dict) -> xr.DataArray:
    files = sorted(project_path(cfg["paths"]["raw_chirps"]).glob("chirps_sumut_*.nc"))
    if not files:
        sys.exit("File CHIRPS belum ada. Jalankan download_chirps.py dulu.")
    ds = xr.open_mfdataset(files, combine="by_coords")
    pr = ds[cfg["chirps"]["variable"]].astype("float32").load()
    print(f"CHIRPS : {len(files)} file, {pr.sizes['time']} hari, "
          f"{pr.sizes['latitude']}x{pr.sizes['longitude']} sel")
    return pr


def kabupaten_table(ind: xr.Dataset, kab: gpd.GeoDataFrame, valid: np.ndarray) -> pd.DataFrame:
    """Rata-rata tiap indeks per kabupaten. Kota kecil yang tidak memuat
    satu pun titik pusat sel memakai sel valid terdekat (dicatat di kolom method)."""
    lon, lat = ind["longitude"].values, ind["latitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    kmask = regionmask.mask_geopandas(kab, ind["longitude"], ind["latitude"]).values

    rows = []
    for k, row in kab.iterrows():
        cells = (kmask == k) & valid
        method = "mean"
        if not cells.any():
            pt = row.geometry.representative_point()
            d = (lon2d - pt.x) ** 2 + (lat2d - pt.y) ** 2
            d[~valid] = np.inf
            cells = np.zeros_like(valid)
            cells[np.unravel_index(np.argmin(d), d.shape)] = True
            method = "nearest"
        for i, year in enumerate(ind["year"].values):
            rec = {"year": int(year), "kabupaten": row["kabupaten"],
                   "n_cells": int(cells.sum()), "method": method,
                   "chirps_coverage": row.get("chirps_coverage", np.nan)}
            for name in ind.data_vars:
                if name == "r95_threshold":
                    continue
                rec[name] = float(ind[name].values[i][cells].mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_regional(df: pd.DataFrame, out: Path) -> None:
    names = [n for n in INDEX_META if n in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, name in zip(axes.flat, names):
        title, unit = INDEX_META[name]
        ax.plot(df["year"], df[name], color=LINE_COLOR, linewidth=1.6)
        ax.set_title(f"{name.upper()}  ({unit})", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    fig.suptitle("Indeks ekstrem curah hujan, rata-rata Sumatera Utara (CHIRPS 1981-2025)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ic = cfg["indices"]
    ext = project_path(cfg["paths"]["external"])
    proc = project_path(cfg["paths"]["processed"])
    figs = project_path(cfg["paths"]["figures"])

    try:
        prov = gpd.read_file(ext / "sumut_province.geojson")
        kab = gpd.read_file(ext / "sumut_kabupaten.geojson")
    except Exception:
        sys.exit("Batas wilayah belum ada. Jalankan download_boundaries.py dulu.")

    pr = load_chirps(cfg)
    arr = pr.values
    years = pr["time"].dt.year.values

    # Masker provinsi: sel yang pusatnya di dalam Sumut dan punya data (bukan laut)
    pmask = regionmask.mask_geopandas(prov, pr["longitude"], pr["latitude"]).notnull().values
    valid = pmask & np.isfinite(arr).all(axis=0)
    print(f"Sel di dalam Sumut: {int(valid.sum())}")

    b0, b1 = ic["base_period"]
    base = (years >= b0) & (years <= b1)
    print(f"Menghitung persentil-{ic['percentile']} hari basah ({b0}-{b1}) ...")
    thr = wet_day_percentile(arr[base], q=ic["percentile"], wet=ic["wet_day_mm"])

    print("Menghitung indeks tahunan ...")
    uy, ind = annual_indices(arr, years, wet=ic["wet_day_mm"], heavy=ic["heavy_mm"], thr95=thr)

    coords = {"year": uy, "latitude": pr["latitude"].values, "longitude": pr["longitude"].values}
    ds = xr.Dataset(coords=coords)
    for name, values in ind.items():
        values[:, ~valid] = np.nan
        long_name, unit = INDEX_META[name]
        ds[name] = (("year", "latitude", "longitude"), values,
                    {"long_name": long_name, "units": unit})
    thr[~valid] = np.nan
    ds["r95_threshold"] = (("latitude", "longitude"), thr,
                           {"long_name": f"{ic['percentile']}th percentile of wet-day precipitation, "
                                         f"{b0}-{b1}", "units": "mm/day"})
    ds.attrs = {"source": "CHIRPS v2.0 daily 0.05 deg", "region": "Sumatera Utara",
                "indices": "ETCCDI (Zhang et al. 2011)", "wet_day_threshold_mm": ic["wet_day_mm"]}
    ds.to_netcdf(proc / "extreme_indices_grid.nc")

    # Rata-rata provinsi berbobot cos(lintang)
    w = np.cos(np.deg2rad(ds["latitude"]))
    regional = (ds[list(ind)].weighted(w).mean(("latitude", "longitude"))
                .to_dataframe().reset_index().round(2))
    regional.to_csv(proc / "indices_regional_annual.csv", index=False)

    kab_df = kabupaten_table(ds, kab, valid).round(2)
    kab_df.to_csv(proc / "indices_kabupaten_annual.csv", index=False)

    plot_regional(regional, figs / "indices_regional_timeseries.png")

    print("\n=== Ringkasan rata-rata provinsi 1981-2025 ===")
    print(regional.drop(columns="year").describe().loc[["mean", "min", "max"]].round(1).to_string())
    for name in ("rx1day", "prcptot"):
        top = regional.loc[regional[name].idxmax()]
        low = regional.loc[regional[name].idxmin()]
        print(f"{name.upper()}: tertinggi {int(top['year'])} ({top[name]:.0f}), "
              f"terendah {int(low['year'])} ({low[name]:.0f})")
    nearest = kab_df.loc[kab_df["method"] == "nearest", "kabupaten"].unique()
    if len(nearest):
        print(f"\nKabupaten/kota kecil yang memakai sel terdekat: {', '.join(nearest)}")
    print(f"\nTersimpan di {proc} dan {figs}")


if __name__ == "__main__":
    main()
