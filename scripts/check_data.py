"""
Cek kualitas data setelah diunduh (Minggu 1). Laporan yang dicetak:
- tahun yang tersedia / hilang, jumlah grid, resolusi
- hari yang hilang, nilai negatif, dan proporsi NaN (laut)
- curah hujan tahunan rata-rata wilayah
- ringkasan indeks iklim
Juga menyimpan peta cepat curah hujan tahunan rata-rata ke outputs/figures/.

Contoh:
    python scripts/check_data.py
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


def check_chirps(cfg: dict) -> None:
    raw = project_path(cfg["paths"]["raw_chirps"])
    var = cfg["chirps"]["variable"]
    files = sorted(raw.glob("chirps_sumut_*.nc"))
    if not files:
        print("Belum ada file CHIRPS. Jalankan download_chirps.py dulu.")
        return

    years_have = {int(f.stem.split("_")[-1]) for f in files}
    years_want = set(range(cfg["period"]["start_year"], cfg["period"]["end_year"] + 1))
    missing_years = sorted(years_want - years_have)

    ds = xr.open_mfdataset(files, combine="by_coords")
    pr = ds[var]

    print("=== CHIRPS ===")
    print(f"File            : {len(files)} tahun "
          f"({min(years_have)}-{max(years_have)})")
    print(f"Tahun hilang    : {missing_years or 'tidak ada'}")
    print(f"Grid            : {pr.sizes['latitude']} lat x {pr.sizes['longitude']} lon")
    print(f"Resolusi        : {float(np.diff(pr.latitude)[0]):.3f} derajat")

    full = pd.date_range(pr.time.values.min(), pr.time.values.max(), freq="D")
    missing_days = full.difference(pd.DatetimeIndex(pr.time.values))
    print(f"Hari hilang     : {len(missing_days)}")

    n_neg = int((pr < 0).sum().compute())
    nan_frac = float(pr.isel(time=0).isnull().mean())
    print(f"Nilai negatif   : {n_neg}")
    print(f"Grid NaN (laut) : {nan_frac:.1%}")

    annual = pr.resample(time="YS").sum(min_count=1)
    regional = annual.mean(["latitude", "longitude"]).compute()
    s = regional.to_series()
    s.index = s.index.year
    print("\nCurah hujan tahunan rata-rata wilayah (mm):")
    print(f"  rata-rata {s.mean():.0f} | min {s.min():.0f} ({s.idxmin()}) "
          f"| max {s.max():.0f} ({s.idxmax()})")

    # Peta cepat untuk sanity check visual
    clim = annual.mean("time").compute()
    fig, ax = plt.subplots(figsize=(6, 6))
    clim.plot(ax=ax, cmap="YlGnBu", cbar_kwargs={"label": "mm/tahun"})
    ax.set_title(f"Curah hujan tahunan rata-rata CHIRPS\n"
                 f"{cfg['region']['name']} {min(years_have)}-{max(years_have)}")
    ax.set_aspect("equal")
    out = project_path(cfg["paths"]["figures"]) / "check_mean_annual_rainfall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPeta cek tersimpan: {out}")


def check_indices(cfg: dict) -> None:
    path = project_path(cfg["paths"]["external"]) / "climate_indices_monthly.csv"
    print("\n=== Indeks iklim ===")
    if not path.exists():
        print("Belum ada. Jalankan download_climate_indices.py dulu.")
        return
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    print(df.describe().round(2).to_string())

    # Korelasi ONI vs Niño3.4 harus tinggi (>0,9) kalau parsing benar
    r = df[["oni", "nino34"]].dropna().corr().iloc[0, 1]
    flag = "OK" if r > 0.9 else "PERIKSA PARSING"
    print(f"\nKorelasi ONI vs Niño3.4: {r:.2f} ({flag})")


if __name__ == "__main__":
    cfg = load_config()
    check_chirps(cfg)
    check_indices(cfg)
