"""
Cari dan unduh data hujan harian stasiun GHCN-Daily (NOAA NCEI) di Sumatera Utara,
sebagai pengganti data BMKG untuk validasi CHIRPS.

Langkah:
 1. Unduh daftar stasiun (ghcnd-stations.txt, ~11 MB) dan inventaris (ghcnd-inventory.txt, ~35 MB)
 2. Pilih stasiun yang punya elemen PRCP dan berada di dalam poligon Sumut + buffer
 3. Unduh data harian tiap stasiun (by_station/<ID>.csv.gz), buang nilai yang gagal QC
 4. Laporkan kelengkapan data per dekade dan jumlah tahun yang layak dipakai

Keluaran:
    data/external/ghcn/<ID>_prcp.csv          date, prcp_mm
    outputs/tables/ghcn_stations.csv          ringkasan stasiun dan kelengkapan

Contoh:
    python scripts/find_ghcn_stations.py
"""

import io
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402

BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
STATION_COLS = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
STATION_NAMES = ["id", "lat", "lon", "elev_m", "state", "name"]
INV_COLS = [(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)]
INV_NAMES = ["id", "lat", "lon", "element", "first_year", "last_year"]


def get(url: str, dest: Path | None = None, binary: bool = False):
    if dest is not None and dest.exists():
        return dest.read_bytes() if binary else dest.read_text(encoding="latin-1")
    resp = requests.get(url, timeout=(30, 600))
    resp.raise_for_status()
    if dest is not None:
        dest.write_bytes(resp.content)
    return resp.content if binary else resp.content.decode("latin-1")


def parse_fwf(text: str, colspecs, names) -> pd.DataFrame:
    return pd.read_fwf(io.StringIO(text), colspecs=colspecs, names=names, header=None)


def parse_station_csv(raw: bytes) -> pd.DataFrame:
    """Data harian by_station: ID,DATE,ELEMENT,VALUE,MFLAG,QFLAG,SFLAG,OBSTIME. PRCP dalam 0,1 mm."""
    df = pd.read_csv(io.BytesIO(raw), compression="gzip", header=None, dtype=str,
                     names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"],
                     keep_default_na=False)
    df = df[(df["element"] == "PRCP") & (df["qflag"] == "")]
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], format="%Y%m%d"),
                        "prcp_mm": df["value"].astype(float) / 10.0})
    out = out[out["prcp_mm"] >= 0]
    return out.sort_values("date").reset_index(drop=True)


def coverage(daily: pd.DataFrame, y0: int, y1: int) -> dict:
    d = daily[(daily["date"].dt.year >= y0) & (daily["date"].dt.year <= y1)]
    per_year = d.groupby(d["date"].dt.year).size()
    days = pd.Series({y: 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365
                      for y in range(y0, y1 + 1)})
    frac = (per_year.reindex(days.index, fill_value=0) / days)
    out = {"years_ge80pct": int((frac >= 0.8).sum())}
    for dec in range(y0 - y0 % 10, y1 + 1, 10):
        sel = frac[(frac.index >= max(dec, y0)) & (frac.index <= min(dec + 9, y1))]
        out[f"cov_{dec}s_pct"] = round(100 * sel.mean(), 0) if len(sel) else np.nan
    return out


def main() -> None:
    cfg = load_config()
    y0, y1 = cfg["period"]["start_year"], cfg["period"]["end_year"]
    ext = project_path(cfg["paths"]["external"])
    gdir = project_path(f"{cfg['paths']['external']}/ghcn")
    tables = project_path("outputs/tables")
    buffer_deg = 0.25

    print("Mengunduh daftar stasiun dan inventaris GHCN-Daily (sekali saja, lalu disimpan) ...")
    stations = parse_fwf(get(f"{BASE}/ghcnd-stations.txt", gdir / "ghcnd-stations.txt"),
                         STATION_COLS, STATION_NAMES)
    inv = parse_fwf(get(f"{BASE}/ghcnd-inventory.txt", gdir / "ghcnd-inventory.txt"),
                    INV_COLS, INV_NAMES)
    prcp = inv[inv["element"] == "PRCP"][["id", "first_year", "last_year"]]

    prov = gpd.read_file(ext / "sumut_province.geojson").geometry.iloc[0].buffer(buffer_deg)
    cand = stations.merge(prcp, on="id")
    cand = cand[cand.apply(lambda r: prov.contains(Point(r["lon"], r["lat"])), axis=1)]
    cand = cand[cand["last_year"] >= y0]
    if cand.empty:
        sys.exit("Tidak ada stasiun GHCN-Daily dengan data hujan di Sumut. "
                 "Lanjutkan dengan validasi lintas dataset (APHRODITE / ERA5).")
    print(f"Stasiun kandidat: {len(cand)}\n")

    rows = []
    for _, st in cand.iterrows():
        print(f"  {st['id']}  {st['name']:<30} ({st['first_year']}-{st['last_year']}) ...", end=" ")
        try:
            daily = parse_station_csv(get(f"{BASE}/by_station/{st['id']}.csv.gz", binary=True))
        except Exception as e:
            print(f"gagal: {e}")
            continue
        daily.to_csv(gdir / f"{st['id']}_prcp.csv", index=False)
        cov = coverage(daily, y0, y1)
        print(f"{len(daily)} hari, {cov['years_ge80pct']} tahun layak (>= 80% hari)")
        rows.append({"id": st["id"], "name": st["name"], "lat": st["lat"], "lon": st["lon"],
                     "elev_m": st["elev_m"], "first_year": st["first_year"],
                     "last_year": st["last_year"], "n_days": len(daily), **cov})

    summary = pd.DataFrame(rows).sort_values("years_ge80pct", ascending=False)
    summary.to_csv(tables / "ghcn_stations.csv", index=False)
    pd.set_option("display.width", 170)
    print("\n=== Kelengkapan data hujan harian (% hari tersedia per dekade) ===")
    print(summary.drop(columns=["lat", "lon", "elev_m"]).to_string(index=False))
    usable = summary[summary["years_ge80pct"] >= 10]
    print(f"\nStasiun dengan >= 10 tahun layak: {len(usable)}  ->  "
          f"{', '.join(usable['name'].str.strip()) if len(usable) else 'tidak ada'}")
    print(f"Tersimpan: {gdir} dan {tables / 'ghcn_stations.csv'}")


if __name__ == "__main__":
    main()
