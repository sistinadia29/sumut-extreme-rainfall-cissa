"""
Unduh indeks iklim bulanan dan simpan dalam satu CSV rapi:
    data/external/climate_indices_monthly.csv
dengan kolom: date, oni, nino34, dmi, tpi

- ONI    : Oceanic Niño Index (NOAA CPC), rata-rata 3 bulanan anomali Niño 3.4.
           Musim "DJF 1982" ditaruh di bulan tengahnya (Januari 1982).
- Niño3.4: anomali SST bulanan (NOAA PSL).
- DMI    : Dipole Mode Index untuk IOD (NOAA PSL, berbasis HadISST).
- TPI    : Tripole Index untuk Interdecadal Pacific Oscillation (IPO), NOAA PSL, ERSSTv5
           (Henley et al. 2015). Dipakai untuk variabilitas dekadal.

Contoh:
    python scripts/download_climate_indices.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.text


def parse_oni(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), sep=r"\s+")
    df["month"] = df["SEAS"].map({s: i + 1 for i, s in enumerate(SEASONS)})
    df["date"] = pd.to_datetime(dict(year=df["YR"], month=df["month"], day=1))
    return df.set_index("date")["ANOM"].rename("oni").sort_index()


def parse_psl(text: str, name: str) -> pd.Series:
    """Format PSL: baris pertama 'tahun_awal tahun_akhir', lalu baris
    'tahun + 12 nilai bulanan', lalu satu baris nilai missing, lalu teks."""
    lines = [ln.split() for ln in text.strip().splitlines()]
    start, end = int(lines[0][0]), int(lines[0][1])

    records, missing = [], None
    for i, tok in enumerate(lines[1:], start=1):
        if len(tok) == 13 and tok[0].lstrip("-").isdigit() and start <= int(tok[0]) <= end:
            records.append([float(x) for x in tok])
        elif records and len(tok) == 1:
            missing = float(tok[0])  # baris nilai missing tepat setelah data
            break

    data = np.array(records)
    years = data[:, 0].astype(int)
    values = data[:, 1:]
    if missing is not None:
        values[np.isclose(values, missing)] = np.nan

    idx = pd.to_datetime([f"{y}-{m:02d}-01" for y in years for m in range(1, 13)])
    return pd.Series(values.ravel(), index=idx, name=name)


def main() -> None:
    cfg = load_config()
    urls = cfg["climate_indices"]

    print("Mengunduh ONI...")
    oni = parse_oni(fetch_text(urls["oni"]))
    print("Mengunduh Niño 3.4...")
    nino34 = parse_psl(fetch_text(urls["nino34"]), "nino34")
    print("Mengunduh DMI...")
    dmi = parse_psl(fetch_text(urls["dmi"]), "dmi")

    print("Mengunduh TPI (IPO)...")
    tpi = parse_psl(fetch_text(urls["tpi"]), "tpi")

    df = pd.concat([oni, nino34, dmi, tpi], axis=1)
    start = f"{cfg['period']['start_year']}-01-01"
    df = df.loc[start:].rename_axis("date")

    out = project_path(cfg["paths"]["external"]) / "climate_indices_monthly.csv"
    df.to_csv(out, float_format="%.3f")

    print(f"\nTersimpan: {out}")
    print(f"Periode  : {df.index.min():%Y-%m} s/d {df.index.max():%Y-%m}")
    print("Jumlah bulan terisi per indeks:")
    print(df.notna().sum().to_string())


if __name__ == "__main__":
    main()
