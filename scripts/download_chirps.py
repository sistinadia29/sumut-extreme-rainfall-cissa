"""
Unduh CHIRPS v2.0 harian (0,05°) untuk wilayah di config.yaml, satu file per tahun.

Sumber utama: NOAA ERDDAP, yang bisa memotong wilayah di server sehingga
file yang diunduh kecil (beberapa MB per tahun untuk Sumut).
Sumber alternatif (--source iri): IRI Data Library, Columbia University,
juga memotong wilayah di server. Pakai ini kalau ERDDAP sedang tidak bisa diakses.
Sumber cadangan terakhir (--source chc): file global tahunan dari CHC UCSB (~1,2 GB),
lalu dipotong secara lokal dan file globalnya dihapus.

Contoh:
    python scripts/download_chirps.py                 # semua tahun di config
    python scripts/download_chirps.py --start 2020 --end 2021   # uji coba dulu
    python scripts/download_chirps.py --source chc    # jika ERDDAP sedang down
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402

TIMEOUT = (30, 600)  # 30 dtk untuk tersambung, 600 dtk untuk mengunduh
MAX_RETRIES = 8


def erddap_query(cfg: dict, year: int) -> str:
    """Bangun URL ERDDAP griddap untuk satu tahun dan satu bounding box.
    Latitude di dataset ini urut naik, jadi urutan (min):(max) sudah benar."""
    r = cfg["region"]
    var = cfg["chirps"]["variable"]
    t = f"[({year}-01-01T00:00:00Z):1:({year}-12-31T00:00:00Z)]"
    lat = f"[({r['lat_min']}):1:({r['lat_max']})]"
    lon = f"[({r['lon_min']}):1:({r['lon_max']})]"
    return f"{cfg['chirps']['erddap_url']}?{var}{t}{lat}{lon}"


def stream_download(url: str, dest: Path) -> None:
    """Unduh dengan retry dan backoff; tulis ke file .part dulu supaya
    file setengah jadi tidak dianggap selesai. Kalau koneksi putus,
    unduhan dilanjutkan dari byte terakhir (HTTP Range), bukan dari nol."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as resp:
                if resp.status_code == 416:      # file .part sudah lengkap
                    break
                resp.raise_for_status()
                if done and resp.status_code != 206:
                    done = 0                     # server tidak mendukung lanjut: mulai ulang
                total = done + int(resp.headers.get("content-length", 0))
                with open(tmp, "ab" if done else "wb") as f, tqdm(
                    total=total, initial=done, unit="B", unit_scale=True,
                    desc=dest.name, leave=False,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except requests.RequestException as e:
            wait = min(10 * attempt, 60)
            got = tmp.stat().st_size / 1e6 if tmp.exists() else 0
            print(f"  terputus (percobaan {attempt}/{MAX_RETRIES}, sudah {got:.0f} MB): "
                  f"{type(e).__name__}. Lanjut lagi {wait} dtk")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Tidak berhasil mengunduh {url}")
    tmp.replace(dest)


def validate(path: Path, year: int, var: str) -> None:
    """Pastikan file bisa dibuka, variabelnya ada, dan jumlah harinya wajar."""
    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise ValueError(f"Variabel '{var}' tidak ada di {path.name}")
        n_days = ds.sizes["time"]
        expected = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        if n_days != expected:
            print(f"  peringatan: {year} punya {n_days} hari (harusnya {expected})")


def download_erddap(cfg: dict, year: int, dest: Path) -> None:
    stream_download(erddap_query(cfg, year), dest)


def download_chc(cfg: dict, year: int, dest: Path) -> None:
    r = cfg["region"]
    global_file = dest.parent / f"_global_{year}.nc"
    stream_download(cfg["chirps"]["chc_url"].format(year=year), global_file)
    with xr.open_dataset(global_file) as ds:
        # CHC memakai nama dimensi latitude/longitude
        lat = ds["latitude"]
        lat_slice = (slice(r["lat_min"], r["lat_max"]) if lat[0] < lat[-1]
                     else slice(r["lat_max"], r["lat_min"]))
        sub = ds.sel(latitude=lat_slice,
                     longitude=slice(r["lon_min"], r["lon_max"]))
        sub = sub.rename({"precip": cfg["chirps"]["variable"]}) if "precip" in sub else sub
        sub.load().to_netcdf(dest)
    global_file.unlink()


def download_iri(cfg: dict, year: int, dest: Path) -> None:
    """IRI Data Library (Columbia University): potong wilayah & tahun di server,
    lalu samakan nama dimensi/variabel dengan format ERDDAP."""
    r = cfg["region"]
    url = (f"{cfg['chirps']['iri_url']}"
           f"/X/{r['lon_min']}/{r['lon_max']}/RANGEEDGES"
           f"/Y/{r['lat_min']}/{r['lat_max']}/RANGEEDGES"
           f"/T/(1%20Jan%20{year})/(31%20Dec%20{year})/RANGEEDGES/data.nc")
    tmp = dest.parent / f"_iri_{year}.nc"
    stream_download(url, tmp)
    with xr.open_dataset(tmp, decode_times=False) as ds:
        ds = ds.load()
    tmp.unlink()

    var_in = "prcp" if "prcp" in ds else list(ds.data_vars)[0]
    ds = ds.rename({"X": "longitude", "Y": "latitude", "T": "time",
                    var_in: cfg["chirps"]["variable"]})
    # Waktu IRI berupa angka hari; ganti dengan tanggal kalender harian
    ds = ds.assign_coords(time=pd.date_range(f"{year}-01-01",
                                             periods=ds.sizes["time"], freq="D"))
    ds = ds.sortby("latitude")
    for c in ("longitude", "latitude", "time"):
        ds[c].attrs = {}
    ds.to_netcdf(dest)


DOWNLOADERS = {"erddap": download_erddap, "iri": download_iri, "chc": download_chc}


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=cfg["period"]["start_year"])
    p.add_argument("--end", type=int, default=cfg["period"]["end_year"])
    p.add_argument("--source", choices=list(DOWNLOADERS), default="erddap")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = project_path(cfg["paths"]["raw_chirps"])
    var = cfg["chirps"]["variable"]
    downloader = DOWNLOADERS[args.source]

    print(f"Wilayah: {cfg['region']['name']} | {args.start}-{args.end} | sumber: {args.source}")
    failed = []
    for year in range(args.start, args.end + 1):
        dest = out_dir / f"chirps_sumut_{year}.nc"
        if dest.exists() and not args.overwrite:
            print(f"{year}: sudah ada, dilewati")
            continue
        print(f"{year}: mengunduh...")
        try:
            downloader(cfg, year, dest)
            validate(dest, year, var)
            print(f"{year}: selesai ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # lanjut ke tahun berikutnya, laporkan di akhir
            print(f"{year}: GAGAL - {e}")
            failed.append(year)

    if failed:
        print(f"\nTahun yang gagal: {failed}. Jalankan ulang script; "
              "tahun yang sudah selesai otomatis dilewati.")
    else:
        print("\nSemua tahun berhasil diunduh.")


if __name__ == "__main__":
    main()
