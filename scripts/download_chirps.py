"""
Unduh CHIRPS v2.0 harian (0,05°) untuk wilayah di config.yaml, satu file per tahun.

Sumber utama: NOAA ERDDAP, yang bisa memotong wilayah di server sehingga
file yang diunduh kecil (beberapa MB per tahun untuk Sumut).
Sumber cadangan (--source chc): file global tahunan dari CHC UCSB (~1,2 GB),
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

import requests
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402

TIMEOUT = (30, 600)  
MAX_RETRIES = 4


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
    file setengah jadi tidak dianggap selesai."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(tmp, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True,
                    desc=dest.name, leave=False,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            return
        except requests.RequestException as e:
            wait = 10 * attempt
            print(f"  gagal (percobaan {attempt}/{MAX_RETRIES}): {e}. Coba lagi {wait} dtk")
            time.sleep(wait)
    raise RuntimeError(f"Tidak berhasil mengunduh {url}")


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


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=cfg["period"]["start_year"])
    p.add_argument("--end", type=int, default=cfg["period"]["end_year"])
    p.add_argument("--source", choices=["erddap", "chc"], default="erddap")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = project_path(cfg["paths"]["raw_chirps"])
    var = cfg["chirps"]["variable"]
    downloader = download_erddap if args.source == "erddap" else download_chc

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
