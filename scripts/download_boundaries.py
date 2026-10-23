"""
Unduh batas administrasi dari geoBoundaries lalu simpan khusus Sumatera Utara:
    data/external/sumut_province.geojson    (1 poligon provinsi)
    data/external/sumut_kabupaten.geojson   (33 kabupaten/kota)

Kabupaten dipilih secara spasial: titik representatif tiap kabupaten harus
berada di dalam poligon provinsi (geoBoundaries ADM2 tidak menyimpan nama provinsi).
Script juga melaporkan berapa persen luas tiap kabupaten yang berada di dalam
kotak unduhan CHIRPS, supaya cakupan data jelas sejak awal.

Contoh:
    python scripts/download_boundaries.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  {dest.name}: sudah ada, dilewati")
        return dest
    print(f"  mengunduh {dest.name} ...")
    resp = requests.get(url, timeout=(30, 300))
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def main() -> None:
    cfg = load_config()
    b = cfg["boundaries"]
    ext = project_path(cfg["paths"]["external"])

    adm1 = gpd.read_file(fetch(b["adm1_url"], ext / "geoBoundaries-IDN-ADM1_simplified.geojson"))
    adm2 = gpd.read_file(fetch(b["adm2_url"], ext / "geoBoundaries-IDN-ADM2_simplified.geojson"))

    prov = adm1[adm1["shapeName"] == b["province_name"]].to_crs(4326)
    if prov.empty:
        sys.exit(f"Provinsi '{b['province_name']}' tidak ditemukan. "
                 f"Nama yang tersedia: {sorted(adm1['shapeName'])}")
    prov_geom = prov.geometry.iloc[0]

    adm2 = adm2.to_crs(4326)
    inside = adm2.geometry.representative_point().within(prov_geom)
    kab = adm2[inside & ~adm2["shapeName"].isin(b["exclude_units"])].copy()
    kab = kab.rename(columns={"shapeName": "kabupaten"})[["kabupaten", "geometry"]]
    kab = kab.sort_values("kabupaten").reset_index(drop=True)

    # Cakupan kotak unduhan CHIRPS per kabupaten, dihitung dalam proyeksi UTM 47N (meter)
    r = cfg["region"]
    bbox = box(r["lon_min"], r["lat_min"], r["lon_max"], r["lat_max"])
    utm = 32647
    kab_m = kab.to_crs(utm)
    bbox_m = gpd.GeoSeries([bbox], crs=4326).to_crs(utm).iloc[0]
    kab["chirps_coverage"] = (kab_m.geometry.intersection(bbox_m).area
                              / kab_m.geometry.area).round(3).values

    prov[["shapeName", "geometry"]].to_file(ext / "sumut_province.geojson", driver="GeoJSON")
    kab.to_file(ext / "sumut_kabupaten.geojson", driver="GeoJSON")

    minx, miny, maxx, maxy = prov_geom.bounds
    prov_m = prov.to_crs(utm).geometry.iloc[0]
    prov_cov = prov_m.intersection(bbox_m).area / prov_m.area
    print(f"\nProvinsi : {b['province_name']}  (BT {minx:.2f}-{maxx:.2f}, LU {miny:.2f}-{maxy:.2f})")
    print(f"Cakupan kotak CHIRPS untuk provinsi: {prov_cov:.1%}")
    print(f"Kabupaten/kota: {len(kab)} (yang diharapkan 33)")
    partial = kab[kab["chirps_coverage"] < 0.99]
    if not partial.empty:
        print("Kabupaten dengan cakupan parsial:")
        print(partial[["kabupaten", "chirps_coverage"]].to_string(index=False))
    print(f"\nTersimpan: {ext / 'sumut_province.geojson'}")
    print(f"           {ext / 'sumut_kabupaten.geojson'}")


if __name__ == "__main__":
    main()
