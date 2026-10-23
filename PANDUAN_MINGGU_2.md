# Panduan Minggu 2 — Batas Wilayah & Indeks Ekstrem

Target: dari 16.000+ hari data harian menjadi **indeks ekstrem tahunan** untuk seluruh Sumut dan setiap kabupaten/kota. Inilah dataset yang nanti dianalisis dengan SSA/CiSSA, Mann-Kendall, dan GEV.

## Langkah

Buka terminal lewat `mulai.bat`, lalu jalankan satu per satu:

```
python tests/test_indices.py
```
Harus muncul **"Semua uji lolos."** Uji ini memeriksa rumus indeks dengan data buatan yang jawabannya sudah diketahui.

```
python scripts/download_boundaries.py
```
Mengunduh batas provinsi dan kabupaten (geoBoundaries), lalu menyaring 33 kabupaten/kota Sumut. Script juga menampilkan kabupaten yang hanya tercakup sebagian oleh data CHIRPS (Nias Selatan dan Mandailing Natal).

```
python scripts/compute_indices.py
```
Menghitung indeks per tahun. Butuh sekitar 1–3 menit.

## Hasil

| File | Isi |
|---|---|
| `data/processed/extreme_indices_grid.nc` | Indeks per tahun per sel grid (hanya di dalam Sumut) |
| `data/processed/indices_regional_annual.csv` | Rata-rata provinsi per tahun → bahan SSA, Mann-Kendall, GEV |
| `data/processed/indices_kabupaten_annual.csv` | Rata-rata per kabupaten per tahun → bahan SQL (Minggu 3) dan dashboard |
| `outputs/figures/indices_regional_timeseries.png` | Grafik 6 indeks, 1981–2025 |

## Indeks yang dihitung (ETCCDI)

| Indeks | Arti | Satuan |
|---|---|---|
| PRCPTOT | Total hujan tahunan dari hari basah (≥ 1 mm) | mm |
| RX1DAY | Hujan harian maksimum | mm |
| RX5DAY | Akumulasi 5 hari maksimum | mm |
| R20MM | Jumlah hari hujan sangat lebat (≥ 20 mm) | hari |
| R95P | Total hujan dari hari sangat basah (> persentil-95, acuan 1991–2020) | mm |
| CDD | Deret hari kering (< 1 mm) terpanjang | hari |

## Catatan metodologi (untuk bagian Methods paper)

- Periode acuan persentil memakai **1991–2020**, normal iklim WMO terbaru. ETCCDI aslinya memakai 1961–1990, tapi data CHIRPS baru tersedia sejak 1981. Sebutkan ini di paper.
- Rata-rata provinsi diberi **bobot cos(lintang)**. Untuk Sumut efeknya sangat kecil, tapi ini praktik standar.
- Kota kecil (misalnya Sibolga) bisa tidak memuat satu pun titik pusat sel 0,05°. Untuk kota seperti ini, script memakai sel terdekat, ditandai `method = nearest`.
- **Keterbatasan cakupan:** sekitar 2,3% wilayah Sumut berada di selatan 0,5°LU dan belum tercakup. Ini mencakup Kepulauan Batu di Nias Selatan (±46% luas kabupaten itu) dan ujung selatan Mandailing Natal (±9%). Saat server ERDDAP NOAA kembali bisa diakses, ubah `lat_min` di `config.yaml` menjadi `-0.7`, lalu unduh ulang (murah, ±3 MB/tahun).

## Tutup minggu ini

```
git add -A
```
```
git commit -m "Week 2: boundaries and ETCCDI extreme indices"
```
```
git push
```

Kirim grafik `indices_regional_timeseries.png` dan ringkasan yang tercetak di terminal untuk dibahas bersama.
