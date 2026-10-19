# Panduan Minggu 1 — Setup & Akuisisi Data

Target akhir minggu ini: repo publik yang rapi di GitHub, data CHIRPS 1981–2025 dan indeks ENSO/IOD sudah terunduh, dan satu peta cek yang masuk akal.

## Hari 1 — Instal perangkat
1. Instal **Miniforge** (conda ringan): https://github.com/conda-forge/miniforge
2. Instal **Git** dan **VS Code** (dengan ekstensi Python dan Jupyter).
3. Buat akun GitHub kalau belum ada. Pakai username yang profesional (misalnya nama lengkap).

## Hari 2 — Buat environment & repo
```bash
cd sumut-extreme-rainfall-cissa
conda env create -f environment.yml
conda activate sumut-rain

git init
git add .
git commit -m "Initial project structure and data download scripts"
```
Lalu buat repo kosong di GitHub dengan nama yang sama dan ikuti petunjuk `git remote add origin ...` dan `git push -u origin main`.

> Repo boleh langsung publik: yang ada baru struktur dan script, belum ada hasil inti paper.

## Hari 3 — Uji coba download (2 tahun saja)
```bash
python scripts/download_chirps.py --start 2020 --end 2021
python scripts/check_data.py
```
Buka `outputs/figures/check_mean_annual_rainfall.png`. Cek apakah:
- bentuk daratan Sumut + Nias terlihat
- nilai curah hujan tahunan masuk akal (kira-kira 1.500–4.500 mm/tahun)

## Hari 4 — Download penuh
```bash
python scripts/download_chirps.py
python scripts/download_climate_indices.py
python scripts/check_data.py
```
Kalau ada tahun yang gagal, jalankan ulang perintah yang sama. Tahun yang sudah selesai otomatis dilewati.
Kalau ERDDAP sedang bermasalah: `python scripts/download_chirps.py --source chc` (lebih lambat, ~1,2 GB per tahun sementara).

Pastikan `check_data.py` melaporkan:
- tidak ada tahun/hari hilang
- tidak ada nilai negatif
- korelasi ONI vs Niño3.4 > 0,9

## Hari 5 — Commit & catatan
```bash
git add .
git commit -m "Download full CHIRPS and climate indices; add data checks"
git push
```
Tulis catatan singkat (untuk dirimu sendiri dan bahan metodologi paper):
- bounding box yang dipakai dan alasannya
- versi dataset dan tanggal unduh
- masalah yang ditemui dan cara mengatasinya

## Latihan Git yang perlu dikuasai minggu ini
- `git status`, `git add`, `git commit`, `git push`, `git log --oneline`
- Pesan commit yang jelas: kata kerja + apa yang berubah

## Persiapan Minggu 2
Minggu depan kita mengolah data harian menjadi indeks ekstrem (Rx1day, Rx5day, R95p) dengan xarray, dan memotong data memakai batas administrasi Sumut (shapefile dari Badan Informasi Geospasial atau GADM).
