# Panduan Minggu 3 — SQL dengan DuckDB

Target: semua hasil Minggu 1–2 dimasukkan ke satu database, lalu dijawab dengan **9 query SQL** yang meniru pertanyaan nyata dari pemda, perusahaan asuransi, atau tim riset. Hasilnya tampil sebagai halaman `RESULTS.md` yang rapi di GitHub. Halaman ini yang akan dilihat recruiter.

## Langkah

Buka terminal lewat `mulai.bat`, lalu jalankan satu per satu:

```
python scripts/build_database.py
```
Membuat `data/processed/sumut_climate.duckdb` dengan 3 tabel dan 1 view.

```
python scripts/run_queries.py
```
Menjalankan semua query di `sql/queries.sql`. Hasilnya disimpan ke `outputs/tables/*.csv` dan `outputs/tables/RESULTS.md`.

Untuk menjalankan satu query saja, misalnya q05:
```
python scripts/run_queries.py q05
```

## Isi database

| Tabel / view | Isi |
|---|---|
| `indices_kabupaten` | 33 kabupaten × 45 tahun, 6 indeks ETCCDI |
| `indices_regional` | 45 tahun, rata-rata provinsi |
| `climate_monthly` | ONI, Niño 3.4, DMI bulanan |
| `climate_annual` (view) | ONI dan DMI rata-rata tahunan + label fase El Niño/La Niña dan IOD |

## Konsep SQL yang dilatih

| Query | Pertanyaan | Konsep SQL |
|---|---|---|
| q01 | Kabupaten dengan hujan harian ekstrem tertinggi | `GROUP BY`, `AVG`, `ORDER BY ... LIMIT` |
| q02 | Tren RX1DAY per kabupaten (mm/dekade) | CTE (`WITH`), regresi OLS ditulis manual dengan `SUM` |
| q03 | Perubahan per dekade | Kolom turunan (`year - year % 10`), agregasi |
| q04 | Tahun paling ekstrem vs ENSO/IOD | `JOIN ... USING` |
| q05 | Hujan saat El Niño / Netral / La Niña | `JOIN` + `GROUP BY` pada kategori |
| q06 | Rata-rata bergerak 5 tahun | Window function `AVG() OVER (ROWS BETWEEN ...)` |
| q07 | Anomali 2025 per kabupaten vs normal 1991–2020 | CTE + `JOIN` + `RANK() OVER` |
| q08 | Tahun rekor tiap kabupaten | `ROW_NUMBER() OVER (PARTITION BY ...)`, `GROUP_CONCAT` |
| q09 | Kabupaten yang perlu catatan cakupan data | `WHERE` dengan `OR`, `DISTINCT` |

Latihan yang disarankan: setelah semua query berjalan, **ubah sendiri** satu query. Misalnya, ganti `rx1day` di q02 dengan `cdd` untuk melihat apakah musim kering makin panjang. Latihan seperti ini lebih efektif daripada mengerjakan soal SQL generik.

## Catatan

- Fase ENSO dan IOD tahunan memakai rata-rata Januari–Desember. Ini penyederhanaan untuk eksplorasi. Analisis paper nanti memakai deret bulanan dan CiSSA.
- Query ditulis dalam SQL standar (CTE, window function, `CASE`), jadi bisa dipakai juga di PostgreSQL atau BigQuery dengan sedikit atau tanpa perubahan.

## Tutup minggu ini

```
git add -A
```
```
git commit -m "Week 3: DuckDB database and SQL analysis"
```
```
git push
```

Lalu buka repo di GitHub, klik `outputs/tables/RESULTS.md`, dan lihat hasil query yang tampil sebagai tabel. Kirim screenshot ringkasan terminal untuk kita bahas bersama.
