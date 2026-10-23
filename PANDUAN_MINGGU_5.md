# Minggu 5 — GEV non-stasioner: apakah ENSO menggeser hujan ekstrem?

## Kenapa GEV?
Korelasi CiSSA menjawab "apakah *pola naik-turun* hujan ikut ENSO". GEV menjawab pertanyaan yang
lebih praktis untuk paper: **"hujan ekstrem 20 tahunan saat La Niña berapa mm, saat El Niño berapa mm,
dan apakah bedanya nyata?"** Angka seperti ini yang dipakai perencana drainase dan BPBD.

## Model yang dibandingkan
| Model | Lokasi (μ) | Arti |
|---|---|---|
| M0 | konstan | iklim ekstrem tidak berubah |
| M1 | b0 + b1·ONI | ENSO menggeser ekstrem |
| M2 | b0 + b2·tahun | ada tren |
| M3 | b0 + b1·ONI + b2·tahun | keduanya |

- σ (sebaran) dan ξ (bentuk ekor) dibuat konstan, karena 45 tahun data tidak cukup untuk lebih.
- ξ diestimasi dengan **GMLE** (Martins & Stedinger 2000), standar untuk sampel pendek.
- Pemilihan model: AIC/BIC + uji rasio likelihood (LRT).
- Selang kepercayaan: bootstrap parametrik 500 kali.
- Diulang untuk **1991–2025** (keputusan dari validasi minggu 4).

## Langkah
`config.yaml` di paket ini sudah berisi bagian `gev:` baru, jadi cukup timpa file lama.

Jalankan satu per satu:
```
python tests/test_gev.py
```
```
python scripts/analyze_gev.py --no-grid
```
Bagian regional selesai ±2–4 menit. Kirim tangkapan layar tiga tabel + `outputs/figures/gev_regional.png`.

Lalu peta per sel grid (±10–20 menit, biarkan jalan, jangan klik di jendela cmd):
```
python scripts/analyze_gev.py
```
Kirim `outputs/figures/gev_grid_oni.png` + tabel ringkasan terakhir.

## Cara membaca hasil
- **p LRT ONI < 0.05** → ENSO nyata menggeser ekstrem tahunan.
- **Rasio La Niña / El Niño**: CI 95% tidak memuat 1 → beda nyata.
- **QQ plot** (bawah): titik dekat garis putus-putus → model cocok.
- **Peta**: cokelat = ekstrem lebih kecil saat El Niño (lebih besar saat La Niña). Garis hitam
  = wilayah yang lolos FDR. Secara acak ~5% luas akan "signifikan" di p < 0.05; yang penting
  persen luas yang lolos **FDR**.
- Semua angka = **hujan rata-rata grid/area**, bukan hujan di satu titik (CHIRPS meremehkan
  puncak titik, lihat validasi minggu 4).

## Catatan metode untuk paper
RX1DAY regional = rata-rata provinsi dari maksimum tahunan tiap sel. Ini bukan maksimum blok
murni, jadi analisis per sel grid (yang benar-benar maksimum tahunan) adalah hasil utama untuk
klaim spasial, dan regional dipakai sebagai ringkasan.

## Bagian 5b — Uji ketahanan + hasil per kabupaten/kota

Jalankan SETELAH `analyze_gev.py` versi lengkap (butuh `gev_grid.nc`). Waktu ±10–15 menit:
```
python scripts/analyze_gev_robustness.py
```

| Bagian | Pertanyaan reviewer yang dijawab | Hasil yang diharapkan kalau temuan kuat |
|---|---|---|
| A. Jackknife | "Jangan-jangan hanya karena El Niño 1997/2015?" | `p_max_jackknife` tetap < 0,05 dan `n_drops_p_lt_005` = 45/45 |
| B. Tanpa tahun ENSO kuat | "Hanya ditarik tahun-tahun ekstrem ONI?" | **Arah dan besar `b_oni` mirip** dengan semua tahun |
| C. Per kabupaten | "Wilayah mana yang paling terdampak?" | Daftar kabupaten dengan rasio La Niña/El Niño > 1 yang lolos FDR |

Catatan B: membuang semua tahun |ONI| ≥ 1 menghilangkan ±13 tahun dan justru tahun yang paling
informatif, jadi p-value wajar naik karena daya uji turun. Yang dinilai adalah **kestabilan
koefisien** `b_oni` dan rasio, bukan apakah p tetap < 0,05.

Kirim tangkapan layar tabel A, B, C + `gev_jackknife.png` dan `gev_kabupaten.png`.
