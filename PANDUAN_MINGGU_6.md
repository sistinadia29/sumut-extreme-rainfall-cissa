# Minggu 6 — Gambar versi publikasi (International Journal of Climatology)

Semua gambar dibuat ulang dari hasil yang sudah tersimpan, tanpa menghitung ulang analisis.
Gaya seragam: tanpa judul di dalam gambar (judul ditulis di caption naskah), label panel (a), (b),
huruf 8 pt, lebar 1 kolom (84 mm) atau 2 kolom (174 mm), PNG 300 dpi + PDF.

Jalankan (±2 menit, karena pita kepercayaan Fig. 5 dihitung ulang dengan bootstrap):
```
python scripts/make_paper_figures.py
```
Hasil: `outputs/figures/paper/` (ditahan .gitignore sampai naskah dikirim).

Kalau mau memperbaiki satu gambar saja:
```
python scripts/make_paper_figures.py --only 6
```

| File | Isi | Di naskah |
|---|---|---|
| Fig01_study_area | Peta rata-rata hujan tahunan + kabupaten + stasiun; siklus musiman | Bagian 2 |
| Fig02_annual_indices | 6 indeks ETCCDI + garis Sen 1981–2025 dan 1991–2025 | 4.1 |
| Fig03_cissa | Dekomposisi CiSSA vs komponen antartahunan ONI dan DMI | 4.2 |
| Fig04_frequency | Varians dan korelasi per frekuensi | 4.2 |
| Fig05_gev_regional | Maksimum tahunan vs ONI + tingkat ulang 2 dan 20 tahun | 4.3 |
| Fig06_gev_map | Peta perubahan tingkat ulang 20 tahun per +1 ONI | 4.3 |
| Fig07_regencies | Rasio La Niña / El Niño per kabupaten/kota | 4.3 |
| Fig08_jackknife | Uji jackknife | 4.4 |
| FigS1_gev_qq, FigS2_window_selection | Diagnostik GEV, pemilihan L | Lampiran |
