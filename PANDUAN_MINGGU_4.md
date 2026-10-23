# Panduan Minggu 4 — Dekomposisi CiSSA & Uji Tren Mann-Kendall

Target: menjawab pertanyaan penelitian pertama. **Berapa besar variabilitas hujan Sumut yang berasal dari tren jangka panjang, dan berapa yang dari siklus ENSO/IOD?** Inilah inti paper.

## Langkah

Buka terminal lewat `mulai.bat`, lalu jalankan satu per satu:

```
python tests/test_cissa.py
```
Harus muncul "Semua uji CiSSA lolos." Uji ini mendekomposisi deret buatan yang komponennya sudah diketahui (tren + siklus 5 tahun + siklus tahunan + derau), lalu memeriksa apakah CiSSA berhasil memisahkannya kembali.

```
python scripts/analyze_cissa.py
```
Saat pertama kali dijalankan, script membuat deret bulanan dari CHIRPS harian (±1–2 menit). Hasilnya disimpan sebagai cache, jadi pemanggilan berikutnya jauh lebih cepat.

## Rancangan analisis

| Unsur | Pilihan | Alasan |
|---|---|---|
| Deret | Bulanan, rata-rata Sumut, 1981–2025 (540 titik) | Data tahunan (45 titik) terlalu pendek untuk memisahkan siklus 2–7 tahun |
| Variabel | `prcp_mm` (total bulanan) dan `rx1day_mm` (maksimum harian per bulan) | Mewakili hujan rata-rata dan hujan ekstrem |
| Window L | **Otomatis**: dipilih dari 96, 108, …, 264 bulan | CiSSA dijalankan berulang untuk setiap L. L terpilih adalah yang memberi keterpisahan terbaik, yaitu w-correlation maksimum antara kelompok tren, antartahunan, dan musiman yang paling kecil (Golyandina dkk. 2001). Semua kandidat kelipatan 12, sehingga harmonik musiman selalu jatuh tepat di frekuensi Fourier |
| Perpanjangan ujung | AR(p), p dipilih dengan AIC (≤ 24), sepanjang L di tiap ujung | Mengurangi efek tepi, seperti pada Bógalo dkk. (2021) |
| Kelompok | Frekuensi rendah (> 8 tahun), antartahunan (1,5–8 tahun), musiman (12, 6, 4, 3, 2,4, 2 bulan), residu | Pita antartahunan = skala waktu ENSO/IOD |
| Korelasi | Komponen antartahunan vs ONI dan DMI, lag 0–6 bulan | Nilai p memakai ukuran sampel efektif (Bretherton dkk. 1999), karena kedua deret berautokorelasi |
| Uji tren | Mann-Kendall termodifikasi Hamed–Rao + Sen's slope pada 6 indeks tahunan | Tahan terhadap autokorelasi. Diulang untuk 1991–2025 sebagai uji sensitivitas terhadap dekade awal CHIRPS |

Semua parameter ada di bagian `cissa:` di `config.yaml`.

## Pemilihan L otomatis

Di `config.yaml`, `window_months: auto` membuat script mencoba setiap L dalam `window_search` (default 96–264 bulan, langkah 12), lalu memilih L dengan keterpisahan terbaik. Untuk memakai L tetap, isi angka, misalnya `window_months: 168`.

Yang tercetak dan tersimpan:
- Tabel skor w-correlation untuk setiap L (`outputs/tables/cissa_L_selection.csv`) dan grafiknya (`outputs/figures/cissa_L_selection.png`)
- **L lain dalam 10% dari skor terbaik.** Kalau daftarnya panjang, berarti kurvanya datar: banyak L yang sama baiknya, dan pilihan L tidak terlalu menentukan hasil. Ini justru bukti ketangguhan (robustness) yang bagus untuk paper.
- Kolom `r_oni_info_only` menunjukkan korelasi dengan ONI untuk setiap L. Kolom ini **sengaja tidak dipakai** untuk memilih L. Kalau L dipilih berdasarkan korelasi dengan ONI, kita memilih L demi hasil yang kita harapkan, dan reviewer akan menganggapnya bias.

**Pencarian dua tahap** (`window_refine_radius: 5`):
1. Tahap kasar: L = 96, 108, …, 264 (kelipatan 12) → misalnya terbaik di 120
2. Tahap halus: L = 115, 116, …, 125 (langkah 1) → L final adalah yang terbaik dari kedua tahap

Pada L yang bukan kelipatan 12, frekuensi musiman tidak jatuh tepat di frekuensi Fourier, sehingga harmonik musiman diambil dari frekuensi terdekat. Pada uji dengan data buatan, keterpisahan memburuk dengan cepat begitu L bergeser dari kelipatan 12 (w-correlation naik dari ±0,001 ke ±0,04), sedangkan korelasi dengan ONI hampir tidak berubah (−0,840 s.d. −0,841). Kalau pola yang sama muncul pada data asli, itu dua temuan sekaligus untuk paper: (1) L kelipatan periode musiman memang optimal untuk CiSSA pada data bulanan, dan (2) hubungan dengan ENSO tangguh terhadap pilihan L.

**Batas bawah 96 bulan** dipakai supaya periode terpanjang pita antartahunan (8 tahun) masih bisa direpresentasikan. **Batas atas** dibatasi T/2 = 270 bulan, aturan umum SSA.

## Hasil

| File | Isi |
|---|---|
| `outputs/tables/cissa_variance.csv` | Porsi varians tiap kelompok (%) → **jawaban RQ1** |
| `outputs/tables/cissa_trend_slope.csv` | Kemiringan komponen frekuensi rendah (mm/dekade) |
| `outputs/tables/cissa_climate_correlation.csv` | Korelasi dengan ONI/DMI per lag |
| `outputs/tables/mann_kendall.csv` | Tren 6 indeks: arah, p-value, Sen's slope per dekade |
| `outputs/figures/cissa_prcp.png`, `cissa_rx1day.png` | Gambar dekomposisi |
| `data/processed/cissa_components_*.csv` | Semua komponen bulanan, untuk GEV di Minggu 5 |

## Yang perlu kamu cek sebagai ahli CiSSA

1. **Bandingkan dengan implementasimu sendiri.** Kalau kamu punya kode CiSSA MATLAB atau R (misalnya dari Bógalo dkk.), jalankan pada `data/processed/monthly_regional.csv` dengan L = 168. Hasilnya seharusnya hampir sama. Kalau ada perbedaan, cek dulu metode perpanjangan ujungnya.
2. **Sensitivitas terhadap L.** Tabel `cissa_L_selection.csv` sudah menjadi analisis sensitivitas, karena menunjukkan skor keterpisahan dan korelasi ONI untuk setiap L. Kalau korelasi ONI stabil di banyak L, hasilmu tangguh.
3. **Penamaan "trend".** Kelompok frekuensi rendah memuat frekuensi nol (tren) dan periode 14 tahun (variasi dekadal). Di paper, sebut kelompok ini "low-frequency component (> 8 yr)", dan laporkan tren linear sebagai kemiringan OLS komponen tersebut.

## Bagian 4b — Hubungan dengan ENSO, IOD, dan IPO

Dijalankan **setelah** `analyze_cissa.py`, karena memakai L yang terpilih di sana. Jalankan satu per satu:

```
python scripts/download_climate_indices.py
```
Diunduh ulang supaya indeks **TPI (IPO)** ikut masuk.

```
python tests/test_stats.py
```
Harus muncul "Semua uji statistik lolos."

```
python scripts/analyze_climate_links.py
```
Butuh sekitar 30 detik (1.000 surrogate untuk setiap uji).

| Analisis | Apa yang dibandingkan | Mengapa |
|---|---|---|
| **A. Pita-setara** | Komponen antartahunan hujan vs komponen antartahunan ONI/DMI. ONI dan DMI didekomposisi dengan CiSSA, L dan pita yang sama | Yang dibandingkan harus setara: hasil penyaringan dengan hasil penyaringan |
| **B. Per frekuensi** | Setiap komponen elementer k di pita 1,5–8 tahun: hujan_k vs ONI_k dan DMI_k | Menunjukkan **periode mana** yang terkait ENSO dan mana yang terkait IOD. Ini keunggulan khas CiSSA, karena setiap komponen terikat ke satu frekuensi yang diketahui |
| **C. Dekadal** | Komponen frekuensi rendah (> 8 tahun) hujan vs TPI (IPO) | Menguji dugaan bahwa gelombang kering 1990-an dan basah 2008–2010 terkait IPO |

**Signifikansi:** semua p-value memakai **surrogate acak fase** (Ebisuzaki 1997). Fase Fourier deret hujan diacak 1.000 kali sambil spektrumnya dipertahankan. Statistik ujinya adalah |r| maksimum di lag 0–6 bulan, sehingga p-value sudah memperhitungkan pemilihan lag terbaik. Kolom `r95_null` adalah |r| yang bisa muncul **hanya karena kebetulan** pada deret sehalus ini. Pada uji dengan data buatan, angkanya mencapai 0,7–0,8. Itu sebabnya p-value biasa menyesatkan untuk komponen CiSSA. Untuk analisis B ditambahkan **q-value Benjamini–Hochberg**, karena banyak frekuensi diuji sekaligus.

**Hasil uji dengan data buatan** (ENSO buatan berperiode 50 bulan, hujan dibuat bergantung pada ENSO dengan jeda 3 bulan, dan sama sekali tidak bergantung pada DMI):
- Analisis B dengan tepat menunjuk periode **4,33 tahun** (r = −0,99, q = 0,016, lag 4 bulan) sebagai satu-satunya frekuensi yang terkait ENSO
- Hubungan dengan DMI tidak signifikan, sesuai dengan cara data dibuat

**Catatan jujur untuk analisis C:** dalam 45 tahun hanya ada 2–3 siklus dekadal. Derajat kebebasannya sangat kecil, sehingga uji signifikansi apa pun di skala ini rapuh. Pada data buatan bahkan muncul satu positif palsu (p = 0,013 untuk deret yang sebenarnya tidak berhubungan). Di paper, sajikan analisis C sebagai **deskriptif/indikatif**, jangan sebagai bukti statistik.

**Keluaran** (ditahan oleh `.gitignore`):
- `outputs/tables/cissa_band_correlation.csv`, `cissa_frequency_correlation.csv`, `cissa_decadal_ipo.csv`
- `outputs/figures/cissa_frequency_prcp.png`, `cissa_frequency_rx1day.png`, `cissa_decadal_ipo.png`

## Tutup minggu ini

```
git add -A
```
```
git commit -m "Week 4: CiSSA decomposition and Mann-Kendall trend tests"
```
```
git push
```

**Catatan untuk paper:** hasil Minggu 4 adalah inti kebaruan paper. `.gitignore` sudah diatur supaya tabel dan gambar CiSSA serta Mann-Kendall **tidak ikut terunggah** ke GitHub. Yang terunggah hanya kode dan ujinya. Setelah naskah dikirim ke jurnal atau terbit sebagai preprint, hapus tiga baris terakhir di `.gitignore`, lalu commit hasilnya. Sampai saat itu, jangan cantumkan angka-angka hasil Minggu 4 di post LinkedIn.

Kirim screenshot ringkasan terminal dan kedua gambar untuk kita bahas bersama.
