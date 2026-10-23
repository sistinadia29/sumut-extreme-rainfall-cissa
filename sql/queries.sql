-- =====================================================================
-- Analisis hujan ekstrem Sumatera Utara dengan SQL (DuckDB)
-- Tabel  : indices_kabupaten  (tahun x kabupaten, 6 indeks ETCCDI)
--          indices_regional   (tahun, rata-rata provinsi)
--          climate_annual     (tahun, ONI & DMI rata-rata tahunan + fase)
-- Setiap blok diawali "-- name:" dan "-- question:"; run_queries.py
-- menjalankannya satu per satu dan menyimpan hasilnya ke outputs/tables/.
-- =====================================================================


-- name: q01_wettest_kabupaten
-- question: Which regencies have the highest average 1-day maximum rainfall (RX1DAY), 1981-2025?
SELECT
    kabupaten,
    ROUND(AVG(rx1day), 1)  AS mean_rx1day_mm,
    ROUND(MAX(rx1day), 1)  AS record_rx1day_mm,
    ROUND(AVG(prcptot), 0) AS mean_annual_mm
FROM indices_kabupaten
GROUP BY kabupaten
ORDER BY mean_rx1day_mm DESC
LIMIT 10;


-- name: q02_rx1day_trend_by_kabupaten
-- question: Where is RX1DAY rising or falling fastest? (OLS slope, mm per decade)
-- Kemiringan OLS ditulis dari rumus jumlah: (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2).
-- Tahun dipusatkan (year - 2000) supaya angka tetap kecil dan presisi terjaga.
WITH d AS (
    SELECT kabupaten, year - 2000 AS x, rx1day AS y
    FROM indices_kabupaten
),
s AS (
    SELECT
        kabupaten,
        COUNT(*)      AS n,
        SUM(x)        AS sx,
        SUM(y)        AS sy,
        SUM(x * y)    AS sxy,
        SUM(x * x)    AS sxx
    FROM d
    GROUP BY kabupaten
)
SELECT
    kabupaten,
    ROUND(10.0 * (n * sxy - sx * sy) / (n * sxx - sx * sx), 2) AS rx1day_trend_mm_per_decade
FROM s
ORDER BY rx1day_trend_mm_per_decade DESC;


-- name: q03_decadal_change
-- question: How have province-wide extremes changed decade by decade?
SELECT
    year - (year % 10)        AS decade,
    COUNT(*)                  AS n_years,
    ROUND(AVG(prcptot), 0)    AS prcptot_mm,
    ROUND(AVG(rx1day), 1)     AS rx1day_mm,
    ROUND(AVG(rx5day), 1)     AS rx5day_mm,
    ROUND(AVG(r20mm), 1)      AS r20mm_days,
    ROUND(AVG(cdd), 1)        AS cdd_days
FROM indices_regional
GROUP BY decade
ORDER BY decade;


-- name: q04_extreme_years_vs_climate_modes
-- question: In the five years with the highest province-wide RX5DAY, what were ENSO and IOD doing?
SELECT
    r.year,
    ROUND(r.rx5day, 1)  AS rx5day_mm,
    ROUND(c.oni, 2)     AS oni,
    c.enso_phase,
    ROUND(c.dmi, 2)     AS dmi,
    c.iod_phase
FROM indices_regional AS r
JOIN climate_annual   AS c USING (year)
ORDER BY r.rx5day DESC
LIMIT 5;


-- name: q05_rainfall_by_enso_phase
-- question: How do annual rainfall and extremes differ between El Nino, neutral and La Nina years?
SELECT
    c.enso_phase,
    COUNT(*)                   AS n_years,
    ROUND(AVG(r.prcptot), 0)   AS prcptot_mm,
    ROUND(AVG(r.r95p), 0)      AS r95p_mm,
    ROUND(AVG(r.r20mm), 1)     AS r20mm_days,
    ROUND(AVG(r.cdd), 1)       AS cdd_days
FROM indices_regional AS r
JOIN climate_annual   AS c USING (year)
GROUP BY c.enso_phase
ORDER BY prcptot_mm;


-- name: q06_rx5day_moving_average
-- question: What does the 5-year moving average of province-wide RX5DAY look like?
SELECT
    year,
    ROUND(rx5day, 1) AS rx5day_mm,
    ROUND(AVG(rx5day) OVER (ORDER BY year ROWS BETWEEN 4 PRECEDING AND CURRENT ROW), 1)
        AS rx5day_5yr_avg_mm
FROM indices_regional
ORDER BY year;


-- name: q07_2025_anomaly_by_kabupaten
-- question: How extreme was 2025 in each regency compared with its 1991-2020 normal?
WITH normal AS (
    SELECT kabupaten, AVG(rx1day) AS rx1day_normal
    FROM indices_kabupaten
    WHERE year BETWEEN 1991 AND 2020
    GROUP BY kabupaten
)
SELECT
    k.kabupaten,
    ROUND(k.rx1day, 1)                                        AS rx1day_2025_mm,
    ROUND(n.rx1day_normal, 1)                                 AS rx1day_normal_mm,
    ROUND(100.0 * (k.rx1day - n.rx1day_normal) / n.rx1day_normal, 0) AS anomaly_pct,
    RANK() OVER (ORDER BY k.rx1day DESC)                      AS rank_2025
FROM indices_kabupaten AS k
JOIN normal            AS n USING (kabupaten)
WHERE k.year = 2025
ORDER BY anomaly_pct DESC;


-- name: q08_record_year_per_kabupaten
-- question: In which year did each regency record its highest RX1DAY?
WITH ranked AS (
    SELECT
        kabupaten, year, rx1day,
        ROW_NUMBER() OVER (PARTITION BY kabupaten ORDER BY rx1day DESC) AS rn
    FROM indices_kabupaten
)
SELECT
    year              AS record_year,
    COUNT(*)          AS n_kabupaten,
    GROUP_CONCAT(kabupaten, ', ') AS kabupaten_list
FROM ranked
WHERE rn = 1
GROUP BY year
ORDER BY n_kabupaten DESC, record_year;


-- name: q09_data_coverage
-- question: Which regencies need a caveat because of partial data coverage?
SELECT DISTINCT
    kabupaten,
    n_cells,
    method,
    chirps_coverage
FROM indices_kabupaten
WHERE method <> 'mean' OR chirps_coverage < 0.99
ORDER BY chirps_coverage;
