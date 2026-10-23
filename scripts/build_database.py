"""
Minggu 3: bangun database DuckDB dari hasil Minggu 2.

    data/processed/sumut_climate.duckdb
        indices_kabupaten   tahun x kabupaten, 6 indeks ETCCDI
        indices_regional    tahun, rata-rata provinsi
        climate_monthly     ONI, Nino 3.4, DMI bulanan
        climate_annual      (view) rata-rata tahunan ONI & DMI + label fase

Fase tahunan memakai rata-rata ONI/DMI Januari-Desember:
    El Nino jika ONI >= 0.5, La Nina jika ONI <= -0.5, selain itu Neutral
    IOD positif jika DMI >= 0.4, negatif jika DMI <= -0.4, selain itu Neutral
Ini penyederhanaan untuk eksplorasi; analisis paper memakai deret bulanan.

Contoh:
    python scripts/build_database.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    proc = project_path(cfg["paths"]["processed"])
    ext = project_path(cfg["paths"]["external"])

    sources = {
        "indices_kabupaten": proc / "indices_kabupaten_annual.csv",
        "indices_regional": proc / "indices_regional_annual.csv",
        "climate_monthly": ext / "climate_indices_monthly.csv",
    }
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        sys.exit("File belum ada (jalankan script Minggu 1-2 dulu):\n  " + "\n  ".join(missing))

    db_path = proc / "sumut_climate.duckdb"
    con = duckdb.connect(str(db_path))
    for table, csv in sources.items():
        con.execute(f"CREATE OR REPLACE TABLE {table} AS "
                    f"SELECT * FROM read_csv_auto('{csv.as_posix()}')")

    con.execute("""
        CREATE OR REPLACE VIEW climate_annual AS
        SELECT
            year(date)  AS year,
            AVG(oni)    AS oni,
            AVG(dmi)    AS dmi,
            CASE WHEN AVG(oni) >=  0.5 THEN 'El Nino'
                 WHEN AVG(oni) <= -0.5 THEN 'La Nina'
                 ELSE 'Neutral' END AS enso_phase,
            CASE WHEN AVG(dmi) >=  0.4 THEN 'Positive IOD'
                 WHEN AVG(dmi) <= -0.4 THEN 'Negative IOD'
                 ELSE 'Neutral' END AS iod_phase
        FROM climate_monthly
        WHERE oni IS NOT NULL
        GROUP BY year(date)
    """)

    print(f"Database: {db_path}\n")
    for name in ["indices_kabupaten", "indices_regional", "climate_monthly", "climate_annual"]:
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        cols = [r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()]
        print(f"{name:<18} {n:>5} baris | kolom: {', '.join(cols)}")
    con.close()


if __name__ == "__main__":
    main()
