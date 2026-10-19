# Extreme Rainfall in North Sumatra: Trend vs. ENSO/IOD Modulation

> Decomposing long-term trends and climate-mode modulation of extreme rainfall in North Sumatra, Indonesia, using Circulant Singular Spectrum Analysis (CiSSA) and non-stationary extreme value analysis.

**Status:** 🚧 Work in progress — Week 1: data pipeline

## Why this matters

Extreme rainfall drives floods and landslides across North Sumatra, affecting cities such as Medan, agriculture, and infrastructure. Part of its variability comes from long-term change, and part from natural climate modes, mainly **ENSO** (Pacific) and the **Indian Ocean Dipole** (Indian Ocean). Separating the two matters for planning: a "trend" estimated over a period dominated by La Niña years can overstate future risk, and vice versa.

This project asks:

1. How much of the variability in rainfall extremes comes from long-term trends versus ENSO/IOD modulation?
2. Does a non-stationary GEV model with CiSSA-derived climate-mode covariates outperform a stationary GEV?
3. How do 10-, 25- and 50-year return levels differ between El Niño, neutral and La Niña phases?

## Data

| Dataset | Description | Source |
|---|---|---|
| CHIRPS v2.0 | Daily rainfall, 0.05°, 1981–present | UCSB CHC, via NOAA ERDDAP |
| ONI | Oceanic Niño Index (ENSO) | NOAA CPC |
| Niño 3.4 | Monthly SST anomaly | NOAA PSL |
| DMI | Dipole Mode Index (IOD) | NOAA PSL |
| BMKG stations | Gauge data for validation | BMKG *(planned)* |

Raw data are not stored in this repository. Every dataset is downloaded with a script.

## Methods (planned)

- ETCCDI extreme indices: Rx1day, Rx5day, R95p
- CiSSA decomposition into trend and ENSO/IOD-band oscillatory components
- Mann–Kendall trend test and Sen's slope on the trend component
- Stationary vs. non-stationary GEV (covariates from CiSSA components), compared with AIC/BIC and likelihood-ratio tests
- Return levels conditioned on climate phase, mapped and served in an interactive dashboard

## Reproduce

```bash
conda env create -f environment.yml
conda activate sumut-rain

python scripts/download_chirps.py --start 2020 --end 2021   # quick test
python scripts/download_chirps.py                            # full period
python scripts/download_climate_indices.py
python scripts/check_data.py
```

Region and period are set in [`config.yaml`](config.yaml).

## Project structure

```
├── config.yaml          # region, period, data sources
├── environment.yml      # conda environment
├── scripts/             # download & data-check scripts
├── src/sumut_rain/      # reusable project code
├── notebooks/           # analysis notebooks
├── data/                # raw / processed / external (git-ignored)
├── outputs/figures/     # figures
└── paper/               # manuscript materials
```

## Author

**Sisti Nadia Amalia** — Lecturer, Department of Statistics, Universitas Negeri Medan.
Research: singular spectrum analysis for climate time series, spatial clustering, climate risk zoning.

## License

Code: MIT. Data remain under their original providers' terms.
