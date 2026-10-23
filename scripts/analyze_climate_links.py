"""
Minggu 4b: hubungan komponen CiSSA dengan ENSO, IOD, dan IPO.

Dijalankan SETELAH analyze_cissa.py (memakai L yang dipilih di sana).

Tiga analisis:
 A. Korelasi pita-setara: komponen antartahunan hujan vs komponen antartahunan ONI/DMI
    (ONI/DMI didekomposisi dengan CiSSA, L dan pengelompokan yang sama).
 B. Korelasi per frekuensi: setiap komponen elementer k di pita antartahunan, hujan_k vs
    indeks_k (frekuensi yang sama). Menunjukkan periode mana yang terkait ENSO dan mana IOD.
 D. Korelasi parsial (pita antartahunan): ENSO dengan IOD dikendalikan dan sebaliknya,
    plus pemisahan R^2 regresi menjadi bagian unik ENSO, unik IOD, dan bersama.
 C. Variabilitas dekadal: komponen frekuensi rendah hujan vs komponen frekuensi rendah TPI (IPO).

Semua p-value memakai surrogate acak fase (Ebisuzaki 1997) dengan statistik |r| maksimum
di lag 0..max_lag. Untuk analisis B ditambahkan q-value Benjamini-Hochberg (banyak uji).

Keluaran (ikut ditahan .gitignore sampai naskah dikirim):
    outputs/tables/cissa_band_correlation.csv
    outputs/tables/cissa_frequency_correlation.csv
    outputs/tables/cissa_decadal_ipo.csv
    outputs/tables/cissa_partial_correlation.csv
    outputs/figures/cissa_frequency_<var>.png
    outputs/figures/cissa_decadal_ipo.png

Contoh:
    python scripts/analyze_climate_links.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.cissa import cissa, group_components  # noqa: E402
from sumut_rain.stats import commonality, fdr_bh, partial_lag_test, surrogate_lag_test  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ab"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
VAR_LABEL = {"prcp_mm": "Monthly rainfall", "rx1day_mm": "Monthly max 1-day rainfall"}


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def zscore(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std()


def chosen_L(cfg, tables) -> dict:
    """L per variabel dari hasil analyze_cissa.py; kalau tidak ada, dari config."""
    f = tables / "cissa_trend_slope.csv"
    if f.exists():
        d = pd.read_csv(f)
        if "L" in d.columns:
            return dict(zip(d["variable"], d["L"].astype(int)))
    L = cfg["cissa"]["window_months"]
    if str(L).lower() == "auto":
        sys.exit("Jalankan analyze_cissa.py dulu supaya L terpilih tersedia.")
    return {"prcp_mm": int(L), "rx1day_mm": int(L)}


def band_masks(freqs, trend_min, band):
    with np.errstate(divide="ignore"):
        period = np.where(freqs > 0, 1 / freqs, np.inf)
    return period, (period >= band[0]) & (period <= band[1])


def main() -> None:
    cfg = load_config()
    cc = cfg["cissa"]
    proc = project_path(cfg["paths"]["processed"])
    tables = project_path("outputs/tables")
    figs = project_path(cfg["paths"]["figures"])
    n_surr = int(cc.get("n_surrogates", 1000))
    max_lag = int(cc["max_lag_months"])
    gkw = {"trend_min_period": cc["trend_min_period_months"],
           "band": tuple(cc["interannual_band_months"])}

    monthly = pd.read_csv(proc / "monthly_regional.csv", parse_dates=["date"])
    clim = pd.read_csv(project_path(cfg["paths"]["external"]) / "climate_indices_monthly.csv",
                       parse_dates=["date"])
    df = monthly.merge(clim, on="date", how="left")
    indices = [c for c in ("oni", "dmi", "tpi") if c in df.columns]
    if "tpi" not in indices:
        print("Kolom TPI belum ada. Jalankan ulang download_climate_indices.py untuk analisis C.")
    for c in indices:
        if df[c].isna().any():
            n = int(df[c].isna().sum())
            print(f"Peringatan: {c.upper()} punya {n} bulan kosong; diisi interpolasi linear.")
            df[c] = df[c].interpolate(limit_direction="both")

    Ls = chosen_L(cfg, tables)
    print(f"L yang dipakai: {Ls}  |  surrogate: {n_surr}  |  lag 0..{max_lag} bulan\n")

    band_rows, freq_rows, dec_rows, partial_rows = [], [], [], []
    dec_series = {}
    for var, L in Ls.items():
        x = df[var].values.astype(float)
        fx, cx, _ = cissa(x, L)
        gx = group_components(fx, cx, **gkw)
        anomaly_var = np.var(x - gx["seasonal"])
        period, in_band = band_masks(fx, cc["trend_min_period_months"], gkw["band"])

        idx_comp = {}
        for c in indices:
            fi, ci, _ = cissa(df[c].values.astype(float), L)
            idx_comp[c] = (ci, group_components(fi, ci, **gkw))

        # A. pita-setara (ONI, DMI)
        band_lags = {}
        for c in ("oni", "dmi"):
            res = surrogate_lag_test(gx["interannual"], idx_comp[c][1]["interannual"],
                                     max_lag, n_surr, seed=1)
            band_lags[c] = res["lag_months"]
            band_rows.append({"variable": var, "L": L, "index": c.upper(), **res})
        targets = [("band 1.5-8 yr", gx["interannual"],
                    {c: idx_comp[c][1]["interannual"] for c in ("oni", "dmi")}, band_lags)]

        # B. per frekuensi
        for k in np.flatnonzero(in_band):
            row = {"variable": var, "L": L, "k": int(k),
                   "period_months": round(period[k], 1), "period_years": round(period[k] / 12, 2),
                   "share_of_anomaly_var_pct": round(100 * np.var(cx[k]) / anomaly_var, 2)}
            for c in ("oni", "dmi"):
                res = surrogate_lag_test(cx[k], idx_comp[c][0][k], max_lag, n_surr, seed=int(k))
                row[f"r_{c}"] = round(res["r"], 3)
                row[f"lag_{c}"] = res["lag_months"]
                row[f"p_{c}"] = round(res["p_surrogate"], 4)
            freq_rows.append(row)

        # D. korelasi parsial: ENSO dengan IOD dikendalikan, dan sebaliknya,
        #    plus pemisahan R^2 menjadi unik ENSO, unik IOD, dan bersama.
        #    Hanya di level pita: pada satu frekuensi tunggal, komponen ONI dan DMI hampir
        #    berupa dua sinus identik (kolinear), sehingga korelasi parsialnya tidak stabil.
        for j, (level, xs, ys, lags) in enumerate(targets):
            com = commonality(xs, ys["oni"], ys["dmi"], lags["oni"], lags["dmi"])
            for c, ctrl, key in (("oni", "dmi", "unique_1"), ("dmi", "oni", "unique_2")):
                res = partial_lag_test(xs, ys[c], ys[ctrl], lags[ctrl], max_lag, n_surr, seed=100 + j)
                partial_rows.append({"variable": var, "L": L, "level": level, "index": c.upper(),
                                     "controlled_for": ctrl.upper(), **res,
                                     "r2_full_pct": round(100 * com["r2_full"], 1),
                                     "unique_pct": round(100 * com[key], 1),
                                     "shared_pct": round(100 * com["shared"], 1)})

        # C. dekadal (TPI)
        if "tpi" in indices:
            res = surrogate_lag_test(gx["trend"], idx_comp["tpi"][1]["trend"], 0, n_surr, seed=2)
            dec_rows.append({"variable": var, "L": L, "index": "TPI (IPO)",
                             "r": round(res["r"], 3), "p_surrogate": round(res["p_surrogate"], 4),
                             "r95_null": round(res["r95_null"], 3)})
            dec_series[var] = (gx["trend"], idx_comp["tpi"][1]["trend"])

    band_df = pd.DataFrame(band_rows).round(4)
    freq_df = pd.DataFrame(freq_rows)
    for c in ("oni", "dmi"):
        freq_df[f"q_{c}"] = np.round(fdr_bh(freq_df[f"p_{c}"].values), 4)
    band_df.to_csv(tables / "cissa_band_correlation.csv", index=False)
    freq_df.to_csv(tables / "cissa_frequency_correlation.csv", index=False)
    partial_df = pd.DataFrame(partial_rows)
    partial_df["partial_r"] = partial_df["partial_r"].round(3)
    partial_df["q_value"] = np.round(fdr_bh(partial_df["p_surrogate"].values), 4)  # 4 uji: 2 variabel x 2 indeks
    partial_df.to_csv(tables / "cissa_partial_correlation.csv", index=False)

    for var in Ls:
        plot_frequency(freq_df[freq_df["variable"] == var], var, figs / f"cissa_frequency_{var.replace('_mm', '')}.png")
    if dec_rows:
        dec_df = pd.DataFrame(dec_rows)
        dec_df.to_csv(tables / "cissa_decadal_ipo.csv", index=False)
        plot_decadal(df["date"].values, dec_series, figs / "cissa_decadal_ipo.png")

    pd.set_option("display.width", 170)
    print("=== A. Komponen antartahunan hujan vs ONI/DMI (pita sama), uji surrogate ===")
    print(band_df.to_string(index=False))
    print("\n=== B. Korelasi per frekuensi (q = Benjamini-Hochberg) ===")
    print(freq_df.drop(columns=["L", "k"]).to_string(index=False))
    print("\n=== D. Korelasi parsial ENSO vs IOD di pita 1.5-8 tahun + pemisahan R^2 (%) ===")
    print(partial_df.drop(columns=["L"]).to_string(index=False))
    if dec_rows:
        print("\n=== C. Komponen frekuensi rendah hujan vs TPI (IPO) ===")
        print(dec_df.to_string(index=False))
    print(f"\nTersimpan di {tables} dan {figs}")


def plot_frequency(d: pd.DataFrame, var: str, out: Path) -> None:
    d = d.sort_values("period_years")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax = axes[0]
    ax.bar(d["period_years"].astype(str), d["share_of_anomaly_var_pct"], color=BLUE, width=0.6)
    ax.set_ylabel("% of anomaly variance", color=MUTED, fontsize=9)
    ax.set_title(f"{VAR_LABEL[var]}: variance by CiSSA frequency in the 1.5-8 yr band",
                 loc="left", fontsize=10, color=INK)
    ax = axes[1]
    for c, color, label in (("oni", ORANGE, "ONI (ENSO)"), ("dmi", BLUE, "DMI (IOD)")):
        sig = d[f"q_{c}"] < 0.05
        ax.plot(d["period_years"].astype(str), d[f"r_{c}"], color=color, linewidth=1.6, label=label)
        ax.scatter(d["period_years"].astype(str)[sig], d[f"r_{c}"][sig], s=60, color=color, zorder=3)
        ax.scatter(d["period_years"].astype(str)[~sig], d[f"r_{c}"][~sig], s=60, facecolors="white",
                   edgecolors=color, linewidths=1.6, zorder=3)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_ylabel("correlation r (best lag)", color=MUTED, fontsize=9)
    ax.set_xlabel("Period (years)", color=MUTED, fontsize=9)
    ax.set_title("Same-frequency correlation with climate indices (filled = q < 0.05)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        style(a)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decadal(dates, series: dict, out: Path) -> None:
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 3.2 * len(series)), sharex=True, squeeze=False)
    for ax, (var, (rain, tpi)) in zip(axes[:, 0], series.items()):
        ax.plot(dates, zscore(rain), color=BLUE, linewidth=2, label=f"{VAR_LABEL[var]}, low-frequency (> 8 yr)")
        ax.plot(dates, -zscore(tpi), color=ORANGE, linewidth=2, label="TPI (IPO), low-frequency, sign reversed")
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_ylabel("z-score", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style(ax)
    axes[0, 0].set_title("Decadal rainfall vs Interdecadal Pacific Oscillation (positive IPO points down)",
                         loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
