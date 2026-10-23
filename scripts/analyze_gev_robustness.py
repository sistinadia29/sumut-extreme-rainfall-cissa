"""
Minggu 5b: uji ketahanan GEV dan hasil per kabupaten/kota.

Dijalankan SETELAH analyze_gev.py (memakai data/processed/gev_grid.nc untuk bagian C).

 A. Jackknife: model M1 (lokasi ~ ONI) dipasang ulang 45 kali, setiap kali tanpa satu tahun.
    Apakah efek ONI bergantung pada satu tahun tertentu (misalnya El Nino 1997 atau 2015)?
 B. Tanpa tahun ENSO kuat: semua tahun dengan |ONI tahunan| >= 1 dibuang sekaligus.
    Apakah efeknya hanya datang dari ujung-ujung sebaran ONI?
 C. Per kabupaten/kota (33):
      - GEV M0 vs M1 pada rata-rata kabupaten, q-value FDR di 33 kabupaten,
        rasio tingkat ulang 20 tahun La Nina / El Nino + CI 95% bootstrap
      - dari peta per sel: % luas kabupaten yang lolos FDR dan median perubahan per +1 ONI

Keluaran (ditahan .gitignore sampai naskah dikirim):
    outputs/tables/gev_jackknife.csv
    outputs/tables/gev_drop_strong_enso.csv
    outputs/tables/gev_kabupaten.csv
    outputs/figures/gev_jackknife.png
    outputs/figures/gev_kabupaten.png

Contoh:
    python scripts/analyze_gev_robustness.py
    python scripts/analyze_gev_robustness.py --n-boot 100     # lebih cepat
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.gev import bootstrap, fit, lrt, return_level  # noqa: E402
from sumut_rain.stats import fdr_bh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gev import annual_oni  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ab"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABEL = {"rx1day": "RX1DAY", "rx5day": "RX5DAY"}


def m1_summary(y, oni, T=20):
    """Pasang M0 dan M1, kembalikan b_oni, p LRT, dan rasio RL_T La Nina / El Nino."""
    X = np.column_stack([np.ones(len(y)), oni])
    f0, f1 = fit(y), fit(y, X)
    _, p = lrt(f0, f1)
    rl = lambda x: return_level(f1["beta"][0] + f1["beta"][1] * x, f1["sigma"], f1["xi"], T)  # noqa: E731
    return {"b_oni": f1["beta"][1], "p_value": p, "ratio_ln_en": rl(-1.0) / rl(1.0),
            "rl20_change_pct": 100 * f1["beta"][1] / rl(0.0)}, f1, X


def style(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


# ---------------------------------------------------------------- A + B

def jackknife(df, indices, periods):
    rows, base = [], {}
    for pname, d in periods.items():
        for idx in indices:
            full, _, _ = m1_summary(d[idx].values, d["oni"].values)
            base[(pname, idx)] = full
            for yr in d["year"]:
                dd = d[d["year"] != yr]
                s, _, _ = m1_summary(dd[idx].values, dd["oni"].values)
                oni_yr = float(d.loc[d["year"] == yr, "oni"].iloc[0])
                rows.append({"period": pname, "index": idx, "dropped_year": int(yr),
                             "oni_of_dropped": oni_yr, **s})
    return pd.DataFrame(rows), base


def drop_strong(df, indices, periods, thr=1.0):
    rows = []
    for pname, d in periods.items():
        dd = d[d["oni"].abs() < thr]
        dropped = ", ".join(str(int(y)) for y in d.loc[d["oni"].abs() >= thr, "year"])
        for idx in indices:
            s, _, _ = m1_summary(dd[idx].values, dd["oni"].values)
            rows.append({"period": pname, "index": idx, "n_years": len(dd),
                         "dropped_years": dropped, **s})
    return pd.DataFrame(rows)


def plot_jackknife(jk, base, out):
    full = jk[jk["period"] == "full"]
    idxs = list(full["index"].unique())
    fig, axes = plt.subplots(1, len(idxs), figsize=(5.6 * len(idxs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, idxs):
        d = full[full["index"] == idx]
        strong = d["oni_of_dropped"].abs() >= 1.0
        ax.scatter(d.loc[~strong, "oni_of_dropped"], d.loc[~strong, "p_value"], s=16, color=GRAY,
                   label="Other years")
        ax.scatter(d.loc[strong, "oni_of_dropped"], d.loc[strong, "p_value"], s=26, color=ORANGE,
                   label="Strong ENSO years (|ONI| >= 1)", zorder=3)
        for _, r in d[strong].iterrows():
            ax.annotate(str(r["dropped_year"]), (r["oni_of_dropped"], r["p_value"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=INK)
        ax.axhline(0.05, color=MUTED, linestyle="--", linewidth=1)
        ax.axhline(base[("full", idx)]["p_value"], color=BLUE, linewidth=1.5, label="All 45 years")
        ax.set_yscale("log")
        ax.set_title(f"{LABEL.get(idx, idx)}: p-value of ONI effect when one year is removed",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Annual mean ONI of the removed year", color=MUTED, fontsize=9)
        style(ax)
    axes[0].set_ylabel("LRT p-value (log scale)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- C

def kabupaten_gev(kab_df, oni, indices, n_boot):
    d = kab_df.merge(oni, on="year", how="inner")
    rows = []
    names = sorted(d["kabupaten"].unique())
    for i, name in enumerate(names, 1):
        g = d[d["kabupaten"] == name].sort_values("year")
        print(f"    [{i:2d}/{len(names)}] {name}", flush=True)
        for idx in indices:
            y = g[idx].values.astype(float)
            s, f1, X = m1_summary(y, g["oni"].values)

            def ratio(f):
                rl = lambda x: return_level(f["beta"][0] + f["beta"][1] * x, f["sigma"], f["xi"], 20)  # noqa: E731
                return rl(-1.0) / rl(1.0)

            B = bootstrap(y, X, f1, ratio, n_boot=n_boot, seed=7)
            lo, hi = np.percentile(B, [2.5, 97.5]) if len(B) else (np.nan, np.nan)
            rows.append({"kabupaten": name, "index": idx, **s, "ratio_ci_low": lo, "ratio_ci_high": hi})
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    for idx in indices:
        m = out["index"] == idx
        out.loc[m, "q_value"] = fdr_bh(out.loc[m, "p_value"].values)
    return out


def grid_by_kabupaten(proc, ext, indices, alpha):
    """% luas kabupaten yang lolos FDR dan median perubahan, dari gev_grid.nc."""
    f = proc / "gev_grid.nc"
    if not f.exists():
        print("  gev_grid.nc belum ada (jalankan analyze_gev.py tanpa --no-grid). Bagian peta dilewati.")
        return None
    import geopandas as gpd
    import regionmask
    import xarray as xr
    ds = xr.open_dataset(f)
    kab = gpd.read_file(ext / "sumut_kabupaten.geojson")
    kmask = regionmask.mask_geopandas(kab, ds["longitude"], ds["latitude"]).values
    w = np.broadcast_to(np.cos(np.deg2rad(ds["latitude"].values))[:, None], kmask.shape)
    rows = []
    for k, row in kab.iterrows():
        for idx in indices:
            q = ds[f"{idx}_q_value"].values
            ch = ds[f"{idx}_rl20_change_pct"].values
            cells = (kmask == k) & np.isfinite(q)
            if not cells.any():
                rows.append({"kabupaten": row["kabupaten"], "index": idx, "grid_cells": 0})
                continue
            ww = w[cells]
            rows.append({"kabupaten": row["kabupaten"], "index": idx, "grid_cells": int(cells.sum()),
                         "grid_area_pct_fdr": 100 * np.sum(ww * (q[cells] < alpha)) / ww.sum(),
                         "grid_median_change_pct": float(np.median(ch[cells]))})
    return pd.DataFrame(rows)


def plot_kabupaten(kb, out):
    idxs = list(kb["index"].unique())
    order = (kb[kb["index"] == idxs[-1]].sort_values("ratio_ln_en")["kabupaten"].tolist())
    ypos = {k: i for i, k in enumerate(order)}
    fig, axes = plt.subplots(1, len(idxs), figsize=(5.0 * len(idxs), 0.24 * len(order) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, idxs):
        d = kb[kb["index"] == idx]
        for _, r in d.iterrows():
            y = ypos[r["kabupaten"]]
            sig = r["q_value"] < 0.10
            color = BLUE if sig else GRAY
            ax.plot([r["ratio_ci_low"], r["ratio_ci_high"]], [y, y], color=color, linewidth=1.4)
            ax.scatter(r["ratio_ln_en"], y, s=22, color=color, zorder=3)
        ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=7.5, color=INK)
        ax.set_title(f"{LABEL.get(idx, idx)}: 20-year return level,\nLa Nina / El Nino (95% CI)",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Ratio (> 1: larger extremes in La Nina years)", color=MUTED, fontsize=9)
        style(ax, "x")
    axes[-1].scatter([], [], color=BLUE, label="FDR q < 0.10")
    axes[-1].scatter([], [], color=GRAY, label="not significant")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap per kabupaten (default 200)")
    args = ap.parse_args()

    cfg = load_config()
    gc = cfg.get("gev", {})
    indices = gc.get("indices", ["rx1day", "rx5day"])
    alpha = gc.get("alpha_fdr", 0.10)
    sens = cfg["cissa"]["sensitivity_start_year"]
    proc, ext = project_path(cfg["paths"]["processed"]), project_path(cfg["paths"]["external"])
    tables, figs = project_path("outputs/tables"), project_path(cfg["paths"]["figures"])
    pd.set_option("display.width", 180)

    oni = annual_oni(ext)
    df = pd.read_csv(proc / "indices_regional_annual.csv").merge(oni, on="year", how="inner")
    periods = {"full": df, f"{sens}+": df[df["year"] >= sens]}

    # A
    print("A. Jackknife (buang satu tahun) ...")
    jk, base = jackknife(df, indices, periods)
    jk.round(4).to_csv(tables / "gev_jackknife.csv", index=False)
    plot_jackknife(jk, base, figs / "gev_jackknife.png")
    rows = []
    for (pname, idx), b in base.items():
        d = jk[(jk["period"] == pname) & (jk["index"] == idx)]
        worst = d.loc[d["p_value"].idxmax()]
        rows.append({"period": pname, "index": idx, "p_all_years": b["p_value"],
                     "p_max_jackknife": worst["p_value"], "worst_year": int(worst["dropped_year"]),
                     "n_drops_p_lt_005": f"{int((d['p_value'] < 0.05).sum())}/{len(d)}",
                     "ratio_all": b["ratio_ln_en"], "ratio_min": d["ratio_ln_en"].min(),
                     "ratio_max": d["ratio_ln_en"].max()})
    print("\n=== A. Jackknife: efek ONI bila satu tahun dibuang ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    strong = jk[(jk["period"] == "full") & (jk["oni_of_dropped"].abs() >= 1.0)]
    print("\n    Tahun ENSO kuat yang dibuang satu per satu (periode penuh):")
    print(strong[["index", "dropped_year", "oni_of_dropped", "b_oni", "p_value", "ratio_ln_en"]]
          .round(3).to_string(index=False))

    # B
    ds_ = drop_strong(df, indices, periods)
    ds_.round(4).to_csv(tables / "gev_drop_strong_enso.csv", index=False)
    print("\n=== B. Tanpa SEMUA tahun ENSO kuat (|ONI| >= 1) ===")
    print(ds_[["period", "index", "n_years", "b_oni", "p_value", "ratio_ln_en", "dropped_years"]]
          .round(3).to_string(index=False))

    # C
    print(f"\nC. GEV per kabupaten/kota (bootstrap {args.n_boot}x) ...")
    kab_df = pd.read_csv(proc / "indices_kabupaten_annual.csv")
    kb = kabupaten_gev(kab_df, oni, indices, args.n_boot)
    gk = grid_by_kabupaten(proc, ext, indices, alpha)
    if gk is not None:
        kb = kb.merge(gk, on=["kabupaten", "index"], how="left")
    kb.round(4).to_csv(tables / "gev_kabupaten.csv", index=False)
    plot_kabupaten(kb, figs / "gev_kabupaten.png")
    cols = ["kabupaten", "ratio_ln_en", "ratio_ci_low", "ratio_ci_high", "p_value", "q_value"]
    if gk is not None:
        cols += ["grid_area_pct_fdr", "grid_median_change_pct"]
    for idx in indices:
        d = kb[kb["index"] == idx].sort_values("ratio_ln_en", ascending=False)
        n_sig = int((d["q_value"] < alpha).sum())
        print(f"\n=== C. {idx.upper()} per kabupaten (urut dari efek La Nina terbesar); "
              f"lolos FDR: {n_sig}/{len(d)} ===")
        print(d[cols].round(3).to_string(index=False))

    print(f"\nTersimpan di {tables} dan {figs}")


if __name__ == "__main__":
    main()
