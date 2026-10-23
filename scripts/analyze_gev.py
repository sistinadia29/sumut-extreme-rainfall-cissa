"""
Minggu 5: GEV non-stasioner untuk hujan ekstrem tahunan (RX1DAY, RX5DAY) dengan ENSO.

Pertanyaan: apakah ENSO menggeser hujan ekstrem tahunan, dan apakah ada tren?
Kovariat ONI = rata-rata ONI tahun kalender (sama dengan fase ENSO di DuckDB/SQL).

Bagian 1 - Regional (rata-rata provinsi dari maksimum tahunan tiap sel grid):
    M0  stasioner              mu = b0
    M1  ENSO                   mu = b0 + b1*ONI
    M2  tren                   mu = b0 + b2*t           (t = dekade, terpusat)
    M3  ENSO + tren            mu = b0 + b1*ONI + b2*t
    Uji LRT: ONI (M1 vs M0), tren (M2 vs M0), ONI setelah tren (M3 vs M2).
    Tingkat ulang 2/20/50 tahun dari M1 pada ONI = -1 (La Nina), 0, +1 (El Nino),
    selang kepercayaan 95% dari bootstrap parametrik. Diulang untuk 1991-2025.

Bagian 2 - Per sel grid (0.05 derajat): M0 vs M1 di setiap sel, peta perubahan
    tingkat ulang 20-tahun per +1 ONI, q-value FDR Benjamini-Hochberg di semua sel
    (alpha_FDR = 0.10, Wilks 2016 BAMS untuk medan yang berkorelasi spasial).

Keluaran (ditahan .gitignore sampai naskah dikirim):
    outputs/tables/gev_models.csv           parameter, logLik, AIC, BIC semua model
    outputs/tables/gev_tests.csv            uji LRT
    outputs/tables/gev_return_levels.csv    tingkat ulang + CI 95% per fase ENSO
    outputs/tables/gev_grid_summary.csv     ringkasan peta per sel
    outputs/figures/gev_regional.png
    outputs/figures/gev_grid_oni.png
    data/processed/gev_grid.nc

Contoh:
    python scripts/analyze_gev.py
    python scripts/analyze_gev.py --no-grid          # hanya regional (cepat)
    python scripts/analyze_gev.py --n-boot 200       # bootstrap lebih sedikit
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.gev import bootstrap, fit, gumbel_residuals, lrt, return_level  # noqa: E402
from sumut_rain.stats import fdr_bh  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABEL = {"rx1day": "RX1DAY (max 1-day)", "rx5day": "RX5DAY (max 5-day)"}
PHASE = {-1.0: "La Nina (ONI -1)", 0.0: "Neutral (ONI 0)", 1.0: "El Nino (ONI +1)"}
MODELS = {"M0_stationary": [], "M1_oni": ["oni"], "M2_trend": ["t"], "M3_oni_trend": ["oni", "t"]}
TESTS = [("ONI", "M0_stationary", "M1_oni"), ("trend", "M0_stationary", "M2_trend"),
         ("ONI | trend", "M2_trend", "M3_oni_trend")]


def design(df, cols):
    return np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])


def annual_oni(ext: Path) -> pd.DataFrame:
    clim = pd.read_csv(ext / "climate_indices_monthly.csv", parse_dates=["date"])
    g = clim.groupby(clim["date"].dt.year)["oni"].agg(["mean", "count"])
    return g[g["count"] == 12]["mean"].rename("oni").rename_axis("year").reset_index()


# ------------------------------------------------------------------ regional

def regional_analysis(df, indices, periods, rps, oni_levels, n_boot):
    model_rows, test_rows, rl_rows, curves = [], [], [], {}
    oni_grid = np.linspace(df["oni"].min(), df["oni"].max(), 41)
    for pname, d in periods.items():
        d = d.copy()
        d["t"] = (d["year"] - d["year"].mean()) / 10.0          # per dekade
        for idx in indices:
            y = d[idx].values
            fits = {m: fit(y, design(d, cols)) for m, cols in MODELS.items()}
            for m, f in fits.items():
                coef = dict(zip(["b0"] + MODELS[m], f["beta"]))
                model_rows.append({"period": pname, "index": idx, "model": m, "n": f["n"],
                                   "b0": coef["b0"], "b_oni": coef.get("oni", np.nan),
                                   "b_trend_per_decade": coef.get("t", np.nan),
                                   "sigma": f["sigma"], "xi": f["xi"], "loglik": f["loglik"],
                                   "aic": f["aic"], "bic": f["bic"]})
            best_aic = min(fits, key=lambda m: fits[m]["aic"])
            for name, small, big in TESTS:
                D, p = lrt(fits[small], fits[big])
                test_rows.append({"period": pname, "index": idx, "test": name,
                                  "models": f"{big} vs {small}", "D": D, "p_value": p,
                                  "best_by_aic": best_aic})

            # tingkat ulang dari M1 + bootstrap
            X1, f1 = design(d, ["oni"]), fits["M1_oni"]
            levels = np.array(oni_levels, float)

            def rl_of(f, xs=levels):
                mu = f["beta"][0] + f["beta"][1] * xs
                return np.array([return_level(mu, f["sigma"], f["xi"], T) for T in rps])

            curve_T = [T for T in (2, 20) if T in rps]
            curve_rows = [list(rps).index(T) for T in curve_T]

            def boot_func(f):
                return np.concatenate([rl_of(f).ravel(), rl_of(f, oni_grid)[curve_rows].ravel()])

            est = rl_of(f1)
            print(f"    {pname:<10} {idx}: bootstrap {n_boot}x ...", flush=True)
            B = bootstrap(y, X1, f1, boot_func, n_boot=n_boot, seed=42)
            nb = est.size
            Bl = B[:, :nb].reshape(len(B), len(rps), len(levels))
            for i, T in enumerate(rps):
                for j, x in enumerate(levels):
                    lo, hi = np.percentile(Bl[:, i, j], [2.5, 97.5])
                    rl_rows.append({"period": pname, "index": idx, "return_period": T,
                                    "phase": PHASE.get(x, f"ONI {x:+g}"), "oni": x,
                                    "return_level_mm": est[i, j], "ci_low": lo, "ci_high": hi})
                # rasio La Nina / El Nino
                jl, je = int(np.argmin(levels)), int(np.argmax(levels))
                ratio = Bl[:, i, jl] / Bl[:, i, je]
                lo, hi = np.percentile(ratio, [2.5, 97.5])
                rl_rows.append({"period": pname, "index": idx, "return_period": T,
                                "phase": "ratio La Nina / El Nino", "oni": np.nan,
                                "return_level_mm": est[i, jl] / est[i, je], "ci_low": lo, "ci_high": hi})
            if pname == "full":
                Bc = B[:, nb:].reshape(len(B), len(curve_T), len(oni_grid))
                curves[idx] = {"d": d, "f1": f1, "oni_grid": oni_grid, "boot": Bc, "curve_T": curve_T,
                               "est": rl_of(f1, oni_grid)[curve_rows]}
    return (pd.DataFrame(model_rows), pd.DataFrame(test_rows), pd.DataFrame(rl_rows), curves)


def plot_regional(curves, rps, tests, out):
    idxs = list(curves)
    fig, axes = plt.subplots(2, len(idxs), figsize=(5.6 * len(idxs), 8.6),
                             gridspec_kw={"height_ratios": [1.5, 1]})
    axes = np.atleast_2d(axes).reshape(2, len(idxs))
    for c, idx in enumerate(idxs):
        cv, ax = curves[idx], axes[0, c]
        d, g = cv["d"], cv["oni_grid"]
        ax.scatter(d["oni"], d[idx], s=18, color=MUTED, alpha=0.75, zorder=3, label="Annual maximum")
        for k, T in enumerate(cv["curve_T"]):
            color = BLUE if T == 2 else ORANGE
            ax.plot(g, cv["est"][k], color=color, linewidth=2, label=f"{T}-year return level (95% CI)")
            lo, hi = np.percentile(cv["boot"][:, k], [2.5, 97.5], axis=0)
            ax.fill_between(g, lo, hi, color=color, alpha=0.15, linewidth=0)
        last = d.sort_values("year").iloc[-1]
        ax.annotate(str(int(last["year"])), (last["oni"], last[idx]), xytext=(5, 3),
                    textcoords="offset points", fontsize=8, color=INK)
        p = tests[(tests["period"] == "full") & (tests["index"] == idx) & (tests["test"] == "ONI")]["p_value"].iloc[0]
        ax.set_title(f"{LABEL.get(idx, idx)}\nLRT for ONI effect: p = {p:.3f}", loc="left",
                     fontsize=10, color=INK)
        ax.set_xlabel("Annual mean ONI (La Nina  <-  0  ->  El Nino)", color=MUTED, fontsize=9)
        ax.set_ylabel("Area-mean annual maximum (mm)", color=MUTED, fontsize=9)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.legend(frameon=False, fontsize=8)

        # diagnostik QQ (residual Gumbel standar)
        ax2, f1 = axes[1, c], cv["f1"]
        mu = f1["beta"][0] + f1["beta"][1] * d["oni"].values
        r = np.sort(gumbel_residuals(d[idx].values, mu, f1["sigma"], f1["xi"]))
        n = len(r)
        theo = -np.log(-np.log((np.arange(1, n + 1) - 0.44) / (n + 0.12)))    # Gringorten
        ax2.scatter(theo, r, s=14, color=BLUE)
        lim = [min(theo.min(), r.min()) - 0.2, max(theo.max(), r.max()) + 0.2]
        ax2.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1)
        ax2.set_title("Model check (M1): residual QQ plot", loc="left", fontsize=9, color=INK)
        ax2.set_xlabel("Standard Gumbel quantile", color=MUTED, fontsize=9)
        ax2.set_ylabel("Transformed residual", color=MUTED, fontsize=9)
    for ax in axes.ravel():
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ grid

def grid_analysis(ds, indices, oni_by_year, T=20):
    years = ds["year"].values
    oni = oni_by_year.set_index("year").reindex(years)["oni"].values
    keep = ~np.isnan(oni)
    X1 = np.column_stack([np.ones(keep.sum()), oni[keep]])
    res = {}
    for idx in indices:
        arr = ds[idx].values[keep]                                 # (tahun, lat, lon)
        valid = np.all(np.isfinite(arr), axis=0)
        cells = np.argwhere(valid)
        shape = valid.shape
        out = {k: np.full(shape, np.nan) for k in ("b_oni_mm", "rl20_mm", "rl20_change_pct", "p_value",
                                                   "xi", "q_value")}
        t0, n = time.time(), len(cells)
        print(f"    {idx}: {n} sel grid ...", flush=True)
        for c, (i, j) in enumerate(cells):
            y = arr[:, i, j].astype(float)
            f0 = fit(y)
            start = np.array([f0["beta"][0], 0.0, np.log(f0["sigma"]), f0["xi"]])
            f1 = fit(y, X1, start=start)
            _, p = lrt(f0, f1)
            rl0 = return_level(f1["beta"][0], f1["sigma"], f1["xi"], T)
            out["b_oni_mm"][i, j] = f1["beta"][1]
            out["rl20_mm"][i, j] = rl0
            out["rl20_change_pct"][i, j] = 100 * f1["beta"][1] / rl0
            out["p_value"][i, j] = p
            out["xi"][i, j] = f1["xi"]
            if (c + 1) % max(1, n // 10) == 0:
                el = time.time() - t0
                print(f"      {100 * (c + 1) / n:3.0f}%  ({el / 60:.1f} menit, sisa ~{el / (c + 1) * (n - c - 1) / 60:.1f})",
                      flush=True)
        out["q_value"][valid] = fdr_bh(out["p_value"][valid])
        res[idx] = out
    return res


def plot_grid(ds, res, prov_path, out, alpha_fdr):
    idxs = list(res)
    lon, lat = ds["longitude"].values, ds["latitude"].values
    vals = np.concatenate([res[k]["rl20_change_pct"][np.isfinite(res[k]["rl20_change_pct"])] for k in idxs])
    vmax = float(np.ceil(np.percentile(np.abs(vals), 98)))
    fig, axes = plt.subplots(1, len(idxs), figsize=(5.4 * len(idxs), 5.6))
    axes = np.atleast_1d(axes)
    prov = None
    if prov_path.exists():
        try:
            import geopandas as gpd
            prov = gpd.read_file(prov_path)
        except Exception:
            prov = None
    for ax, idx in zip(axes, idxs):
        r = res[idx]
        im = ax.pcolormesh(lon, lat, r["rl20_change_pct"], cmap="BrBG", vmin=-vmax, vmax=vmax,
                           shading="nearest")
        sig = r["q_value"] < alpha_fdr
        if sig.any():                                              # garis batas area signifikan
            ax.contour(lon, lat, sig.astype(float), levels=[0.5], colors=INK, linewidths=0.7)
        if prov is not None:
            prov.boundary.plot(ax=ax, color=INK, linewidth=0.6)
        frac = 100 * sig.sum() / np.isfinite(r["q_value"]).sum()
        ax.set_title(f"{LABEL.get(idx, idx)}\nOutlined: FDR q < {alpha_fdr} ({frac:.0f}% of area)",
                     loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal")
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    cb = fig.colorbar(im, ax=axes.tolist(), orientation="horizontal", fraction=0.05, pad=0.08)
    cb.set_label("Change in 20-year return level per +1 ONI (%)  -  brown: drier extremes in El Nino",
                 color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid", action="store_true", help="lewati analisis per sel grid")
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    gc = cfg.get("gev", {})
    indices = gc.get("indices", ["rx1day", "rx5day"])
    rps = gc.get("return_periods", [2, 20, 50])
    oni_levels = gc.get("oni_levels", [-1.0, 0.0, 1.0])
    n_boot = args.n_boot or gc.get("n_boot", 500)
    alpha_fdr = gc.get("alpha_fdr", 0.10)
    sens = cfg["cissa"]["sensitivity_start_year"]

    proc = project_path(cfg["paths"]["processed"])
    ext = project_path(cfg["paths"]["external"])
    tables, figs = project_path("outputs/tables"), project_path(cfg["paths"]["figures"])

    oni = annual_oni(ext)
    df = pd.read_csv(proc / "indices_regional_annual.csv").merge(oni, on="year", how="inner")
    periods = {"full": df, f"{sens}+": df[df["year"] >= sens]}
    print(f"Regional: {df['year'].min()}-{df['year'].max()} ({len(df)} tahun), indeks: {', '.join(indices)}")

    models, tests, rls, curves = regional_analysis(df, indices, periods, rps, oni_levels, n_boot)
    models.round(4).to_csv(tables / "gev_models.csv", index=False)
    tests.round(4).to_csv(tables / "gev_tests.csv", index=False)
    rls.round(3).to_csv(tables / "gev_return_levels.csv", index=False)
    plot_regional(curves, rps, tests, figs / "gev_regional.png")

    pd.set_option("display.width", 180)
    print("\n=== Perbandingan model (AIC/BIC lebih kecil = lebih baik) ===")
    print(models[["period", "index", "model", "b0", "b_oni", "b_trend_per_decade", "sigma", "xi",
                  "aic", "bic"]].round(2).to_string(index=False))
    print("\n=== Uji rasio likelihood ===")
    print(tests[["period", "index", "test", "D", "p_value", "best_by_aic"]].round(3).to_string(index=False))
    print("\n=== Tingkat ulang 20 tahun dari M1 (mm, CI 95% bootstrap) ===")
    t20 = rls[rls["return_period"] == 20]
    print(t20[["period", "index", "phase", "return_level_mm", "ci_low", "ci_high"]].round(2).to_string(index=False))

    if not args.no_grid:
        import xarray as xr
        print("\nGEV per sel grid (M0 vs M1). Ini bisa makan waktu 5-15 menit ...")
        ds = xr.open_dataset(proc / "extreme_indices_grid.nc")
        res = grid_analysis(ds, indices, oni)
        out = xr.Dataset(coords={"latitude": ds["latitude"], "longitude": ds["longitude"]})
        rows = []
        w = np.cos(np.deg2rad(ds["latitude"].values))[:, None]
        for idx, r in res.items():
            for k, v in r.items():
                out[f"{idx}_{k}"] = (("latitude", "longitude"), v)
            ok = np.isfinite(r["q_value"])
            ww = np.broadcast_to(w, ok.shape)[ok]
            ch = r["rl20_change_pct"][ok]
            sig = r["q_value"][ok] < alpha_fdr
            rows.append({"index": idx, "n_cells": int(ok.sum()),
                         "area_pct_negative": 100 * np.sum(ww * (ch < 0)) / ww.sum(),
                         "area_pct_p_lt_005": 100 * np.sum(ww * (r["p_value"][ok] < 0.05)) / ww.sum(),
                         f"area_pct_fdr_q_lt_{alpha_fdr}": 100 * np.sum(ww * sig) / ww.sum(),
                         "median_rl20_change_pct": float(np.median(ch)),
                         "median_xi": float(np.median(r["xi"][ok]))})
        out.to_netcdf(proc / "gev_grid.nc")
        summ = pd.DataFrame(rows).round(2)
        summ.to_csv(tables / "gev_grid_summary.csv", index=False)
        plot_grid(ds, res, ext / "sumut_province.geojson", figs / "gev_grid_oni.png", alpha_fdr)
        print("\n=== Ringkasan per sel grid (% luas, bobot cos lintang) ===")
        print(summ.to_string(index=False))
        print("  (Secara acak, ~5% luas akan punya p < 0.05 walau ENSO tidak berpengaruh.)")

    print(f"\nTersimpan di {tables} dan {figs}")


if __name__ == "__main__":
    main()
