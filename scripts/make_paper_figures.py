"""
Gambar versi publikasi (International Journal of Climatology) dari hasil yang sudah tersimpan.

Tidak menghitung ulang analisis: semua dibaca dari data/processed dan outputs/tables,
kecuali selang kepercayaan Fig. 5 (bootstrap ulang, +-1 menit).

Gaya: tanpa judul di dalam gambar (judul ada di caption naskah), label panel (a), (b),
huruf sans 8 pt, lebar 1 kolom 84 mm atau 2 kolom 174 mm, PNG 300 dpi + PDF vektor.

Keluaran: outputs/figures/paper/FigNN_<nama>.png dan .pdf (ditahan .gitignore)

    Fig01_study_area        rata-rata hujan tahunan + kabupaten + stasiun GHCN; siklus tahunan
    Fig02_annual_indices    6 indeks ETCCDI 1981-2025 + garis Sen
    Fig03_cissa             dekomposisi CiSSA hujan bulanan vs ONI dan DMI
    Fig04_frequency         varians dan korelasi per frekuensi CiSSA
    Fig05_gev_regional      maksimum tahunan vs ONI + tingkat ulang 2 dan 20 tahun
    Fig06_gev_map           peta perubahan tingkat ulang 20 tahun per +1 ONI
    Fig07_regencies         rasio La Nina / El Nino per kabupaten/kota
    Fig08_jackknife         uji jackknife
    FigS1_gev_qq            diagnostik QQ model GEV (lampiran)
    FigS2_window_selection  pemilihan L CiSSA (lampiran)

Contoh:
    python scripts/make_paper_figures.py              # semua
    python scripts/make_paper_figures.py --only 3 6   # hanya Fig. 3 dan 6
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sumut_rain import load_config, project_path  # noqa: E402
from sumut_rain.gev import bootstrap, fit, gumbel_residuals, return_level  # noqa: E402

MM = 1 / 25.4
SINGLE, DOUBLE = 84 * MM, 174 * MM
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9c9a92"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
INDEX_LABEL = {"prcptot": ("PRCPTOT", "mm"), "rx1day": ("RX1DAY", "mm"), "rx5day": ("RX5DAY", "mm"),
               "r20mm": ("R20mm", "days"), "r95p": ("R95p", "mm"), "cdd": ("CDD", "days")}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "savefig.dpi": 300,
})


# ------------------------------------------------------------------ util
def style(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom", color=INK)


def save(fig, out_dir: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def zscore(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / np.nanstd(a)


def lon_fmt(v, _):
    return f"{v:g}°E"


def lat_fmt(v, _):
    return f"{abs(v):g}°{'N' if v >= 0 else 'S'}"


def map_axes(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lon_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_fmt))
    ax.set_xticks([97, 98, 99, 100])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_aspect("equal")
    ax.tick_params(length=2)
    ax.set_xlabel("")
    ax.set_ylabel("")


def english_name(n: str) -> str:
    return f"{n[5:]} City" if n.startswith("Kota ") else n


def annual_oni(ext: Path) -> pd.DataFrame:
    clim = pd.read_csv(ext / "climate_indices_monthly.csv", parse_dates=["date"])
    g = clim.groupby(clim["date"].dt.year)["oni"].agg(["mean", "count"])
    return g[g["count"] == 12]["mean"].rename("oni").rename_axis("year").reset_index()


def load_boundaries(ext: Path):
    try:
        import geopandas as gpd
        return (gpd.read_file(ext / "sumut_province.geojson"),
                gpd.read_file(ext / "sumut_kabupaten.geojson"))
    except Exception as e:                                     # noqa: BLE001
        print(f"  (batas wilayah tidak dimuat: {e})")
        return None, None


def draw_boundaries(ax, prov, kab):
    if kab is not None:
        kab.boundary.plot(ax=ax, color="white", linewidth=0.3)
    if prov is not None:
        prov.boundary.plot(ax=ax, color=INK, linewidth=0.6)


# ------------------------------------------------------------------ Fig. 1
def fig01(P):
    import xarray as xr
    ds = xr.open_dataset(P["proc"] / "extreme_indices_grid.nc")
    clim = ds["prcptot"].mean("year").values
    monthly = pd.read_csv(P["proc"] / "monthly_regional.csv", parse_dates=["date"])
    cyc = monthly.groupby(monthly["date"].dt.month)["prcp_mm"].agg(["mean", "std"])
    st = pd.read_csv(P["tables"] / "ghcn_stations.csv") if (P["tables"] / "ghcn_stations.csv").exists() else None

    fig = plt.figure(figsize=(DOUBLE, 3.6))
    ax = fig.add_axes([0.03, 0.17, 0.50, 0.80])
    im = ax.pcolormesh(ds["longitude"].values, ds["latitude"].values, clim, cmap="YlGnBu",
                       shading="nearest",
                       vmin=float(np.nanpercentile(clim, 2)), vmax=float(np.nanpercentile(clim, 98)))
    draw_boundaries(ax, P["prov"], P["kab"])
    if st is not None:
        ax.scatter(st["lon"], st["lat"], marker="^", s=22, color=ORANGE, edgecolor=INK,
                   linewidth=0.5, zorder=4, label="GHCN-Daily station")
        for _, r in st.iterrows():
            ax.annotate(str(r["name"]).strip().split("/")[-1].title(), (r["lon"], r["lat"]),
                        xytext=(3, 2), textcoords="offset points", fontsize=6, color=INK)
        ax.legend(loc="upper right", frameon=False)
    ax.annotate("Lake Toba", (98.85, 2.6), fontsize=6, color=INK, ha="center", style="italic")
    ax.set_xlim(97.0, 100.5)
    ax.set_ylim(0.5, 4.5)
    map_axes(ax)
    cax = fig.add_axes([0.12, 0.07, 0.32, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Mean annual rainfall, 1981–2025 (mm)")
    cb.outline.set_linewidth(0.4)
    panel(ax, "a")

    ax2 = fig.add_axes([0.64, 0.17, 0.35, 0.72])
    ax2.bar(cyc.index, cyc["mean"], color=BLUE, width=0.7)
    ax2.errorbar(cyc.index, cyc["mean"], yerr=cyc["std"], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(list("JFMAMJJASOND"))
    ax2.set_ylabel("Area-mean monthly rainfall (mm)")
    style(ax2)
    panel(ax2, "b")
    save(fig, P["out"], "Fig01_study_area")


# ------------------------------------------------------------------ Fig. 2
def fig02(P):
    reg = pd.read_csv(P["proc"] / "indices_regional_annual.csv")
    mk_path = P["tables"] / "mann_kendall.csv"
    mk = pd.read_csv(mk_path) if mk_path.exists() else None
    sens = P["sens"]
    names = [n for n in INDEX_LABEL if n in reg.columns]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.9), sharex=True)
    for k, (ax, n) in enumerate(zip(axes.ravel(), names)):
        y, yr = reg[n].values, reg["year"].values
        ax.plot(yr, y, color=GRAY, linewidth=0.9, marker="o", markersize=2)
        for sub, color, ls in ((reg, BLUE, "-"), (reg[reg["year"] >= sens], ORANGE, "--")):
            s, i, *_ = stats.theilslopes(sub[n].values, sub["year"].values)
            xx = np.array([sub["year"].min(), sub["year"].max()])
            ax.plot(xx, i + s * xx, color=color, linewidth=1.3, linestyle=ls)
        label, unit = INDEX_LABEL[n]
        ax.set_ylabel(f"{label} ({unit})")
        if mk is not None:
            lines = []
            for sub, tag in (("full", "1981–2025"), (f"{sens}+", f"{sens}–2025")):
                r = mk[(mk["index"] == n) & (mk["subset"] == sub)]
                if len(r):
                    r = r.iloc[0]
                    pv = "p < 0.001" if r["p_value"] < 0.001 else f"p = {r['p_value']:.3f}"
                    lines.append(f"{tag}: {r['sen_slope_per_decade']:+.1f}/dec, {pv}".replace("-", "−"))
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=5.8, va="top",
                    color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.28 * (hi - lo))                      # ruang untuk teks Sen/MK
        style(ax)
        panel(ax, "abcdef"[k])
    axes[0, 0].plot([], [], color=BLUE, label="Sen slope 1981–2025")
    axes[0, 0].plot([], [], color=ORANGE, linestyle="--", label=f"Sen slope {sens}–2025")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, P["out"], "Fig02_annual_indices")


# ------------------------------------------------------------------ Fig. 3
def fig03(P):
    comp = pd.read_csv(P["proc"] / "cissa_components_prcp.csv", parse_dates=["date"])
    clim = pd.read_csv(P["ext"] / "climate_indices_monthly.csv", parse_dates=["date"])
    df = comp.merge(clim, on="date", how="left")
    for c in ("oni", "dmi"):
        df[c] = df[c].interpolate(limit_direction="both")
    band = pd.read_csv(P["tables"] / "cissa_band_correlation.csv")
    band = band[band["variable"] == "prcp_mm"].set_index("index")
    # ONI dan DMI didekomposisi dengan CiSSA (L dan pita yang sama), seperti pada uji korelasi
    from sumut_rain.cissa import cissa, group_components
    cc = P["cfg"]["cissa"]
    L = int(band["L"].iloc[0]) if "L" in band.columns else int(cc["window_months"])
    gkw = {"trend_min_period": cc["trend_min_period_months"], "band": tuple(cc["interannual_band_months"])}
    for c in ("oni", "dmi"):
        f, comps, _ = cissa(df[c].values.astype(float), L)
        df[f"{c}_inter"] = group_components(f, comps, **gkw)["interannual"]

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1]})
    ax = axes[0]
    ax.plot(df["date"], df["original"], color=GRAY, linewidth=0.5, label="Monthly rainfall")
    ax.plot(df["date"], df["trend"], color=BLUE, linewidth=1.6, label="Low-frequency component (> 8 yr)")
    ax.set_ylabel("mm month$^{-1}$")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    for ax, c, name in ((axes[1], "oni", "ONI"), (axes[2], "dmi", "DMI")):
        ax.plot(df["date"], zscore(df["interannual"]), color=BLUE, linewidth=1.1,
                label="Interannual component (1.5–8 yr)")
        ax.plot(df["date"], -zscore(df[f"{c}_inter"]), color=ORANGE, linewidth=1.1,
                label=f"−{name}, interannual component")
        ax.axhline(0, color=MUTED, linewidth=0.4)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 1.2)
        ax.set_ylabel("Standardised")
        if name in band.index:
            b = band.loc[name]
            ax.text(0.995, 0.03, f"r = {b['r']:.2f} (lag {int(b['lag_months'])} mo), "
                                 f"surrogate p = {b['p_surrogate']:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
        ax.legend(loc="upper left", frameon=False, ncol=2)
    for k, ax in enumerate(axes):
        style(ax)
        panel(ax, "abc"[k])
    fig.tight_layout()
    save(fig, P["out"], "Fig03_cissa")


# ------------------------------------------------------------------ Fig. 4
def fig04(P):
    d = pd.read_csv(P["tables"] / "cissa_frequency_correlation.csv")
    d = d[d["variable"] == "prcp_mm"].sort_values("period_years")
    x = np.arange(len(d))
    labels = [f"{v:.1f}" for v in d["period_years"]]
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE * 1.35, 3.6), sharex=True)
    ax = axes[0]
    ax.bar(x, d["share_of_anomaly_var_pct"], color=GRAY, width=0.6)
    ax.set_ylabel("Anomaly variance (%)")
    ax = axes[1]
    for c, color, name, dx in (("oni", ORANGE, "ONI", -0.12), ("dmi", BLUE, "DMI", 0.12)):
        sig = (d[f"q_{c}"] < 0.05).values
        r = d[f"r_{c}"].values
        ax.plot(x + dx, r, color=color, linewidth=0.8, alpha=0.6)
        ax.scatter((x + dx)[sig], r[sig], s=20, color=color, zorder=3, label=f"{name}, q < 0.05")
        ax.scatter((x + dx)[~sig], r[~sig], s=20, facecolors="white", edgecolors=color, linewidths=1,
                   zorder=3, label=f"{name}, not significant")
    ax.axhline(0, color=MUTED, linewidth=0.4)
    ax.set_ylabel("Correlation r")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Period of CiSSA component (years)")
    for k, ax in enumerate(axes):
        style(ax)
        panel(ax, "ab"[k])
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, P["out"], "Fig04_frequency")


# ------------------------------------------------------------------ Fig. 5 + S1
def gev_curves(P, indices, n_boot):
    df = pd.read_csv(P["proc"] / "indices_regional_annual.csv").merge(annual_oni(P["ext"]), on="year")
    g = np.linspace(df["oni"].min(), df["oni"].max(), 41)
    res = {}
    for idx in indices:
        y = df[idx].values
        X = np.column_stack([np.ones(len(y)), df["oni"].values])
        f = fit(y, X)

        def curves(ff):
            mu = ff["beta"][0] + ff["beta"][1] * g
            return np.array([return_level(mu, ff["sigma"], ff["xi"], T) for T in (2, 20)])

        B = bootstrap(y, X, f, curves, n_boot=n_boot, seed=42)
        res[idx] = {"df": df, "g": g, "fit": f, "est": curves(f), "boot": B}
    return res


def fig05(P, cur):
    tests = pd.read_csv(P["tables"] / "gev_tests.csv")
    fig, axes = plt.subplots(1, len(cur), figsize=(DOUBLE, 2.9))
    for k, (ax, (idx, c)) in enumerate(zip(np.atleast_1d(axes), cur.items())):
        d, g = c["df"], c["g"]
        ax.scatter(d["oni"], d[idx], s=9, color=GRAY, zorder=3, label="Annual maximum")
        for j, (T, color) in enumerate(((2, BLUE), (20, ORANGE))):
            lo, hi = np.percentile(c["boot"][:, j], [2.5, 97.5], axis=0)
            ax.fill_between(g, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(g, c["est"][j], color=color, linewidth=1.4, label=f"{T}-year return level")
        top = d.loc[d[idx].idxmax()]
        ax.annotate(str(int(top["year"])), (top["oni"], top[idx]), xytext=(4, -2),
                    textcoords="offset points", fontsize=6, color=INK)
        p = tests[(tests["period"] == "full") & (tests["index"] == idx) & (tests["test"] == "ONI")]["p_value"]
        if len(p):
            ax.text(0.98, 0.97, f"LRT for ONI: p = {p.iloc[0]:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color=INK)
        ax.axvline(0, color=MUTED, linewidth=0.4)
        ax.set_xlabel("Annual mean ONI (°C)")
        ax.set_ylabel(f"{INDEX_LABEL[idx][0]} (mm)")
        style(ax, "both")
        panel(ax, "ab"[k])
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save(fig, P["out"], "Fig05_gev_regional")


def figS1(P, cur):
    fig, axes = plt.subplots(1, len(cur), figsize=(DOUBLE * 0.7, 2.4))
    for k, (ax, (idx, c)) in enumerate(zip(np.atleast_1d(axes), cur.items())):
        d, f = c["df"], c["fit"]
        mu = f["beta"][0] + f["beta"][1] * d["oni"].values
        r = np.sort(gumbel_residuals(d[idx].values, mu, f["sigma"], f["xi"]))
        n = len(r)
        theo = -np.log(-np.log((np.arange(1, n + 1) - 0.44) / (n + 0.12)))
        lim = [min(theo.min(), r.min()) - 0.2, max(theo.max(), r.max()) + 0.2]
        ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=0.7)
        ax.scatter(theo, r, s=8, color=BLUE)
        ax.set_xlabel("Standard Gumbel quantile")
        ax.set_ylabel(f"Residual, {INDEX_LABEL[idx][0]}")
        style(ax, "both")
        panel(ax, "ab"[k])
    fig.tight_layout()
    save(fig, P["out"], "FigS1_gev_qq")


# ------------------------------------------------------------------ Fig. 6
def fig06(P, indices, alpha):
    import xarray as xr
    ds = xr.open_dataset(P["proc"] / "gev_grid.nc")
    lon, lat = ds["longitude"].values, ds["latitude"].values
    vals = np.concatenate([ds[f"{i}_rl20_change_pct"].values.ravel() for i in indices])
    vmax = float(np.ceil(np.nanpercentile(np.abs(vals), 98)))
    fig, axes = plt.subplots(1, len(indices), figsize=(DOUBLE, 3.7))
    for k, (ax, idx) in enumerate(zip(np.atleast_1d(axes), indices)):
        ch = ds[f"{idx}_rl20_change_pct"].values
        q = ds[f"{idx}_q_value"].values
        im = ax.pcolormesh(lon, lat, ch, cmap="BrBG", vmin=-vmax, vmax=vmax, shading="nearest")
        sig = np.where(np.isfinite(q), (q < alpha).astype(float), 0.0)
        if sig.any():
            ax.contour(lon, lat, sig, levels=[0.5], colors=INK, linewidths=0.6)
        draw_boundaries(ax, P["prov"], None)
        if P["kab"] is not None:
            P["kab"].boundary.plot(ax=ax, color=MUTED, linewidth=0.2)
        ok = np.isfinite(q)
        w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], q.shape)
        frac = 100 * np.sum(w[ok] * (q[ok] < alpha)) / np.sum(w[ok])
        ax.text(0.97, 0.97, f"{INDEX_LABEL[idx][0]}\nFDR q < {alpha:g}:\n{frac:.0f}% of area",
                transform=ax.transAxes, fontsize=6.5, color=INK, va="top", ha="right",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
        ax.set_xlim(97.0, 100.5)
        ax.set_ylim(0.5, 4.5)
        map_axes(ax)
        panel(ax, "ab"[k])
    cb = fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), orientation="horizontal", fraction=0.05,
                      pad=0.09, aspect=40)
    cb.set_label("Change in 20-year return level per +1 °C ONI (%)")
    cb.outline.set_linewidth(0.4)
    save(fig, P["out"], "Fig06_gev_map")


# ------------------------------------------------------------------ Fig. 7
def fig07(P, alpha):
    kb = pd.read_csv(P["tables"] / "gev_kabupaten.csv")
    kb["name"] = kb["kabupaten"].map(english_name)
    idxs = list(kb["index"].unique())
    order = kb[kb["index"] == idxs[-1]].sort_values("ratio_ln_en")["name"].tolist()
    ypos = {n: i for i, n in enumerate(order)}
    fig, axes = plt.subplots(1, len(idxs), figsize=(DOUBLE, 0.13 * len(order) + 0.9), sharey=True)
    for k, (ax, idx) in enumerate(zip(np.atleast_1d(axes), idxs)):
        d = kb[kb["index"] == idx]
        for _, r in d.iterrows():
            y = ypos[r["name"]]
            sig = r["q_value"] < alpha
            color = BLUE if sig else GRAY
            ax.plot([r["ratio_ci_low"], r["ratio_ci_high"]], [y, y], color=color, linewidth=0.9)
            ax.scatter(r["ratio_ln_en"], y, s=12, color=color, zorder=3)
        ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=0.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=6.3)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlabel(f"{INDEX_LABEL.get(idx, (idx,))[0]}: 20-yr return level, La Niña / El Niño")
        style(ax, "x")
        panel(ax, "ab"[k], x=0.0)
    ax = np.atleast_1d(axes)[-1]
    ax.scatter([], [], s=12, color=BLUE, label=f"FDR q < {alpha:g}")
    ax.scatter([], [], s=12, color=GRAY, label="Not significant")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, P["out"], "Fig07_regencies")


# ------------------------------------------------------------------ Fig. 8
def fig08(P):
    jk = pd.read_csv(P["tables"] / "gev_jackknife.csv")
    tests = pd.read_csv(P["tables"] / "gev_tests.csv")
    jk = jk[jk["period"] == "full"]
    idxs = list(jk["index"].unique())
    fig, axes = plt.subplots(1, len(idxs), figsize=(DOUBLE, 2.6), sharey=True)
    for k, (ax, idx) in enumerate(zip(np.atleast_1d(axes), idxs)):
        d = jk[jk["index"] == idx]
        strong = d["oni_of_dropped"].abs() >= 1.0
        ax.scatter(d.loc[~strong, "oni_of_dropped"], d.loc[~strong, "p_value"], s=9, color=GRAY,
                   label="One year removed")
        ax.scatter(d.loc[strong, "oni_of_dropped"], d.loc[strong, "p_value"], s=16, color=ORANGE,
                   zorder=3, label="Strong ENSO year removed (|ONI| ≥ 1)")
        for _, r in d[strong].iterrows():
            ax.annotate(str(int(r["dropped_year"])), (r["oni_of_dropped"], r["p_value"]), xytext=(3, 2),
                        textcoords="offset points", fontsize=6, color=INK)
        base = tests[(tests["period"] == "full") & (tests["index"] == idx) & (tests["test"] == "ONI")]
        if len(base):
            ax.axhline(base["p_value"].iloc[0], color=BLUE, linewidth=1, label="All 45 years")
        ax.axhline(0.05, color=MUTED, linestyle="--", linewidth=0.6)
        ax.text(0.01, 0.05, "p = 0.05", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=6, color=MUTED)
        ax.set_yscale("log")
        ax.set_xlabel("Annual mean ONI of the removed year (°C)")
        ax.set_title(INDEX_LABEL[idx][0], loc="left", fontsize=8, color=INK)
        style(ax, "both")
        panel(ax, "ab"[k], x=-0.04)
    np.atleast_1d(axes)[0].set_ylabel("LRT p-value for the ONI effect")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, P["out"], "Fig08_jackknife")


# ------------------------------------------------------------------ Fig. S2
def figS2(P):
    sel = pd.read_csv(P["tables"] / "cissa_L_selection.csv")
    fig, ax = plt.subplots(figsize=(SINGLE * 1.4, 2.4))
    for var, color, name in (("prcp_mm", BLUE, "Monthly rainfall"), ("rx1day_mm", ORANGE, "Monthly RX1DAY")):
        d = sel[sel["variable"] == var].sort_values("L")
        if d.empty:
            continue
        ax.plot(d["L"], d["max_wcorr"], color=color, linewidth=1, marker="o", markersize=2.5, label=name)
        pick = d[d["selected"]]
        ax.scatter(pick["L"], pick["max_wcorr"], s=50, facecolors="none", edgecolors=color, linewidths=1.2,
                   zorder=3)
    ax.set_xlabel("Window length L (months)")
    ax.set_ylabel("Max |w-correlation| between groups")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    save(fig, P["out"], "FigS2_window_selection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="nomor gambar, misalnya 3 6 S1")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config()
    gc = cfg.get("gev", {})
    indices = gc.get("indices", ["rx1day", "rx5day"])
    alpha = gc.get("alpha_fdr", 0.10)
    P = {"proc": project_path(cfg["paths"]["processed"]), "ext": project_path(cfg["paths"]["external"]),
         "tables": project_path("outputs/tables"),
         "out": project_path(f"{cfg['paths']['figures']}/paper"),
         "sens": cfg["cissa"]["sensitivity_start_year"], "cfg": cfg}
    P["out"].mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else None

    def run(key, func, *a):
        if want and key not in want:
            return
        try:
            func(*a)
        except FileNotFoundError as e:
            print(f"  Fig {key} dilewati, file belum ada: {Path(e.filename).name}")

    if not want or want & {"1", "6"}:
        P["prov"], P["kab"] = load_boundaries(P["ext"])
    else:
        P["prov"] = P["kab"] = None

    print(f"Menyimpan ke {P['out']}")
    run("1", fig01, P)
    run("2", fig02, P)
    run("3", fig03, P)
    run("4", fig04, P)
    if not want or want & {"5", "S1"}:
        print(f"  (bootstrap {args.n_boot}x untuk pita kepercayaan Fig. 5 ...)")
        cur = gev_curves(P, indices, args.n_boot)
        run("5", fig05, P, cur)
        run("S1", figS1, P, cur)
    run("6", fig06, P, indices, alpha)
    run("7", fig07, P, alpha)
    run("8", fig08, P)
    run("S2", figS2, P)
    print("Selesai.")


if __name__ == "__main__":
    main()
