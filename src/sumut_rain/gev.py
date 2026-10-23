"""
GEV (Generalized Extreme Value) non-stasioner untuk maksimum tahunan.

Model:  Y_t ~ GEV(mu_t, sigma, xi),  mu_t = b0 + b1*x1_t + b2*x2_t + ...
        sigma dan xi konstan. xi > 0 = ekor berat (Frechet), xi = 0 = Gumbel.

Estimasi memakai GMLE (Martins & Stedinger 2000, Water Resour. Res. 36, 737-744):
likelihood dikali prior Beta(6, 9) pada xi di (-0.5, 0.5) (rata-rata xi = 0.10).
Untuk sampel pendek (n ~ 45 tahun) MLE biasa sering memberi xi yang tidak masuk akal;
GMLE menstabilkannya tanpa memaksa nilai tertentu.

AIC, BIC, dan uji rasio likelihood (LRT) memakai log-likelihood TANPA prior,
dihitung di titik estimasi GMLE.
"""

import numpy as np
from scipy import optimize, stats

PRIOR_P, PRIOR_Q = 6.0, 9.0
XI_EPS = 1e-6


def _loglik_terms(y, mu, sigma, xi):
    z = (y - mu) / sigma
    if abs(xi) < XI_EPS:                                   # Gumbel
        return -np.log(sigma) - z - np.exp(-z)
    t = 1 + xi * z
    if np.any(t <= 0):
        return None
    return -np.log(sigma) - (1 + 1 / xi) * np.log(t) - t ** (-1 / xi)


def loglik(params, y, X):
    k = X.shape[1]
    mu = X @ params[:k]
    sigma, xi = np.exp(params[k]), params[k + 1]
    ll = _loglik_terms(y, mu, sigma, xi)
    return -np.inf if ll is None else float(ll.sum())


def log_prior(xi):
    if not -0.5 < xi < 0.5:
        return -np.inf
    return (PRIOR_P - 1) * np.log(0.5 - xi) + (PRIOR_Q - 1) * np.log(0.5 + xi)


def _start(y, X):
    """Awal dari momen Gumbel + regresi OLS untuk koefisien lokasi."""
    s = np.sqrt(6) * np.std(y) / np.pi
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    b[0] -= 0.5772 * s
    return np.concatenate([b, [np.log(s), 0.1]])


def fit(y, X=None, start=None) -> dict:
    """GMLE. X = matriks desain lokasi (kolom pertama = 1). Tanpa X = stasioner."""
    y = np.asarray(y, float)
    X = np.ones((len(y), 1)) if X is None else np.asarray(X, float)
    k = X.shape[1]

    def obj(p):
        lp = log_prior(p[k + 1])
        if not np.isfinite(lp):
            return 1e10
        ll = loglik(p, y, X)
        return 1e10 if not np.isfinite(ll) else -(ll + lp)

    p0 = _start(y, X) if start is None else np.asarray(start, float)
    best = None
    for p in (p0, _start(y, X)):
        r = optimize.minimize(obj, p, method="Nelder-Mead",
                              options={"maxiter": 4000 * (k + 2), "xatol": 1e-6, "fatol": 1e-8})
        r = optimize.minimize(obj, r.x, method="Nelder-Mead",
                              options={"maxiter": 4000 * (k + 2), "xatol": 1e-7, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    ll = loglik(p, y, X)
    n, npar = len(y), k + 2
    return {"beta": p[:k], "sigma": float(np.exp(p[k])), "xi": float(p[k + 1]),
            "params": p, "loglik": ll, "aic": 2 * npar - 2 * ll,
            "bic": npar * np.log(n) - 2 * ll, "n": n, "k": k, "ok": bool(np.isfinite(ll))}


def lrt(f_small: dict, f_big: dict) -> tuple[float, float]:
    """Uji rasio likelihood model bersarang. Kembalikan (statistik D, p-value chi2)."""
    D = max(0.0, 2 * (f_big["loglik"] - f_small["loglik"]))
    df = f_big["k"] - f_small["k"]
    return D, float(stats.chi2.sf(D, df))


def return_level(mu, sigma, xi, T):
    """Tingkat ulang T-tahun: kuantil 1 - 1/T."""
    yp = -np.log(1 - 1 / np.asarray(T, float))
    if abs(xi) < XI_EPS:
        return mu - sigma * np.log(yp)
    return mu + sigma / xi * (yp ** (-xi) - 1)


def simulate(mu, sigma, xi, rng):
    """Sampel GEV (scipy memakai c = -xi)."""
    return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng)


def gumbel_residuals(y, mu, sigma, xi):
    """Transformasi ke Gumbel standar untuk diagnostik QQ model non-stasioner."""
    z = (np.asarray(y) - mu) / sigma
    if abs(xi) < XI_EPS:
        return z
    return np.log(1 + xi * z) / xi


def bootstrap(y, X, fitted: dict, func, n_boot: int = 500, seed: int = 0):
    """Bootstrap parametrik: simulasi dari model terpasang, pasang ulang, hitung func(fit)."""
    rng = np.random.default_rng(seed)
    mu = X @ fitted["beta"]
    out = []
    for _ in range(n_boot):
        yb = simulate(mu, fitted["sigma"], fitted["xi"], rng)
        fb = fit(yb, X, start=fitted["params"])
        if fb["ok"]:
            out.append(func(fb))
    return np.array(out)
