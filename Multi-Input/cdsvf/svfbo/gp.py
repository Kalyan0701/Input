"""
GP surrogate, expected improvement and candidate generation.

Every method calls these same functions with the same settings, so differences between
methods come from WHERE the GP lives (x-space vs h-space), not from how it is fitted or
how the acquisition function is optimized.
"""
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def fit_gp(X, y, ard=True, seed=0):
    """Matern-5/2 (ARD or isotropic) + learned noise. normalize_y standardizes y internally."""
    length_scale = np.ones(X.shape[1]) if ard else 1.0
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-4, (1e-8, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, np.asarray(y, dtype=float).ravel())
    return gp


def expected_improvement(gp, X, y_best, xi=0.01):
    """EI for MINIMIZATION: E[max(y_best - xi - f(x), 0)]."""
    mu, sd = gp.predict(X, return_std=True)
    sd = np.maximum(sd, 1e-12)
    imp = y_best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def make_candidates(lo, hi, X_obs, y_obs, n, local_frac, local_sigma, rng):
    """
    Uniform candidates in [lo, hi] plus Gaussian perturbations around the best observed points.
    Uniform sampling alone is weak in 10-20 dimensions; the local part makes acquisition
    optimization reasonable, and it is applied identically to every method.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    d = lo.size
    n_local = int(n * local_frac)
    cands = rng.uniform(lo, hi, size=(n - n_local, d))
    if n_local > 0:
        k = min(5, len(y_obs))
        best = np.asarray(X_obs)[np.argsort(y_obs)[:k]]
        centers = best[rng.integers(0, k, size=n_local)]
        local = centers + local_sigma * (hi - lo) * rng.standard_normal((n_local, d))
        cands = np.vstack([cands, np.clip(local, lo, hi)])
    return cands


def propose(gp, candidates, y_best, xi):
    ei = expected_improvement(gp, candidates, y_best, xi)
    i = int(np.argmax(ei))
    return candidates[i], float(ei[i])