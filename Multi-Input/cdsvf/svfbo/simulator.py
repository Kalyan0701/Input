"""
Synthetic heterogeneous-facility simulator (same math as new_simulator.py, rewritten).

Facility l, with d_l parameters:
    historical designs:  h ~ N(0, I_{d_h}),   x = A_l phi(h) + eps,   phi(h) = [h, tanh(h), h^2]
    objective:           u = C_l x  (in R^r),  f_l(x) = q(u - c_l),
                         q(v) = 100 * (1 - exp(-||v||^2 / 10))
    where                C_l = B^T (A_l^T A_l + lambda I)^{-1} A_l^T.

So every facility's objective factors through a linear map to a shared r-dim latent u:
the SVF hypothesis holds exactly, with E_l* = C_l.

Differences from the original:
  * The structure (B, A_l, C_l, c_l, boxes) uses its own RNG and depends only on `seed`.
  * Sampling and noise take an explicit rng, so each method has its own reproducible stream.
  * f_true() is noiseless; it is used only to measure regret.
  * Each facility has a FIXED search box: central 99% range of many reference designs.
    Historical designs are clipped into it, and every method searches the same box.
  * c_l = delta * e_l shifts each facility's optimum in u. delta = 0 reproduces the original
    objective (all optima at u = 0). ||u|| on typical designs is about 2, so delta = 1 moves
    the optima by roughly one typical design-spread.
"""
import warnings

import numpy as np


class FacilitySimulator:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.L = len(cfg.d_l_list)
        self.d_l_list = list(cfg.d_l_list)
        self.d_phi = 3 * cfg.d_h
        rng = np.random.default_rng([seed, 999])  # structure stream

        self.B = rng.standard_normal((self.d_phi, cfg.r)) / np.sqrt(self.d_phi)
        A_shared_full = rng.standard_normal((max(self.d_l_list), self.d_phi))

        self.A, self.C, self.centers = [], [], []
        for d_l in self.d_l_list:
            if d_l < self.d_phi:
                warnings.warn(
                    f"d_l={d_l} < 3*d_h={self.d_phi}: A_l^T A_l is singular, C_l has rank "
                    f"{min(d_l, cfg.r)}, so this facility only sees a slice of the objective.")
            A_private = rng.standard_normal((d_l, self.d_phi))
            # sqrt-weights keep each entry's variance at 1 for any rho
            A_l = np.sqrt(cfg.rho) * A_shared_full[:d_l] + np.sqrt(1 - cfg.rho) * A_private
            reg = cfg.ridge_lambda * np.eye(self.d_phi)
            C_l = self.B.T @ np.linalg.solve(A_l.T @ A_l + reg, A_l.T)  # (r, d_l)
            e_l = rng.standard_normal(cfg.r)  # drawn even if delta = 0, so delta only rescales
            self.A.append(A_l)
            self.C.append(C_l)
            self.centers.append(cfg.delta * e_l)

        self.bounds = []
        q_lo, q_hi = cfg.box_quantiles
        for l in range(self.L):
            R = self._raw_designs(l, cfg.n_ref, rng)
            self.bounds.append((np.quantile(R, q_lo, axis=0), np.quantile(R, q_hi, axis=0)))

        # Global minimum value of q is 0. Exact for delta = 0 (x = 0 attains it);
        # for delta > 0 it is a lower bound if C_l x = c_l is not reachable inside the box.
        self.y_star = 0.0

    @staticmethod
    def phi(h):
        return np.concatenate([h, np.tanh(h), h ** 2], axis=1)

    def _raw_designs(self, l, n, rng):
        h = rng.standard_normal((n, self.cfg.d_h))
        x = self.phi(h) @ self.A[l].T
        return x + self.cfg.sigma_x * rng.standard_normal(x.shape)

    def sample_designs(self, l, n, rng):
        """Historical experiments: designs from the facility's design distribution, inside its box."""
        lo, hi = self.bounds[l]
        return np.clip(self._raw_designs(l, n, rng), lo, hi)

    def f_true(self, l, x):
        """Noiseless objective (lower is better)."""
        x = np.atleast_2d(x)
        v = x @ self.C[l].T - self.centers[l]
        # A larger denominator than 10 widens the bowl, so the plateau at 100 is reached later.
        return 100.0 * (1.0 - np.exp(-np.sum(v ** 2, axis=1) / 10.0))

    def evaluate(self, l, x, rng):
        """What an experiment returns: f plus Gaussian noise."""
        f = self.f_true(l, x)
        return f + self.cfg.sigma_y * rng.standard_normal(f.shape)