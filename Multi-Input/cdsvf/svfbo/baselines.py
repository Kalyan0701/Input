"""
Baselines.

  random : uniform designs in the facility's box (sanity check: BO should beat this).
  indep  : the null model M_o of the proposal, used for BO. Each facility fits a GP on its
           own (x, y) in [0,1]^{d_l} and maximizes EI over the same box SVF-BO searches.

Each returns one dict per facility: X (all designs so far, physical units),
y (noisy observations) and f (noiseless values, for regret only).
"""
import numpy as np

from .gp import fit_gp, make_candidates, propose
from .normalization import BoxScaler


def _copy(init_l):
    return init_l["X"].copy(), init_l["y"].copy(), init_l["f"].copy()


def run_random(sim, init, budget, rng):
    out = []
    for l in range(sim.L):
        X, y, f = _copy(init[l])
        lo, hi = sim.bounds[l]
        X_new = rng.uniform(lo, hi, size=(budget, lo.size))
        out.append({"X": np.vstack([X, X_new]),
                    "y": np.append(y, sim.evaluate(l, X_new, rng)),
                    "f": np.append(f, sim.f_true(l, X_new))})
    return out


def run_independent_bo(sim, init, bo, rng, seed):
    out = []
    for l in range(sim.L):
        X, y, f = _copy(init[l])
        scaler = BoxScaler(*sim.bounds[l])
        d = X.shape[1]
        zeros, ones = np.zeros(d), np.ones(d)
        for t in range(bo.budget):
            X01 = scaler.to_unit(X)
            gp = fit_gp(X01, y, ard=bo.ard, seed=seed + t)
            cands = make_candidates(zeros, ones, X01, y, bo.n_candidates,
                                    bo.local_frac, bo.local_sigma, rng)
            x01_next, _ = propose(gp, cands, y.min(), bo.xi)
            x_next = scaler.from_unit(x01_next)
            X = np.vstack([X, x_next])
            y = np.append(y, sim.evaluate(l, x_next, rng))
            f = np.append(f, sim.f_true(l, x_next))
        out.append({"X": X, "y": y, "f": f})
    return out