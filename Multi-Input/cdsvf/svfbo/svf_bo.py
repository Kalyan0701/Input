"""
SVF-BO, Step 1: the method as written in the proposal (Task 1 + Task 2 + M_cd),
with the bugs fixed. Each BO iteration:

  1. Task 1  fit each facility's encoder on its box-normalized designs:  x01 -> z
  2. Task 2  train SVF on standardized z and shared-standardized y
  3.         map every facility's data to h; standardize h with ONE pooled transform
  4. M_cd    per-facility GP on (h, y); maximize EI over candidate h  ->  h*
  5.         decode h* -> z~ -> x01, clip to the box, run the experiment

Task 3 (Bayes-factor model comparison) is NOT part of Step 1.

Variant (svf_cfg.proposal == "encode", method "svf_x"): identical Tasks 1-2 and GP in h, but
step 5 is replaced by: draw candidate designs in the x-box (as independent BO does), map them
to h with the trained encoders, and score them with the GP in h. No decoder, no clipping.

Diagnostics stored per facility and iteration:
  cycle_err : || E(D(h*)) - h* || in standardized h units. Large values mean the point we
              ran is not the point the GP evaluated.
  clipped   : whether the decoded design left the box and had to be clipped.
"""
import numpy as np
import torch

from .gp import expected_improvement, fit_gp, make_candidates, propose
from .normalization import BoxScaler, Standardizer
from .svf_model import NNAutoencoder, PCAEncoder, SVFNet, train_svf


def choose_q(X01_list, h_dim, var=0.99):
    """
    Task 1 latent size from the initial designs.
    For each facility: the smallest k whose principal components keep `var` of the variance.
    q = max over facilities (no facility loses more than 1 - var), then clipped to
        q >= h_dim          so Task 2 still compresses z -> h,
        q <= min_l d_l - 1  so Task 1 actually compresses x -> z (whiteboard: q < min d_l).
    """
    ks = []
    for X in X01_list:
        s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2
        ks.append(int(np.searchsorted(np.cumsum(s) / s.sum(), var) + 1))
    d_min = min(X.shape[1] for X in X01_list)
    q = int(np.clip(max(ks), h_dim, d_min - 1))
    return q, ks


class SVFBO:
    def __init__(self, sim, svf_cfg, bo_cfg, seed, rng):
        self.sim, self.c, self.bo, self.seed, self.rng = sim, svf_cfg, bo_cfg, seed, rng
        self.scalers = [BoxScaler(*b) for b in sim.bounds]
        self.net = None
        self.q, self.q_per_facility = None, None

    # ------------------------------------------------------------------ setup
    def _new_models(self, X01_list, t):
        c = self.c
        torch.manual_seed(self.seed * 1000 + t)
        if c.task1 == "pca":
            self.task1 = [PCAEncoder(self.q) for _ in X01_list]
        elif c.task1 == "nn":
            self.task1 = [NNAutoencoder(X.shape[1], self.q, c.hidden, c.lr, c.weight_decay)
                          for X in X01_list]
        else:
            raise ValueError(f"unknown task1 encoder: {c.task1}")
        self.net = SVFNet(len(X01_list), self.q, c.h_dim, c.hidden, c.predictor_uses_z)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)

    # ------------------------------------------------------------------ maps
    def _encode(self, x01, l):
        """x01 -> standardized h (Task 1 encoder, z standardizer, u_l, h standardizer)."""
        z_s = self.z_std[l].transform(self.task1[l].encode(x01))
        with torch.no_grad():
            h = self.net.encode(torch.as_tensor(z_s, dtype=torch.float32), l).numpy().astype(float)
        return self.h_std.transform(h)

    def _decode(self, h_s, l):
        """standardized h -> x01 (inverse chain: v_l, z standardizer, Task 1 decoder)."""
        h = self.h_std.inverse(np.atleast_2d(h_s))
        with torch.no_grad():
            z_s = self.net.decode(torch.as_tensor(h, dtype=torch.float32), l).numpy().astype(float)
        return self.task1[l].decode(self.z_std[l].inverse(z_s))

    # ------------------------------------------------------------------ one iteration
    def step(self, data, t):
        c = self.c
        X01 = [self.scalers[l].to_unit(d["X"]) for l, d in enumerate(data)]

        first = self.net is None
        if first:
            if c.q == "auto":
                self.q, self.q_per_facility = choose_q(X01, c.h_dim, c.q_var)
            else:
                self.q = int(c.q)
            # One y transform for ALL facilities, fixed from the initial data.
            self.y_std = Standardizer().fit(np.concatenate([d["y"] for d in data]))
        retrain = first or not c.warm_start
        if retrain:
            self._new_models(X01, t)
        epochs = c.epochs_first if retrain else c.epochs_refit

        # 1. Task 1: x01 -> z, then standardize z per facility
        Z = []
        for l, X in enumerate(X01):
            self.task1[l].fit(X, epochs)
            Z.append(self.task1[l].encode(X))
        self.z_std = [Standardizer().fit(z) for z in Z]
        Zs = [s.transform(z) for s, z in zip(self.z_std, Z)]

        # 2. Task 2: train SVF on standardized z and shared-standardized y
        Ys = [self.y_std.transform(d["y"]) for d in data]
        loss_hist = train_svf(self.net, Zs, Ys, c.lam, epochs, self.opt)

        # 3. h for every facility, standardized with ONE pooled transform
        with torch.no_grad():
            H = [self.net.encode(torch.as_tensor(z, dtype=torch.float32), l).numpy().astype(float)
                 for l, z in enumerate(Zs)]
        self.h_std = Standardizer().fit(np.vstack(H))
        Hs = [self.h_std.transform(h) for h in H]

        # 4-5. M_cd per facility: GP on (h, y), then pick the next design
        proposals, diag = [], []
        for l, d in enumerate(data):
            gp = fit_gp(Hs[l], d["y"], ard=self.bo.ard, seed=self.seed + t)

            if c.proposal == "encode":
                # No decoder: candidates live in the real design box (same generator as
                # independent BO), are mapped to h, and the GP in h scores them.
                d_l = X01[l].shape[1]
                cands = make_candidates(np.zeros(d_l), np.ones(d_l), X01[l], d["y"],
                                        self.bo.n_candidates, self.bo.local_frac,
                                        self.bo.local_sigma, self.rng)
                ei = expected_improvement(gp, self._encode(cands, l), d["y"].min(), self.bo.xi)
                i = int(np.argmax(ei))
                proposals.append(self.scalers[l].from_unit(cands[i]))
                diag.append({"cycle_err": 0.0, "clipped": False, "ei": float(ei[i])})
                continue

            lo, hi = Hs[l].min(axis=0), Hs[l].max(axis=0)
            pad = c.h_box_expansion * np.maximum(hi - lo, 1e-6)
            cands = make_candidates(lo - pad, hi + pad, Hs[l], d["y"], self.bo.n_candidates,
                                    self.bo.local_frac, self.bo.local_sigma, self.rng)
            h_star, ei = propose(gp, cands, d["y"].min(), self.bo.xi)

            x01 = self._decode(h_star, l)
            clipped = bool(np.any((x01 < 0.0) | (x01 > 1.0)))
            x01 = np.clip(x01, 0.0, 1.0)
            cycle_err = float(np.linalg.norm(self._encode(x01, l) - h_star))

            proposals.append(self.scalers[l].from_unit(x01))
            diag.append({"cycle_err": cycle_err, "clipped": clipped, "ei": ei})

        return proposals, {"facility": diag, "svf_loss": loss_hist}


def run_svf_bo(sim, init, svf_cfg, bo_cfg, rng, seed):
    data = [{k: v.copy() for k, v in d.items()} for d in init]
    algo = SVFBO(sim, svf_cfg, bo_cfg, seed, rng)
    per_iter = []
    for t in range(bo_cfg.budget):
        proposals, diag = algo.step(data, t)
        for l, x in enumerate(proposals):
            data[l]["X"] = np.vstack([data[l]["X"], x])
            data[l]["y"] = np.append(data[l]["y"], sim.evaluate(l, x, rng))
            data[l]["f"] = np.append(data[l]["f"], sim.f_true(l, x))
        per_iter.append(diag)
    return data, {"per_iter": per_iter, "q": algo.q, "q_per_facility": algo.q_per_facility}