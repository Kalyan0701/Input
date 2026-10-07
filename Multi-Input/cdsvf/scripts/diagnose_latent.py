"""
Diagnostic (no BO loop): which INPUT SPACE lets a GP predict the objective at unseen designs?

For each seed and each n_init, every facility has n_init historical designs. We then predict
the noiseless objective at held-out designs:
  "on-surface": from the facility's own design distribution (like the historical data)
  "box":        uniform in the facility's search box (where BO actually searches)

GP input spaces (same GP code and settings as the BO methods):
  x_indiv    raw design in [0,1]^d, one GP per facility           (independent BO)
  h_indiv    as-written SVF h, one GP per facility                (SVF-BO; needs torch)
  h_pool     as-written SVF h, one GP on all facilities           (needs torch)
  sup_indiv  supervised encoder h, one GP per facility            (recommended method, encoder only)
  sup_pool   supervised encoder h, one GP on all facilities       (recommended method)
  u_indiv    TRUE latent u = C_l x, one GP per facility           (oracle)
  u_pool     TRUE latent u = C_l x, one pooled GP                 (oracle)

Metrics against f_true:
  top_box_f  mean true f of the box designs the GP ranks best (what BO needs; lower is better;
             random picks give the median f of the box, about 85-99)
  sp_surf / sp_box   Spearman rank correlation on-surface / in the box
Also R^2 of a linear map to the true u, fitted on on-surface designs, tested on box designs,
from: z (Task 1 output), h (as-written SVF), sup (supervised encoder).

Run from hetero-input/cdsvf:
  python scripts/diagnose_latent.py --smoke --n-jobs 1
  python scripts/diagnose_latent.py --tag feas --n-init 10 20 40 80 --n-jobs 16
  python scripts/diagnose_latent.py --tag feas_wellcond --d-l 15 15 15 20 20 20 --n-init 10 20 40 80 --n-jobs 16
  add --skip-svf to leave out the (slow) as-written SVF rows
"""
import argparse
import os
import warnings
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from svfbo.config import BOConfig, SimConfig, SVFConfig
from svfbo.experiment import make_initial_data
from svfbo.gp import fit_gp
from svfbo.normalization import BoxScaler, Standardizer
from svfbo.simulator import FacilitySimulator
from svfbo.supervised_latent import SupervisedLatent
from svfbo.utils import save_pickle_atomic, set_global_seeds, stream_rng

ALL_SPACES = ["x_indiv", "h_indiv", "h_pool", "sup_indiv", "sup_pool", "u_indiv", "u_pool"]


def _rho(a, b):
    """Spearman; a constant prediction carries no ranking information, so score it 0."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a, b).correlation)


def _metrics(pred, truth, n_surface):
    pb, tb = pred[n_surface:], truth[n_surface:]
    k = max(3, len(pb) // 20)
    return {"sp_surf": _rho(pred[:n_surface], truth[:n_surface]),
            "sp_box": _rho(pb, tb),
            "top_box_f": float(np.mean(tb[np.argsort(pb)[:k]])),
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2)))}


def diagnose(seed, n_init, d_l, n_test, svf_cfg, task1, skip_svf):
    warnings.filterwarnings("ignore")
    set_global_seeds(seed)
    sim = FacilitySimulator(SimConfig(d_l_list=d_l), seed)
    init = make_initial_data(sim, n_init, stream_rng(seed, "init"))
    L = sim.L
    scalers = [BoxScaler(*b) for b in sim.bounds]
    X01 = [scalers[l].to_unit(init[l]["X"]) for l in range(L)]
    Y = [init[l]["y"] for l in range(L)]

    rng = np.random.default_rng([seed, 777])
    test, lin_fit = [], []
    for l in range(L):
        lo, hi = sim.bounds[l]
        Xt = np.vstack([sim.sample_designs(l, n_test, rng), rng.uniform(lo, hi, size=(n_test, lo.size))])
        test.append({"X01": scalers[l].to_unit(Xt), "u": Xt @ sim.C[l].T, "f": sim.f_true(l, Xt)})
        Xf = sim.sample_designs(l, 300, rng)
        lin_fit.append({"X01": scalers[l].to_unit(Xf), "u": Xf @ sim.C[l].T})

    # feature maps: space -> function(x01, l)
    maps = {"x": lambda x, l: x}
    u_std = Standardizer().fit(np.vstack([(scalers[l].from_unit(X01[l])) @ sim.C[l].T for l in range(L)]))
    maps["u"] = lambda x, l: u_std.transform(scalers[l].from_unit(x) @ sim.C[l].T)

    sup = SupervisedLatent([X.shape[1] for X in X01], h_dim=svf_cfg.h_dim, seed=seed).fit(X01, Y)
    maps["sup"] = sup.encode

    q = None
    if not skip_svf:
        import torch
        torch.set_num_threads(1)
        from svfbo.svf_bo import SVFBO
        algo = SVFBO(sim, replace(svf_cfg, task1=task1), BOConfig(n_init=n_init, n_candidates=200),
                     seed, stream_rng(seed, "svf"))
        algo.step([{k: v.copy() for k, v in d.items()} for d in init], 0)  # trains Tasks 1-2
        maps["h"] = algo._encode
        maps["z"] = lambda x, l: algo.z_std[l].transform(algo.task1[l].encode(x))
        q = algo.q

    out = {}
    for space in ("x", "h", "sup", "u"):
        if space not in maps:
            continue
        F = maps[space]
        out[f"{space}_indiv"] = []
        for l in range(L):
            gp = fit_gp(F(X01[l], l), Y[l], ard=False, seed=seed)
            out[f"{space}_indiv"].append(_metrics(gp.predict(F(test[l]["X01"], l)), test[l]["f"], n_test))
        if space != "x":
            gp = fit_gp(np.vstack([F(X01[l], l) for l in range(L)]), np.concatenate(Y), ard=False, seed=seed)
            out[f"{space}_pool"] = [_metrics(gp.predict(F(test[l]["X01"], l)), test[l]["f"], n_test)
                                    for l in range(L)]

    r2 = {}
    for space in ("z", "h", "sup"):
        if space not in maps:
            continue
        F = maps[space]
        r2[space] = []
        for l in range(L):
            reg = LinearRegression().fit(F(lin_fit[l]["X01"], l), lin_fit[l]["u"])
            Xb, ub = test[l]["X01"][n_test:], test[l]["u"][n_test:]
            r2[space].append(r2_score(ub, reg.predict(F(Xb, l)), multioutput="variance_weighted"))
    return {"seed": seed, "n_init": n_init, "d_l_list": sim.d_l_list, "q": q, "metrics": out, "r2_box": r2}


def report(results, n_init):
    rs = [r for r in results if r["n_init"] == n_init]
    d_l = rs[0]["d_l_list"]
    groups = {"all": list(range(len(d_l)))}
    for d in sorted(set(d_l)):
        groups[f"d={d}"] = [l for l, dd in enumerate(d_l) if dd == d]
    cols = [("top_box_f", g) for g in groups] + [("sp_box", g) for g in groups if g != "all"]

    def cell(vals):
        v = np.array(vals)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return f"{v.mean():6.2f}±{se:4.2f}".rjust(14)

    print(f"\n=== n_init = {n_init} per facility, {len(rs)} seeds (mean ± SE) ===")
    print("space".ljust(10) + "".join(f"{k}[{g}]".rjust(14) for k, g in cols))
    for sp in ALL_SPACES:
        if sp not in rs[0]["metrics"]:
            continue
        row = sp.ljust(10)
        for k, g in cols:
            row += cell([np.mean([r["metrics"][sp][l][k] for l in groups[g]]) for r in rs])
        print(row)
    for space, name in [("z", "z (Task 1)"), ("h", "h (SVF)"), ("sup", "sup (supervised)")]:
        if space in rs[0]["r2_box"]:
            vals = [np.mean(r["r2_box"][space]) for r in rs]
            print(f"R2 linear map to true u, box designs, from {name}:".ljust(52) + cell(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="feas")
    p.add_argument("--task1", choices=["nn", "pca"], default="pca")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-init", type=int, nargs="+", default=[10])
    p.add_argument("--d-l", type=int, nargs="+", default=[10, 10, 10, 20, 20, 20])
    p.add_argument("--n-test", type=int, default=200, help="held-out designs per facility, per kind")
    p.add_argument("--skip-svf", action="store_true", help="leave out the as-written SVF rows")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    svf_cfg = SVFConfig()
    if args.smoke:
        args.seeds, args.n_test, args.tag = 2, 40, "smoke_diag"
        svf_cfg = replace(svf_cfg, epochs_first=100)

    jobs = [(s, n) for n in args.n_init for s in range(args.seeds)]
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(diagnose)(s, n, args.d_l, args.n_test, svf_cfg, args.task1, args.skip_svf)
        for s, n in jobs)

    os.makedirs("results", exist_ok=True)
    save_pickle_atomic(results, os.path.join("results", f"diag_{args.tag}.pkl"))
    print(f"\nfacilities d = {args.d_l}, Task 1 (SVF rows) = {args.task1}")
    for n in args.n_init:
        report(results, n)


if __name__ == "__main__":
    main()