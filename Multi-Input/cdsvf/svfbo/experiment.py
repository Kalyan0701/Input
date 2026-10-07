"""
One run = one seed: build the simulator, draw the shared historical data, run each method.

Fairness: every method starts from the SAME initial data, searches the SAME boxes and uses
the SAME GP / EI / candidate code. Each method draws its own noise from its own stream.
"""
import time
from dataclasses import replace

from .baselines import run_independent_bo, run_random
from .simulator import FacilitySimulator
from .utils import set_global_seeds, stream_rng


def make_initial_data(sim, n_init, rng):
    init = []
    for l in range(sim.L):
        X = sim.sample_designs(l, n_init, rng)
        init.append({"X": X, "y": sim.evaluate(l, X, rng), "f": sim.f_true(l, X)})
    return init


def run_one(cfg, seed):
    set_global_seeds(seed)
    try:
        import torch
        torch.set_num_threads(1)  # parallelism comes from running seeds side by side
    except ImportError:
        pass

    sim = FacilitySimulator(cfg.sim, seed)
    init = make_initial_data(sim, cfg.bo.n_init, stream_rng(seed, "init"))
    out = {"config": cfg.to_dict(), "seed": seed, "d_l_list": sim.d_l_list,
           "y_star": sim.y_star, "init": init, "methods": {}}

    for m in cfg.methods:
        t0, rng = time.time(), stream_rng(seed, m)
        if m == "random":
            data, diag = run_random(sim, init, cfg.bo.budget, rng), None
        elif m == "indep":
            data, diag = run_independent_bo(sim, init, cfg.bo, rng, seed), None
        elif m in ("svf", "svf_x"):
            from .svf_bo import run_svf_bo  # imports torch only when SVF is run
            svf_cfg = replace(cfg.svf, proposal="encode") if m == "svf_x" else cfg.svf
            data, diag = run_svf_bo(sim, init, svf_cfg, cfg.bo, rng, seed)
        else:
            raise ValueError(f"unknown method: {m}")
        out["methods"][m] = {"data": data, "diag": diag, "seconds": time.time() - t0}
    return out