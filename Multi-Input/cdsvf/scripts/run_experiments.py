"""
Run the comparison over seeds (and optionally several n_init), in parallel and resumable.
Finished runs are skipped, so an interrupted sweep can simply be restarted.

Run from hetero-input/cdsvf:
  python scripts/run_experiments.py --smoke --n-jobs 1                  # ~1-2 min check
  python scripts/run_experiments.py --tag step1_nn --n-jobs 20          # 20 seeds, NN Task 1
  python scripts/run_experiments.py --tag step1_pca --task1 pca --n-jobs 20
  python scripts/run_experiments.py --tag nsweep --n-init 5 10 20 40 --n-jobs 20
  python scripts/run_experiments.py --tag step1_pca_x --task1 pca --methods random indep svf svf_x --n-jobs 16
"""
import argparse
import os
import shutil
import traceback
from dataclasses import replace

from joblib import Parallel, delayed

from svfbo.config import BOConfig, ExperimentConfig, SimConfig, SVFConfig
from svfbo.experiment import run_one
from svfbo.utils import save_pickle_atomic


def build_config(args, n_init):
    sim = SimConfig(d_l_list=args.d_l, delta=args.delta)
    bo = BOConfig(n_init=n_init, budget=args.budget, ard=args.ard)
    q = args.q if args.q == "auto" else int(args.q)
    svf = SVFConfig(task1=args.task1, q=q, lam=args.lam, predictor_uses_z=args.predictor_uses_z,
                    warm_start=not args.no_warm_start)
    if args.smoke:
        bo = replace(bo, budget=3, n_candidates=500)
        svf = replace(svf, epochs_first=200, epochs_refit=50)
    return ExperimentConfig(sim=sim, bo=bo, svf=svf, methods=args.methods)


def job(cfg, seed, path):
    if os.path.exists(path):
        return "skipped"
    try:
        save_pickle_atomic(run_one(cfg, seed), path)
        return "done"
    except Exception:
        print(f"\n--- error in {path} ---")
        traceback.print_exc()
        return "error"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="step1", help="results go to results/<tag>/")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-init", type=int, nargs="+", default=[10])
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--d-l", type=int, nargs="+", default=[10, 10, 10, 20, 20, 20])
    p.add_argument("--methods", nargs="+", default=["random", "indep", "svf"])
    p.add_argument("--task1", choices=["nn", "pca"], default="nn")
    p.add_argument("--q", default="auto", help='"auto" or an integer')
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.0, help="facility relatedness knob")
    p.add_argument("--predictor-uses-z", action="store_true", help="ablation: w([h, z])")
    p.add_argument("--no-warm-start", action="store_true", help="re-initialize every iteration")
    p.add_argument("--ard", action="store_true", help="ARD kernels for every GP (default: isotropic)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--smoke", action="store_true", help="2 seeds, 3 iterations, short training")
    args = p.parse_args()

    if args.smoke:
        args.seeds, args.tag = 2, "smoke"
        shutil.rmtree(os.path.join("results", "smoke"), ignore_errors=True)

    jobs = []
    for n_init in args.n_init:
        cfg = build_config(args, n_init)
        name = f"d{'-'.join(map(str, args.d_l))}_N{n_init}_B{cfg.bo.budget}"
        for seed in range(args.seeds):
            jobs.append((cfg, seed, os.path.join("results", args.tag, f"{name}_seed{seed}.pkl")))

    print(f"{len(jobs)} runs -> results/{args.tag}/")
    status = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(job)(*j) for j in jobs)
    print({s: status.count(s) for s in ("done", "skipped", "error")})


if __name__ == "__main__":
    main()