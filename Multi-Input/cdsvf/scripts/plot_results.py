"""
Best-so-far simple regret, computed with the NOISELESS objective:
    r_t = min_{i <= n_init + t} f(x_i) - f*,   t = 0 is the best initial design.
Curves show mean +/- standard error of log10(r_t) over seeds.

Also prints final log10 regret per facility, a paired Wilcoxon test (SVF-BO vs independent
BO, same seeds), and SVF diagnostics (q, cycle error, share of clipped designs).

Run from hetero-input/cdsvf:
  python scripts/plot_results.py --tag step1_nn
"""
import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

COLORS = {"random": "tab:gray", "indep": "tab:blue", "svf": "tab:orange"}
LABELS = {"random": "Random search", "indep": "Independent BO", "svf": "SVF-BO"}


def log_regret(f, n_init, y_star):
    best = np.minimum.accumulate(f)[n_init - 1:]
    return np.log10(np.maximum(best - y_star, 1e-6))


def mean_se(A):
    mu = A.mean(axis=0)
    se = A.std(axis=0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros_like(mu)
    return mu, se


def summarize(key, runs, out_dir):
    methods = list(runs[0]["methods"].keys())
    d_l = runs[0]["d_l_list"]
    L, n_init = len(d_l), runs[0]["config"]["bo"]["n_init"]
    # R[m]: (seeds, L, budget + 1)
    R = {m: np.array([[log_regret(r["methods"][m]["data"][l]["f"], n_init, r["y_star"])
                       for l in range(L)] for r in runs]) for m in methods}

    n_pan = L + 1
    ncols = min(4, n_pan)
    nrows = int(np.ceil(n_pan / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for i in range(n_pan):
        ax = axes[i]
        for m in methods:
            A = R[m][:, i, :] if i < L else R[m].mean(axis=1)
            mu, se = mean_se(A)
            t = np.arange(len(mu))
            ax.plot(t, mu, color=COLORS.get(m), label=LABELS.get(m, m))
            ax.fill_between(t, mu - se, mu + se, color=COLORS.get(m), alpha=0.2)
        ax.set_title(f"facility {i} (d={d_l[i]})" if i < L else "mean over facilities")
        ax.set_xlabel("BO iteration")
        ax.set_ylabel("log10 simple regret")
        ax.grid(alpha=0.3)
    for j in range(n_pan, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{key}   ({len(runs)} seeds)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{key}.{ext}"), bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== {key}: final log10 regret, mean ± SE over {len(runs)} seeds ===")
    print("facility".ljust(12) + "".join(LABELS.get(m, m).rjust(20) for m in methods)
          + ("   p (SVF vs indep)" if {"svf", "indep"} <= set(methods) else ""))
    for l in range(L + 1):
        name = f"{l} (d={d_l[l]})" if l < L else "mean"
        row = name.ljust(12)
        finals = {}
        for m in methods:
            v = R[m][:, l, -1] if l < L else R[m][:, :, -1].mean(axis=1)
            finals[m] = v
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            row += f"{v.mean():9.2f} ± {se:5.2f}".rjust(20)
        if {"svf", "indep"} <= set(methods) and len(runs) >= 6:
            try:
                row += f"   {wilcoxon(finals['svf'], finals['indep']).pvalue:.3g}"
            except ValueError:
                row += "   n/a"
        print(row)

    if "svf" in methods:
        diags = [r["methods"]["svf"]["diag"] for r in runs]
        cyc = np.array([[[f["cycle_err"] for f in it["facility"]] for it in d["per_iter"]]
                        for d in diags])                       # (seeds, T, L)
        clp = np.array([[[f["clipped"] for f in it["facility"]] for it in d["per_iter"]]
                        for d in diags], dtype=float)
        print(f"SVF q used: {sorted({d['q'] for d in diags})}")
        print("SVF cycle error |E(D(h*)) - h*| per facility:", np.round(cyc.mean(axis=(0, 1)), 3))
        print("SVF share of decoded designs clipped to box:", np.round(clp.mean(axis=(0, 1)), 3))

    secs = {m: np.mean([r["methods"][m]["seconds"] for r in runs]) for m in methods}
    print("mean seconds per run:", {m: round(float(s), 1) for m, s in secs.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="step1")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join("results", args.tag, "*.pkl")))
    if not files:
        raise SystemExit(f"no results in results/{args.tag}/")
    groups = {}
    for fp in files:
        groups.setdefault(re.sub(r"_seed\d+\.pkl$", "", os.path.basename(fp)), []).append(fp)

    out_dir = os.path.join("figures", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    for key, fps in sorted(groups.items()):
        runs = []
        for fp in fps:
            with open(fp, "rb") as f:
                runs.append(pickle.load(f))
        summarize(key, runs, out_dir)
    print(f"\nfigures saved to {out_dir}/")


if __name__ == "__main__":
    main()