# SVF-BO vs. independent BO — Step 1

Step 1 runs the advisor's method as written in the proposal (Task 1 + Task 2 + M_cd),
with bugs fixed, against independent BO and random search on the synthetic simulator.
The old code is not imported anywhere; keep it in its own folder.

## Layout

```
hetero-input/cdsvf/
├── pyproject.toml          # makes `svfbo` importable everywhere (pip install -e .)
├── README.md
├── .gitignore              # results/ and figures/ are not committed
├── .vscode/launch.json     # VS Code: "Smoke test" and "Plot smoke results" run configs
├── svfbo/                  # the library
│   ├── config.py           # every setting, saved with every run
│   ├── simulator.py        # facilities, fixed search boxes, noiseless f for regret
│   ├── normalization.py    # x: fixed box -> [0,1]; y and h: ONE shared standardizer
│   ├── gp.py               # GP, EI, candidates: identical for every method
│   ├── baselines.py        # random search, independent BO (M_o)
│   ├── svf_model.py        # Task 1 encoders (NN / PCA), Task 2 SVF network + loss
│   ├── svf_bo.py           # SVF-BO loop, choice of q, decode diagnostics
│   ├── experiment.py       # one seed: shared initial data, then each method
│   └── utils.py            # seeding, random streams, atomic save
├── scripts/
│   ├── run_experiments.py  # parallel, resumable sweeps
│   └── plot_results.py     # regret curves, summary table, Wilcoxon test
├── results/                # created by runs  (results/<tag>/*.pkl)
└── figures/                # created by plots (figures/<tag>/*.png, *.pdf)
```

## Setup (once per machine)

Open `hetero-input/cdsvf` as the folder in VS Code, then in its terminal:

macOS (zsh)
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Windows (PowerShell)
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```
If PowerShell refuses to run Activate.ps1:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

In VS Code: `Ctrl/Cmd+Shift+P` -> "Python: Select Interpreter" -> pick `.venv`.

## Run

```
python scripts/run_experiments.py --smoke --n-jobs 1      # 2 seeds, 3 iterations: checks everything runs
python scripts/plot_results.py --tag smoke

python scripts/run_experiments.py --tag step1_nn  --n-jobs 20              # Task 1 = NN autoencoder
python scripts/run_experiments.py --tag step1_pca --n-jobs 20 --task1 pca  # Task 1 = PCA (small data)
python scripts/plot_results.py --tag step1_nn
python scripts/plot_results.py --tag step1_pca
```

Other useful sweeps:
```
--n-init 5 10 20 40        # data regimes: how the advantage changes with more historical data
--q 5 / --q 9              # sensitivity to the Task 1 latent size
--predictor-uses-z         # ablation: old predictor w([h, z])
--no-warm-start            # ablation: re-initialize networks every iteration (old behaviour)
--ard                      # ARD kernels for every GP instead of isotropic
--delta 1.0                # facilities with different optima (for Task 3 / negative transfer)
```

Finished runs are skipped, so an interrupted sweep can be restarted with the same command.
Use a new `--tag` whenever you change settings, so results never mix.