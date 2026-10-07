"""
All experiment settings in one place.

Every run stores ExperimentConfig.to_dict() next to its results, so any figure can be
traced back to the exact settings that produced it.
"""
from dataclasses import asdict, dataclass, field
from typing import List, Tuple, Union


@dataclass
class SimConfig:
    d_l_list: List[int] = field(default_factory=lambda: [10, 10, 10, 20, 20, 20])
    d_h: int = 3              # latent that GENERATES the designs (simulator-internal)
    r: int = 5                # latent the OBJECTIVE depends on: u = C_l x in R^r (the "true h")
    rho: float = 0.5          # similarity of the facilities' x-coordinates (not objective relatedness)
    delta: float = 0.0        # objective relatedness: 0 = identical optimum in u for every facility
    sigma_x: float = 0.01     # noise on historical designs
    sigma_y: float = 0.01     # observation noise on y
    ridge_lambda: float = 1e-3
    n_ref: int = 10_000       # reference designs used once to fix each facility's search box
    box_quantiles: Tuple[float, float] = (0.005, 0.995)


@dataclass
class BOConfig:
    n_init: int = 10          # historical experiments per facility
    budget: int = 30          # BO iterations (one new experiment per facility per iteration)
    n_candidates: int = 5000  # candidates scored by EI
    local_frac: float = 0.5   # share of candidates drawn around the current best points
    local_sigma: float = 0.1  # size of those perturbations, as a fraction of the box width
    xi: float = 0.01          # EI exploration margin, in y units
    # One kernel family for every GP in every method. Isotropic is the default because it is
    # the STRONGER baseline: with 10-40 points in 10-20 dims, ARD length scales overfit and
    # independent BO then proposes mostly plateau designs (checked on seed 0: median proposal
    # f = 94-100 with ARD vs 35-50 isotropic on the d=10 facilities). Use --ard to compare.
    ard: bool = False


@dataclass
class SVFConfig:
    task1: str = "nn"         # "nn": Task 1 NN autoencoder (proposal);  "pca": linear, small-data
    q: Union[int, str] = "auto"   # Task 1 latent size; "auto" -> svf_bo.choose_q
    q_var: float = 0.99       # variance kept by the "auto" rule
    h_dim: int = 5            # shared latent size (matches r in the simulator)
    hidden: int = 32          # hidden width of every MLP
    lam: float = 1.0          # weight of the prediction loss in Eq. (4); z and y are standardized
    predictor_uses_z: bool = False  # True = w([h, z]) as in the old code (ablation only)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_first: int = 3000  # training epochs at the first BO iteration
    epochs_refit: int = 500   # epochs at later iterations (models are warm-started)
    warm_start: bool = True   # False = re-initialize every iteration (old behaviour)
    h_box_expansion: float = 0.1  # candidate box = observed h range widened by 10% per side


@dataclass
class ExperimentConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    svf: SVFConfig = field(default_factory=SVFConfig)
    methods: List[str] = field(default_factory=lambda: ["random", "indep", "svf"])

    def to_dict(self):
        return asdict(self)