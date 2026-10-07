import os
import pickle
import random

import numpy as np

# One independent random stream per purpose, so that e.g. adding a method
# never changes the initial data or another method's noise.
STREAMS = {"init": 0, "random": 1, "indep": 2, "svf": 3, "svf_x": 4}


def stream_rng(seed, name):
    """Reproducible generator for (seed, purpose)."""
    return np.random.default_rng([seed, STREAMS[name]])


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_pickle_atomic(obj, path):
    """Write to a temp file, then rename: a crash never leaves a half-written result."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)