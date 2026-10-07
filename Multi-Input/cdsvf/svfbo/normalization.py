"""
Normalization, and why each quantity is scaled the way it is.

  x : per facility, mapped to [0, 1] with the facility's FIXED box (BoxScaler).
      Facilities have different parameters, so each needs its own scaler. The box never
      moves as data arrives, so the search space is the same at every iteration.
  y : ONE Standardizer shared by all facilities, fit once on the pooled initial data.
      The hypothesis is "same h => same y", so a given physical y must map to the same
      number in every facility. Per-facility scaling would break that.
  z : per facility. Task 1 coordinates are facility-specific anyway.
  h : ONE Standardizer shared by all facilities, because h is meant to be a common space.
"""
import numpy as np


class BoxScaler:
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.span = np.asarray(hi, dtype=float) - self.lo

    def to_unit(self, x):
        return (np.atleast_2d(x) - self.lo) / self.span

    def from_unit(self, x01):
        return np.atleast_2d(x01) * self.span + self.lo


class Standardizer:
    def __init__(self, min_std=1e-8):
        self.min_std = min_std
        self.mean = None
        self.std = None

    def fit(self, a):
        a = np.asarray(a, dtype=float)
        self.mean = a.mean(axis=0)
        # Floor on std: a constant column (e.g. a zero-padded PCA direction) maps to 0, not NaN.
        self.std = np.maximum(a.std(axis=0), self.min_std)
        return self

    def transform(self, a):
        return (np.asarray(a, dtype=float) - self.mean) / self.std

    def inverse(self, a):
        return np.asarray(a, dtype=float) * self.std + self.mean