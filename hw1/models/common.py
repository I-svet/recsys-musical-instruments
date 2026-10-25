from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from recsys.core import FitSnapshot, Model

def _indices(values: np.ndarray, size: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(f"{name} outside the fitted mapping")
    return values.astype(np.int64, copy=False)

class _ClassicModel(Model):
    def _remember_snapshot(self, snapshot: FitSnapshot) -> None:
        self.bind_snapshot(snapshot)
        self.n_users = snapshot.n_users
        self.n_items = snapshot.n_items
        self.as_of = snapshot.as_of

    def _score_indices(self, users, items, as_of):
        if not hasattr(self, "n_users"):
            raise RuntimeError("Call fit before scoring")
        if as_of < self.as_of:
            raise ValueError("Cannot score a time before the fitted snapshot")
        return (_indices(users, self.n_users, "users"),
                _indices(items, self.n_items, "items"))
