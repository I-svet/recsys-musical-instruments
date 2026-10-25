from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from recsys.core import FitSnapshot, Model

from .common import _ClassicModel, _indices

class EASE(_ClassicModel):
    """Constrained ridge reconstruction of a binary positive-history matrix.

min ||X-XB||_F^2 + reg ||B||_F^2, subject to diag(B)=0.
The dense coefficient matrix is exact; negative coefficients are retained.
"""

    def __init__(self, reg: float = 100.0):
        if not np.isfinite(reg) or reg <= 0:
            raise ValueError("reg must be finite and positive")
        self.reg = float(reg)

    def fit(self, snapshot: FitSnapshot, seed: int = 42, callback=None):
        self._remember_snapshot(snapshot)
        self.history = snapshot.matrix(positive=True)
        gram = (self.history.T @ self.history).toarray(order="F")
        diagonal = np.diag_indices(self.n_items)
        gram[diagonal] += self.reg
        factor = linalg.cho_factor(gram, lower=True, overwrite_a=True,
                                  check_finite=False)
        inverse = linalg.cho_solve(factor, np.eye(self.n_items, order="F"),
                                   overwrite_b=True, check_finite=False)
        inverse /= -inverse.diagonal().copy()[None, :]
        inverse[diagonal] = 0.0
        self.weights = inverse
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Non-finite EASE coefficients")
        return self

    def score_users(self, users, items, as_of):
        users, items = self._score_indices(users, items, as_of)
        # Avoid copying the full dense coefficient matrix for every user batch.
        weights = (self.weights if np.array_equal(items, np.arange(self.n_items))
                   else self.weights[:, items])
        return np.asarray(self.history[users] @ weights)

    def has_signal(self, users):
        users = _indices(users, self.n_users, "users")
        return np.diff(self.history.indptr)[users] > 0
