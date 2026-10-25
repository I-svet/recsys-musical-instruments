from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from recsys.core import FitSnapshot, Model

from .common import _ClassicModel, _indices

class ItemKNN(_ClassicModel):
    """Cosine similarity with overlap shrinkage and incoming top-k neighbors.

For each target column j, retain k largest s_ij. Equal similarities are
resolved by the source item ID. Scores are X @ W, so pruning by columns is
intentional and can produce an asymmetric graph.
"""

    def __init__(self, k: int = 100, gamma: float = 10.0):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.k = int(k)
        self.gamma = float(gamma)

    def fit(self, snapshot: FitSnapshot, seed: int = 42, callback=None):
        self._remember_snapshot(snapshot)
        self.history = snapshot.matrix(positive=True)
        counts = np.asarray(self.history.sum(axis=0)).ravel()
        overlaps = (self.history.T @ self.history).tocsc()
        sources, targets, values = [], [], []
        for target in range(self.n_items):
            start, end = overlaps.indptr[target:target + 2]
            source = overlaps.indices[start:end]
            overlap = overlaps.data[start:end]
            keep = (source != target) & (overlap > 0) & (counts[source] > 0)
            source, overlap = source[keep], overlap[keep]
            if counts[target] == 0 or len(source) == 0:
                continue
            similarity = (overlap / np.sqrt(counts[source] * counts[target])
                          * overlap / (overlap + self.gamma))
            chosen = np.lexsort((source, -similarity))[:self.k]
            sources.extend(source[chosen])
            targets.extend([target] * len(chosen))
            values.extend(similarity[chosen])
        self.weights = sparse.csr_matrix((values, (sources, targets)),
                                         shape=(self.n_items, self.n_items),
                                         dtype=np.float64)
        self.weights.sort_indices()
        return self

    def score_users(self, users, items, as_of):
        users, items = self._score_indices(users, items, as_of)
        if np.array_equal(items, np.arange(self.n_items)):
            return (self.history[users] @ self.weights).toarray()
        return (self.history[users] @ self.weights[:, items]).toarray()

    def has_signal(self, users):
        users = _indices(users, self.n_users, "users")
        return np.diff(self.history.indptr)[users] > 0
