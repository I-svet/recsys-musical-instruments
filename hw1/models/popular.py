from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from recsys.core import FitSnapshot, Model

from .common import _ClassicModel

class MostPopular(_ClassicModel):
    """Number of distinct positive users per item; rating >= 4 is positive."""

    is_popularity_baseline = True

    def fit(self, snapshot: FitSnapshot, seed: int = 42, callback=None):
        self._remember_snapshot(snapshot)
        self.popularity = np.asarray(snapshot.matrix(positive=True).sum(axis=0)).ravel()
        return self

    def score_users(self, users, items, as_of):
        users, items = self._score_indices(users, items, as_of)
        return np.broadcast_to(self.popularity[items], (len(users), len(items))).copy()
