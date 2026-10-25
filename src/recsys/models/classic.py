"""Transparent NumPy/SciPy implementations of four classical recommenders.

The classes return unmasked scores. Candidate eligibility, seen-item filtering and
the common popularity fallback belong to the recommendation engine.
"""
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


def _ridge_blocks(matrix: sparse.csr_matrix, rank: int):
    """Group rows by power-of-two degree to bound padding during batched solves."""
    degrees = np.diff(matrix.indptr)
    positive_rows = np.flatnonzero(degrees)
    if len(positive_rows) == 0:
        return []
    widths = 2 ** np.ceil(np.log2(degrees[positive_rows])).astype(np.int64)
    blocks = []
    for width in np.unique(widths):
        rows = positive_rows[widths == width]
        # Bound the temporary padded design matrix to about 16 MB.
        batch_size = min(512, max(1, 2_000_000 // (int(width) * (rank + 1))))
        for offset in range(0, len(rows), batch_size):
            entities = rows[offset:offset + batch_size]
            mask = np.arange(width)[None, :] < degrees[entities, None]
            positions = matrix.indptr[entities, None] + np.arange(width)[None, :]
            positions = np.where(mask, positions, 0)
            neighbors = matrix.indices[positions]
            ratings = matrix.data[positions]
            blocks.append((entities, neighbors, ratings, mask))
    return blocks


def _ridge_sweep(blocks, other_factors, other_bias, factors, bias,
                 global_mean, reg, bias_reg):
    factors.fill(0.0)
    bias.fill(0.0)
    dimensions = factors.shape[1] + 1
    penalties = np.full(dimensions, reg, dtype=np.float64)
    penalties[0] = bias_reg
    diagonal = np.arange(dimensions)
    for entities, neighbors, ratings, mask in blocks:
        design = np.empty((*neighbors.shape, dimensions), dtype=np.float64)
        design[..., 0] = mask
        design[..., 1:] = other_factors[neighbors] * mask[..., None]
        target = (ratings - global_mean - other_bias[neighbors]) * mask
        gram = design.transpose(0, 2, 1) @ design
        gram[:, diagonal, diagonal] += penalties
        rhs = np.einsum("bdr,bd->br", design, target)
        solution = np.linalg.solve(gram, rhs[..., None])[..., 0]
        bias[entities] = solution[:, 0]
        factors[entities] = solution[:, 1:]


class ExplicitALS(_ClassicModel):
    """Observed-rating sum-MSE ALS, jointly updating each entity bias and factors.

The mean is fixed at the observed fit-set mean. Missing ratings are absent
from the objective. reg and bias_reg penalize the unnormalized sum loss.
"""

    def __init__(self, rank: int = 32, reg: float = 10.0,
                 bias_reg: float = 10.0, epochs: int = 20):
        if not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(reg) or reg <= 0 or not np.isfinite(bias_reg) or bias_reg <= 0:
            raise ValueError("reg and bias_reg must be finite and positive")
        self.rank, self.epochs = int(rank), int(epochs)
        self.reg, self.bias_reg = float(reg), float(bias_reg)

    def fit(self, snapshot: FitSnapshot, seed: int = 42, callback=None):
        self._remember_snapshot(snapshot)
        frame = snapshot.interactions
        users = frame.user.to_numpy(dtype=np.int64)
        items = frame.item.to_numpy(dtype=np.int64)
        ratings = frame.rating.to_numpy(dtype=np.float64)
        observed = sparse.csr_matrix((ratings, (users, items)),
                                     shape=(self.n_users, self.n_items))
        user_blocks = _ridge_blocks(observed, self.rank)
        item_blocks = _ridge_blocks(observed.T.tocsr(), self.rank)
        rng = np.random.default_rng(seed)
        scale = 0.1 / np.sqrt(self.rank)
        self.user_factors = rng.normal(0, scale, (self.n_users, self.rank))
        self.item_factors = rng.normal(0, scale, (self.n_items, self.rank))
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)
        self.global_mean = float(ratings.mean())
        self.loss_history = [self._objective(users, items, ratings)]
        self.n_iter_ = 0
        for iteration in range(1, self.epochs + 1):
            _ridge_sweep(user_blocks, self.item_factors, self.item_bias,
                         self.user_factors, self.user_bias, self.global_mean,
                         self.reg, self.bias_reg)
            _ridge_sweep(item_blocks, self.user_factors, self.user_bias,
                         self.item_factors, self.item_bias, self.global_mean,
                         self.reg, self.bias_reg)
            loss = self._objective(users, items, ratings)
            if not np.isfinite(loss):
                raise FloatingPointError("Non-finite ALS objective")
            self.loss_history.append(loss)
            self.n_iter_ = iteration
            if callback is not None and callback(self, iteration):
                break
        return self

    def _objective(self, users, items, ratings):
        prediction = (self.global_mean + self.user_bias[users] + self.item_bias[items]
                      + np.einsum("ij,ij->i", self.user_factors[users],
                                  self.item_factors[items]))
        residual = ratings - prediction
        return float(residual @ residual
                     + self.reg * (np.square(self.user_factors).sum()
                                   + np.square(self.item_factors).sum())
                     + self.bias_reg * (self.user_bias @ self.user_bias
                                        + self.item_bias @ self.item_bias))

    def score_users(self, users, items, as_of):
        users, items = self._score_indices(users, items, as_of)
        return (self.global_mean + self.user_bias[users, None]
                + self.item_bias[None, items]
                + self.user_factors[users] @ self.item_factors[items].T)
