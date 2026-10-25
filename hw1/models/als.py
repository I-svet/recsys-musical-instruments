from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from recsys.core import FitSnapshot, Model

from .common import _ClassicModel

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
