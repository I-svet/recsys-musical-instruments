"""Second-order FM, implemented directly with PyTorch tensor operations."""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np
import torch
from torch import nn

from ..core import FitSnapshot, Model
from ..features import HistoricalFeatures


def seed_torch(seed: int):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def check_integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def check_real(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) \
            or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def mean_square(parameters):
    """Mean squared parameter value over all elements, excluding global mean."""
    parameters = list(parameters)
    return sum(p.square().sum() for p in parameters) / sum(p.numel() for p in parameters)


class _FMNet(nn.Module):
    def __init__(self, n_features, rank, mean):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(mean), dtype=torch.float32))
        self.linear = nn.Embedding(n_features, 1)
        self.factors = nn.Embedding(n_features, rank)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

    def group(self, indices, values):
        latent = self.factors(indices) * values[..., None]
        summed = latent.sum(dim=1)
        scalar = (self.linear(indices).squeeze(-1) * values).sum(dim=1)
        scalar = scalar + 0.5 * (summed.square() - latent.square().sum(dim=1)).sum(dim=1)
        return scalar, summed

    def forward(self, indices, values):
        scalar, _ = self.group(indices, values)
        return self.bias + scalar


class FM(Model):
    """MSE + reg * global mean-square(linear weights, factor weights).

    All observed ratings are used. Missing interactions are never labelled zero.
    Full-catalog scores use the exact disjoint-group identity for a second-order FM.
    """
    def __init__(self, rank=32, lr=0.01, reg=0.0001, epochs=30,
                 batch_size=4096, metadata=True):
        self.rank = check_integer("rank", rank, 1)
        self.epochs = check_integer("epochs", epochs, 0)
        self.batch_size = check_integer("batch_size", batch_size, 1)
        self.lr = check_real("lr", lr, positive=True)
        self.reg = check_real("reg", reg)
        if not isinstance(metadata, (bool, np.bool_)):
            raise ValueError("metadata must be a boolean")
        self.metadata = bool(metadata)

    def fit(self, snapshot: FitSnapshot, seed=42, callback=None):
        self.bind_snapshot(snapshot)
        seed_torch(seed)
        rng = np.random.default_rng(seed)
        self.n_users, self.n_items, self.as_of = snapshot.n_users, snapshot.n_items, snapshot.as_of
        self.features = HistoricalFeatures(self.metadata)
        ui, uv, ii, iv = self.features.fit_transform(snapshot)
        indices = torch.from_numpy(np.concatenate([ui, ii], axis=1))
        values = torch.from_numpy(np.concatenate([uv, iv], axis=1))
        ratings = torch.tensor(snapshot.interactions.rating.to_numpy(), dtype=torch.float32)
        self.net = _FMNet(self.features.n_features, self.rank, ratings.mean().item())
        self.invalidate_score_cache()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.history_, self.epochs_trained_ = [], 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(ratings))
            total = 0.0
            self.net.train()
            for start in range(0, len(order), self.batch_size):
                rows = torch.from_numpy(order[start:start + self.batch_size])
                prediction = self.net(indices[rows], values[rows])
                mse = (prediction - ratings[rows]).square().mean()
                penalty = mean_square([self.net.linear.weight, self.net.factors.weight])
                loss = mse + self.reg * penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite FM loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += mse.item() * len(rows)
            self.history_.append({"epoch": epoch + 1, "mse": total / len(ratings)})
            self.epochs_trained_ = epoch + 1
            self.invalidate_score_cache()
            self.net.eval()
            if callback is not None and callback(self, epoch + 1):
                break
        self.net.eval()
        return self

    def invalidate_score_cache(self):
        """Call after external parameter edits; fit invalidates at each epoch."""
        self._item_score_cache = None

    def score_users(self, users, items, as_of):
        ui, uv = self.features.user_features(users, as_of)
        items = np.asarray(items, dtype=np.int64)
        key = int(as_of), items.tobytes()
        with torch.no_grad():
            us, uz = self.net.group(torch.from_numpy(ui), torch.from_numpy(uv))
            if self._item_score_cache is None or self._item_score_cache[0] != key:
                ii, iv = self.features.item_features(items, as_of)
                its, iz = self.net.group(torch.from_numpy(ii), torch.from_numpy(iv))
                self._item_score_cache = key, its, iz
            _, its, iz = self._item_score_cache
            scores = self.net.bias + us[:, None] + its[None, :] + uz @ iz.T
        return scores.numpy().astype(np.float64)

    def score_pairs(self, users, items, as_of):
        """Direct reference formula; useful for equivalence tests and rating RMSE."""
        ui, uv = self.features.user_features(users, as_of)
        ii, iv = self.features.item_features(items, as_of)
        with torch.no_grad():
            return self.net(torch.from_numpy(np.concatenate([ui, ii], axis=1)),
                            torch.from_numpy(np.concatenate([uv, iv], axis=1))).numpy().astype(np.float64)
