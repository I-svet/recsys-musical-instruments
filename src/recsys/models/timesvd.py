"""Causal timeSVD++ with differentiable histories and sparse observed time keys.

The implicit history for a training event contains only earlier UTC days. This
is a documented causal adaptation, rather than an exact Netflix replication.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core import FitSnapshot, Model
from ..features import DAY
from .fm import check_integer, check_real, mean_square, seed_torch


def key_lookup(keys, query):
    """Observed-key lookup, with index zero reserved for an absent correction."""
    query = np.asarray(query, dtype=np.int64)
    positions = np.searchsorted(keys, query)
    matched = positions < len(keys)
    matched[matched] &= keys[positions[matched]] == query[matched]
    return np.where(matched, positions + 1, 0).astype(np.int64)


def causal_history(embedding, items, user_starts, prior_starts, prior_counts):
    """Segmented prefix sum. Every batch must contain each selected user's history.

    Both prefix terms remain in autograd: later rating errors update their past
    item vectors. Same-day items are excluded by prior_starts, not row order.
    """
    vectors = embedding(items)
    prefix = torch.cat([vectors.new_zeros((1, vectors.shape[1])), vectors.cumsum(dim=0)], dim=0)
    summed = prefix[prior_starts] - prefix[user_starts]
    return summed / prior_counts.clamp(min=1).sqrt()[:, None]


class _TimeNet(nn.Module):
    def __init__(self, n_users, n_items, rank, n_days, n_bins, mean, temporal):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.user_bias, self.item_bias = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
        self.user_factors, self.item_factors = nn.Embedding(n_users, rank), nn.Embedding(n_items, rank)
        self.implicit = nn.Embedding(n_items, rank)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for layer in [self.user_factors, self.item_factors, self.implicit]:
            nn.init.normal_(layer.weight, std=0.02)
        self.temporal = temporal
        if temporal:
            self.user_drift_bias, self.user_drift = nn.Embedding(n_users, 1), nn.Embedding(n_users, rank)
            self.day_bias, self.day_factors = nn.Embedding(n_days + 1, 1, padding_idx=0), \
                nn.Embedding(n_days + 1, rank, padding_idx=0)
            self.bin_bias = nn.Embedding(n_bins + 1, 1, padding_idx=0)
            for parameter in self.temporal_parameters():
                nn.init.zeros_(parameter)

    def base_parameters(self):
        return [self.user_bias.weight, self.item_bias.weight, self.user_factors.weight,
                self.item_factors.weight, self.implicit.weight]

    def temporal_parameters(self):
        if not self.temporal:
            return []
        return [self.user_drift_bias.weight, self.user_drift.weight, self.day_bias.weight,
                self.day_factors.weight, self.bin_bias.weight]

    def user_components(self, users, history, deviation, day_index):
        bias = self.user_bias(users).squeeze(-1)
        factors = self.user_factors(users) + history
        if self.temporal:
            bias = bias + self.user_drift_bias(users).squeeze(-1) * deviation + self.day_bias(day_index).squeeze(-1)
            factors = factors + self.user_drift(users) * deviation[:, None] + self.day_factors(day_index)
        return bias, factors

    def forward(self, users, items, history, deviation, day_index, bin_index):
        bias, factors = self.user_components(users, history, deviation, day_index)
        score = self.mean + bias + self.item_bias(items).squeeze(-1) + (factors * self.item_factors(items)).sum(dim=1)
        if self.temporal:
            score = score + self.bin_bias(bin_index).squeeze(-1)
        return score


class TimeSVDPP(Model):
    """MSE + reg * mean-square(base) + temporal_reg * mean-square(temporal).

    Means are over parameter elements separately in the two parameter groups.
    Training batches contain complete users and aim for batch_size events.
    Unknown user-day and item-bin keys have exactly zero temporal corrections.
    """
    def __init__(self, rank=32, lr=0.01, reg=0.0001, temporal_reg=0.001,
                 epochs=30, bin_days=90, beta=0.4, temporal=True, batch_size=4096):
        self.rank = check_integer("rank", rank, 1)
        self.epochs = check_integer("epochs", epochs, 0)
        self.batch_size = check_integer("batch_size", batch_size, 1)
        self.bin_days = check_integer("bin_days", bin_days, 1)
        self.lr = check_real("lr", lr, positive=True)
        self.reg = check_real("reg", reg)
        self.temporal_reg = check_real("temporal_reg", temporal_reg)
        self.beta = check_real("beta", beta, positive=True)
        if self.beta > 1:
            raise ValueError("beta must be <= 1")
        if not isinstance(temporal, (bool, np.bool_)):
            raise ValueError("temporal must be a boolean")
        self.temporal = bool(temporal)

    def _deviation(self, users, days):
        delta = np.asarray(days) - self.mean_days[users]
        return (np.sign(delta) * np.abs(delta) ** self.beta).astype(np.float32)

    def _day_indices(self, users, days):
        relative = np.asarray(days, dtype=np.int64) - self.first_day
        valid = (relative >= 0) & (relative < self.day_stride)
        result = key_lookup(self.day_keys, np.asarray(users) * self.day_stride + relative)
        return np.where(valid, result, 0).astype(np.int64)

    def _bin_indices(self, items, days):
        bins = (np.asarray(days, dtype=np.int64) - self.first_day) // self.bin_days
        valid = (bins >= 0) & (bins < self.bin_stride)
        result = key_lookup(self.bin_keys, np.asarray(items) * self.bin_stride + bins)
        return np.where(valid, result, 0).astype(np.int64)

    def _prepare(self, snapshot):
        self.n_users, self.n_items, self.as_of = snapshot.n_users, snapshot.n_items, snapshot.as_of
        frame = snapshot.interactions.sort_values(["user", "timestamp", "item"], kind="stable")
        self.users = frame.user.to_numpy(dtype=np.int64)
        self.items = frame.item.to_numpy(dtype=np.int64)
        self.days = frame.timestamp.to_numpy(dtype=np.int64) // DAY
        self.ratings = frame.rating.to_numpy(dtype=np.float32)
        self.counts = np.bincount(self.users, minlength=self.n_users)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.active_users = np.flatnonzero(self.counts)
        self.first_day, self.last_day = int(self.days.min()), int(self.days.max())
        self.day_stride = self.last_day - self.first_day + 1
        self.bin_stride = (self.last_day - self.first_day) // self.bin_days + 1
        self.mean_days = np.divide(np.bincount(self.users, weights=self.days, minlength=self.n_users),
                                   self.counts, out=np.full(self.n_users, float(self.first_day)),
                                   where=self.counts > 0)
        self.day_keys = np.unique(self.users * self.day_stride + self.days - self.first_day)
        self.bin_keys = np.unique(self.items * self.bin_stride + (self.days - self.first_day) // self.bin_days)
        # First row for each (user, day); all same-day rows share the same prefix.
        new_group = np.r_[True, (self.users[1:] != self.users[:-1]) | (self.days[1:] != self.days[:-1])]
        day_starts = np.maximum.accumulate(np.where(new_group, np.arange(len(frame)), 0))
        self.prior_counts = day_starts - self.offsets[self.users]

    def _batches(self, rng):
        packed, total = [], 0
        for user in rng.permutation(self.active_users):
            if packed and total + self.counts[user] > self.batch_size:
                yield np.asarray(packed, dtype=np.int64)
                packed, total = [], 0
            packed.append(user)
            total += self.counts[user]
        if packed:
            yield np.asarray(packed, dtype=np.int64)

    def _batch_rows(self, users):
        lengths = self.counts[users]
        rows = np.concatenate([np.arange(self.offsets[u], self.offsets[u + 1]) for u in users])
        local_starts = np.repeat(np.r_[0, np.cumsum(lengths)[:-1]], lengths)
        return rows, local_starts, local_starts + self.prior_counts[rows]

    def fit(self, snapshot: FitSnapshot, seed=42, callback=None):
        self.bind_snapshot(snapshot)
        seed_torch(seed)
        rng = np.random.default_rng(seed)
        self._prepare(snapshot)
        self.net = _TimeNet(self.n_users, self.n_items, self.rank, len(self.day_keys), len(self.bin_keys),
                            self.ratings.mean(), self.temporal)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        users_t, items_t = torch.from_numpy(self.users), torch.from_numpy(self.items)
        ratings_t = torch.from_numpy(self.ratings)
        deviation_t = torch.from_numpy(self._deviation(self.users, self.days))
        day_t = torch.from_numpy(self._day_indices(self.users, self.days))
        bin_t = torch.from_numpy(self._bin_indices(self.items, self.days))
        self.history_, self.epochs_trained_, self._score_cache = [], 0, None
        for epoch in range(self.epochs):
            total = 0.0
            self.net.train()
            for packed in self._batches(rng):
                rows, starts, priors = self._batch_rows(packed)
                indices = torch.from_numpy(rows)
                history = causal_history(self.net.implicit, items_t[indices], torch.from_numpy(starts),
                                         torch.from_numpy(priors), torch.tensor(self.prior_counts[rows], dtype=torch.float32))
                prediction = self.net(users_t[indices], items_t[indices], history,
                                      deviation_t[indices], day_t[indices], bin_t[indices])
                mse = (prediction - ratings_t[indices]).square().mean()
                loss = mse + self.reg * mean_square(self.net.base_parameters())
                if self.temporal:
                    loss = loss + self.temporal_reg * mean_square(self.net.temporal_parameters())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite timeSVD++ loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += mse.item() * len(rows)
            self.history_.append({"epoch": epoch + 1, "mse": total / len(self.users)})
            self.epochs_trained_, self._score_cache = epoch + 1, None
            self.net.eval()
            if callback is not None and callback(self, epoch + 1):
                break
        self.net.eval()
        return self

    def _history_at(self, as_of):
        if as_of < self.as_of:
            raise ValueError("Scoring before fitted snapshot")
        day = int(as_of) // DAY
        if self._score_cache is None or self._score_cache[0] != day:
            mask = self.days < day
            with torch.no_grad():
                history = torch.zeros((self.n_users, self.rank), dtype=torch.float32)
                history.index_add_(0, torch.from_numpy(self.users[mask]),
                                   self.net.implicit(torch.from_numpy(self.items[mask])))
                count = torch.from_numpy(np.bincount(self.users[mask], minlength=self.n_users).astype(np.float32))
                history /= count.clamp(min=1).sqrt()[:, None]
            self._score_cache = day, history
        return self._score_cache[1]

    def score_users(self, users, items, as_of):
        users, items = np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        day = int(as_of) // DAY
        history = self._history_at(as_of)[torch.from_numpy(users)]
        deviation = torch.from_numpy(self._deviation(users, day))
        day_index = torch.from_numpy(self._day_indices(users, day))
        bin_index = torch.from_numpy(self._bin_indices(items, day))
        with torch.no_grad():
            bias, factors = self.net.user_components(torch.from_numpy(users), history, deviation, day_index)
            item_bias = self.net.item_bias(torch.from_numpy(items)).squeeze(-1)
            if self.temporal:
                item_bias = item_bias + self.net.bin_bias(bin_index).squeeze(-1)
            result = self.net.mean + bias[:, None] + item_bias[None, :] + factors @ self.net.item_factors(torch.from_numpy(items)).T
        return result.numpy().astype(np.float64)

    def score_pairs(self, users, items, as_of):
        users, items = np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        day = int(as_of) // DAY
        with torch.no_grad():
            result = self.net(torch.from_numpy(users), torch.from_numpy(items),
                              self._history_at(as_of)[torch.from_numpy(users)],
                              torch.from_numpy(self._deviation(users, day)),
                              torch.from_numpy(self._day_indices(users, day)),
                              torch.from_numpy(self._bin_indices(items, day)))
        return result.numpy().astype(np.float64)
