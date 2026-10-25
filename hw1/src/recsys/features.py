"""Causal, day-granularity historical features for a factorization machine.

An event never contributes to its own features, nor to other events on its day.
The numeric scaler is fitted on training rows only. No catalogue crawl fields
or future review text, helpfulness, popularity or prices are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import FitSnapshot

DAY = 86_400


def prior_history(frame: pd.DataFrame, entity: str) -> np.ndarray:
    """Return [log1p(prior count), prior mean, log1p(age days)] in row order."""
    work = frame[[entity, "rating", "timestamp"]].copy()
    work["day"] = work.timestamp.to_numpy(dtype=np.int64) // DAY
    work["row"] = np.arange(len(work))
    grouped = work.groupby([entity, "day"], sort=True).rating.agg(["count", "sum"]).reset_index()
    totals = grouped.groupby(entity, sort=False)[["count", "sum"]].cumsum()
    grouped["prior_count"] = totals["count"] - grouped["count"]
    grouped["prior_sum"] = totals["sum"] - grouped["sum"]
    grouped["first_day"] = grouped.groupby(entity, sort=False).day.transform("min")
    joined = work.merge(grouped[[entity, "day", "prior_count", "prior_sum", "first_day"]],
                        on=[entity, "day"], validate="many_to_one").sort_values("row")
    count = joined.prior_count.to_numpy(dtype=float)
    mean = np.divide(joined.prior_sum.to_numpy(), count,
                     out=np.full(len(work), 3.0), where=count > 0)
    age = np.where(count > 0, joined.day.to_numpy() - joined.first_day.to_numpy(), 0)
    return np.column_stack([np.log1p(count), mean, np.log1p(age)]).astype(np.float32)


class HistoricalFeatures:
    """Small fitted local feature builder, not a learned external algorithm."""

    def __init__(self, metadata: bool = True):
        self.metadata = metadata

    def fit_transform(self, snapshot: FitSnapshot):
        self.n_users, self.n_items = snapshot.n_users, snapshot.n_items
        self.as_of = snapshot.as_of
        self.frame = snapshot.interactions[["user", "item", "rating", "timestamp"]].copy()
        self.base = self.n_users + self.n_items
        self.n_features = self.base + (25 if self.metadata else 0)  # 6 numeric + 12 month + 7 weekday
        users = self.frame.user.to_numpy(dtype=np.int64)
        items = self.frame.item.to_numpy(dtype=np.int64)
        if not self.metadata:
            return users[:, None], np.ones((len(users), 1), np.float32), \
                (self.n_users + items[:, None]), np.ones((len(items), 1), np.float32)
        user_raw, item_raw = prior_history(self.frame, "user"), prior_history(self.frame, "item")
        self.user_mean, self.user_scale = self._scaler(user_raw)
        self.item_mean, self.item_scale = self._scaler(item_raw)
        ui, uv = self._user(users, (user_raw - self.user_mean) / self.user_scale,
                            self.frame.timestamp.to_numpy(dtype=np.int64))
        ii, iv = self._item(items, (item_raw - self.item_mean) / self.item_scale)
        self._stats_cache = {}
        return ui, uv, ii, iv

    @staticmethod
    def _scaler(values):
        mean, scale = values.mean(axis=0), values.std(axis=0)
        return mean, np.where(scale > 1e-7, scale, 1.0).astype(np.float32)

    def _user(self, users, numeric, timestamps):
        dates = pd.to_datetime(timestamps, unit="s", utc=True)
        month, weekday = np.asarray(dates.month) - 1, np.asarray(dates.dayofweek)
        indices = np.column_stack([users, np.broadcast_to(np.arange(self.base, self.base + 3),
                                                         (len(users), 3)),
                                   self.base + 6 + month, self.base + 18 + weekday])
        values = np.column_stack([np.ones(len(users)), numeric, np.ones((len(users), 2))])
        return indices.astype(np.int64), values.astype(np.float32)

    def _item(self, items, numeric):
        indices = np.column_stack([self.n_users + items,
                                   np.broadcast_to(np.arange(self.base + 3, self.base + 6),
                                                   (len(items), 3))])
        return indices.astype(np.int64), np.column_stack([np.ones(len(items)), numeric]).astype(np.float32)

    def _stats(self, entity, as_of):
        if as_of < self.as_of:
            raise ValueError("Scoring before the fitted snapshot would expose future features")
        day = int(as_of) // DAY
        key = (entity, day)
        if key not in self._stats_cache:
            # Freeze at fit, and exclude the query day even for an intraday cutoff.
            frame = self.frame[self.frame.timestamp // DAY < day]
            n = self.n_users if entity == "user" else self.n_items
            ids = frame[entity].to_numpy(dtype=np.int64)
            count = np.bincount(ids, minlength=n)
            sums = np.bincount(ids, weights=frame.rating.to_numpy(), minlength=n)
            first = np.full(n, day, dtype=np.int64)
            np.minimum.at(first, ids, frame.timestamp.to_numpy(dtype=np.int64) // DAY)
            mean = np.divide(sums, count, out=np.full(n, 3.0), where=count > 0)
            self._stats_cache[key] = np.column_stack([np.log1p(count), mean,
                                                      np.log1p(np.where(count > 0, day - first, 0))]).astype(np.float32)
        return self._stats_cache[key]

    def user_features(self, users: np.ndarray, as_of: int):
        users = np.asarray(users, dtype=np.int64)
        if as_of < self.as_of:
            raise ValueError("Scoring before fitted snapshot")
        if not self.metadata:
            return users[:, None], np.ones((len(users), 1), np.float32)
        numeric = (self._stats("user", as_of)[users] - self.user_mean) / self.user_scale
        return self._user(users, numeric, np.full(len(users), as_of, dtype=np.int64))

    def item_features(self, items: np.ndarray, as_of: int):
        items = np.asarray(items, dtype=np.int64)
        if as_of < self.as_of:
            raise ValueError("Scoring before fitted snapshot")
        if not self.metadata:
            return (self.n_users + items[:, None]), np.ones((len(items), 1), np.float32)
        numeric = (self._stats("item", as_of)[items] - self.item_mean) / self.item_scale
        return self._item(items, numeric)
