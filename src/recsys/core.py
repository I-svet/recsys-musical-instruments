"""Shared contracts. A fit snapshot never contains validation/test labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import os
import pickle
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class FitSnapshot:
    interactions: pd.DataFrame  # user:int, item:int, rating:float, timestamp:int (UTC seconds)
    n_users: int
    n_items: int
    as_of: int  # exclusive upper bound on observed timestamps
    mapping_hash: str = ""

    def __post_init__(self):
        required = {"user", "item", "rating", "timestamp"}
        if not required.issubset(self.interactions.columns):
            raise ValueError(f"Missing columns: {required - set(self.interactions.columns)}")
        frame = self.interactions
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError("Fit snapshot must contain non-missing interactions")
        if not np.isfinite(frame[["rating", "timestamp"]].to_numpy()).all():
            raise ValueError("Non-finite snapshot values")
        for column in ["user", "item", "timestamp"]:
            values = frame[column].to_numpy()
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{column} must contain finite integers")
        if not frame.rating.between(1, 5).all():
            raise ValueError("Ratings must be in [1, 5]")
        if not frame.user.between(0, self.n_users - 1).all() or not frame.item.between(0, self.n_items - 1).all():
            raise ValueError("IDs outside frozen mapping")
        if (frame.timestamp >= self.as_of).any():
            raise ValueError("Snapshot contains current or future events")
        if frame.duplicated(["user", "item"]).any():
            raise ValueError("Snapshot contains repeated user-item pairs")

    def matrix(self, positive: bool = False) -> sparse.csr_matrix:
        frame = self.interactions
        if positive:
            frame = frame[frame.rating >= 4]
        return sparse.csr_matrix((np.ones(len(frame), dtype=np.float64),
                                  (frame.user.to_numpy(), frame.item.to_numpy())),
                                 shape=(self.n_users, self.n_items))

    def fingerprint(self) -> str:
        digest = hashlib.sha256(f"{self.n_users}:{self.n_items}:{self.as_of}:{self.mapping_hash}".encode())
        digest.update(pd.util.hash_pandas_object(
            self.interactions[["user", "item", "rating", "timestamp"]], index=False).to_numpy().tobytes())
        return digest.hexdigest()


class Model:
    """Models implement fit and score_users; serialization is for trusted local files."""
    def fit(self, snapshot: FitSnapshot, seed: int = 42,
            callback: Callable[[Model, int], bool] | None = None) -> Model:
        raise NotImplementedError

    def score_users(self, users: np.ndarray, items: np.ndarray, as_of: int) -> np.ndarray:
        raise NotImplementedError

    def has_signal(self, users: np.ndarray) -> np.ndarray:
        return np.ones(len(users), dtype=bool)

    def score_pairs(self, users: np.ndarray, items: np.ndarray, as_of: int) -> np.ndarray:
        users, items = np.asarray(users), np.asarray(items)
        if users.shape != items.shape or users.ndim != 1:
            raise ValueError("Pair arrays must be equal one-dimensional shapes")
        result = np.empty(len(users), dtype=np.float64)
        for start in range(0, len(users), 128):
            stop = min(start + 128, len(users))
            unique_u, index_u = np.unique(users[start:stop], return_inverse=True)
            unique_i, index_i = np.unique(items[start:stop], return_inverse=True)
            result[start:stop] = self.score_users(unique_u, unique_i, as_of)[index_u, index_i]
        return result

    def bind_snapshot(self, snapshot: FitSnapshot) -> None:
        self.fit_fingerprint = snapshot.fingerprint()
        self.fit_as_of = snapshot.as_of
        self.n_users, self.n_items = snapshot.n_users, snapshot.n_items

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        with temporary.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    @staticmethod
    def load(path: str | Path) -> Model:
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, Model):
            raise TypeError("Artifact does not contain a recommender")
        return model
