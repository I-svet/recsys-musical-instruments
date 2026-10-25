"""One exact candidate/ranking implementation shared by every model."""
from __future__ import annotations

import numpy as np

from .core import FitSnapshot, Model


def _stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Exact score-descending/ID-ascending top-k without a full catalog sort."""
    thresholds = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k]
    selected = np.empty((len(scores), k), dtype=np.int64)
    for row, threshold in enumerate(thresholds):
        above = np.flatnonzero(scores[row] > threshold)
        # flatnonzero preserves item-ID order, including a tie at the boundary.
        tied = np.flatnonzero(scores[row] == threshold)[:k - len(above)]
        candidates = np.concatenate([above, tied])
        selected[row] = candidates[np.lexsort((candidates, -scores[row, candidates]))]
    return selected


def recommend(
    model: Model,
    snapshot: FitSnapshot,
    users: np.ndarray,
    k: int = 10,
    batch_size: int = 128,
    as_of: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return item IDs, scores and fallback flags in the supplied user order.

    All observed interactions (including low ratings) are excluded. Each row is
    ordered by decreasing score, then increasing frozen item ID. Users without
    model signal, or with identical scores for every unseen candidate, receive
    popularity scores learned from the same fit snapshot.
    """
    users = np.asarray(users, dtype=np.int64)
    if users.ndim != 1 or len(np.unique(users)) != len(users):
        raise ValueError("users must be a unique one-dimensional array")
    if np.any((users < 0) | (users >= snapshot.n_users)):
        raise ValueError("User outside frozen mapping")
    if k < 1 or k > snapshot.n_items or batch_size < 1:
        raise ValueError("Invalid k or batch_size")
    if hasattr(model, "fit_fingerprint") and model.fit_fingerprint != snapshot.fingerprint():
        raise ValueError("Model was fitted on a different snapshot or mapping")
    for field, expected in (("n_users", snapshot.n_users), ("n_items", snapshot.n_items), ("fit_as_of", snapshot.as_of)):
        if hasattr(model, field) and getattr(model, field) != expected:
            raise ValueError(f"Model {field} does not match the fit snapshot")
    cutoff = snapshot.as_of if as_of is None else int(as_of)
    if cutoff < snapshot.as_of:
        raise ValueError("Cannot score before the fit snapshot cutoff")
    candidate_ids = np.arange(snapshot.n_items, dtype=np.int64)
    seen = snapshot.matrix()
    available = snapshot.n_items - np.diff(seen.indptr)[users]
    if (available < k).any():
        raise ValueError("Fewer than k unseen candidates for at least one user")
    popularity = np.asarray(snapshot.matrix(positive=True).sum(axis=0)).ravel()
    selected_items = np.empty((len(users), k), dtype=np.int64)
    selected_scores = np.empty((len(users), k), dtype=np.float64)
    fallback = np.zeros(len(users), dtype=bool)
    for start in range(0, len(users), batch_size):
        stop = min(start + batch_size, len(users))
        batch_users = users[start:stop]
        scores = np.asarray(model.score_users(batch_users, candidate_ids, cutoff), dtype=np.float64)
        if scores.shape != (len(batch_users), snapshot.n_items):
            raise ValueError(f"Unexpected model score shape: {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError("Model returned non-finite scores")
        # Some scorers return a broadcast or view of learned arrays.
        scores = scores.copy()
        signal = np.asarray(model.has_signal(batch_users), dtype=bool)
        if signal.shape != (len(batch_users),):
            raise ValueError("Unexpected has_signal shape")
        for row, user in enumerate(batch_users):
            prior_items = seen.indices[seen.indptr[user]:seen.indptr[user + 1]]
            eligible = np.ones(snapshot.n_items, dtype=bool)
            eligible[prior_items] = False
            values = scores[row, eligible]
            flat = values.min() == values.max()
            use_fallback = not signal[row] or (flat and not getattr(model, "is_popularity_baseline", False))
            if use_fallback:
                scores[row] = popularity
            fallback[start + row] = use_fallback
            scores[row, prior_items] = -np.inf
        order = _stable_topk(scores, k)
        selected_items[start:stop] = order
        selected_scores[start:stop] = np.take_along_axis(scores, order, axis=1)
    return selected_items, selected_scores, fallback
