"""Binary top-k metrics, coverage diagnostics and paired user bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import FitSnapshot, Model
from .ranking import recommend


def evaluate(
    recommendations: np.ndarray,
    users: np.ndarray,
    holdout: pd.DataFrame,
    n_items: int,
    k: int = 10,
    allfuture_counts: np.ndarray | dict[int, int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Main metrics average over users with >=1 positive within the catalog.

    Outside-only users get NaN main metrics, but zero Recall_all_future. The
    optional count vector is indexed by frozen user ID and includes positives
    outside the available item catalog. It never changes training or ranking.
    """
    recommendations, users = np.asarray(recommendations), np.asarray(users, dtype=np.int64)
    if k < 1 or recommendations.shape != (len(users), k):
        raise ValueError("Expected exactly k recommendations per user")
    if users.ndim != 1 or len(np.unique(users)) != len(users):
        raise ValueError("Evaluation user IDs must be unique")
    if recommendations.dtype.kind not in "iu" or np.any((recommendations < 0) | (recommendations >= n_items)):
        raise ValueError("Recommendation IDs outside catalog")
    if len(users) and np.any(np.diff(np.sort(recommendations, axis=1), axis=1) == 0):
        raise ValueError("A recommendation row contains duplicate items")
    positives = holdout.loc[holdout.rating >= 4, ["user", "item"]].drop_duplicates()
    relevant = {int(user): set(group.item.tolist()) for user, group in positives.groupby("user")}
    missing = set(relevant) - set(users.tolist())
    if missing:
        raise ValueError("Recommendations omit users with positive in-catalog labels")
    if allfuture_counts is not None:
        if isinstance(allfuture_counts, dict):
            future = {int(user): int(count) for user, count in allfuture_counts.items()}
        else:
            array = np.asarray(allfuture_counts)
            if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any() or (array % 1 != 0).any():
                raise ValueError("Invalid all-future counts")
            future = dict(enumerate(array.astype(np.int64).tolist()))
        if any(count < 0 for count in future.values()):
            raise ValueError("Invalid all-future counts")
        if {user for user, count in future.items() if count > 0} - set(users.tolist()):
            raise ValueError("Recommendations omit users with positives outside catalog")
    else:
        future = None
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal_dcgs = np.r_[0., np.cumsum(discounts)]
    rows = []
    for user, recs in zip(users, recommendations):
        targets = relevant.get(int(user), set())
        count = len(targets)
        hits = np.array([item in targets for item in recs], dtype=np.float64)
        n_hits = int(hits.sum())
        if count:
            ndcg = float(hits @ discounts / ideal_dcgs[min(k, count)])
            recall = n_hits / count
            precision = n_hits / k
            ap = float(((np.cumsum(hits) / np.arange(1, k + 1)) * hits).sum() / min(k, count))
        else:
            ndcg = recall = precision = ap = float("nan")
        row = {"user": int(user), "relevant_count": count, "hits": n_hits,
               f"ndcg@{k}": ndcg, f"recall@{k}": recall, f"precision@{k}": precision, f"map@{k}": ap}
        if future is not None:
            full_count = future.get(int(user), 0)
            if full_count < count:
                raise ValueError("All-future count smaller than in-catalog positives")
            row["allfuture_count"] = full_count
            row[f"recall_all_future@{k}"] = n_hits / full_count if full_count else float("nan")
        rows.append(row)
    columns = ["user", "relevant_count", "hits", f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"map@{k}"]
    if future is not None:
        columns += ["allfuture_count", f"recall_all_future@{k}"]
    per_user = pd.DataFrame(rows, columns=columns)
    main_mask = per_user.relevant_count.to_numpy() > 0
    aggregates = {name: float(per_user[name].mean()) for name in columns if "@" in name}
    aggregates["n_users"] = int(main_mask.sum())
    aggregates["n_users_recommended"] = len(users)
    aggregates[f"catalog_coverage@{k}"] = len(np.unique(recommendations[main_mask])) / n_items if n_items else 0.
    if future is not None:
        aggregates["n_users_all_future"] = int((per_user.allfuture_count > 0).sum())
        all_positive = int(per_user.allfuture_count.sum())
        aggregates["positive_event_catalog_coverage"] = int(per_user.relevant_count.sum()) / all_positive if all_positive else float("nan")
    return per_user, aggregates


def evaluate_model(
    model: Model,
    snapshot: FitSnapshot,
    holdout: pd.DataFrame,
    k: int = 10,
    batch_size: int = 128,
    allfuture_counts: np.ndarray | dict[int, int] | None = None,
) -> dict:
    if (holdout.timestamp < snapshot.as_of).any():
        raise ValueError("Holdout contains events before recommendation cutoff")
    users = np.sort(holdout.loc[holdout.rating >= 4, "user"].unique()).astype(np.int64)
    if allfuture_counts is not None:
        outside = np.array([user for user, count in allfuture_counts.items() if count > 0], dtype=np.int64) if isinstance(allfuture_counts, dict) else np.flatnonzero(np.asarray(allfuture_counts) > 0)
        users = np.union1d(users, outside)
    items, scores, fallback = recommend(model, snapshot, users, k=k, batch_size=batch_size)
    per_user, aggregates = evaluate(items, users, holdout, snapshot.n_items, k=k, allfuture_counts=allfuture_counts)
    per_user["fallback"] = fallback
    main_mask = per_user.relevant_count.to_numpy() > 0
    aggregates["fallback_fraction"] = float(fallback[main_mask].mean()) if main_mask.any() else float("nan")
    popularity = np.asarray(snapshot.matrix(positive=True).sum(axis=0)).ravel()
    aggregates[f"mean_recommended_popularity@{k}"] = float(popularity[items[main_mask]].mean()) if main_mask.any() else float("nan")
    history_count = np.diff(snapshot.matrix().indptr)
    per_user["history_count"] = history_count[users]
    return {"metrics": aggregates, "per_user": per_user, "users": users,
            "items": items, "scores": scores, "fallback": fallback}


def paired_bootstrap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    metric: str = "ndcg@10",
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> dict:
    """CI for first minus second; collapse repeated seeds before resampling users."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    left = first.groupby("user")[metric].mean().dropna()
    right = second.groupby("user")[metric].mean().dropna()
    if not left.index.equals(right.index) or left.empty:
        raise ValueError("Paired bootstrap requires the same non-empty user cohort")
    delta = (left - right).to_numpy()
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite bootstrap scores")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    # Bounded memory even for large cohorts; the RNG sequence is deterministic.
    for start in range(0, n_bootstrap, 100):
        stop = min(start + 100, n_bootstrap)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        samples[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"metric": metric, "difference": float(delta.mean()), "ci95_low": float(low),
            "ci95_high": float(high), "n_users": len(delta), "n_bootstrap": n_bootstrap, "seed": seed}
