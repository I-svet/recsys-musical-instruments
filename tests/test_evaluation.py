import numpy as np
import pandas as pd
import pytest

from recsys.core import FitSnapshot, Model
from recsys.metrics import evaluate, evaluate_model, paired_bootstrap
from recsys.ranking import _stable_topk, recommend


class FixedModel(Model):
    def __init__(self, scores, signal=None):
        self.scores = np.asarray(scores, dtype=float)
        self.signal = np.ones(len(scores), dtype=bool) if signal is None else np.asarray(signal)

    def score_users(self, users, items, as_of):
        return self.scores[np.ix_(users, items)]

    def has_signal(self, users):
        return self.signal[users]


def _snapshot():
    frame = pd.DataFrame({"user": [0, 0, 1, 1, 2], "item": [0, 1, 2, 3, 4],
                          "rating": [1., 5., 5., 5., 2.], "timestamp": [1, 2, 1, 2, 1]})
    return FitSnapshot(frame, n_users=4, n_items=6, as_of=10)


def test_ranking_masks_negative_ratings_and_breaks_ties_by_item_id():
    snapshot = _snapshot()
    model = FixedModel([[9, 9, 3, 3, 2, 1], [3, 3, 9, 9, 2, 1],
                        [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    items, scores, fallback = recommend(model, snapshot, np.array([1, 0]), k=3, batch_size=1)
    np.testing.assert_array_equal(items, [[0, 1, 4], [2, 3, 4]])
    np.testing.assert_array_equal(scores, [[3, 3, 2], [3, 3, 2]])
    assert not fallback.any()
    # Scoring arrays are not mutated by seen masking.
    assert np.isfinite(model.scores).all()


@pytest.mark.parametrize("k", [1, 3, 10, 50])
def test_partial_topk_matches_full_stable_sort_with_boundary_ties(k):
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(50, 100))
    # Rounded and constant rows exercise ties that straddle the k-th boundary.
    scores[10:20] = np.round(scores[10:20])
    scores[20:30] = 1
    scores[:, 50:60] = -np.inf
    scores[30, :20] = 9
    expected = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(_stable_topk(scores, k), expected)


def test_missing_signal_uses_same_snapshot_popularity_and_retains_user():
    snapshot = _snapshot()
    model = FixedModel(np.zeros((4, 6)), signal=[True, True, False, False])
    items, scores, fallback = recommend(model, snapshot, np.array([2, 3]), k=3)
    np.testing.assert_array_equal(items, [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(scores, np.ones((2, 3)))
    assert fallback.all()


def test_flat_unseen_scores_fall_back_even_when_has_signal_is_true():
    snapshot = _snapshot()
    model = FixedModel(np.ones((4, 6)))
    items, _, fallback = recommend(model, snapshot, np.array([3]), k=3)
    np.testing.assert_array_equal(items, [[1, 2, 3]])
    assert fallback.tolist() == [True]
    model.is_popularity_baseline = True
    items, _, fallback = recommend(model, snapshot, np.array([3]), k=3)
    np.testing.assert_array_equal(items, [[0, 1, 2]])
    assert fallback.tolist() == [False]


def test_ranking_rejects_nonfinite_and_insufficient_candidates():
    snapshot = _snapshot()
    bad = np.ones((4, 6))
    bad[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        recommend(FixedModel(bad), snapshot, np.array([0]), k=2)
    with pytest.raises(ValueError, match="Fewer than k"):
        recommend(FixedModel(np.ones((4, 6))), snapshot, np.array([0]), k=5)
    with pytest.raises(ValueError, match="before"):
        recommend(FixedModel(np.ones((4, 6))), snapshot, np.array([0]), k=2, as_of=9)


def test_ranking_rejects_model_with_wrong_mapping_cutoff_or_fit_labels():
    snapshot = _snapshot()
    model = FixedModel(np.ones((4, 6)))
    model.bind_snapshot(snapshot)
    changed = FitSnapshot(snapshot.interactions.copy(), 4, 6, 10, mapping_hash="swapped_external_ids")
    with pytest.raises(ValueError, match="different snapshot or mapping"):
        recommend(model, changed, np.array([0]), k=2)
    changed = FitSnapshot(snapshot.interactions.copy(), 4, 6, 11)
    with pytest.raises(ValueError, match="different snapshot or mapping"):
        recommend(model, changed, np.array([0]), k=2)
    changed_frame = snapshot.interactions.copy()
    changed_frame.loc[0, "rating"] = 5
    changed = FitSnapshot(changed_frame, 4, 6, 10)
    with pytest.raises(ValueError, match="different snapshot or mapping"):
        recommend(model, changed, np.array([0]), k=2)


def test_metric_oracle_and_outside_catalog_users():
    # User 0 hits at ranks 1 and 3; user 1 only at rank 2.
    # User 2 has positives only outside catalog, excluded from main metrics.
    users = np.array([0, 1, 2])
    recommendations = np.array([[1, 3, 2], [0, 4, 5], [1, 2, 3]])
    holdout = pd.DataFrame({"user": [0, 0, 1, 0], "item": [1, 2, 4, 5], "rating": [5, 4, 5, 2]})
    per_user, metrics = evaluate(recommendations, users, holdout, n_items=8, k=3, allfuture_counts=np.array([3, 1, 2]))
    ideal_two = 1 + 1 / np.log2(3)
    assert per_user.loc[0, "ndcg@3"] == pytest.approx(1.5 / ideal_two)
    assert per_user.loc[0, "map@3"] == pytest.approx((1 + 2 / 3) / 2)
    assert per_user.loc[1, "map@3"] == pytest.approx(.5)
    assert metrics["recall@3"] == pytest.approx(1.)
    assert metrics["precision@3"] == pytest.approx(.5)
    assert metrics["ndcg@3"] == pytest.approx((1.5 / ideal_two + 1 / np.log2(3)) / 2)
    assert metrics["recall_all_future@3"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert np.isnan(per_user.loc[2, "ndcg@3"])
    assert per_user.loc[2, "recall_all_future@3"] == 0
    assert metrics["n_users"] == 2
    assert metrics["n_users_all_future"] == 3
    assert metrics["catalog_coverage@3"] == pytest.approx(6 / 8)
    assert metrics["positive_event_catalog_coverage"] == pytest.approx(3 / 6)


def test_ap_denominator_is_min_k_and_relevance_is_binary():
    holdout = pd.DataFrame({"user": [0] * 5, "item": range(5), "rating": [4, 5, 4, 5, 4]})
    _, metrics = evaluate(np.array([[0, 1, 2]]), np.array([0]), holdout, n_items=8, k=3)
    assert metrics["map@3"] == 1
    assert metrics["ndcg@3"] == 1
    assert metrics["recall@3"] == pytest.approx(3 / 5)


def test_evaluation_rejects_dropped_users_and_duplicate_recommendations():
    holdout = pd.DataFrame({"user": [0, 1], "item": [1, 2], "rating": [5, 4]})
    with pytest.raises(ValueError, match="omit users"):
        evaluate(np.array([[1, 2]]), np.array([0]), holdout, n_items=3, k=2)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(np.array([[1, 1], [1, 2]]), np.array([0, 1]), holdout, n_items=3, k=2)


def test_evaluate_model_uses_cutoff_and_includes_outside_only_diagnostic():
    snapshot = _snapshot()
    model = FixedModel(np.ones((4, 6)))
    holdout = pd.DataFrame({"user": [0], "item": [2], "rating": [5.], "timestamp": [11]})
    result = evaluate_model(model, snapshot, holdout, k=2, allfuture_counts=np.array([1, 0, 0, 1]))
    assert result["users"].tolist() == [0, 3]
    assert result["metrics"]["ndcg@2"] == 1
    assert result["metrics"]["recall_all_future@2"] == .5
    assert result["per_user"].history_count.tolist() == [2, 0]


def test_paired_bootstrap_pairs_users_and_averages_seeds():
    a = pd.DataFrame({"user": [0, 1, 0, 1], "ndcg@10": [.2, .6, .4, .8]})
    b = pd.DataFrame({"user": [1, 0], "ndcg@10": [.6, .2]})
    result = paired_bootstrap(a, b, n_bootstrap=100)
    assert result["n_users"] == 2
    assert result["difference"] == pytest.approx(.1)
    assert result["ci95_low"] == pytest.approx(.1)
    assert result["ci95_high"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="same non-empty"):
        paired_bootstrap(a, b[b.user == 0])
