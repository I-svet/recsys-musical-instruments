import numpy as np
import pandas as pd
import pytest

from recsys.core import FitSnapshot, Model
from recsys.models.classic import EASE, ExplicitALS, ItemKNN, MostPopular


@pytest.fixture
def snapshot():
    # User 4 has no positive history. Item 5 has no positive interactions.
    rows = [(0, 0, 5), (0, 1, 4), (0, 2, 2), (1, 0, 4), (1, 2, 5),
            (2, 0, 5), (2, 1, 4), (2, 3, 5), (3, 1, 5), (3, 4, 4),
            (4, 3, 1), (4, 5, 2)]
    frame = pd.DataFrame(rows, columns=["user", "item", "rating"])
    frame["timestamp"] = np.arange(len(frame))
    return FitSnapshot(frame, n_users=6, n_items=7, as_of=100)


def dense_positive(snapshot):
    result = np.zeros((snapshot.n_users, snapshot.n_items))
    for row in snapshot.interactions.itertuples():
        if row.rating >= 4:
            result[row.user, row.item] = 1
    return result


@pytest.mark.parametrize("gamma", [0, 10])
def test_itemknn_dense_oracle_and_incoming_topk(snapshot, gamma):
    x = dense_positive(snapshot)
    expected = np.zeros((snapshot.n_items, snapshot.n_items))
    for target in range(snapshot.n_items):
        candidates = []
        for source in range(snapshot.n_items):
            overlap = sum(x[:, source] * x[:, target])
            norm = np.linalg.norm(x[:, source]) * np.linalg.norm(x[:, target])
            if source != target and overlap and norm:
                candidates.append((source, overlap / norm * overlap / (overlap + gamma)))
        for source, score in sorted(candidates, key=lambda row: (-row[1], row[0]))[:1]:
            expected[source, target] = score
    model = ItemKNN(k=1, gamma=gamma).fit(snapshot)
    np.testing.assert_allclose(model.weights.toarray(), expected)
    assert np.all(np.count_nonzero(expected, axis=0) <= 1)
    # Incoming pruning is generally asymmetric, unlike the unpruned similarities.
    assert not np.allclose(expected, expected.T)
    np.testing.assert_allclose(model.score_users(np.arange(6), np.arange(7), 100), x @ expected)
    np.testing.assert_array_equal(model.has_signal(np.array([0, 4, 5])), [True, False, False])


def test_ease_independent_constrained_regressions(snapshot):
    x = dense_positive(snapshot)
    reg = 0.7
    model = EASE(reg=reg).fit(snapshot)
    # Solve each constrained target regression directly, without the inverse formula.
    expected = np.zeros((snapshot.n_items, snapshot.n_items))
    for target in range(snapshot.n_items):
        others = np.delete(np.arange(snapshot.n_items), target)
        a = x[:, others]
        expected[others, target] = np.linalg.solve(a.T @ a + reg * np.eye(len(others)),
                                                   a.T @ x[:, target])
    np.testing.assert_allclose(model.weights, expected, atol=1e-13)
    np.testing.assert_array_equal(np.diag(model.weights), 0)
    assert np.any(model.weights < -1e-8)
    assert model.weights.dtype == np.float64
    np.testing.assert_allclose(model.score_users(np.arange(6), np.arange(7), 100), x @ expected)
    np.testing.assert_array_equal(model.has_signal(np.array([4, 5])), [False, False])


def test_als_one_sweep_matches_per_entity_ridge_oracle(snapshot):
    rank, seed, reg, bias_reg = 3, 123, 0.8, 1.3
    model = ExplicitALS(rank=rank, reg=reg, bias_reg=bias_reg, epochs=1).fit(snapshot, seed=seed)
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.1 / np.sqrt(rank), (snapshot.n_users, rank))
    q = rng.normal(0, 0.1 / np.sqrt(rank), (snapshot.n_items, rank))
    bu = np.zeros(snapshot.n_users)
    bi = np.zeros(snapshot.n_items)
    mean = snapshot.interactions.rating.mean()
    penalty = np.diag([bias_reg] + [reg] * rank)
    for user in range(snapshot.n_users):
        rows = snapshot.interactions.query("user == @user")
        a = np.column_stack((np.ones(len(rows)), q[rows.item]))
        y = rows.rating.to_numpy() - mean - bi[rows.item]
        solution = np.linalg.solve(a.T @ a + penalty, a.T @ y)
        bu[user], p[user] = solution[0], solution[1:]
    for item in range(snapshot.n_items):
        rows = snapshot.interactions.query("item == @item")
        a = np.column_stack((np.ones(len(rows)), p[rows.user]))
        y = rows.rating.to_numpy() - mean - bu[rows.user]
        solution = np.linalg.solve(a.T @ a + penalty, a.T @ y)
        bi[item], q[item] = solution[0], solution[1:]
    np.testing.assert_allclose(model.user_bias, bu, atol=1e-13)
    np.testing.assert_allclose(model.item_bias, bi, atol=1e-13)
    np.testing.assert_allclose(model.user_factors, p, atol=1e-13)
    np.testing.assert_allclose(model.item_factors, q, atol=1e-13)


def test_als_objective_descends_and_is_observed_only(snapshot):
    model = ExplicitALS(rank=4, reg=0.2, bias_reg=0.3, epochs=8).fit(snapshot)
    assert len(model.loss_history) == 9
    assert np.all(np.diff(model.loss_history) <= 1e-10)
    assert model.loss_history[-1] < model.loss_history[0]
    scores = model.score_users(np.arange(6), np.arange(7), 100)
    observed_sse = sum((row.rating - scores[row.user, row.item]) ** 2
                       for row in snapshot.interactions.itertuples())
    penalty = (0.2 * (np.square(model.user_factors).sum() + np.square(model.item_factors).sum())
               + 0.3 * (np.square(model.user_bias).sum() + np.square(model.item_bias).sum()))
    assert model.loss_history[-1] == pytest.approx(observed_sse + penalty)
    np.testing.assert_array_equal(model.user_factors[5], 0)
    np.testing.assert_array_equal(model.item_factors[6], 0)


def test_als_callback_and_repeatability(snapshot):
    steps = []
    def stop(model, step):
        steps.append((step, len(model.loss_history)))
        return step == 2
    first = ExplicitALS(rank=2, epochs=5).fit(snapshot, seed=17, callback=stop)
    second = ExplicitALS(rank=2, epochs=2).fit(snapshot, seed=17)
    assert steps == [(1, 2), (2, 3)]
    assert first.n_iter_ == 2
    np.testing.assert_array_equal(first.user_factors, second.user_factors)
    np.testing.assert_array_equal(first.item_factors, second.item_factors)


@pytest.mark.parametrize("factory", [MostPopular, ItemKNN, EASE,
                                     lambda: ExplicitALS(rank=2, epochs=2)])
def test_score_slicing_save_load_and_input_validation(snapshot, tmp_path, factory):
    model = factory().fit(snapshot)
    full = model.score_users(np.arange(6), np.arange(7), 100)
    users, items = np.array([4, 0, 2]), np.array([5, 1, 0])
    np.testing.assert_allclose(model.score_users(users, items, 200), full[np.ix_(users, items)])
    artifact = tmp_path / "model.pkl"
    model.save(artifact)
    loaded = Model.load(artifact)
    np.testing.assert_array_equal(loaded.score_users(users, items, 100),
                                  model.score_users(users, items, 100))
    with pytest.raises(ValueError, match="before"):
        model.score_users(users, items, 99)
    with pytest.raises(ValueError, match="mapping"):
        model.score_users(np.array([6]), items, 100)
    assert model.score_users(np.array([], dtype=int), items, 100).shape == (0, 3)


def test_popularity_counts_only_positive_reviews(snapshot):
    model = MostPopular().fit(snapshot)
    np.testing.assert_array_equal(model.popularity, [3, 3, 1, 1, 1, 0, 0])


@pytest.mark.parametrize("factory", [lambda: ItemKNN(gamma=0), EASE])
def test_empty_positive_matrix_has_finite_zero_scores(snapshot, factory):
    frame = snapshot.interactions.copy()
    frame["rating"] = 1.0
    empty_positive = FitSnapshot(frame, snapshot.n_users, snapshot.n_items, snapshot.as_of)
    model = factory().fit(empty_positive)
    users, items = np.arange(snapshot.n_users), np.arange(snapshot.n_items)
    np.testing.assert_array_equal(model.score_users(users, items, snapshot.as_of), 0)
    assert not model.has_signal(users).any()


@pytest.mark.parametrize("factory", [lambda: ItemKNN(k=0), lambda: ItemKNN(gamma=-1),
                                     lambda: EASE(reg=0), lambda: EASE(reg=np.nan),
                                     lambda: ExplicitALS(reg=0),
                                     lambda: ExplicitALS(bias_reg=-1),
                                     lambda: ExplicitALS(rank=0)])
def test_invalid_hyperparameters(factory):
    with pytest.raises(ValueError):
        factory()
