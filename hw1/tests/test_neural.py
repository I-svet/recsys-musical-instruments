import numpy as np
import pandas as pd
import pytest
import torch

from recsys.core import FitSnapshot, Model
from recsys.features import DAY
from recsys.models.fm import FM
from recsys.models.timesvd import TimeSVDPP, causal_history, key_lookup


@pytest.fixture
def snapshot():
    return FitSnapshot(pd.DataFrame({
        "user": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "item": [0, 1, 2, 1, 2, 3, 0, 2, 3],
        "rating": [5., 2., 4., 5., 1., 4., 1., 5., 2.],
        "timestamp": np.array([1, 1, 3, 1, 2, 4, 1, 3, 4]) * DAY,
    }), 3, 4, 5 * DAY)


@pytest.mark.parametrize("metadata", [False, True])
def test_fm_factorized_scoring_matches_direct_pair_formula(snapshot, metadata):
    model = FM(rank=4, epochs=2, metadata=metadata, batch_size=4).fit(snapshot, seed=7)
    users, items = np.array([2, 0]), np.array([3, 1, 0])
    full = model.score_users(users, items, snapshot.as_of)
    direct = model.score_pairs(np.repeat(users, len(items)), np.tile(items, len(users)), snapshot.as_of)
    np.testing.assert_allclose(full.ravel(), direct, rtol=1e-6, atol=1e-6)


def test_prefix_history_excludes_same_day_and_other_users_and_has_gradient():
    embedding = torch.nn.Embedding(5, 2)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(10).reshape(5, 2))
    # User 0: two same-day reviews, then a later one. User 1: two different days.
    result = causal_history(embedding, torch.arange(5), torch.tensor([0, 0, 0, 3, 3]),
                             torch.tensor([0, 0, 2, 3, 4]), torch.tensor([0., 0., 2., 0., 1.]))
    np.testing.assert_array_equal(result[[0, 1, 3]].detach().numpy(), np.zeros((3, 2)))
    np.testing.assert_allclose(result[2].detach().numpy(), [2 / np.sqrt(2), 4 / np.sqrt(2)])
    np.testing.assert_array_equal(result[4].detach().numpy(), [6, 7])
    result[2].sum().backward()
    assert torch.all(embedding.weight.grad[:2] > 0)
    assert torch.count_nonzero(embedding.weight.grad[2:]) == 0


@pytest.mark.parametrize("temporal", [False, True])
def test_timesvd_pair_scoring_matches_matrix_and_serialization(snapshot, tmp_path, temporal):
    model = TimeSVDPP(rank=4, epochs=2, batch_size=4, temporal=temporal).fit(snapshot, seed=7)
    users, items = np.array([2, 0]), np.array([3, 1, 0])
    scores = model.score_users(users, items, snapshot.as_of)
    direct = model.score_pairs(np.repeat(users, len(items)), np.tile(items, len(users)), snapshot.as_of)
    np.testing.assert_allclose(scores.ravel(), direct, rtol=1e-6, atol=1e-6)
    model.save(tmp_path / "time.pkl")
    restored = Model.load(tmp_path / "time.pkl")
    np.testing.assert_array_equal(scores, restored.score_users(users, items, snapshot.as_of))
    assert np.all(model._day_indices(users, 1000) == 0)
    assert np.all(model._bin_indices(items, 1000) == 0)
    if temporal:
        for layer in [model.net.day_bias, model.net.day_factors, model.net.bin_bias]:
            assert torch.count_nonzero(layer.weight[0]) == 0
    else:
        np.testing.assert_array_equal(scores, model.score_users(users, items, snapshot.as_of + 1000 * DAY))


@pytest.mark.parametrize("factory", [FM, TimeSVDPP])
def test_neural_models_determinism_and_callback(snapshot, factory, tmp_path):
    calls = []
    first = factory(rank=3, epochs=5, batch_size=4).fit(snapshot, seed=11,
                callback=lambda model, step: calls.append(step) or step == 2)
    assert calls == [1, 2]
    assert first.epochs_trained_ == 2
    second = factory(rank=3, epochs=2, batch_size=4).fit(snapshot, seed=11)
    users, items = np.arange(3), np.arange(4)
    np.testing.assert_array_equal(first.score_users(users, items, snapshot.as_of),
                                  second.score_users(users, items, snapshot.as_of))
    first.save(tmp_path / "model.pkl")
    np.testing.assert_array_equal(Model.load(tmp_path / "model.pkl").score_users(users, items, snapshot.as_of),
                                  first.score_users(users, items, snapshot.as_of))


def test_sparse_time_keys_cannot_alias_next_entity():
    np.testing.assert_array_equal(key_lookup(np.array([3, 6, 8]), np.array([0, 3, 8, 9])), [0, 1, 3, 0])


@pytest.mark.parametrize("factory", [FM, TimeSVDPP])
@pytest.mark.parametrize("parameters", [{"rank": 1.5}, {"epochs": True}, {"lr": np.nan},
                                        {"reg": np.inf}, {"batch_size": 0}])
def test_invalid_neural_parameters_fail_before_training(factory, parameters):
    with pytest.raises(ValueError):
        factory(**parameters)


@pytest.mark.parametrize("parameters", [{"bin_days": 1.5}, {"beta": np.inf}, {"temporal_reg": np.nan}])
def test_invalid_time_parameters(parameters):
    with pytest.raises(ValueError):
        TimeSVDPP(**parameters)


def test_fm_catalog_cache_invalidates_between_epochs_and_after_external_edits(snapshot):
    users, items = np.arange(3), np.arange(4)

    def compare(model, step):
        matrix = model.score_users(users, items, snapshot.as_of)
        pairs = model.score_pairs(np.repeat(users, len(items)), np.tile(items, len(users)), snapshot.as_of)
        np.testing.assert_allclose(matrix.ravel(), pairs, rtol=1e-6, atol=1e-6)
        return False

    model = FM(rank=3, epochs=2).fit(snapshot, callback=compare)
    with torch.no_grad():
        model.net.factors.weight.add_(0.05)
    model.invalidate_score_cache()
    compare(model, 0)
    # A different item order/subset must not reuse cached catalogue columns.
    chosen = np.array([3, 0])
    matrix = model.score_users(users, chosen, snapshot.as_of)
    pairs = model.score_pairs(np.repeat(users, 2), np.tile(chosen, 3), snapshot.as_of)
    np.testing.assert_allclose(matrix.ravel(), pairs, rtol=1e-6, atol=1e-6)
