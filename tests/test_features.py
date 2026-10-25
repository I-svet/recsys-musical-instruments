import numpy as np
import pandas as pd
import pytest

from recsys.core import FitSnapshot
from recsys.features import DAY, HistoricalFeatures, prior_history


def frame():
    return pd.DataFrame({"user": [0, 0, 1, 0, 1], "item": [0, 1, 0, 2, 1],
                         "rating": [5., 1., 4., 2., 5.],
                         "timestamp": np.array([1, 1, 2, 3, 4]) * DAY})


def test_prior_features_exclude_entire_current_day():
    result = prior_history(frame(), "user")
    np.testing.assert_allclose(result[:2], [[0, 3, 0], [0, 3, 0]])
    np.testing.assert_allclose(result[3], [np.log(3), 3, np.log(3)])
    changed = frame()
    changed.loc[3, "rating"] = 5
    np.testing.assert_array_equal(prior_history(changed, "user")[3], result[3])


def test_item_features_use_only_past_and_preserve_row_order():
    source = frame().iloc[[4, 3, 1, 2, 0]]
    values = prior_history(source, "item")
    np.testing.assert_allclose(values[0], [np.log(2), 1, np.log(4)])
    np.testing.assert_allclose(values[-1], [0, 3, 0])


def test_frozen_features_are_independent_of_later_labels():
    snapshot = FitSnapshot(frame(), 3, 4, 5 * DAY)
    builder = HistoricalFeatures()
    ui, uv, ii, iv = builder.fit_transform(snapshot)
    assert ui.shape == uv.shape == (5, 6)
    assert ii.shape == iv.shape == (5, 4)
    before = builder.user_features(np.array([0, 1, 2]), 5 * DAY)
    snapshot.interactions.loc[:, "rating"] = 99
    after = builder.user_features(np.array([0, 1, 2]), 5 * DAY)
    np.testing.assert_array_equal(before[1], after[1])
    unknown = builder._stats("item", 5 * DAY)[3]
    np.testing.assert_array_equal(unknown, [0, 3, 0])
    with pytest.raises(ValueError):
        builder.user_features(np.array([0]), 2 * DAY)


def test_intraday_query_excludes_same_day():
    builder = HistoricalFeatures()
    builder.fit_transform(FitSnapshot(frame(), 3, 4, 4 * DAY + 1))
    raw = builder._stats("user", 4 * DAY + 1)
    np.testing.assert_allclose(raw[1, :2], [np.log(2), 4])
