import json

import numpy as np
import pandas as pd
import pytest

from recsys.data import load_dataset, prepare, train_core


def _raw(tmp_path, rows, name="raw.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    return path


def _tiny_rows():
    # A 2-core with one extra fringe edge, a low rating, repeated pairs,
    # boundary events and a selected user's inaccessible positive item.
    return [
        ["a", "u1", 1, 1], ["b", "u1", 5, 2],
        ["a", "u2", 5, 3], ["b", "u2", 4, 4],
        ["a", "u3", 5, 1],
        ["a", "u1", 5, 12],  # Earliest low rating must survive.
        ["outside", "u1", 5, 10],
        ["outside", "u2", 5, 20],
        ["b", "new_user", 5, 11],
        ["invalid", "u1", 9, 1],
    ]


def test_cleaning_train_only_core_and_boundaries(tmp_path):
    output = tmp_path / "prepared"
    manifest = prepare(_raw(tmp_path, _tiny_rows()), output,
                       validation_start=10, test_start=20, core_k=2,
                       enforce_requirements=False)
    data = load_dataset(output)
    assert data.users == ["u1", "u2"]
    assert data.items == ["a", "b"]
    assert len(data.train) == 4
    assert data.train.loc[(data.train.user == 0) & (data.train.item == 0), "rating"].item() == 1
    assert manifest["cleaning"]["invalid_rows"] == 1
    assert manifest["cleaning"]["duplicate_pair_rows_removed"] == 1
    assert not data.validation.shape[0] and not data.test.shape[0]
    assert data.allfuture_counts["validation"].tolist() == [1, 0]
    assert data.allfuture_counts["test"].tolist() == [0, 1]
    assert data.snapshot("validation").as_of == 10
    assert data.snapshot("test").as_of == 20
    assert data.evaluation_users("test", include_outside=True).tolist() == [1]
    assert data.evaluation_users("test").tolist() == []


def test_future_events_cannot_change_training_or_mappings(tmp_path):
    rows = _tiny_rows()
    base = tmp_path / "base"
    modified = tmp_path / "modified"
    settings = dict(validation_start=10, test_start=20, core_k=2, enforce_requirements=False)
    prepare(_raw(tmp_path, rows), base, **settings)
    # Very active future users and items cannot enter the frozen training core.
    rows += [[f"future_item_{i}", f"future_user_{j}", 5, 25] for i in range(5) for j in range(5)]
    rows += [["a", "u1", 4, 30]]
    prepare(_raw(tmp_path, rows, "modified_raw.csv"), modified, **settings)
    for filename in ("train.csv", "users.json", "items.json"):
        assert (base / filename).read_bytes() == (modified / filename).read_bytes()


def test_core_requires_convergence_not_one_pass():
    frame = pd.DataFrame({"user_id": ["a", "a", "b", "b", "c"],
                          "item_id": ["x", "y", "y", "z", "z"], "rating": 5})
    selected, trace = train_core(frame, k=2)
    assert selected.empty
    assert len(trace) >= 3


def test_prepare_is_byte_reproducible_and_load_rejects_corruption(tmp_path):
    raw = _raw(tmp_path, _tiny_rows())
    output = tmp_path / "prepared"
    settings = dict(validation_start=10, test_start=20, core_k=2, enforce_requirements=False)
    first = prepare(raw, output, **settings)
    second = prepare(raw, output, **settings)
    assert first == second
    users = output / "users.json"
    users.write_text(json.dumps(["u2", "u1"]), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(output)


def test_refit_includes_validation_but_never_test(tmp_path):
    # 3 users/3 items, each with 2 train ratings. Remaining pairs cross cutoffs.
    rows = [["a", "u1", 5, 1], ["b", "u1", 4, 2],
            ["b", "u2", 3, 1], ["c", "u2", 4, 2],
            ["a", "u3", 2, 1], ["c", "u3", 5, 2],
            ["c", "u1", 5, 10], ["a", "u2", 4, 20]]
    out = tmp_path / "prepared"
    prepare(_raw(tmp_path, rows), out, validation_start=10, test_start=20, core_k=2, enforce_requirements=False)
    data = load_dataset(out)
    assert len(data.snapshot("validation").interactions) == 6
    refit = data.snapshot("test")
    assert len(refit.interactions) == 7
    assert refit.interactions.timestamp.max() == 10
    assert data.test.timestamp.tolist() == [20]
    assert np.array_equal(data.evaluation_users("validation"), [0])


def test_small_dataset_cannot_silently_claim_assignment_requirements(tmp_path):
    with pytest.raises(ValueError, match="size constraints"):
        prepare(_raw(tmp_path, _tiny_rows()), tmp_path / "prepared",
                validation_start=10, test_start=20, core_k=2)
