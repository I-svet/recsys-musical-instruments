"""Small real-model integration runs for the temporal experiment protocol."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from recsys.core import Model
from recsys.data import prepare
from recsys.experiment import Experiment
from recsys.io import sha256
from recsys.ranking import recommend


def _case(tmp_path):
    rows = []
    for user in range(12):
        for offset in range(5):
            rows.append([f"i{(2 * user + offset) % 24:02}", f"u{user:02}",
                         1 if offset == 0 else 4 + (offset % 2), 100 + offset])
        rows.append([f"i{(2 * user + 8) % 24:02}", f"u{user:02}", 5, 1000])
        rows.append([f"i{(2 * user + 9) % 24:02}", f"u{user:02}", 4, 2000])
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, header=False, index=False)
    data_dir = tmp_path / "prepared"
    prepare(raw, data_dir, validation_start=1000, test_start=2000,
            core_k=1, enforce_requirements=False)
    config = {
        "k": 10, "threads": 1, "batch_size": 4, "patience": 2,
        "tuning_seed": 42, "seeds": [42], "ablations": False,
        "search": {
            "popular": [{}],
            "knn": [{"k": 3, "gamma": 0}, {"k": 8, "gamma": 5}],
            "als": [{"rank": 3, "reg": 1., "bias_reg": 1., "epochs": 2}],
            "ease": [{"reg": 3.}],
            "fm": [{"rank": 3, "lr": .01, "reg": .01, "epochs": 2, "batch_size": 64}],
            "timesvd": [{"rank": 3, "lr": .01, "reg": .01, "temporal_reg": .1,
                         "epochs": 2, "batch_size": 64, "bin_days": 30}],
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return data_dir, config_path, config


def test_all_five_models_end_to_end_and_verified_resume(tmp_path, monkeypatch):
    data_dir, config_path, config = _case(tmp_path)
    output = tmp_path / "experiment"
    fits = []
    original_bind = Model.bind_snapshot

    def record_snapshot(model, snapshot):
        # Records actual data supplied to every model, including final refits.
        fits.append((type(model).__name__, snapshot.as_of, snapshot.interactions.copy()))
        return original_bind(model, snapshot)

    monkeypatch.setattr(Model, "bind_snapshot", record_snapshot)
    with threadpool_limits(limits=1):
        experiment = Experiment(data_dir, config_path, output)
        selected = experiment.tune()
        experiment.final()
    assert set(selected) == {"popular", "knn", "als", "ease", "fm", "timesvd"}
    assert {cutoff for _, cutoff, _ in fits} == {1000, 2000}
    assert all((frame.timestamp < cutoff).all() for _, cutoff, frame in fits)
    assert all(len(frame) == (60 if cutoff == 1000 else 72) for _, cutoff, frame in fits)
    assert len({name for name, _, _ in fits}) == 6
    snapshot = experiment.data.snapshot("test")
    observed = snapshot.interactions.groupby("user").item.agg(set).to_dict()
    saved_hashes = {}
    for name in config["search"]:
        run_dir = output / "final" / name / "42"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run["status"] == "complete"
        assert run["fingerprint"] == experiment.fingerprint
        assert run["metrics"]["n_users"] == 12
        for metric in ("ndcg@10", "recall@10", "precision@10", "map@10", "recall_all_future@10"):
            assert 0 <= run["metrics"][metric] <= 1
        if name in {"als", "fm", "timesvd"}:
            assert np.isfinite(run["metrics"]["rmse"])
            assert run["parameters"]["epochs"] == selected[name]["best_epoch"]
        for filename, digest in run["files"].items():
            assert sha256(run_dir / filename) == digest
            saved_hashes[(name, filename)] = digest
        recommendations = pd.read_csv(run_dir / "recommendations.csv")
        assert len(recommendations) == 120
        for user, group in recommendations.groupby("user"):
            assert group["rank"].tolist() == list(range(1, 11))
            assert len(set(group.item)) == 10
            assert set(group.item).isdisjoint(observed[user])
        # A persisted checkpoint must reproduce the exported recommendation list.
        restored = Model.load(run_dir / "model.pkl")
        items, scores, _ = recommend(restored, snapshot, np.arange(12))
        np.testing.assert_array_equal(items.ravel(), recommendations.item.to_numpy())
        np.testing.assert_allclose(scores.ravel(), recommendations.score.to_numpy())

    # Existing complete runs should not instantiate or train another model.
    def forbidden_model(*args, **kwargs):
        raise AssertionError("A completed experiment retrained instead of resuming")

    monkeypatch.setattr("recsys.experiment.make_model", forbidden_model)
    resumed = Experiment(data_dir, config_path, output)
    assert resumed.tune() == selected
    resumed.final()
    for (name, filename), digest in saved_hashes.items():
        assert sha256(output / "final" / name / "42" / filename) == digest

    # Corruption of a completed run is a failure, never a cache hit.
    damaged = output / "final" / "popular" / "42" / "recommendations.csv"
    damaged.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        resumed.final(["popular"])


def test_test_labels_cannot_change_validation_selection(tmp_path, monkeypatch):
    data_dir, config_path, _ = _case(tmp_path)
    first = Experiment(data_dir, config_path, tmp_path / "first")
    changed = Experiment(data_dir, config_path, tmp_path / "changed")
    changed.data.test.loc[:, "rating"] = 1.
    changed.data.test.loc[:, "item"] = 0
    changed.data.test.loc[:, "timestamp"] = 999999
    changed.data.allfuture_counts["test"][:] = 0
    fits = []
    original_bind = Model.bind_snapshot

    def require_training_only(model, snapshot):
        assert snapshot.as_of == 1000
        assert len(snapshot.interactions) == 60
        assert snapshot.interactions.timestamp.max() < 1000
        fits.append(type(model).__name__)
        return original_bind(model, snapshot)

    monkeypatch.setattr(Model, "bind_snapshot", require_training_only)
    with threadpool_limits(limits=1):
        selected_first = first.tune()
        selected_changed = changed.tune()
    assert selected_first == selected_changed
    assert len(set(fits)) == 6


def test_changed_configuration_or_prepared_file_cannot_reuse_experiment(tmp_path):
    data_dir, config_path, config = _case(tmp_path)
    output = tmp_path / "experiment"
    Experiment(data_dir, config_path, output)
    config["search"]["ease"][0]["reg"] = 4.
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        Experiment(data_dir, config_path, output)
    # Data corruption fails before the experiment's cache can be reused.
    with (data_dir / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("0,1,5,999\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        Experiment(data_dir, config_path, tmp_path / "new_output")


def test_selection_and_cached_trial_parameters_cannot_be_tampered(tmp_path):
    data_dir, config_path, _ = _case(tmp_path)
    output = tmp_path / "experiment"
    experiment = Experiment(data_dir, config_path, output)
    experiment.tune(["popular"])
    selection_path = output / "selected" / "popular.json"
    selection_original = selection_path.read_text(encoding="utf-8")
    selection = json.loads(selection_original)
    selection["parameters"] = {"unexpected": 1}
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="Selection disagrees"):
        experiment.final(["popular"])

    selection_path.write_text(selection_original, encoding="utf-8")
    trial_path = output / "trials" / "popular" / "000.json"
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    trial["parameters"] = {"unexpected": 1}
    trial_path.write_text(json.dumps(trial), encoding="utf-8")
    with pytest.raises(ValueError, match="Cached trial does not match"):
        experiment.tune(["popular"])
