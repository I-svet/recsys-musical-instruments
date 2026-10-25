import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.data import load_dataset, prepare
from recsys.io import sha256, write_json
from recsys.metrics import evaluate
from recsys.report import _example_users, build_report


@pytest.fixture
def completed_experiment(tmp_path):
    raw = tmp_path / "source.csv"
    events = []
    for user in range(3):
        for offset in range(5):
            item = user * 5 + offset
            events.append((f"PRODUCT_{item:02}", f"REVIEWER_SECRET_{user}", 5, item + 1))
    for user in range(3):
        events.append((f"PRODUCT_{((user + 1) % 3) * 5:02}", f"REVIEWER_SECRET_{user}", 4, 20 + user))
        events.append((f"PRODUCT_{((user + 1) % 3) * 5 + 1:02}", f"REVIEWER_SECRET_{user}", 5, 30 + user))
    pd.DataFrame(events).to_csv(raw, index=False, header=False)
    data_dir, runs_dir = tmp_path / "data", tmp_path / "runs"
    prepare(raw, data_dir, validation_start=20, test_start=30, core_k=1, enforce_requirements=False)
    data = load_dataset(data_dir)
    models = ["popular", "knn", "als", "ease", "fm", "timesvd"]
    config = {"search": {model: [{}] for model in models}, "k": 3,
              "tuning_seed": 42, "seeds": [17, 42], "ablations": True}
    experiment = {"fingerprint": "synthetic-test-fingerprint", "config": config,
                  "environment": {"python": "test", "device": "cpu"}, "data": data.manifest}
    write_json(runs_dir / "experiment.json", experiment)
    users = data.evaluation_users("test", include_outside=True)
    seen = data.snapshot("test").matrix()
    for model in models + ["fm_id", "svdpp"]:
        for seed in ([42] if model in {"popular", "knn", "ease"} else [17, 42]):
            matrix = []
            for user in users:
                candidates = np.setdiff1d(np.arange(len(data.items)), seen[user].indices)
                positive = int(data.test.loc[data.test.user == user, "item"].iloc[0])
                # Genuine lists with model-dependent quality, not fabricated aggregate metrics.
                if model == "timesvd" or (model == "als" and user == 0):
                    chosen = np.r_[positive, candidates[candidates != positive][:2]]
                else:
                    chosen = candidates[candidates != positive][:3]
                matrix.append(chosen)
            matrix = np.array(matrix)
            per_user, metrics = evaluate(matrix, users, data.test, len(data.items), k=3,
                                         allfuture_counts=data.allfuture_counts["test"])
            per_user["history_count"], per_user["fallback"] = 6, False
            directory = runs_dir / "final" / model / str(seed)
            directory.mkdir(parents=True)
            per_user.to_csv(directory / "per_user.csv", index=False)
            pd.DataFrame({"user": np.repeat(users, 3), "rank": np.tile([1, 2, 3], len(users)),
                          "item": matrix.ravel(), "score": np.tile([3., 2., 1.], len(users)),
                          "fallback": False}).to_csv(directory / "recommendations.csv", index=False)
            (directory / "model.pkl").write_bytes(b"Not deserialized by report")
            run = {"status": "complete", "fingerprint": experiment["fingerprint"], "model": model,
                   "seed": seed, "parameters": {}, "metrics": metrics, "validation": metrics,
                   "fit_seconds": 0.1 + seed / 100, "test_seconds": 0.02, "peak_rss_mib": 100.,
                   "files": {p: sha256(directory / p) for p in ["model.pkl", "per_user.csv", "recommendations.csv"]}}
            write_json(directory / "run.json", run)
        if model in models:
            trial = {"status": "complete", "fingerprint": experiment["fingerprint"],
                     "parameters": {}, "best_epoch": 1, "validation": metrics, "fit_seconds": 0.1}
            write_json(runs_dir / "trials" / model / "000.json", trial)
            write_json(runs_dir / "selected" / f"{model}.json",
                       {"trial": 0, "parameters": {}, "best_epoch": 1, "validation": metrics})
    return data_dir, runs_dir, tmp_path / "report"


def test_report_recomputes_metrics_exports_no_review_ids_and_keeps_expected_seeds(completed_experiment):
    data_dir, runs_dir, output = completed_experiment
    build_report(data_dir, runs_dir, output)
    summary = pd.read_csv(output / "summary.csv").set_index("model")
    assert summary.loc["timesvd", "ndcg@3_mean"] == pytest.approx(1.)
    assert summary.loc["als", "ndcg@3_mean"] == pytest.approx(1 / 3)
    assert summary.loc["timesvd", "n_seeds"] == 2
    assert summary.loc["ease", "n_seeds"] == 1
    contrasts = json.loads((output / "contrasts.json").read_text())
    assert contrasts[0]["role"] == "primary"
    assert contrasts[0]["difference"] == pytest.approx(2 / 3)
    assert contrasts[0]["n_bootstrap"] == 2000
    assert contrasts[0]["n_users"] == 3
    assert len(list(output.rglob("model.pkl"))) == 0
    assert not (output / "users.json").exists()
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md"}:
            assert "REVIEWER_SECRET" not in path.read_text(encoding="utf-8")
    assert (output / "ndcg.png").stat().st_size > 1000
    assert (output / "quality_vs_time.png").stat().st_size > 1000
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["all_five_models"] is True
    assert manifest["n_final_runs"] == 13
    for filename, digest in manifest["files"].items():
        assert sha256(output / filename) == digest
    assert json.loads((output / "example_selection.json").read_text())["users"] == _example_users(load_dataset(data_dir))


def test_report_refuses_missing_final_seed(completed_experiment):
    data_dir, runs_dir, output = completed_experiment
    (runs_dir / "final" / "als" / "17" / "run.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        build_report(data_dir, runs_dir, output)
    assert not (output / "report_manifest.json").exists()


def test_report_refuses_corrupted_artifact(completed_experiment):
    data_dir, runs_dir, output = completed_experiment
    with (runs_dir / "final" / "ease" / "42" / "recommendations.csv").open("a") as stream:
        stream.write("corruption")
    with pytest.raises(ValueError, match="checksum"):
        build_report(data_dir, runs_dir, output)


def test_report_refuses_mixed_fingerprints(completed_experiment):
    data_dir, runs_dir, output = completed_experiment
    path = runs_dir / "final" / "fm" / "42" / "run.json"
    run = json.loads(path.read_text())
    run["fingerprint"] = "other-experiment"
    write_json(path, run)
    with pytest.raises(ValueError, match="fingerprint"):
        build_report(data_dir, runs_dir, output)


def test_report_checks_metrics_against_recommendation_lists(completed_experiment):
    data_dir, runs_dir, output = completed_experiment
    path = runs_dir / "final" / "fm" / "42" / "run.json"
    run = json.loads(path.read_text())
    run["metrics"]["ndcg@3"] = 0.7
    write_json(path, run)
    with pytest.raises(ValueError, match="Aggregate metric"):
        build_report(data_dir, runs_dir, output)
