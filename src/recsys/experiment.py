"""Validation-only search, fixed-seed refits, and full-catalog final evaluation."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np

from .data import load_dataset
from .io import ResourceMonitor, code_hash, environment, object_hash, sha256, write_json
from .metrics import evaluate_model
from .registry import make_model

DETERMINISTIC = {"popular", "knn", "ease"}


def configure_runtime(threads: int):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Experiment:
    def __init__(self, data_dir: Path, config_path: Path, output_dir: Path):
        self.data_dir, self.config_path, self.output = Path(data_dir), Path(config_path), Path(output_dir)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        configure_runtime(self.config["threads"])
        self.data = load_dataset(self.data_dir)
        self.snapshots = {split: self.data.snapshot(split) for split in ["validation", "test"]}
        self.output.mkdir(parents=True, exist_ok=True)
        self.provenance = environment()
        self.fingerprint = object_hash({"data": self.data.manifest, "config": self.config,
                                       "code": code_hash(), "versions": self.provenance["versions"],
                                       "python": self.provenance["python"],
                                       "platform": self.provenance["platform"]})
        manifest_path = self.output / "experiment.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous["fingerprint"] != self.fingerprint:
                raise ValueError("Data/code/config/environment changed. Use a new output directory.")
        else:
            write_json(manifest_path, {"fingerprint": self.fingerprint, "config": self.config,
                                      "environment": self.provenance, "data": self.data.manifest})

    def _evaluate(self, model, split):
        return evaluate_model(model, self.snapshots[split], self.data.holdout(split),
                              k=self.config["k"], batch_size=self.config["batch_size"],
                              allfuture_counts=self.data.allfuture_counts[split])

    def _fit_validate(self, name: str, params: dict, seed: int, path: Path) -> dict:
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("model") != name or previous.get("parameters") != params or previous.get("seed") != seed:
                raise ValueError(f"Cached trial does not match requested parameters: {path}")
            if previous.get("status") == "complete" and previous.get("fingerprint") == self.fingerprint:
                return previous
        snapshot = self.snapshots["validation"]
        model = make_model(name, params)
        curve = []
        best_value, best_epoch, best_metrics, stale, evaluation_seconds = -np.inf, 1, None, 0, 0.0
        def callback(current, epoch):
            nonlocal best_value, best_epoch, best_metrics, stale, evaluation_seconds
            started = time.perf_counter()
            metrics = self._evaluate(current, "validation")["metrics"]
            evaluation_seconds += time.perf_counter() - started
            value = metrics[f"ndcg@{self.config['k']}"]
            if not np.isfinite(value):
                raise ValueError("Non-finite validation metric")
            curve.append({"epoch": epoch, "ndcg": value})
            if value > best_value + 1e-12:
                best_value, best_epoch, best_metrics, stale = value, epoch, metrics, 0
            else:
                stale += 1
            write_json(path.with_name(path.stem + ".progress.json"), {
                "model": name, "seed": seed, "epoch": epoch, "best_epoch": best_epoch,
                "best_ndcg": best_value, "curve": curve})
            return stale >= self.config["patience"]
        try:
            with ResourceMonitor() as monitor:
                model.fit(snapshot, seed=seed, callback=None if name in DETERMINISTIC else callback)
                if not curve:
                    callback(model, 1)
            result = {"status": "complete", "fingerprint": self.fingerprint, "model": name,
                      "parameters": params, "seed": seed, "best_epoch": best_epoch,
                      "validation": best_metrics, "curve": curve,
                      "fit_seconds": monitor.seconds - evaluation_seconds,
                      "validation_seconds": evaluation_seconds,
                      "peak_rss_mib": monitor.peak_rss_mib}
            write_json(path, result)
            print(f"{name} seed={seed} best_epoch={best_epoch} val_NDCG={best_value:.6f} "
                  f"fit={result['fit_seconds']:.1f}s", flush=True)
            return result
        except Exception as exc:
            write_json(path, {"status": "failed", "fingerprint": self.fingerprint,
                              "model": name, "parameters": params, "seed": seed,
                              "error": f"{type(exc).__name__}: {exc}", "curve": curve})
            raise
        finally:
            del model
            gc.collect()

    def pilot(self, names=None):
        for name in names or self.config["search"]:
            params = dict(self.config["search"][name][0])
            if "epochs" in params:
                params["epochs"] = min(2, params["epochs"])
            self._fit_validate(name, params, self.config["tuning_seed"], self.output / "pilot" / f"{name}.json")

    def tune(self, names=None):
        selected = {}
        for name in names or self.config["search"]:
            trials = []
            for index, params in enumerate(self.config["search"][name]):
                print(f"Tune {name} {index + 1}/{len(self.config['search'][name])}", flush=True)
                trials.append(self._fit_validate(name, params, self.config["tuning_seed"],
                                                self.output / "trials" / name / f"{index:03d}.json"))
            best_index = max(range(len(trials)), key=lambda i: trials[i]["validation"][f"ndcg@{self.config['k']}"])
            best = trials[best_index]
            selected[name] = {"fingerprint": self.fingerprint, "trial": best_index, "parameters": best["parameters"],
                              "validation": best["validation"], "best_epoch": best["best_epoch"]}
            write_json(self.output / "selected" / f"{name}.json", selected[name])
        return selected

    def _selection(self, name):
        selected = json.loads((self.output / "selected" / f"{name}.json").read_text(encoding="utf-8"))
        index = selected["trial"]
        if not isinstance(index, int) or not 0 <= index < len(self.config["search"][name]):
            raise ValueError("Selected trial outside registered search")
        expected = self.config["search"][name][index]
        trial = json.loads((self.output / "trials" / name / f"{index:03d}.json").read_text(encoding="utf-8"))
        if (selected.get("fingerprint") != self.fingerprint or selected["parameters"] != expected
                or trial.get("status") != "complete" or trial.get("fingerprint") != self.fingerprint
                or trial["model"] != name or trial["seed"] != self.config["tuning_seed"]
                or trial["parameters"] != expected or selected["validation"] != trial["validation"]
                or selected["best_epoch"] != trial["best_epoch"]):
            raise ValueError("Selection disagrees with registered completed trial")
        return selected, trial

    def _final_one(self, name, params, seed, validation_result=None):
        run_dir = self.output / "final" / name / str(seed)
        result_path = run_dir / "run.json"
        if result_path.exists():
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") == self.fingerprint and old.get("status") == "complete":
                for filename, digest in old["files"].items():
                    if sha256(run_dir / filename) != digest:
                        raise ValueError(f"Artifact corrupted: {run_dir / filename}")
                return old
        if validation_result is None:
            validation_result = self._fit_validate(name, params, seed,
                                                   self.output / "seed_validation" / name / f"{seed}.json")
        fit_params = dict(params)
        if "epochs" in fit_params:
            fit_params["epochs"] = validation_result["best_epoch"]
        model = make_model(name, fit_params)
        snapshot = self.snapshots["test"]
        with ResourceMonitor() as monitor:
            start = time.perf_counter()
            model.fit(snapshot, seed=seed)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            evaluated = self._evaluate(model, "test")
            if name in {"als", "fm", "timesvd", "fm_id", "svdpp"}:
                holdout = self.data.test
                predicted = model.score_pairs(holdout.user.to_numpy(), holdout.item.to_numpy(), snapshot.as_of)
                evaluated["metrics"]["rmse"] = float(np.sqrt(np.mean((predicted - holdout.rating.to_numpy()) ** 2)))
            test_seconds = time.perf_counter() - start
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save(run_dir / "model.pkl")
        evaluated["per_user"].to_csv(run_dir / "per_user.csv", index=False)
        import pandas as pd
        recommendations = pd.DataFrame({
            "user": np.repeat(evaluated["users"], self.config["k"]),
            "rank": np.tile(np.arange(1, self.config["k"] + 1), len(evaluated["users"])),
            "item": evaluated["items"].ravel(), "score": evaluated["scores"].ravel(),
            "fallback": np.repeat(evaluated["fallback"], self.config["k"])})
        recommendations.to_csv(run_dir / "recommendations.csv", index=False)
        result = {"status": "complete", "fingerprint": self.fingerprint, "model": name,
                  "seed": seed, "parameters": fit_params,
                  "validation": validation_result["validation"], "metrics": evaluated["metrics"],
                  "fit_seconds": fit_seconds, "test_seconds": test_seconds,
                  "peak_rss_mib": monitor.peak_rss_mib,
                  "fallback_users": int(evaluated["fallback"].sum()),
                  "evaluated_users_including_outside": len(evaluated["users"]),
                  "loss_history": getattr(model, "loss_history", []),
                  "training_history": getattr(model, "history_", []),
                  "files": {p: sha256(run_dir / p) for p in ["model.pkl", "per_user.csv", "recommendations.csv"]}}
        write_json(result_path, result)
        metric_key = f"ndcg@{self.config['k']}"
        print(f"FINAL {name} seed={seed} NDCG={result['metrics'][metric_key]:.6f}", flush=True)
        del model
        gc.collect()
        return result

    def final(self, names=None):
        names = list(names or self.config["search"])
        for name in names:
            selected, selected_trial = self._selection(name)
            for seed in ([self.config["tuning_seed"]] if name in DETERMINISTIC else self.config["seeds"]):
                validation_result = None
                if seed == self.config["tuning_seed"]:
                    validation_result = selected_trial
                self._final_one(name, selected["parameters"], seed, validation_result)
        if self.config.get("ablations", True):
            for name, parent in [("fm_id", "fm"), ("svdpp", "timesvd")]:
                if parent not in names:
                    continue
                selected, _ = self._selection(parent)
                for seed in self.config["seeds"]:
                    self._final_one(name, selected["parameters"], seed)
