"""Deterministic temporal preparation of Amazon ratings with a train-only cohort."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .core import FitSnapshot


VALIDATION_START = 1492387200  # 2017-04-17 00:00 UTC
TEST_START = 1512691200  # 2017-12-08 00:00 UTC
SOURCE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
    "categoryFilesSmall/Musical_Instruments.csv"
)
SCHEMA_VERSION = 1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value, path: Path) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "users": int(frame.user_id.nunique()),
        "items": int(frame.item_id.nunique()),
        "positive_rows": int((frame.rating >= 4).sum()),
        "positive_users": int(frame.loc[frame.rating >= 4, "user_id"].nunique()),
    }


def train_core(frame: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, list[dict]]:
    """Repeated simultaneous degree filtering; caller must pass training events only."""
    if k < 1:
        raise ValueError("core_k must be positive")
    trace = []
    while True:
        previous_rows = len(frame)
        user_count = frame.user_id.value_counts()
        item_count = frame.item_id.value_counts()
        frame = frame[
            frame.user_id.isin(user_count[user_count >= k].index)
            & frame.item_id.isin(item_count[item_count >= k].index)
        ]
        trace.append(_summary(frame))
        if len(frame) == previous_rows:
            return frame.copy(), trace


def prepare(
    raw_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_start: int = VALIDATION_START,
    test_start: int = TEST_START,
    core_k: int = 5,
    enforce_requirements: bool = True,
) -> dict:
    """Create frozen IDs, clean splits and outside-catalog coverage diagnostics.

    Cutoffs are the pre-registered UTC day boundaries from the design audit. They
    are parameters to permit small synthetic tests, not recomputed on each run.
    The source is never modified. ``manifest.json`` is the completion marker;
    every output is hashed and load_dataset refuses a partial or altered dataset.
    """
    raw_csv, output_dir = Path(raw_csv), Path(output_dir)
    if not 0 < validation_start < test_start:
        raise ValueError("Require 0 < validation_start < test_start")
    frame = pd.read_csv(
        raw_csv,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
        dtype={"item_id": "string", "user_id": "string"},
    )
    raw_rows = len(frame)
    frame["source_row"] = np.arange(raw_rows)
    frame["rating"] = pd.to_numeric(frame.rating, errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame.timestamp, errors="coerce")
    valid = (
        frame[["user_id", "item_id", "rating", "timestamp"]].notna().all(axis=1)
        & frame.user_id.str.len().gt(0).fillna(False)
        & frame.item_id.str.len().gt(0).fillna(False)
        & frame.rating.isin([1, 2, 3, 4, 5])
        & np.isfinite(frame.timestamp)
        & frame.timestamp.ge(0)
        & frame.timestamp.lt(np.iinfo(np.int64).max)
        & frame.timestamp.mod(1).eq(0)
    )
    invalid_rows = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["timestamp"] = frame.timestamp.astype(np.int64)
    frame["rating"] = frame.rating.astype(np.float64)
    frame = frame.sort_values(["timestamp", "source_row"], kind="stable")
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="first")
    duplicate_rows = raw_rows - invalid_rows - len(frame)
    raw_splits = {
        "train": frame.loc[frame.timestamp < validation_start],
        "validation": frame.loc[
            (frame.timestamp >= validation_start) & (frame.timestamp < test_start)
        ],
        "test": frame.loc[frame.timestamp >= test_start],
    }
    selected_train, trace = train_core(raw_splits["train"], core_k)
    if selected_train.empty:
        raise ValueError("Training core is empty")
    users = sorted(selected_train.user_id.unique().tolist())
    items = sorted(selected_train.item_id.unique().tolist())
    user_map, item_map = dict(zip(users, range(len(users)))), dict(zip(items, range(len(items))))
    selected_splits = {"train": selected_train}
    diagnostic_counts = pd.DataFrame({"user": np.arange(len(users), dtype=np.int64)})
    exclusions = {}
    for split in ("validation", "test"):
        raw_split = raw_splits[split]
        known_user, known_item = raw_split.user_id.isin(users), raw_split.item_id.isin(items)
        selected_splits[split] = raw_split.loc[known_user & known_item]
        all_positive = raw_split.loc[known_user & raw_split.rating.ge(4)]
        all_counts = all_positive.groupby("user_id").size()
        diagnostic_counts[split] = all_counts.reindex(users, fill_value=0).to_numpy(dtype=np.int64)
        reachable = selected_splits[split].loc[selected_splits[split].rating >= 4]
        outside_only = set(all_positive.user_id) - set(reachable.user_id)
        exclusions[split] = {
            "outside_user_cohort_rows": int((~known_user).sum()),
            "outside_item_cohort_rows": int((~known_item).sum()),
            "outside_either_cohort_rows": int((~(known_user & known_item)).sum()),
            "selected_users_all_catalog_events": _summary(raw_split.loc[known_user]),
            "selected_users_outside_catalog_positive_rows": len(all_positive) - len(reachable),
            "selected_users_positive_only_outside_catalog": len(outside_only),
            "positive_event_catalog_coverage": len(reachable) / len(all_positive) if len(all_positive) else None,
        }
    total = sum(map(len, selected_splits.values()))
    meets_requirements = 100_000 <= total <= 2_000_000 and len(users) >= 1000 and len(items) >= 1000
    if enforce_requirements and not meets_requirements:
        raise ValueError(f"Assignment size constraints not met: {total} events, {len(users)} users, {len(items)} items")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(users, output_dir / "users.json")
    _write_json(items, output_dir / "items.json")
    _write_csv(diagnostic_counts, output_dir / "allfuture_counts.csv")
    for name, selected in selected_splits.items():
        encoded = pd.DataFrame({
            "user": selected.user_id.map(user_map).to_numpy(dtype=np.int64),
            "item": selected.item_id.map(item_map).to_numpy(dtype=np.int64),
            "rating": selected.rating.to_numpy(dtype=np.float64),
            "timestamp": selected.timestamp.to_numpy(dtype=np.int64),
        })
        encoded = encoded.sort_values(["timestamp", "user", "item"], kind="stable")
        _write_csv(encoded, output_dir / f"{name}.csv")
    filenames = ["users.json", "items.json", "allfuture_counts.csv", "train.csv", "validation.csv", "test.csv"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {"name": raw_csv.name, "url": SOURCE_URL, "bytes": raw_csv.stat().st_size, "sha256": sha256(raw_csv)},
        "protocol": {
            "validation_start": int(validation_start), "test_start": int(test_start),
            "intervals": "train < validation_start; validation_start <= validation < test_start; test >= test_start",
            "cutoff_origin": "fixed UTC day boundaries pre-registered by full-data 0.8/0.9 lower-quantile audit",
            "core_k": core_k, "core_source": "train_only", "positive_rating_threshold": 4,
            "deduplication": "earliest timestamp then source row for each user-item pair",
            "id_order": "lexicographic external IDs",
        },
        "cleaning": {"raw_rows": raw_rows, "invalid_rows": invalid_rows, "duplicate_pair_rows_removed": duplicate_rows, "clean_rows": len(frame)},
        "n_users": len(users), "n_items": len(items), "total_interactions": total,
        "meets_assignment_requirements": meets_requirements,
        "splits": {name: _summary(selected) for name, selected in selected_splits.items()},
        "raw_splits": {name: _summary(selected) for name, selected in raw_splits.items()},
        "core_trace": trace, "exclusions": exclusions,
        "exclusions_note": "outside cohort does not necessarily mean temporally unseen/cold",
        "files": {filename: sha256(output_dir / filename) for filename in filenames},
    }
    _write_json(manifest, output_dir / "manifest.json")
    return manifest


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    users: list[str]
    items: list[str]
    manifest: dict
    allfuture_counts: dict[str, np.ndarray]

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def validation_start(self) -> int:
        return self.manifest["protocol"]["validation_start"]

    @property
    def test_start(self) -> int:
        return self.manifest["protocol"]["test_start"]

    def snapshot(self, split: str = "validation") -> FitSnapshot:
        if split == "validation":
            frame, cutoff = self.train.copy(), self.validation_start
        elif split == "test":
            frame, cutoff = pd.concat([self.train, self.validation], ignore_index=True), self.test_start
        else:
            raise ValueError("split must be validation or test")
        mapping_hash = hashlib.sha256(json.dumps([self.users, self.items], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        return FitSnapshot(frame, len(self.users), len(self.items), cutoff, mapping_hash=mapping_hash)

    def holdout(self, split: str) -> pd.DataFrame:
        if split not in ("validation", "test"):
            raise ValueError("split must be validation or test")
        return getattr(self, split)

    def evaluation_users(self, split: str, include_outside: bool = False) -> np.ndarray:
        if include_outside:
            self.holdout(split)  # Validate name before accessing diagnostics.
            return np.flatnonzero(self.allfuture_counts[split] > 0)
        frame = self.holdout(split)
        return np.sort(frame.loc[frame.rating >= 4, "user"].unique()).astype(np.int64)


def load_dataset(path: str | Path, *, verify_hashes: bool = True) -> PreparedData:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema")
    if verify_hashes:
        for filename, expected in manifest["files"].items():
            if Path(filename).name != filename:
                raise ValueError("Manifest filename must be local")
            if sha256(path / filename) != expected:
                raise ValueError(f"Dataset hash mismatch: {filename}")
    users = json.loads((path / "users.json").read_text(encoding="utf-8"))
    items = json.loads((path / "items.json").read_text(encoding="utf-8"))
    if users != sorted(set(users)) or items != sorted(set(items)):
        raise ValueError("Mappings are not unique lexicographic IDs")
    if len(users) != manifest["n_users"] or len(items) != manifest["n_items"]:
        raise ValueError("Mapping sizes do not match manifest")
    frames = {split: pd.read_csv(path / f"{split}.csv", dtype={"user": np.int64, "item": np.int64, "rating": np.float64, "timestamp": np.int64}) for split in ("train", "validation", "test")}
    counts = pd.read_csv(path / "allfuture_counts.csv", dtype=np.int64)
    if not np.array_equal(counts.user.to_numpy(), np.arange(len(users))) or (counts[["validation", "test"]] < 0).any().any():
        raise ValueError("Invalid all-future diagnostic counts")
    result = PreparedData(**frames, users=users, items=items, manifest=manifest,
                          allfuture_counts={split: counts[split].to_numpy() for split in ("validation", "test")})
    result.snapshot("validation")
    result.snapshot("test")
    if (result.validation.timestamp < result.validation_start).any() or (result.test.timestamp < result.test_start).any():
        raise ValueError("Holdout contains past events")
    all_events = pd.concat(frames.values(), ignore_index=True)
    if all_events.duplicated(["user", "item"]).any():
        raise ValueError("Repeated user-item pair crosses splits")
    for split, frame in frames.items():
        if len(frame) != manifest["splits"][split]["rows"]:
            raise ValueError("Split size mismatch")
        if not frame.user.between(0, len(users) - 1).all() or not frame.item.between(0, len(items) - 1).all():
            raise ValueError("Holdout IDs outside frozen mapping")
        if not frame.rating.isin([1, 2, 3, 4, 5]).all():
            raise ValueError("Invalid rating")
    return result
