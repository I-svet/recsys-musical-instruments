"""One-off design feasibility audit; no fitted recommender or test score."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "Musical_Instruments.csv", header=None,
                 names=["item_id", "user_id", "rating", "timestamp"],
                 dtype={"item_id": "string", "user_id": "string"})

def summary(frame):
    return {"rows": len(frame), "users": int(frame.user_id.nunique()),
            "items": int(frame.item_id.nunique()),
            "positive_rows_rating_ge4": int((frame.rating >= 4).sum()),
            "rating_counts": {str(k): int(v) for k, v in frame.rating.value_counts().sort_index().items()}}

def iso(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()

raw = summary(df)
invalid_mask = df.isna().any(axis=1) | ~df.rating.isin([1, 2, 3, 4, 5]) | (df.timestamp < 0)
invalid = int(invalid_mask.sum())
df = df[~invalid_mask].copy()
df["source_row"] = np.arange(len(df))
df = df.sort_values(["timestamp", "source_row"], kind="stable").drop_duplicates(["user_id", "item_id"], keep="first")
clean = summary(df)
cut80 = int(df.timestamp.quantile(.8, interpolation="lower"))
cut90 = int(df.timestamp.quantile(.9, interpolation="lower"))
train = df[df.timestamp <= cut80].copy()
val = df[(df.timestamp > cut80) & (df.timestamp <= cut90)].copy()
test = df[df.timestamp > cut90].copy()

def core(frame, k=5):
    trace = []
    while True:
        before = len(frame)
        users = frame.user_id.value_counts()
        items = frame.item_id.value_counts()
        frame = frame[frame.user_id.isin(users[users >= k].index) & frame.item_id.isin(items[items >= k].index)]
        trace.append(summary(frame))
        if len(frame) == before:
            return frame, trace

def cohort(train_subset):
    users, items = set(train_subset.user_id), set(train_subset.item_id)
    retained = {}
    for name, frame in [("train", train_subset), ("validation", val), ("test", test)]:
        known_u, known_i = frame.user_id.isin(users), frame.item_id.isin(items)
        selected = frame[known_u & known_i]
        stats = summary(selected)
        stats["eligible_users_with_positive"] = int(selected[selected.rating >= 4].user_id.nunique())
        if name != "train":
            stats["excluded_unknown_user_rows"] = int((~known_u).sum())
            stats["excluded_unknown_item_rows"] = int((~known_i).sum())
            stats["excluded_either_rows"] = int((~(known_u & known_i)).sum())
            # These names mean outside the frozen cohort, not necessarily cold-start.
            stats["excluded_rows_label_note"] = "unknown means not in frozen train-only core; not proof of temporal cold-start"
            all_selected_user_events = frame[known_u]
            all_positive_users = set(all_selected_user_events.loc[all_selected_user_events.rating >= 4, "user_id"])
            in_catalog_positive_users = set(selected.loc[selected.rating >= 4, "user_id"])
            stats["selected_users_all_catalog_events"] = summary(all_selected_user_events)
            stats["selected_users_outside_cohort_item_positive_rows"] = int((known_u & ~known_i & (frame.rating >= 4)).sum())
            stats["selected_users_with_positive_only_outside_catalog"] = len(all_positive_users - in_catalog_positive_users)
        retained[name] = stats
    retained["train_users_without_positive"] = retained["train"]["users"] - retained["train"]["eligible_users_with_positive"]
    retained["train_users_without_positive_share"] = retained["train_users_without_positive"] / retained["train"]["users"]
    retained["all_retained_rows"] = sum(retained[name]["rows"] for name in ["train", "validation", "test"])
    retained["meets_assignment_counts"] = retained["all_retained_rows"] >= 100_000 and len(users) >= 1000 and len(items) >= 1000
    n_items = len(items)
    retained["ease_one_float64_matrix_gib"] = n_items * n_items * 8 / 1024**3
    return retained

train_core, trace = core(train)
result = {
    "download": json.loads((root / "download_manifest.json").read_text()),
    "raw": raw, "invalid_rows": invalid, "unique_user_item_earliest": clean,
    "duplicate_user_item_rows_removed": raw["rows"] - invalid - clean["rows"],
    "time_range": [iso(df.timestamp.min()), iso(df.timestamp.max())],
    "all_timestamps_are_utc_midnight": bool((df.timestamp % 86400 == 0).all()),
    "cutoffs": {"quantiles_on": "deduplicated complete rating events, before cohort selection", "train_end_inclusive_utc": iso(cut80), "validation_end_inclusive_utc": iso(cut90), "train_end_timestamp": cut80, "validation_end_timestamp": cut90, "boundary_rule": "whole UTC day; <= end included in preceding split"},
    "raw_time_split": {"train": summary(train), "validation": summary(val), "test": summary(test)},
    "train_only_5core_trace": trace,
    "train_only_5core_cohort": cohort(train_core),
}
result["cutoffs"]["half_open_validation_start_utc"] = iso(cut80 + 86400)
result["cutoffs"]["half_open_test_start_utc"] = iso(cut90 + 86400)
result["cutoffs"]["half_open_rule"] = "train timestamp < validation_start; validation_start <= timestamp < test_start; test timestamp >= test_start"
if train_core.item_id.nunique() > 12000:
    ranking = train_core.groupby("item_id").size().reset_index(name="count").sort_values(["count", "item_id"], ascending=[False, True], kind="stable")
    allowed = ranking.item_id.head(12000)
    capped, capped_trace = core(train_core[train_core.item_id.isin(allowed)])
    result["train_top12000_then_recore"] = {"trace": capped_trace, "cohort": cohort(capped)}
out = root / "ratings_audit.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
