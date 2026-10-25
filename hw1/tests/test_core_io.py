import numpy as np
import pandas as pd
import pytest

from recsys.core import FitSnapshot
from recsys.io import object_hash, sha256, write_json


def test_snapshot_rejects_fractional_ids_and_future():
    frame = pd.DataFrame({"user": [0.5], "item": [0], "rating": [4], "timestamp": [1]})
    with pytest.raises(ValueError, match="integers"):
        FitSnapshot(frame, 2, 2, 2)
    frame.user = 0
    with pytest.raises(ValueError, match="future"):
        FitSnapshot(frame, 2, 2, 1)


def test_fingerprint_covers_history_and_mapping():
    frame = pd.DataFrame({"user": [0], "item": [0], "rating": [4.0], "timestamp": [1]})
    first = FitSnapshot(frame, 2, 2, 2, "mapping1").fingerprint()
    assert first != FitSnapshot(frame, 2, 2, 2, "mapping2").fingerprint()
    assert first != FitSnapshot(frame.assign(rating=5.0), 2, 2, 2, "mapping1").fingerprint()


def test_json_is_atomic_and_rejects_nonfinite(tmp_path):
    target = tmp_path / "record.json"
    write_json(target, {"value": np.int64(7)})
    digest = sha256(target)
    with pytest.raises(ValueError):
        write_json(target, {"value": float("nan")})
    assert sha256(target) == digest
    assert object_hash({"b": 2, "a": 1}) == object_hash({"a": 1, "b": 2})
