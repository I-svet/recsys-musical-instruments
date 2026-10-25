"""Atomic JSON and provenance helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import psutil


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default,
                                    separators=(",", ":")).encode()).hexdigest()


def code_hash() -> str:
    root = Path(__file__).parent
    return object_hash({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))})


def environment() -> dict:
    packages = ["numpy", "scipy", "pandas", "torch", "matplotlib", "psutil", "threadpoolctl"]
    versions = {p: importlib.metadata.version(p) for p in packages}
    def git(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    from threadpoolctl import threadpool_info
    pools = [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 2), "versions": versions,
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "code_sha256": code_hash(), "threadpools": pools, "device": "cpu"}


class ResourceMonitor:
    """Sample process RSS every 50ms; includes libraries and resident datasets."""
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.started = time.perf_counter()
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        self.seconds = time.perf_counter() - self.started
        self.peak_rss_mib = self.peak / 2**20
