"""Downloads from the authors' source; raw data are never committed to Git."""
from __future__ import annotations

import gzip
import html
import json
import os
from pathlib import Path
import urllib.request

import pandas as pd

from .io import sha256, write_json

RATINGS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments.csv"
METADATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz"
RATINGS_SHA256 = "8933b17434e4984bcce8cf4ec43d847e0af6aa6d46add28bbb37b80e06da1b06"
METADATA_SHA256 = "8e3f0dbe4c6c1aa5b32d350aa26072232beb6160ff3c5dadb44bca2c473a8b41"


def fetch(url: str, target: Path, expected: str | None = None,
          max_bytes: int = 1_000_000_000) -> dict:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(target.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "RecsysCourseBenchmark/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > max_bytes:
                    raise ValueError(f"Download exceeds {max_bytes} bytes")
                out.write(block)
        if expected and sha256(temporary) != expected:
            raise ValueError("Source checksum mismatch; partial file preserved for inspection")
        os.replace(temporary, target)
    digest = sha256(target)
    if expected and digest != expected:
        raise ValueError("Existing source file checksum mismatch")
    manifest = {"url": url, "file": target.name, "bytes": target.stat().st_size, "sha256": digest}
    write_json(target.with_name(target.name + ".manifest.json"), manifest)
    return manifest


def download_ratings(directory: Path) -> Path:
    target = Path(directory) / "Musical_Instruments.csv"
    fetch(RATINGS_URL, target, RATINGS_SHA256, max_bytes=100_000_000)
    return target


def prepare_metadata(raw_path: Path, items: list[str], output_dir: Path) -> dict:
    """Left join titles to the frozen catalog; absent metadata never removes events."""
    if sha256(raw_path) != METADATA_SHA256:
        raise ValueError("Metadata source checksum mismatch")
    wanted = set(items)
    records = {}
    source_rows = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source_rows += 1
            asin = row.get("asin")
            if asin in wanted and asin not in records:
                title = row.get("title") or ""
                if isinstance(title, list):
                    title = " ".join(str(x) for x in title)
                categories = row.get("category", [])
                records[asin] = {"title": " ".join(html.unescape(str(title)).split()),
                                 "brand": str(row.get("brand") or ""),
                                 "categories": json.dumps(categories, ensure_ascii=False)}
    rows = [{"item": idx, "asin": asin, **records.get(asin, {
        "title": "", "brand": "", "categories": "[]"})} for idx, asin in enumerate(items)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)
    manifest = {"source_rows": source_rows, "catalog_items": len(items),
                "matched_items": len(records), "title_items": sum(bool(r["title"]) for r in rows),
                "source_sha256": sha256(raw_path), "file_sha256": sha256(output_dir / "metadata.csv"),
                "use": "Report labels only. Snapshot metadata are not historical training features."}
    write_json(output_dir / "metadata_manifest.json", manifest)
    return manifest
