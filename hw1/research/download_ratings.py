"""Bounded official data download for a one-off design audit, not model code."""
from pathlib import Path
import hashlib
import json
import urllib.request

URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments.csv"
LIMIT = 100_000_000
root = Path(__file__).resolve().parent
target = root / "data" / "Musical_Instruments.csv"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    with urllib.request.urlopen(URL, timeout=30) as response:
        if int(response.headers.get("Content-Length", LIMIT + 1)) > LIMIT:
            raise RuntimeError("Download exceeds 100 MB limit")
        total = 0
        with target.with_suffix(".csv.part").open("wb") as out:
            while block := response.read(1_048_576):
                total += len(block)
                if total > LIMIT:
                    raise RuntimeError("Download exceeds 100 MB limit")
                out.write(block)
        target.with_suffix(".csv.part").replace(target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
manifest = {"url": URL, "bytes": target.stat().st_size, "sha256": digest}
(root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest), flush=True)
