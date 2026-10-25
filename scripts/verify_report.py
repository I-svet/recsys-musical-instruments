"""Verify published report bytes without downloading data or loading checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?", default="reports")
args = parser.parse_args()
root = Path(args.directory).resolve()


def verify(directory, filename):
    manifest = json.loads((directory / filename).read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Manifest path escapes report directory: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    return manifest


report = verify(root, "report_manifest.json")
audit = verify(root / "audit", "manifest.json")
if report["experiment_fingerprint"] != audit["experiment_fingerprint"]:
    raise ValueError("Report and audit belong to different experiments")
if report["status"] != "complete" or not report["all_five_models"]:
    raise ValueError("The report does not contain all five completed models")
print(json.dumps({"status": "verified", "report_files": len(report["files"]),
                  "audit_files": len(audit["files"]), "final_runs": report["n_final_runs"],
                  "experiment_fingerprint": report["experiment_fingerprint"]}, indent=2))
