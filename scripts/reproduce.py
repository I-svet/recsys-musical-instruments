"""One-command reproduction after installation; stops on the first failed step."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/benchmark.json")
parser.add_argument("--output", default="artifacts/benchmark")
parser.add_argument("--report", default="reports")
parser.add_argument("--skip-metadata", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
def run(*arguments):
    subprocess.run([sys.executable, "-m", "recsys", *arguments], cwd=root, check=True)

run("download", *([] if args.skip_metadata else ["--metadata"]))
run("prepare")
if not args.skip_metadata:
    run("metadata")
run("run", "--config", args.config, "--output", args.output)
run("report", "--runs", args.output, "--output", args.report)
