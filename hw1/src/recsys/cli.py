"""Command-line entry points. Every path is relative to the current directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Amazon Musical Instruments: own top-10 recommenders")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download and verify official data")
    p.add_argument("--output", type=Path, default=Path("data/raw"))
    p.add_argument("--metadata", action="store_true")
    p = sub.add_parser("prepare", help="Clean, split globally by time, freeze train-only 5-core")
    p.add_argument("--raw", type=Path, default=Path("data/raw/Musical_Instruments.csv"))
    p.add_argument("--output", type=Path, default=Path("data/prepared"))
    p = sub.add_parser("metadata", help="Left-join names for the report; not historical features")
    p.add_argument("--raw", type=Path, default=Path("data/raw/meta_Musical_Instruments.json.gz"))
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p = sub.add_parser("run", help="Pilot, validation search and final refits")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    p.add_argument("--stage", choices=["pilot", "tune", "final", "all"], default="all")
    p.add_argument("--models", nargs="+")
    p = sub.add_parser("report", help="Create tables, bootstrap intervals, plots and examples")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--runs", type=Path, default=Path("artifacts/benchmark"))
    p.add_argument("--output", type=Path, default=Path("reports"))
    p = sub.add_parser("recommend", help="Use a trusted local model checkpoint")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--user", type=int, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args(argv)
    if args.command == "download":
        from .download import download_ratings, fetch, METADATA_URL, METADATA_SHA256
        print(download_ratings(args.output))
        if args.metadata:
            print(fetch(METADATA_URL, args.output / "meta_Musical_Instruments.json.gz", METADATA_SHA256))
    elif args.command == "prepare":
        from .data import prepare
        from .download import RATINGS_SHA256
        from .io import sha256
        if sha256(args.raw) != RATINGS_SHA256:
            raise ValueError("Expected the verified complete 2018 Musical Instruments CSV")
        print(json.dumps(prepare(args.raw, args.output), ensure_ascii=False, indent=2))
    elif args.command == "metadata":
        from .data import load_dataset
        from .download import prepare_metadata
        print(prepare_metadata(args.raw, load_dataset(args.data).items, args.data))
    elif args.command == "run":
        from threadpoolctl import threadpool_limits
        from .experiment import Experiment
        config = json.loads(args.config.read_text(encoding="utf-8"))
        with threadpool_limits(limits=config["threads"]):
            experiment = Experiment(args.data, args.config, args.output)
            for stage in (["pilot", "tune", "final"] if args.stage == "all" else [args.stage]):
                getattr(experiment, stage)(args.models)
    elif args.command == "report":
        from .report import build_report
        build_report(args.data, args.runs, args.output)
    elif args.command == "recommend":
        import numpy as np
        from .core import Model
        from .data import load_dataset
        from .ranking import recommend
        data = load_dataset(args.data)
        items, scores, _ = recommend(Model.load(args.model), data.snapshot(args.split), np.array([args.user]))
        for rank, (item, score) in enumerate(zip(items[0], scores[0]), start=1):
            print(json.dumps({"rank": rank, "item": int(item), "asin": data.items[item], "score": float(score)}))


if __name__ == "__main__":
    main()
