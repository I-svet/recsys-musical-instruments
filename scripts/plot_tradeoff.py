"""Render the report's cost/quality plot with separated labels; no model changes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?", default="reports-local")
args = parser.parse_args()
root = Path(args.directory)
manifest_path = root / "report_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source = root / "summary.csv"
if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["files"]["summary.csv"]:
    raise ValueError("Summary differs from the verified report")
summary = pd.read_csv(source)
metric = next(c for c in summary if c.startswith("ndcg@") and c.endswith("_mean"))
metric_std = metric.removesuffix("_mean") + "_std"
offsets = {"knn": (-75, 25), "fm_id": (12, -26), "fm": (14, 14),
           "popular": (10, 12), "ease": (-55, 12), "als": (-100, -24),
           "timesvd": (-90, -24), "svdpp": (16, -5)}
fig, ax = plt.subplots(figsize=(10, 5.8))
for _, row in summary.iterrows():
    point = (row.fit_seconds_mean, row[metric])
    ax.errorbar(*point, xerr=row.fit_seconds_std, yerr=row[metric_std],
                fmt="o", capsize=4, markersize=6, alpha=0.85)
    ax.annotate(row.label, point, xytext=offsets[row.model],
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.7},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
ax.set_xscale("symlog", linthresh=0.1)
lower = float((summary[metric] - summary[metric_std]).min())
upper = float((summary[metric] + summary[metric_std]).max())
padding = max((upper - lower) * 0.25, 0.001)
ax.set_ylim(max(0, lower - padding), upper + padding)
ax.set_xlabel("Время обучения, с (симметричная логарифмическая шкала)")
ax.set_ylabel(metric.removesuffix("_mean").upper())
ax.set_title("Качество и время обучения\n"
             "Среднее и стандартное отклонение между запусками", fontsize=12, pad=12)
ax.grid(alpha=0.2)
fig.tight_layout()
destination = root / "quality_vs_time.png"
fig.savefig(destination, dpi=180)
plt.close(fig)
manifest["files"][destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
