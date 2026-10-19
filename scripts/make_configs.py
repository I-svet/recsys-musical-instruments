"""Generate the preregistered search using only a fixed RNG, before any metrics."""
import json
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
rng = random.Random(2026)
def logdraw(low, high):
    return 10 ** rng.uniform(low, high)

search = {
    "popular": [{}],
    "knn": [{"k": k, "gamma": g} for k in [20, 50, 100, 200] for g in [0, 10, 50, 100]],
    "als": [{"rank": 32, "reg": 10, "bias_reg": 10, "epochs": 40}],
    "ease": [{"reg": r} for r in [1, 3, 10, 30, 100, 300, 1000, 3000]],
    "fm": [{"rank": 32, "lr": .005, "reg": .0001, "epochs": 50, "batch_size": 4096}],
    "timesvd": [{"rank": 32, "lr": .005, "reg": .0001, "temporal_reg": .001,
                 "epochs": 50, "bin_days": 90, "batch_size": 4096}],
}
for _ in range(15):
    search["als"].append({"rank": rng.choice([16, 32, 64]), "reg": logdraw(-2, 2),
                          "bias_reg": logdraw(-2, 2), "epochs": 40})
    search["fm"].append({"rank": rng.choice([16, 32, 64]), "lr": logdraw(-4, -2),
                         "reg": logdraw(-6, -1), "epochs": 50, "batch_size": 4096})
    base = logdraw(-6, -1)
    search["timesvd"].append({"rank": rng.choice([16, 32, 64]), "lr": logdraw(-4, -2),
                              "reg": base, "temporal_reg": max(base, logdraw(-6, -1)),
                              "epochs": 50, "bin_days": rng.choice([30, 90]), "batch_size": 4096})
config = {"protocol": "global-time-train-5core-v1", "k": 10, "threads": 4,
          "batch_size": 128, "patience": 5, "tuning_seed": 42,
          "seeds": [17, 42, 2026], "bootstrap_seed": 2026, "bootstrap_replicates": 2000,
          "ablations": True, "search": search}
(root / "configs").mkdir(exist_ok=True)
(root / "configs" / "benchmark.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
smoke = dict(config, seeds=[42], ablations=False)
smoke["search"] = {k: [dict(v[0], **({"epochs": 2} if "epochs" in v[0] else {}))] for k, v in search.items()}
(root / "configs" / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
