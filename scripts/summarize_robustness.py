"""Summarise the fine-tuning robustness runs (scripts/ft_robustness.sh) into experiments/v1.0/robustness/summary.json.

  uv run python scripts/summarize_robustness.py --exp experiments/v1.0
Seed 13 of the pre-registered and date-matched conditions is the main run (experiments/v1.0/runs); seeds 14 and 15
come from experiments/v1.0/robustness/runs. The two random-negative controls have one seed each. Reports MRR@10 on
dev and test per seed, plus the mean and sample sd over seeds. Zero-shot mE5-base is included for reference.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

CONDS = {
    "prereg": {"label": "BM25 hard negatives, any date (pre-registered)",
               "main": "me5-base-ft", "seeds": {14: "prereg_s14", 15: "prereg_s15"}},
    "dm": {"label": "BM25 hard negatives within ±365 days (post hoc)",
           "main": "me5-base-ftdm", "seeds": {14: "dm_s14", 15: "dm_s15"}},
    "rnd": {"label": "random negatives, any date (control)", "main": None,
            "seeds": {13: "rnd_s13", 14: "rnd_s14", 15: "rnd_s15"}},
    "rndwin": {"label": "random negatives within ±365 days (control)", "main": None,
               "seeds": {13: "rndwin_s13", 14: "rndwin_s14", 15: "rndwin_s15"}},
}

def agg(path: Path):
    return json.loads(path.read_text())["aggregate"] if path.exists() else None


def rr(path: Path):
    a = agg(path)
    return a["RR@10"] if a else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, required=True)
    a = ap.parse_args()
    R = a.exp / "robustness" / "runs"
    out = {"zero_shot": {s: rr(a.exp / "runs" / f"answer_{s}_me5-base_text.metrics.json") for s in ("dev", "test")}}
    for c, spec in CONDS.items():
        rec = {"label": spec["label"], "per_seed": {}, "test_aggregate_per_seed": {}}
        if spec["main"]:
            rec["per_seed"][13] = {s: rr(a.exp / "runs" / f"answer_{s}_{spec['main']}_text.metrics.json") for s in ("dev", "test")}
            rec["test_aggregate_per_seed"][13] = agg(a.exp / "runs" / f"answer_test_{spec['main']}_text.metrics.json")
        for seed, name in spec["seeds"].items():
            rec["per_seed"][seed] = {s: rr(R / f"answer_{s}_{name}.metrics.json") for s in ("dev", "test")}
            rec["test_aggregate_per_seed"][seed] = agg(R / f"answer_test_{name}.metrics.json")
        for s in ("dev", "test"):
            v = [x[s] for x in rec["per_seed"].values() if x[s] is not None]
            rec[s] = {"n_seeds": len(v), "mean": st.mean(v) if v else None, "sd": st.stdev(v) if len(v) > 1 else None,
                      "min": min(v) if v else None, "max": max(v) if v else None}
        out[c] = rec
    (a.exp / "robustness" / "summary.json").write_text(json.dumps(out, indent=1))
    for c in CONDS:
        t = out[c]["test"]
        print(f"{c:7s} test mean {t['mean']}  sd {t['sd']}  n={t['n_seeds']}  dev mean {out[c]['dev']['mean']}")


if __name__ == "__main__":
    main()
