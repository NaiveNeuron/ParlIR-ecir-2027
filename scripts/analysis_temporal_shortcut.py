"""Temporal-shortcut diagnosis for fine-tuning on ParlIR's temporal splits.

  uv run python scripts/analysis_temporal_shortcut.py --build data/build/v1.0 --exp experiments/v1.0

(1) Date skew of the BM25 hard negatives that the pre-registered fine-tuning protocol uses, by training period:
    share of negatives dated >= 2 calendar years after the question vs. earlier than it.
(2) For each dev run: share of top-1 documents dated in the 2014-2016 training period, i.e. whether a model learned
    to prefer answers that look like its training period.
(3) MRR@10 of every run analysed (from its metrics file).
Writes <exp>/temporal_shortcut.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PERIODS = {"2014-2016": ("2014", "2015", "2016"), "2019-2021": ("2019", "2020", "2021")}
DEV_RUNS = {  # label -> run path relative to <exp>
    "zero-shot mE5-base": "runs/answer_dev_me5-base_text",
    "fine-tuned, whole split (pre-registered)": "runs/answer_dev_me5-base-ft_text",
    "fine-tuned, whole split, date-matched negatives (post hoc)": "runs/answer_dev_me5-base-ftdm_text",
    "fine-tuned, 2019-2021 (exploratory)": "runs/answer_dev_me5-base-ft2019_text",
    "fine-tuned, 2015-2016, 16,873 pairs (diagnostic)": "exploratory/runs/answer_dev_ft-old-2015-16_text",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--exp", type=Path, required=True)
    a = ap.parse_args()
    qs = {q["_id"]: q for q in map(json.loads, (a.build / "queries.jsonl").open())}
    dyear = {d["_id"]: (d.get("date") or "????")[:4] for d in map(json.loads, (a.build / "corpus.jsonl").open())}

    first_neg = {}
    for line in (a.exp / "runs" / "answer_train_bm25_text.run").open():
        q, _, d, _, _, _ = line.split()
        if d != qs[q]["answer"] and q not in first_neg:
            first_neg[q] = d
    skew = {}
    for name, yrs in PERIODS.items():
        c = Counter()
        for q, d in first_neg.items():
            y = qs[q]["date"][:4]
            if y not in yrs:
                continue
            c["later_2y_plus" if int(dyear[d]) >= int(y) + 2 else ("earlier" if dyear[d] < y else "same_or_next_year")] += 1
        n = sum(c.values())
        skew[name] = {"n": n, **{k: v / n for k, v in c.items()}}

    top1 = {}
    for label, rel in DEV_RUNS.items():
        run = a.exp / (rel + ".run")
        if not run.exists():
            continue
        seen, c = set(), Counter()
        for line in run.open():
            q, _, d, r, _, _ = line.split()
            if r == "1" and q not in seen:
                seen.add(q)
                c[dyear[d]] += 1
        n = sum(c.values())
        met = a.exp / (rel + ".metrics.json")
        top1[label] = {"n": n, "top1_2014_2016_share": sum(c[y] for y in PERIODS["2014-2016"]) / n,
                       "mrr10": json.loads(met.read_text())["aggregate"]["RR@10"] if met.exists() else None,
                       "top1_by_year": dict(sorted(c.items()))}
    out = {"hard_negative_date_skew": skew, "dev_top1_period": top1}
    (a.exp / "temporal_shortcut.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"skew": skew, "top1": {k: (round(v["top1_2014_2016_share"], 3), v["mrr10"]) for k, v in top1.items()}}, indent=1))


if __name__ == "__main__":
    main()
