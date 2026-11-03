"""Metadata-shortcut probe: how much do non-text fields give away the official answer?

  uv run python scripts/analysis_metadata_probe.py --build data/build/v1.0 --exp experiments/v1.0
(1) Date window: re-rank each released top-100 test run keeping only answers dated 0..W days after the question
    (W = 180 by default; in v1.0 official replies come a median 64 days, p99 173 days after their question), then MRR@10.
(2) Reply-gap distribution of the official answers.
(3) Identifier rule: rank the answer `<question id>-ASW` first if it exists (answer ids reuse the id of the first
    question they reply to, so joint replies break the rule); MRR@10 measured, not assumed. The `answer` field of
    queries.jsonl is the qrel itself and is never a permitted input.
These are forbidden inputs (paper, README: systems may use only the `text` fields; dates only for the documented
as-of filters of the citation tasks). Writes <exp>/metadata_probe.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import date
from pathlib import Path

from parlir.evaluate import _ranked, evaluate, read_qrels, read_run

SYSTEMS = ["bm25_text", "bge-m3_text", "rrf_text", "me5-base-ft_text", "me5-base-ftdm_text"]


def d(s):
    return date.fromisoformat(s[:10])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--window", type=int, default=180)
    a = ap.parse_args()
    qs = {q["_id"]: q for q in map(json.loads, (a.build / "queries.jsonl").open()) if q["split"] == "test"}
    ddate = {x["_id"]: x.get("date") for x in map(json.loads, (a.build / "corpus.jsonl").open())}
    gaps = [(d(ddate[q["answer"]]) - d(q["date"])).days for q in qs.values() if ddate.get(q["answer"]) and q.get("date")]
    out = {"window_days": a.window,
           "official_reply_gap_days": {"min": min(gaps), "median": st.median(gaps), "p99": sorted(gaps)[int(0.99 * len(gaps))],
                                       "max": max(gaps), "n": len(gaps)},
           "date_window_mrr10": {}}
    # identifier rule, measured: put '<qid>-ASW' at rank 1 when that answer exists (MRR@10 = hit rate)
    hits = sum(f"{qid}-ASW" in ddate and q["answer"] == f"{qid}-ASW" for qid, q in qs.items())
    out["identifier_rule_mrr10"] = hits / len(qs)
    out["identifier_rule_misses"] = len(qs) - hits
    qrels = read_qrels(a.build / "qrels" / "answer_test.tsv")
    out["date_window_metrics"] = {}
    for s in SYSTEMS:
        p = a.exp / "runs" / f"answer_test_{s}.run"
        if not p.exists():
            continue
        run = read_run(p)  # canonical order (score desc, ties by docid desc), as the reported metrics
        base, filt, filt_run = [], [], {}
        for qid, q in qs.items():
            ranked = _ranked(run.get(qid, {}))
            qd = d(q["date"])
            keep = [doc for doc in ranked if ddate.get(doc) and 0 <= (d(ddate[doc]) - qd).days <= a.window]
            filt_run[qid] = {doc: run[qid][doc] for doc in keep}  # original scores: order is unchanged
            for lst, acc in ((ranked, base), (keep, filt)):
                rr = next((1 / (i + 1) for i, doc in enumerate(lst[:10]) if doc == q["answer"]), 0.0)
                acc.append(rr)
        out["date_window_mrr10"][s] = {"unfiltered": sum(base) / len(base), "filtered": sum(filt) / len(filt)}
        out["date_window_metrics"][s] = evaluate(qrels, filt_run)[0]  # all reported metrics, canonical evaluator
    (a.exp / "metadata_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
