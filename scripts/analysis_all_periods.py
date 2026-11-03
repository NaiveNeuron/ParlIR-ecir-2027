# /// script
# requires-python = ">=3.10"
# ///
"""Per-term known-citation R@10 of zero-shot systems on PrecedentRet and CiteRet over ALL periods' queries.

Non-held-out (post hoc): train, dev and test citation queries are pooled and split by the parliamentary term of
the question (queries.jsonl `term`). Only zero-shot systems: none was trained or tuned on these queries (fine-tuned
models trained on the train split; dev-tuned BM25 is in-sample on dev). Held-out test results stay the official
ones. Each cell: mean R@10 over the term's queries with a bootstrap 95% CI over joint-answer groups (questions sharing
one reply are resampled together, as elsewhere in the paper); 2,000 resamples, seed 0. Also reports, per term, the
paired difference BM25 question->question minus BGE-M3 (answer side).

  uv run scripts/analysis_all_periods.py --build data/build/v1.0 --exp experiments/v1.0
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SYSTEMS = ["bm25_text", "bm25q2q_text", "bge-m3_text", "bge-m3q2q_text", "me5-base_text", "me5-large_text",
           "qwen3-emb-8b_text", "rrf_text"]
TERMS = {8: "term8", 9: "term9", 10: "term10"}


def grouped_boot(vals: dict, group_of: dict, n=2000, seed=0):
    groups = defaultdict(list)
    for q, v in vals.items():
        groups[group_of[q]].append(v)
    keys = sorted(groups)
    rng = random.Random(seed)
    mean = sum(vals.values()) / len(vals)
    ms = []
    for _ in range(n):
        tot = cnt = 0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            tot += sum(g)
            cnt += len(g)
        ms.append(tot / cnt)
    ms.sort()
    return [mean, ms[int(0.025 * n)], ms[int(0.975 * n)]]


ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--exp", type=Path, required=True)
a = ap.parse_args()
Q = {r["_id"]: r for r in map(json.loads, open(a.build / "queries.jsonl"))}
group_of = {q: r["answer"] for q, r in Q.items()}
runs = a.exp / "runs"
out = {"systems": SYSTEMS, "terms": list(TERMS.values()), "tasks": {}}
for task in ("precedent", "cite"):
    per = {s: {} for s in SYSTEMS}
    for s in SYSTEMS:
        for split in ("train", "dev", "test"):
            p = runs / f"{task}_{split}_{s}.metrics.json"
            if not p.exists():
                raise SystemExit(f"missing run {p.name}: run `scripts/run_experiments.py --only allperiods` first")
            per[s].update(json.loads(p.read_text())["per_query"]["R@10"])
    qids = sorted(per[SYSTEMS[0]])
    by_term = defaultdict(list)
    for q in qids:
        t = TERMS.get(int(Q[q]["term"]))
        if t:
            by_term[t].append(q)
    rec = {"n": {t: len(by_term[t]) for t in TERMS.values()}, "r10": {}, "q2q_minus_bgem3": {}}
    for s in SYSTEMS:
        rec["r10"][s] = {t: grouped_boot({q: per[s].get(q, 0.0) for q in by_term[t]}, group_of)
                         for t in TERMS.values() if by_term[t]}
    for t in TERMS.values():
        if by_term[t]:
            rec["q2q_minus_bgem3"][t] = grouped_boot(
                {q: per["bm25q2q_text"].get(q, 0.0) - per["bge-m3_text"].get(q, 0.0) for q in by_term[t]}, group_of)
    out["tasks"][task] = rec
(a.exp / "all_periods.json").write_text(json.dumps(out, indent=1))
for task, rec in out["tasks"].items():
    print(task, "n", rec["n"])
    for s in SYSTEMS:
        print(f"   {s:20s}", {t: round(v[0], 3) for t, v in rec["r10"][s].items()})
    print("   q2q - bge-m3      ", {t: [round(x, 3) for x in v] for t, v in rec["q2q_minus_bgem3"].items()})
