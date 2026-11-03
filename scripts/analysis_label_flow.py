# /// script
# requires-python = ">=3.10"
# ///
"""Task-and-label flow of the citation tasks, and descriptive precedent statistics (paper §Tasks / §Statistics).

Per split: AnswerRet queries, and the queries and qrels of PrecedentRet (pre-submission: cited answers published
before the QUESTION) and CiteRet (drafting time: cited answers published before the REPLY), each with its
high-confidence (explicit-referral) subset, read from the released qrels files. Edge attrition by status from
precedent_meta.jsonl. Descriptive statistics over the PrecedentRet-usable edges (status "ok": the cited answer
predates the citing question): the answer->question gap, and how often the citing and cited questions have
different authors (when both author fields are known). These describe how much citable precedent the archive holds;
they are NOT evidence that any duplicate check failed.

  uv run scripts/analysis_label_flow.py --build data/build/v1.0 --exp experiments/v1.0
"""
import argparse
import collections
import json
from datetime import date
from pathlib import Path

TASKS = {"answer": "answer", "precedent": "precedent", "precedenthc": "precedenthc", "cite": "cite",
         "citehc": "citehc"}


def qrels(p: Path):
    q = collections.defaultdict(set)
    for line in open(p):
        parts = line.split()
        if len(parts) == 4 and parts[3].lstrip("-").isdigit() and int(parts[3]) > 0:
            q[parts[0]].add(parts[2])
    return q


ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--exp", type=Path, required=True)
a = ap.parse_args()

Q = {r["_id"]: r for r in map(json.loads, open(a.build / "queries.jsonl"))}
C = {r["_id"]: r for r in map(json.loads, open(a.build / "corpus.jsonl"))}
meta = [json.loads(l) for l in open(a.build / "precedent_meta.jsonl")]
out = {"splits": {}, "edges": {}, "precedent_edges": {}}
for split in ("train", "dev", "test"):
    row = {}
    for name, f in TASKS.items():
        q = qrels(a.build / "qrels" / f"{f}_{split}.tsv")
        row[name] = {"queries": len(q), "qrels": sum(len(v) for v in q.values())}
    out["splits"][split] = row
out["edges"] = dict(collections.Counter(e.get("status") for r in meta for e in r["edges"]))

d = lambda s: date.fromisoformat(s[:10])
author = lambda q: (q.get("author") or "").strip().lower()
gaps, known, diff = [], 0, 0
for r in meta:
    q = Q[r["qid"]]
    for e in r["edges"]:
        if e.get("status") != "ok" or e.get("answer") not in C:
            continue
        gaps.append((d(q["date"]) - d(C[e["answer"]]["date"])).days)
        c = Q.get(e["cited"])
        if c and author(q) and author(c):
            known += 1
            diff += author(q) != author(c)
gaps.sort()
out["precedent_edges"] = {
    "n": len(gaps), "answer_to_question_gap_days_median": gaps[len(gaps) // 2],
    "share_within_183_days": sum(g <= 183 for g in gaps) / len(gaps),
    "both_authors_known": known, "share_different_author": diff / known if known else None}
out["empty_author_share"] = sum(not author(q) for q in Q.values()) / len(Q)
(a.exp / "label_flow.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
