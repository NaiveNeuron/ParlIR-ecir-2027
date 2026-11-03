# /// script
# requires-python = ">=3.10"
# ///
"""Contamination check for a post-hoc zero-shot embedder.

Splits AnswerRet test questions by the publication date of their official answer, relative to the model's public
release, and reports the model's MRR@10 gain over BM25 per period with a bootstrap 95% CI (2000 resamples, seed 0).
Answers published after the release cannot be in its training data, so a memorisation advantage would show up as a
smaller gain on that period. BM25 controls for period difficulty.

Usage: uv run scripts/analysis_release_split.py --build data/build/v1.0 --exp experiments/v1.0 \
           --system qwen3-emb-8b_text --release 2025-06-05
"""
import argparse
import json
import random
from pathlib import Path


def boot(x, n=2000, seed=0):
    r, k = random.Random(seed), len(x)
    ms = sorted(sum(x[r.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [sum(x) / k, ms[int(0.025 * n)], ms[int(0.975 * n)]]


ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--exp", type=Path, required=True)
ap.add_argument("--system", required=True)
ap.add_argument("--release", required=True, help="model release date, YYYY-MM-DD")
a = ap.parse_args()

date = {d["_id"]: d["date"] for d in map(json.loads, open(a.build / "corpus.jsonl"))}
target = {}
for line in open(a.build / "qrels" / "answer_test.tsv"):
    q, _, d, s = line.split()
    if int(s) > 0:
        target[q] = d
rr = lambda s: json.load(open(a.exp / "runs" / f"answer_test_{s}.metrics.json"))["per_query"]["RR@10"]
sysrr, bm = rr(a.system), rr("bm25_text")
out = {"system": a.system, "release": a.release, "periods": {}}
for name, keep in (("before", lambda d: d < a.release), ("after", lambda d: d >= a.release)):
    qs = sorted(q for q, d in target.items() if keep(date[d]))
    out["periods"][name] = {"n": len(qs), "system_mrr10": sum(sysrr.get(q, 0.0) for q in qs) / len(qs),
                            "bm25_mrr10": sum(bm.get(q, 0.0) for q in qs) / len(qs),
                            "gain_over_bm25": boot([sysrr.get(q, 0.0) - bm.get(q, 0.0) for q in qs])}
(a.exp / "release_split.json").write_text(json.dumps(out, indent=1))
for k, v in out["periods"].items():
    g = v["gain_over_bm25"]
    print(f"{k:6s} n={v['n']:5d} system {v['system_mrr10']:.3f} bm25 {v['bm25_mrr10']:.3f} "
          f"gain {g[0]:+.3f} ({g[1]:+.3f}, {g[2]:+.3f})")
