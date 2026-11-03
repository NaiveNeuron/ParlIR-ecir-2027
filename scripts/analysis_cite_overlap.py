# /// script
# requires-python = ">=3.10"
# ///
"""CiteRet near-repeat analysis (post hoc).

For each CiteRet test query, near-repeat strength = max token-set Jaccard between the query question and the
questions whose answers its official reply cites. Reference: Jaccard of the query with random questions (5 per query,
seed 0). The low-overlap slice holds queries whose cited questions are no more similar than the random pairs' 99th
percentile. Reports per-system R@10 on all queries and on the slice (bootstrap 95% CIs, 2000 resamples, seed 0), and
the paired Q->Q minus BGE-M3 difference on the slice.

Usage: uv run scripts/analysis_cite_overlap.py --build data/build/v1.0 --exp experiments/v1.0
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

SYSTEMS = ["bm25_text", "bm25tuned_text", "bm25q2q_text", "bge-m3q2q_text", "me5-base_text", "me5-large_text",
           "bge-m3_text", "rrf_text", "rerank_text", "me5-base-ft_text", "me5-base-ftdm_text", "qwen3-emb-8b_text"]
TOK = re.compile(r"[a-z0-9]+")


def tok(r):
    return set(TOK.findall(((r.get("subject") or "") + " " + (r.get("text") or "")).lower()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def boot(x, n=2000, seed=0):
    r, k = random.Random(seed), len(x)
    ms = sorted(sum(x[r.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [sum(x) / k, ms[int(0.025 * n)], ms[int(0.975 * n)]]


ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--exp", type=Path, required=True)
a = ap.parse_args()

Q = {r["_id"]: r for r in map(json.loads, open(a.build / "queries.jsonl"))}
T = {q: tok(r) for q, r in Q.items()}
rel = collections.defaultdict(set)
for line in open(a.build / "qrels" / "cite_test.tsv"):
    q, _, d, s = line.split()
    if int(s) > 0:
        rel[q].add(d)
meta = {r["qid"]: r for r in map(json.loads, open(a.build / "precedent_meta.jsonl"))}

near = {}
for q in sorted(rel):
    cited = [e["cited"] for e in meta[q]["edges"] if e.get("answer") in rel[q] and e["cited"] in Q]
    near[q] = max(jaccard(T[q], T[c]) for c in cited)
rng, ids = random.Random(0), sorted(Q)
rand = sorted(jaccard(T[q], T[c]) for q in sorted(rel) for c in rng.sample(ids, 5) if c != q)
thr = rand[int(0.99 * len(rand))]
low = [q for q in sorted(rel) if near[q] < thr]
vals = sorted(near.values())

out = {"n_queries": len(rel), "threshold_random_p99": thr, "n_low": len(low),
       "near_jaccard_median": vals[len(vals) // 2], "random_jaccard_median": rand[len(rand) // 2],
       "r10": {}}
runs = a.exp / "runs"
for s in SYSTEMS:
    p = runs / f"cite_test_{s}.metrics.json"
    if p.exists():
        d = json.load(open(p))["per_query"]["R@10"]
        out["r10"][s] = {"all": boot([d.get(q, 0.0) for q in sorted(rel)]), "low": boot([d.get(q, 0.0) for q in low])}
def r10(s):
    return json.load(open(runs / f"cite_test_{s}.metrics.json"))["per_query"]["R@10"]


qq, bg = r10("bm25q2q_text"), r10("bge-m3_text")
out["low_paired_q2q_minus_bgem3"] = boot([qq.get(q, 0.0) - bg.get(q, 0.0) for q in low])
if (runs / "cite_test_bge-m3q2q_text.metrics.json").exists():  # dense question->question (post hoc)
    dq = r10("bge-m3q2q_text")
    out["all_paired_denseq2q_minus_bm25q2q"] = boot([dq.get(q, 0.0) - qq.get(q, 0.0) for q in sorted(rel)])
    out["low_paired_denseq2q_minus_bgem3"] = boot([dq.get(q, 0.0) - bg.get(q, 0.0) for q in low])
(a.exp / "cite_overlap.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "r10"}, indent=1))
for s, v in out["r10"].items():
    print(f"{s:22s} all {v['all'][0]:.3f}  low {v['low'][0]:.3f} ({v['low'][1]:.3f}-{v['low'][2]:.3f})")
