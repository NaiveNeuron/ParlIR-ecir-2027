# /// script
# requires-python = ">=3.10"
# ///
"""Paired system contrasts under official vs audit-augmented labels (post hoc).

On the merged audit's sampled test queries, for close system pairs (adjacent pairs in the population-weighted
official ordering, plus fusion vs the date-matched fine-tuned model), reports the MRR@10 difference A-B under
official labels, under augmented labels (official + non-official answers whose adjudicated grade is 2), and the
change between the two. Each comes with a stratified bootstrap 95% CI: queries are resampled within the audit's
sampling strata, and estimates use the stratum (population) weights, as in all audit estimates. 2,000 resamples,
seed 0. Descriptive: it shows how much uncertainty close comparisons carry on this sample. It is not a significance
test of the full test split.

Usage: uv run scripts/analysis_audit_contrasts.py --audit experiments/v1.0/audit_full3/audit \
           --runs-dir experiments/v1.0/runs experiments/v1.0/robustness/runs \
           --out experiments/v1.0/audit_full3/audit/contrasts.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SYSTEMS = ["bm25_text", "bm25tuned_text", "me5-base_text", "me5-large_text", "bge-m3_text", "me5-base-ft_text",
           "me5-base-ftdm_text", "rrf_text", "rerank_text", "rndwin_s13", "rnd_s13", "qwen3-emb-8b_text"]
FAMILIES = ["judge-claude", "judge-gpt"]
EXTRA_PAIRS = [("rrf_text", "me5-base-ftdm_text")]


def load_judgments(p):
    g = {}
    for r in map(json.loads, open(p)):
        for j in r["judgments"]:
            g[(r["qid"], j["doc"])] = j["grade"]
    return g


def read_run(p, k=10):
    """Canonical order of parlir.evaluate (score desc, ties by docid desc, as trec_eval), as the audit uses."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parlir.evaluate import _ranked, read_run as _rr
    return {q: _ranked(scores)[:k] for q, scores in _rr(p).items()}


ap = argparse.ArgumentParser()
ap.add_argument("--audit", type=Path, required=True)
ap.add_argument("--runs-dir", type=Path, nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()

pools = [json.loads(l) for l in open(a.audit / "pools.jsonl")]
ga, gb = (load_judgments(a.audit / f"judgments_{f}.jsonl") for f in FAMILIES)
adj = {}
for r in map(json.loads, open(a.audit / "adjudication.jsonl")):
    g = (r.get("answer") or {}).get("grade") if isinstance(r.get("answer"), dict) else None
    if isinstance(g, dict) and "choice" in g:
        adj[(r["qid"], r["doc"])] = int(g["choice"])
final = {k: v if gb.get(k) in (None, v) else adj.get(k, min(v, gb[k])) for k, v in ga.items()}
popn = json.loads((a.audit / "sample.json").read_text())["strata_population"]
ns = Counter(p["stratum"] for p in pools)
w = [popn[p["stratum"]] / ns[p["stratum"]] for p in pools]
runs = {s: read_run(next(d / f"answer_test_{s}.run" for d in a.runs_dir if (d / f"answer_test_{s}.run").exists()))
        for s in SYSTEMS}


def rr(run, qid, rel):
    return next((1 / (i + 1) for i, d in enumerate(run.get(qid, [])) if d in rel), 0.0)


off = {s: [rr(runs[s], p["qid"], {p["official"]}) for p in pools] for s in SYSTEMS}
aug = {s: [rr(runs[s], p["qid"], {p["official"]} | {it["doc"] for it in p["items"]
                                                    if final.get((p["qid"], it["doc"])) == 2}) for p in pools]
       for s in SYSTEMS}


def wmean(x, idx):
    return sum(w[i] * x[i] for i in idx) / sum(w[i] for i in idx)


full = list(range(len(pools)))
order = sorted(SYSTEMS, key=lambda s: -wmean(off[s], full))
pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
pairs += [p for p in EXTRA_PAIRS if p not in pairs and p[::-1] not in pairs]
by_stratum = defaultdict(list)
for i, p in enumerate(pools):
    by_stratum[p["stratum"]].append(i)
rng = random.Random(0)
reps = [[i for ix in by_stratum.values() for i in (rng.choice(ix) for _ in ix)] for _ in range(a.n_boot)]


def ci(f):
    vals = sorted(f(idx) for idx in reps)
    return [f(full), vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


res = []
for s, t in pairs:
    d_off = lambda idx, s=s, t=t: wmean(off[s], idx) - wmean(off[t], idx)
    d_aug = lambda idx, s=s, t=t: wmean(aug[s], idx) - wmean(aug[t], idx)
    res.append({"a": s, "b": t, "official": ci(d_off), "augmented": ci(d_aug),
                "change": ci(lambda idx: d_aug(idx) - d_off(idx))})
excl = lambda c: c[1] > 0 or c[2] < 0
out = {"n_queries": len(pools), "n_boot": a.n_boot, "order_official_weighted": order, "pairs": res,
       "n_pairs": len(res),
       "n_official_ci_excludes_0": sum(excl(r["official"]) for r in res),
       "n_augmented_ci_excludes_0": sum(excl(r["augmented"]) for r in res),
       "n_change_ci_excludes_0": sum(excl(r["change"]) for r in res)}
a.out.write_text(json.dumps(out, indent=1))
fmt = lambda c: f"{c[0]:+.3f} ({c[1]:+.3f},{c[2]:+.3f})"
for r in res:
    print(f"{r['a']:20s} - {r['b']:20s} official {fmt(r['official'])}  augmented {fmt(r['augmented'])}  change {fmt(r['change'])}")
print({k: v for k, v in out.items() if k.startswith("n_")})
