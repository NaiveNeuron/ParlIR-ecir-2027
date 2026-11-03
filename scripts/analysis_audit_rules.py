# /// script
# requires-python = ">=3.10"
# ///
"""System-ordering stability under alternative judge-aggregation rules (post hoc).

Uses the merged 11-system audit (audit_full2) without new LLM calls. A non-official pooled answer counts as relevant
in the augmented labels under:
  final  - the adjudicated grade is 2 (the paper's rule);
  both   - both judge families grade it 2;
  either - at least one family grades it 2.
For each rule: Kendall tau_b between official and augmented MRR@10 orderings, and the reversing pairs, both
population-weighted (stratum weights, the audit's estimand) and unweighted (plain sample means).

Usage: uv run scripts/analysis_audit_rules.py --audit experiments/v1.0/audit_full2/audit \
           --runs-dir experiments/v1.0/runs experiments/v1.0/robustness/runs --out experiments/v1.0/audit_full2/audit/rules.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SYSTEMS = ["bm25_text", "bm25tuned_text", "me5-base_text", "me5-large_text", "bge-m3_text", "me5-base-ft_text",
           "me5-base-ftdm_text", "rrf_text", "rerank_text", "rndwin_s13", "rnd_s13"]
FAMILIES = ["judge-claude", "judge-gpt"]


def load_judgments(p):
    g = {}
    for r in map(json.loads, open(p)):
        for j in r["judgments"]:
            g[(r["qid"], j["doc"])] = j["grade"]
    return g


def read_run(p, k=10):
    """Canonical order of parlir.evaluate (score desc, ties by docid desc, as trec_eval), as the audit uses."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parlir.evaluate import _ranked, read_run as _rr
    return {q: _ranked(scores)[:k] for q, scores in _rr(p).items()}


def tau_b(x, y):
    n = len(x)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif a * b > 0:
                c += 1
            else:
                d += 1
    den = ((c + d + tx) * (c + d + ty)) ** 0.5
    return (c - d) / den if den else None


ap = argparse.ArgumentParser()
ap.add_argument("--audit", type=Path, required=True)
ap.add_argument("--runs-dir", type=Path, nargs="+", required=True, help="directories searched for answer_test_<s>.run")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--systems", nargs="+", default=None, help="run names to rank (default: the 11 systems of audit_full2)")
a = ap.parse_args()
SYSTEMS = a.systems or SYSTEMS

pools = [json.loads(l) for l in open(a.audit / "pools.jsonl")]
ga, gb = (load_judgments(a.audit / f"judgments_{f}.jsonl") for f in FAMILIES)
adj = {}
for r in map(json.loads, open(a.audit / "adjudication.jsonl")):
    g = (r.get("answer") or {}).get("grade") if isinstance(r.get("answer"), dict) else None
    if isinstance(g, dict) and "choice" in g:
        adj[(r["qid"], r["doc"])] = int(g["choice"])
final = {k: v if gb.get(k) in (None, v) else adj.get(k, min(v, gb[k])) for k, v in ga.items()}
rules = {"final": lambda k: final.get(k) == 2,
         "both": lambda k: ga.get(k) == 2 and gb.get(k) == 2,
         "either": lambda k: ga.get(k) == 2 or gb.get(k) == 2}

popn = json.loads((a.audit / "sample.json").read_text())["strata_population"]
ns = Counter(p["stratum"] for p in pools)
w = {p["qid"]: popn[p["stratum"]] / ns[p["stratum"]] for p in pools}
runs = {s: read_run(next(d / f"answer_test_{s}.run" for d in a.runs_dir if (d / f"answer_test_{s}.run").exists()))
        for s in SYSTEMS}


def mrr(rel_of, weighted):
    out = {}
    for s, run in runs.items():
        tot = W = 0.0
        for p in pools:
            rel = rel_of(p)
            rr = next((1 / (i + 1) for i, d in enumerate(run.get(p["qid"], [])) if d in rel), 0.0)
            wt = w[p["qid"]] if weighted else 1.0
            tot += wt * rr
            W += wt
        out[s] = tot / W
    return out


res = {}
for name, is_rel in rules.items():
    aug = lambda p, f=is_rel: {p["official"]} | {it["doc"] for it in p["items"]
                                                  if "control:official" not in it["sources"] and f((p["qid"], it["doc"]))}
    for weighted in (True, False):
        o, g = mrr(lambda p: {p["official"]}, weighted), mrr(aug, weighted)
        rev = [[i, j] for x, i in enumerate(SYSTEMS) for j in SYSTEMS[x + 1:] if (o[i] - o[j]) * (g[i] - g[j]) < 0]
        res[f"{name}_{'weighted' if weighted else 'unweighted'}"] = {
            "tau_b": tau_b([o[s] for s in SYSTEMS], [g[s] for s in SYSTEMS]), "reversals": rev,
            "mrr_official": o, "mrr_augmented": g}
a.out.write_text(json.dumps({"systems": SYSTEMS, "n_queries": len(pools), "rules": res}, indent=1))
for k, v in res.items():
    print(f"{k:20s} tau_b {v['tau_b']:.3f}  reversals {len(v['reversals'])}  {v['reversals']}")
