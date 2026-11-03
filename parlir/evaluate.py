"""Deterministic evaluation for ParlIR runs (TREC run/qrels format).

Metrics follow paper Sec. 4: MRR@10, Recall@{10,100}, nDCG@10, Success@1, implemented directly
(standard trec_eval definitions; cross-checked against pytrec_eval in tests/test_evaluate.py).
Ties in scores are broken by docid (descending), as trec_eval does.
Significance: paired two-sided t-test per query vs. a baseline, Holm-corrected across systems.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_MEASURES = ["RR@10", "R@10", "R@30", "R@100", "nDCG@10", "Success@1"]


def _ranked(run_q: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(run_q.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def query_metrics(qrels_q: dict[str, int], run_q: dict[str, float]) -> dict[str, float]:
    ranked = _ranked(run_q)
    rel = {d: g for d, g in qrels_q.items() if g > 0}
    nrel = len(rel)
    rr = 0.0
    for i, d in enumerate(ranked[:10]):
        if d in rel:
            rr = 1.0 / (i + 1)
            break
    # trec_eval ndcg_cut uses linear gain (gain = relevance grade)
    dcg = sum(rel[d] / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return {
        "RR@10": rr,
        "R@10": sum(d in rel for d in ranked[:10]) / nrel if nrel else 0.0,
        "R@30": sum(d in rel for d in ranked[:30]) / nrel if nrel else 0.0,
        "R@100": sum(d in rel for d in ranked[:100]) / nrel if nrel else 0.0,
        "nDCG@10": dcg / idcg if idcg else 0.0,
        "Success@1": float(bool(ranked) and ranked[0] in rel),
    }

def paired_bootstrap(a: dict[str, float], b: dict[str, float], groups: dict[str, str] | None = None,
                     n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean difference a-b with a 95% percentile CI, resampling query *groups* (e.g. all questions that
    share one joint answer) with replacement. Returns {diff, lo, hi, p_two_sided}."""
    import numpy as np
    qids = sorted(set(a) | set(b))
    g = groups or {q: q for q in qids}
    by: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        by[g.get(q, q)].append(q)
    keys = sorted(by)
    d = np.array([sum(a.get(q, 0.0) - b.get(q, 0.0) for q in by[k]) for k in keys])
    cnt = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    boots = d[idx].sum(1) / cnt[idx].sum(1)
    diff = d.sum() / cnt.sum()
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(diff), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "p_two_sided": float(min(1.0, p))}


def bootstrap_ci(a: dict[str, float], groups: dict[str, str] | None = None, n: int = 2000, seed: int = 0):
    """95% percentile CI of a mean metric, resampling query groups."""
    r = paired_bootstrap(a, {}, groups, n, seed)
    return {"mean": r["diff"], "lo": r["lo"], "hi": r["hi"]}

def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            qid, _, did, rel = parts[:4]
            qrels[qid][did] = int(rel)
    return dict(qrels)


def read_run(path: str | Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            qid, _, did, _rank, score, _tag = line.split()
            run[qid][did] = float(score)
    return dict(run)


def write_run(run: dict[str, list[tuple[str, float]]], path: str | Path, tag: str) -> None:
    """run: qid -> ranked list of (docid, score), best first."""
    with open(path, "w") as fh:
        for qid, ranked in run.items():
            for rank, (did, score) in enumerate(ranked, 1):
                fh.write(f"{qid} Q0 {did} {rank} {score:.6f} {tag}\n")

def evaluate(qrels, run, measures=DEFAULT_MEASURES) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Returns (aggregate, per_query). Every judged query counts; a query missing from the run scores 0."""
    per_query: dict[str, dict[str, float]] = defaultdict(dict)
    for q, rq in qrels.items():
        m = query_metrics(rq, run.get(q, {}))
        for name in measures:
            per_query[name][q] = m[name]
    agg = {name: sum(v.values()) / len(qrels) for name, v in per_query.items()}
    return agg, dict(per_query)


def paired_test(a: dict[str, float], b: dict[str, float]) -> float:
    qids = sorted(set(a) | set(b))
    x = [a.get(q, 0.0) for q in qids]
    y = [b.get(q, 0.0) for q in qids]
    if x == y:
        return 1.0
    from scipy import stats  # only this test needs scipy; ranking/metrics stay importable without it
    return float(stats.ttest_rel(x, y).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def save_metrics(agg: dict, per_query: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps({"aggregate": agg, "per_query": per_query}, indent=1))
