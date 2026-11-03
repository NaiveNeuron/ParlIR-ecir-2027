"""Question-restatement analysis (paper H2b): lexical overlap between questions and answers.

For every query of a split that is covered by a first-stage run, compare the masked question text
(`queries.jsonl:text`) with the masked text (`corpus.jsonl:text`) of three answers:

- ``official``: the query's official answer (``queries.jsonl:answer``);
- ``hardneg``: the highest-ranked answer in the run that is not relevant for the query in
  ``qrels/answer_<split>.tsv`` (score desc, docid desc tie-break, as in ``parlir.evaluate``);
- ``random``: a uniformly random non-relevant corpus answer (``random.Random(0)``, queries in sorted order).

Measures, on lowercased alphanumeric tokens (``[^\\W_]+``); the mask token ``[REF]`` never matches:

- (a) ``jaccard``: Jaccard of token *sets* with a small English stopword list removed;
- (b) ``lcs``: longest common contiguous token substring (all tokens, stopwords kept);
- (c) ``quote8``: the answer contains a contiguous 8-gram of the question (equivalently ``lcs >= 8``).

The run's per-query RR@10 (``parlir.evaluate.query_metrics``) is stratified into quintiles of the
official-answer Jaccard (equal-count bins over the evaluated queries, ties broken by qid).
Queries of the split that are absent from the run are skipped and counted.

Usage::

    uv run python scripts/analysis_overlap.py \\
        --build data/build/scratch \\
        --run experiments/_scratch/answer_test_bm25_text.run \\
        --split test \\
        --out experiments/_scratch/overlap

Outputs in ``--out``:

- ``overlap.json``: n / skipped counts, per condition median, IQR (q25, q75) and mean of jaccard and
  lcs, quote8 rate, RR@10 per official-Jaccard quintile (with bin edges), overall MRR@10 of covered queries;
- ``overlap_per_query.jsonl``: one row per evaluated query (docids, measures per condition, RR@10, quintile);
- ``overlap.pdf``: Jaccard distributions for the three conditions, and RR@10 by quintile.

The summary is also printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parlir.evaluate import query_metrics, read_qrels, read_run  # noqa: E402

CONDITIONS = ("official", "hardneg", "random")
QUOTE_N = 8
N_BINS = 5

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours yourself yourselves
may must shall upon whether within without per via
""".split())

_TOKEN = re.compile(r"\[REF\]|[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; each ``[REF]`` mask becomes ``None`` (matches nothing)."""
    return [None if t == "[REF]" else t.lower() for t in _TOKEN.findall(text or "")]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ngrams(seq: list, n: int) -> set[tuple]:
    out = set()
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        if None not in g:
            out.add(g)
    return out


def lcs_len(a: list, b: list) -> int:
    """Longest common contiguous token substring; binary search over n-gram set intersection
    (monotone: a common substring of length L implies one of every length < L)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _ngrams(a, mid) & _ngrams(b, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def measures(q_toks: list, q_set: set[str], d_toks: list) -> dict:
    d_set = {t for t in d_toks if t is not None and t not in STOPWORDS}
    lcs = lcs_len(q_toks, d_toks)
    return {"jaccard": jaccard(q_set, d_set), "lcs": lcs, "quote8": lcs >= QUOTE_N}


def ranked(run_q: dict[str, float]) -> list[str]:
    # Same ordering as parlir.evaluate (score desc, docid desc).
    return [d for d, _ in sorted(run_q.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def describe(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
            "mean": float(a.mean()), "n": int(a.size)}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plot(rows: list[dict], bins: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    edges = np.linspace(0, max(max(r[c]["jaccard"] for r in rows) for c in CONDITIONS) + 1e-9, 41)
    labels = {"official": "official answer", "hardneg": "top non-official (run)", "random": "random answer"}
    for c in CONDITIONS:
        ax1.hist([r[c]["jaccard"] for r in rows], bins=edges, histtype="step", linewidth=1.2, label=labels[c])
    ax1.set_xlabel("Token Jaccard (question, answer)")
    ax1.set_ylabel("# queries")
    ax1.legend(fontsize=7, frameon=False)

    x = np.arange(len(bins))
    bars = ax2.bar(x, [b["mrr10"] for b in bins], color="0.5")
    ax2.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    # Bin edges are in overlap.json; on the axis they collided at this panel width.
    ax2.set_xticks(x, [f"Q{b['quintile']}" for b in bins], fontsize=8)
    ax2.set_xlabel("Jaccard quintile (Q1 = lowest)")
    ax2.set_ylabel("MRR@10")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path, help="TREC run on answer_<split>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    qrels = read_qrels(args.build / "qrels" / f"answer_{args.split}.tsv")
    run = read_run(args.run)
    queries = {q["_id"]: q for q in load_jsonl(args.build / "queries.jsonl")}
    corpus = {d["_id"]: d["text"] for d in load_jsonl(args.build / "corpus.jsonl")}
    corpus_ids = sorted(corpus)

    split_qids = sorted(q for q in qrels if q in queries)
    skipped = {"not_in_run": 0, "answer_not_in_corpus": 0, "no_hardneg_in_run": 0}
    rng = random.Random(0)
    tok_cache: dict[str, list] = {}

    def doc_toks(did: str) -> list:
        if did not in tok_cache:
            tok_cache[did] = tokenize(corpus[did])
        return tok_cache[did]

    rows = []
    for qid in split_qids:
        q = queries[qid]
        rel = {d for d, g in qrels[qid].items() if g > 0} | {q["answer"]}
        # Draw the random answer for every split query so the sample does not depend on run coverage.
        while True:
            rand_id = corpus_ids[rng.randrange(len(corpus_ids))]
            if rand_id not in rel:
                break
        if qid not in run:
            skipped["not_in_run"] += 1
            continue
        if q["answer"] not in corpus:
            skipped["answer_not_in_corpus"] += 1
            continue
        hard_id = next((d for d in ranked(run[qid]) if d not in rel and d in corpus), None)
        if hard_id is None:
            skipped["no_hardneg_in_run"] += 1
            continue
        q_toks = tokenize(q["text"])
        q_set = {t for t in q_toks if t is not None and t not in STOPWORDS}
        row = {"qid": qid, "rr10": query_metrics(qrels[qid], run[qid])["RR@10"]}
        for cond, did in (("official", q["answer"]), ("hardneg", hard_id), ("random", rand_id)):
            row[cond] = {"docid": did, **measures(q_toks, q_set, doc_toks(did))}
        rows.append(row)

    if not rows:
        sys.exit("no evaluable queries (run does not cover the split?)")

    # Equal-count quintiles of official Jaccard (ties broken by qid -> deterministic).
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["official"]["jaccard"], rows[i]["qid"]))
    for rank, i in enumerate(order):
        rows[i]["quintile"] = rank * N_BINS // len(rows) + 1
    bins = []
    for b in range(1, N_BINS + 1):
        members = [r for r in rows if r["quintile"] == b]
        if not members:
            continue
        js = [r["official"]["jaccard"] for r in members]
        bins.append({"quintile": b, "n": len(members), "jaccard_min": min(js), "jaccard_max": max(js),
                     "mrr10": float(np.mean([r["rr10"] for r in members]))})

    summary = {
        "build": str(args.build), "run": str(args.run), "split": args.split,
        "n_split_queries": len(split_qids), "n": len(rows), "skipped": skipped,
        "random_seed": 0, "quote_ngram": QUOTE_N,
        "mrr10_covered": float(np.mean([r["rr10"] for r in rows])),
        "conditions": {
            c: {"jaccard": describe([r[c]["jaccard"] for r in rows]),
                "lcs": describe([r[c]["lcs"] for r in rows]),
                "quote8_rate": float(np.mean([r[c]["quote8"] for r in rows]))}
            for c in CONDITIONS
        },
        "rr10_by_official_jaccard_quintile": bins,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(args.out / "overlap_per_query.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    plot(rows, bins, args.out / "overlap.pdf")

    print(f"split={args.split}  queries={len(split_qids)}  evaluated n={len(rows)}  skipped={skipped}")
    print(f"MRR@10 (covered queries) = {summary['mrr10_covered']:.4f}")
    print(f"{'condition':<10} {'Jacc med [q25,q75]':<24} {'LCS med [q25,q75]':<20} quote8")
    for c in CONDITIONS:
        s = summary["conditions"][c]
        j, l = s["jaccard"], s["lcs"]
        print(f"{c:<10} {j['median']:.3f} [{j['q25']:.3f},{j['q75']:.3f}]      "
              f"{l['median']:>4.0f} [{l['q25']:.0f},{l['q75']:.0f}]{'':<9} {s['quote8_rate']:.3f}")
    print("RR@10 by official-Jaccard quintile:")
    for b in bins:
        print(f"  Q{b['quintile']} (n={b['n']}, J {b['jaccard_min']:.3f}-{b['jaccard_max']:.3f}): MRR@10={b['mrr10']:.4f}")
    print(f"wrote {args.out}/overlap.json, overlap_per_query.jsonl, overlap.pdf")
    return summary


if __name__ == "__main__":
    main()
