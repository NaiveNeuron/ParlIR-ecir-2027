"""ParlIR quickstart: load a build, run BM25 for `answer`, `precedent` and `cite` (test split), evaluate, peek at xling queries.

    uv run python examples/quickstart.py --build data/build/<v>
"""
from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from datetime import date
from itertools import islice
from pathlib import Path

from parlir.evaluate import write_run
from parlir.loader import evaluate_run, iter_docs, iter_queries, iter_xling_queries, load
from parlir.retrieve import bm25_search, field_text

K = 100


def bm25_run(build: Path, task: str, docs: list[dict], doc_texts: list[str]) -> dict[str, list[tuple[str, float]]]:
    _, queries, qrels = load(build, task=task, split="test")
    qids = sorted(queries)
    # Citation tasks: rank the whole corpus, then keep only answers dated before the cutoff: the question date
    # (pre-submission, `precedent`) or the official reply's date (drafting time, `cite`); `answer` ranks the full
    # corpus to depth K.
    depth = K if task == "answer" else len(docs)
    hits, scores = bm25_search(doc_texts, [field_text(queries[q], "text", "query") for q in qids], depth)
    reply_date = {d["_id"]: d["date"] for d in docs}
    run = {}
    for i, qid in enumerate(qids):
        ranked = zip(hits[i].tolist(), scores[i].tolist())
        if task in ("precedent", "cite"):
            cut = queries[qid]["date"] if task == "precedent" else reply_date[queries[qid]["answer"]]
            cutoff = date.fromisoformat(cut)
            ranked = (r for r in ranked if docs[r[0]]["date"] and date.fromisoformat(docs[r[0]]["date"]) < cutoff)
        run[qid] = [(docs[j]["_id"], float(s)) for j, s in islice(ranked, K)]
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    a = ap.parse_args()

    docs = list(iter_docs(a.build))
    print(f"corpus: {len(docs)} answers; test queries: {sum(1 for _ in iter_queries(a.build, 'test'))}")
    doc_texts = [field_text(d, "text", "doc") for d in docs]  # masked canonical text

    with tempfile.TemporaryDirectory() as tmp:
        for task in ("answer", "precedent", "cite"):
            run = bm25_run(a.build, task, docs, doc_texts)
            path = Path(tmp) / f"{task}_test_bm25.run"
            write_run(run, path, f"{task}_test_bm25")
            agg, _ = evaluate_run(path, a.build, task=task, split="test")
            print(f"{task:9s} BM25  queries={len(run):6d}  MRR@10={agg['RR@10']:.4f}  R@10={agg['R@10']:.4f}  "
                  f"R@100={agg['R@100']:.4f}")

    # Cross-lingual queries: official non-English versions of questions (masked `text`, unmasked `raw_text`).
    langs = Counter(x["lang"] for x in iter_xling_queries(a.build))
    print("xling queries per language:", dict(langs.most_common(8)))
    for x in islice(iter_xling_queries(a.build, lang="de"), 2):
        print(f"  {x['_id']} [{x['lang']}] {x['text'][:100]!r}")


if __name__ == "__main__":
    main()
