"""Second-stage baselines (paper Sec. 7): cross-encoder re-ranking and reciprocal-rank fusion.

  rerank: uv run python -m parlir.rerank rerank --build B --run R --depth 50 --model BAAI/bge-reranker-v2-m3 --out O
  fuse:   uv run python -m parlir.rerank fuse --runs R1 R2 --k 60 --out O
Outputs a TREC run plus metrics/meta JSON alongside (same conventions as parlir.retrieve).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .evaluate import evaluate, read_qrels, read_run, save_metrics, write_run


def rrf(runs: list[dict[str, dict[str, float]]], k: int = 60, depth: int = 100) -> dict[str, list[tuple[str, float]]]:
    fused: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for q, docs in run.items():
            ranked = sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            for r, (d, _) in enumerate(ranked[:depth], 1):
                fused[q][d] += 1.0 / (k + r)
    return {q: sorted(v.items(), key=lambda kv: -kv[1])[:depth] for q, v in fused.items()}


def rerank(build: Path, run: dict[str, dict[str, float]], model_name: str, depth: int, max_length: int,
           batch_size: int) -> dict[str, list[tuple[str, float]]]:
    import torch
    from sentence_transformers import CrossEncoder
    corpus = {d["_id"]: d["text"] for d in map(json.loads, (build / "corpus.jsonl").open())}
    queries = {q["_id"]: q["text"] for q in map(json.loads, (build / "queries.jsonl").open())}
    pairs, keys = [], []
    for q, docs in run.items():
        for d, _ in sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:depth]:
            pairs.append((queries[q], corpus[d]))
            keys.append((q, d))
    model = CrossEncoder(model_name, max_length=max_length, device="cuda" if torch.cuda.is_available() else "cpu",
                         model_kwargs={"torch_dtype": torch.float16})
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0]) - len(pairs[i][1]))
    scores = [0.0] * len(pairs)
    s = model.predict([pairs[i] for i in order], batch_size=batch_size, show_progress_bar=True)
    for i, v in zip(order, s):
        scores[i] = float(v)
    out: dict[str, list] = defaultdict(list)
    for (q, d), v in zip(keys, scores):
        out[q].append((d, v))
    # Re-ranked head (top `depth`) followed by the untouched first-stage tail, so recall beyond `depth`
    # is not artificially truncated (standard re-ranking evaluation).
    res = {}
    for q, docs in run.items():
        head = sorted(out.get(q, []), key=lambda kv: -kv[1])
        tail = [d for d, _ in sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[depth:]]
        floor = min((v for _, v in head), default=0.0)
        res[q] = head + [(d, floor - 1.0 - i * 1e-6) for i, d in enumerate(tail)]
    return res


def _finish(out_run, qrels_path: Path, out: Path, tag: str, meta: dict) -> None:
    qrels = read_qrels(qrels_path)
    cov = len(set(out_run) & set(qrels)) / max(1, len(qrels))
    if cov < 0.99:  # guards against fusing/re-ranking stale runs built on another query set
        raise SystemExit(f"run covers only {cov:.1%} of qrels queries in {qrels_path}; inputs are stale")
    meta["query_coverage"] = cov
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out_run, out.with_suffix(".run"), tag)
    agg, per_q = evaluate(qrels, {q: dict(v) for q, v in out_run.items()})
    save_metrics(agg, per_q, out.with_suffix(".metrics.json"))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"tag": tag, **{k: round(v, 4) for k, v in agg.items()}}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rerank")
    r.add_argument("--build", type=Path, required=True)
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--qrels", type=Path, required=True)
    r.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    r.add_argument("--depth", type=int, default=50)
    r.add_argument("--max-length", type=int, default=1024)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fuse")
    f.add_argument("--runs", type=Path, nargs="+", required=True)
    f.add_argument("--qrels", type=Path, required=True)
    f.add_argument("--k", type=int, default=60)
    f.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "rerank":
        res = rerank(a.build, read_run(a.run), a.model, a.depth, a.max_length, a.batch_size)
        meta = {"first_stage": str(a.run), "model": a.model, "depth": a.depth, "max_length": a.max_length}
    else:
        res = rrf([read_run(p) for p in a.runs], a.k)
        meta = {"runs": [str(p) for p in a.runs], "rrf_k": a.k}
    meta["seconds"] = round(time.time() - t0, 1)
    _finish(res, a.qrels, a.out, a.out.stem, meta)


if __name__ == "__main__":
    main()
