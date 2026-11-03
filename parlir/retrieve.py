"""Retrieval baselines over a ParlIR build (paper Sec. 7).

  uv run python -m parlir.retrieve --build data/build/<v> --task answer --split test --system bm25 --field text
  fields: text (masked, canonical) | body (header-stripped, IDs kept) | full (header + body, raw)
Tasks: answer (corpus = all answers) | precedent (corpus restricted per query to answers dated
strictly before the question).
Writes experiments/<exp>/<task>_<split>_<system>_<field>.{run,metrics.json}.
"""
from __future__ import annotations

import argparse
import bisect
import json
import time
from datetime import date
from pathlib import Path


from .evaluate import evaluate, read_qrels, save_metrics, write_run


def load_build(build: Path):
    corpus = [json.loads(l) for l in (build / "corpus.jsonl").open()]
    queries = {q["_id"]: q for q in map(json.loads, (build / "queries.jsonl").open())}
    return corpus, queries


def field_text(rec: dict, field: str, kind: str) -> str:
    if field == "text":
        return rec["text"]
    if field == "body":
        return rec["raw_text"]
    if field == "full":
        return "\n".join(rec.get("header", [])) + "\n" + rec["raw_text"]
    if field == "header":  # shortcut probe: document headers only
        return "\n".join(rec.get("header", [])) or "."
    if field == "subject":  # probe: question subject only vs masked answer text
        return rec.get("subject") or "." if kind == "query" else rec["text"]
    raise ValueError(field)


def bm25_search(doc_texts: list[str], query_texts: list[str], k: int, k1: float = 0.9, b: float = 0.4,
                lang: str = "english"):
    import bm25s
    import Stemmer
    stemmer = Stemmer.Stemmer(lang)
    retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
    retriever.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=False), show_progress=False)
    qtok = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    docs, scores = retriever.retrieve(qtok, k=min(k, len(doc_texts)), show_progress=False, n_threads=16)
    return docs, scores


def id_oracle(corpus, queries_list, field):
    """Ranks answers by the number of shared identifier-like tokens (shortcut probe, Sec. 6.1)."""
    import re
    from collections import defaultdict
    pat = re.compile(r"[EPO]-\d{1,2}-\d{4}-\d{4,6}|[EPO]\s?-\s?\d{3,6}\s?/\s?\d{2,4}")
    inv = defaultdict(set)
    for j, d in enumerate(corpus):
        for t in set(x.replace(" ", "") for x in pat.findall(field_text(d, field, "doc"))):
            inv[t].add(j)
    out = []
    for q in queries_list:
        c = defaultdict(int)
        for t in set(x.replace(" ", "") for x in pat.findall(field_text(q, field, "query"))):
            for j in inv.get(t, ()):
                c[j] += 1
        out.append(sorted(c.items(), key=lambda kv: -kv[1])[:100])
    return out


def precedent_filter(ranked, corpus, qdate: str, k: int):
    qd = date.fromisoformat(qdate)
    keep = [(i, s) for i, s in ranked if corpus[i]["date"] and date.fromisoformat(corpus[i]["date"]) < qd]
    return keep[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--task", choices=["answer", "answerlenient", "precedent", "precedenthc", "cite", "citehc"],
                    default="answer")
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", default="bm25")
    ap.add_argument("--field", default="text", choices=["text", "body", "full", "header", "subject"])
    ap.add_argument("--model-path", default=None, help="local fine-tuned bi-encoder (system name used as tag)")
    ap.add_argument("--tag", default=None, help="override output tag (default <task>_<split>_<system>_<field>)")
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--xling", default=None, help="restrict to queries originally in this language (or 'all')")
    ap.add_argument("--xling-text", choices=["orig", "en"], default="orig",
                    help="with --xling: use the original-language text (L->EN) or the English text (EN->EN, same queries)")
    a = ap.parse_args()
    a.exp.mkdir(parents=True, exist_ok=True)
    corpus, queries = load_build(a.build)
    qrels = read_qrels(a.build / "qrels" / f"{a.task}_{a.split}.tsv")
    if a.xling:
        xl = {}
        for x in map(json.loads, (a.build / "queries_xling.jsonl").open()):
            if (a.xling == "all" or x["lang"] == a.xling) and x["_id"] in qrels:
                xl.setdefault(x["_id"], x)  # first original language if several
        qrels = {q: r for q, r in qrels.items() if q in xl}
        if a.xling_text == "orig":
            queries = {**queries, **{q: {**queries[q], "text": xl[q]["text"], "raw_text": xl[q]["raw_text"]} for q in xl}}
    qids = sorted(qrels)
    qlist = [queries[q] for q in qids]
    t0 = time.time()
    q2q_deepened = None  # question->question only: queries re-searched over all questions (as-of completeness)
    # Precedent tasks: score the whole corpus, then keep only answers dated before the question, so that
    # later documents can never displace eligible ones (exact as-of ranking with full-corpus statistics).
    depth = a.k if a.task.startswith("answer") else len(corpus)
    if a.system == "bm25":
        docs, scores = bm25_search([field_text(d, a.field, "doc") for d in corpus],
                                   [field_text(q, a.field, "query") for q in qlist], depth, a.k1, a.b)
        ranked = [list(zip(docs[i].tolist(), scores[i].tolist())) for i in range(len(qids))]
    elif a.system.endswith("q2q"):
        # Question-to-question baselines (citation tasks only): rank earlier *questions* against the query question
        # (BM25, or a dense model: <model>q2q, e.g. bge-m3q2q) and return their official answers (max score per
        # answer). The query's own question and its joint siblings map to the query's own answer, which the date
        # filter below always removes.
        if a.task.startswith("answer"):
            raise SystemExit(f"{a.system} is defined for citation tasks only (on AnswerRet it would retrieve itself)")
        base = a.system[:-3]
        idx = {d["_id"]: i for i, d in enumerate(corpus)}
        qall = [q for q in queries.values() if q["answer"] in idx]
        qall_texts = [field_text(q, a.field, "query") for q in qall]
        qtexts = [field_text(q, a.field, "query") for q in qlist]
        def search(qidx, depth):
            qt = [qtexts[i] for i in qidx]
            if base == "bm25":
                docs, scores = bm25_search(qall_texts, qt, depth, a.k1, a.b)
                return [zip(docs[i].tolist(), scores[i].tolist()) for i in range(len(qidx))]
            from .dense import dense_search
            return dense_search(base, qall_texts, qt, depth, cache_dir=a.exp / "cache", model_path=a.model_path)

        def to_answers(h):
            best = {}
            for j, s in h:
                di = idx[qall[j]["answer"]]
                best[di] = max(best.get(di, float("-inf")), s)
            return sorted(best.items(), key=lambda kv: -kv[1])

        ranked = [to_answers(h) for h in search(range(len(qids)), min(len(qall), 5000))]
        # As-of completeness: the question pool is searched to depth 5,000 BEFORE the date filter below. A query's
        # filtered list is exact iff at least min(k, #eligible answers) eligible answers survive (anything beyond the
        # depth scores lower than every survivor); otherwise re-search it over all questions.
        ddate = [d["date"] or "" for d in corpus]
        all_dates = sorted(x for x in ddate if x)
        cutoff = [q["date"] if a.task.startswith("precedent") else corpus[idx[q["answer"]]]["date"] for q in qlist]
        short = [i for i, r in enumerate(ranked)
                 if sum(1 for di, _ in r if ddate[di] and ddate[di] < cutoff[i]) < min(a.k, bisect.bisect_left(all_dates, cutoff[i]))]
        if short and len(qall) > 5000:
            for i, h in zip(short, search(short, len(qall))):
                ranked[i] = to_answers(h)
        q2q_deepened = len(short)
    elif a.system == "idoracle":
        ranked = id_oracle(corpus, qlist, a.field)
    else:
        from .dense import dense_search
        ranked = dense_search(a.system, [field_text(d, a.field, "doc") for d in corpus],
                              [field_text(q, a.field, "query") for q in qlist], depth, cache_dir=a.exp / "cache",
                              model_path=a.model_path)
    run = {}
    doc_date = {d["_id"]: d["date"] for d in corpus}
    for qid, q, r in zip(qids, qlist, ranked):
        if a.task.startswith("precedent"):  # pre-submission: answers published before the question
            r = precedent_filter(r, corpus, q["date"], a.k)
        elif a.task.startswith("cite"):  # drafting time: answers published before the official reply
            r = precedent_filter(r, corpus, doc_date[q["answer"]], a.k)
        run[qid] = [(corpus[i]["_id"], float(s)) for i, s in r[:a.k]]
    tag = a.tag or f"{a.task}_{a.split}_{a.system}_{a.field}" + (f"_x{a.xling}-{a.xling_text}" if a.xling else "")
    write_run(run, a.exp / f"{tag}.run", tag)
    agg, per_q = evaluate(qrels, {q: dict(v) for q, v in run.items()})
    meta = {"build": str(a.build), "build_manifest_sha": None, "task": a.task, "split": a.split,
            "system": a.system, "field": a.field, "k": a.k, "k1": a.k1, "b": a.b,
            "n_queries": len(qids), "n_docs": len(corpus), "seconds": round(time.time() - t0, 1),
            "q2q_deepened": q2q_deepened}
    mf = a.build / "build_manifest.json"
    if mf.exists():
        import hashlib
        meta["build_manifest_sha"] = hashlib.sha256(mf.read_bytes()).hexdigest()
    save_metrics(agg, per_q, a.exp / f"{tag}.metrics.json")
    (a.exp / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"tag": tag, **{k: round(v, 4) for k, v in agg.items()}, **meta}))


if __name__ == "__main__":
    main()
