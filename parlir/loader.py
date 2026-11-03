"""Load a ParlIR build (`data/build/<version>/`) for retrieval experiments.

    from parlir.loader import load
    corpus, queries, qrels = load("data/build/v1.0", task="answer", split="test")

`corpus`: docid -> record; `queries`: qid -> record (only queries judged in the chosen qrels);
`qrels`: qid -> {docid: rel}. Tasks: `answer`, `answerlenient`, `cite`, `citehc`, `precedent`, `precedenthc`;
splits: `train`, `dev`, `test`. Cross-lingual query texts: `iter_xling_queries(build_dir, lang)`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .evaluate import DEFAULT_MEASURES, evaluate, read_qrels, read_run

TASKS = ("answer", "answerlenient", "cite", "citehc", "precedent", "precedenthc")
REL_LEVELS = {2: "official answer", 1: "relevant (citation edge, or twin/duplicate in answerlenient)"}
SPLITS = ("train", "dev", "test")


def _jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def iter_docs(build_dir: str | Path) -> Iterator[dict]:
    """Yield corpus records (official answers) in file order."""
    return _jsonl(Path(build_dir) / "corpus.jsonl")


def iter_queries(build_dir: str | Path, split: str | None = None) -> Iterator[dict]:
    """Yield query records; with `split`, only those assigned to that split."""
    for q in _jsonl(Path(build_dir) / "queries.jsonl"):
        if split is None or q.get("split") == split:
            yield q


def iter_xling_queries(build_dir: str | Path, lang: str | None = None) -> Iterator[dict]:
    """Original-language (L != en) versions of questions: {_id, lang, text (masked), raw_text}."""
    p = Path(build_dir) / "queries_xling.jsonl"
    if not p.exists():
        return
    for x in _jsonl(p):
        if lang is None or x["lang"] == lang:
            yield x


def qrels_path(build_dir: str | Path, task: str, split: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return Path(build_dir) / "qrels" / f"{task}_{split}.tsv"


def load(build_dir: str | Path, task: str = "answer", split: str = "test") -> tuple[dict, dict, dict]:
    """Return (corpus, queries, qrels) for one task/split. Queries are restricted to judged qids."""
    qrels = read_qrels(qrels_path(build_dir, task, split))
    corpus = {d["_id"]: d for d in iter_docs(build_dir)}
    queries = {q["_id"]: q for q in iter_queries(build_dir) if q["_id"] in qrels}
    missing = set(qrels) - set(queries)
    if missing:
        raise ValueError(f"{len(missing)} qrels qids absent from queries.jsonl, e.g. {sorted(missing)[:3]}")
    return corpus, queries, qrels


def evaluate_run(run_path: str | Path, build_dir: str | Path, task: str = "answer", split: str = "test",
                 measures=DEFAULT_MEASURES) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score a TREC run file against the build's qrels; returns (aggregate, per_query) as `parlir.evaluate.evaluate`."""
    return evaluate(read_qrels(qrels_path(build_dir, task, split)), read_run(run_path), measures)


def register_ir_datasets(build_dir: str | Path, name: str = "parlir") -> list[str]:
    """Register `<name>/<task>/<split>` datasets with ir_datasets (optional dependency).

    Returns the registered dataset ids. Raises ImportError if ir_datasets is not installed.
    """
    try:
        import ir_datasets
        from ir_datasets.formats import BaseDocs, BaseQueries, TrecQrels
        from ir_datasets.util import LocalDownload
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError("register_ir_datasets needs `ir_datasets` (pip install ir_datasets)") from e
    from typing import NamedTuple

    class ParlIRDoc(NamedTuple):
        doc_id: str
        text: str
        date: str
        institution: str

    class ParlIRQuery(NamedTuple):
        query_id: str
        text: str
        subject: str
        date: str

    build_dir = Path(build_dir)

    class Docs(BaseDocs):
        def docs_iter(self):
            for d in iter_docs(build_dir):
                yield ParlIRDoc(d["_id"], d["text"], d.get("date", ""), d.get("institution", ""))

        def docs_cls(self):
            return ParlIRDoc

        def docs_count(self):
            return sum(1 for _ in iter_docs(build_dir))

    class Queries(BaseQueries):
        """Queries of one task/split: restricted to judged qids, as in `load()`."""

        def __init__(self, split: str, qrels_file: Path):
            self.split = split
            self.qrels_file = qrels_file

        def queries_iter(self):
            judged = read_qrels(self.qrels_file)
            for q in iter_queries(build_dir, self.split):
                if q["_id"] in judged:
                    yield ParlIRQuery(q["_id"], q["text"], q.get("subject", ""), q.get("date", ""))

        def queries_cls(self):
            return ParlIRQuery

    docs = Docs()
    ids = []
    for task in TASKS:
        for split in SPLITS:
            path = qrels_path(build_dir, task, split)
            if not path.exists():
                continue
            ds_id = f"{name}/{task}/{split}"
            if ds_id not in ir_datasets.registry:
                ir_datasets.registry.register(ds_id, ir_datasets.Dataset(
                    docs, Queries(split, path), TrecQrels(LocalDownload(path), REL_LEVELS)))
            ids.append(ds_id)
    return ids
