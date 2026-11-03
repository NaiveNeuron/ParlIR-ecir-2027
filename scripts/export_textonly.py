# /// script
# requires-python = ">=3.10"
# ///
"""Derived text-only evaluation export of a frozen build (post hoc). The build itself is NOT modified.

System-facing files expose only an opaque `_id` and the masked `text`; nothing in an id reveals a link:
  <out>/corpus.jsonl, <out>/queries.jsonl      {"_id", "text"}
  <out>/qrels/<task>_<split>.tsv               TREC qrels on opaque ids (all tasks and splits of the build)
Evaluator-side files (forbidden as system input; needed for the as-of rule of the citation tasks):
  <out>/eval/doc_dates.tsv                     opaque doc id, publication date
  <out>/eval/cutoffs_<task>.tsv                opaque query id, cutoff date (precedent*: question date; cite*: reply
                                               date): only answers dated strictly before the cutoff are eligible
Provenance, a SEPARATE download (never needed to evaluate):
  <out>-provenance/mapping.json                source id -> opaque id (queries, docs)
Ids are random (secrets), drawn once and frozen: if <out>-provenance/mapping.json exists, export reuses it. The
released mapping (HF tag v1.0-textonly) is the reference; never re-draw it for a published version.

  uv run scripts/export_textonly.py export --build data/build/v1.0 --out data/build/v1.0-textonly
  uv run scripts/export_textonly.py verify --build data/build/v1.0 --out data/build/v1.0-textonly --runs experiments/v1.0/runs
  uv run scripts/export_textonly.py score  --out data/build/v1.0-textonly --task precedent_test --run my.run
"""
import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parlir.evaluate import _ranked, evaluate, read_qrels, read_run  # noqa: E402

TASK_CUTOFF = {"precedent": "question", "precedenthc": "question", "cite": "reply", "citehc": "reply"}


def prov_dir(out: Path) -> Path:
    return out.parent / f"{out.name}-provenance"


def load_mapping(out: Path):
    p = prov_dir(out) / "mapping.json"
    return json.loads(p.read_text()) if p.exists() else None


def export(build: Path, out: Path) -> None:
    corpus = [json.loads(l) for l in open(build / "corpus.jsonl")]
    queries = [json.loads(l) for l in open(build / "queries.jsonl")]
    mapping = load_mapping(out)
    if mapping is None:  # draw once; frozen afterwards
        used, mapping = set(), {"queries": {}, "docs": {}}
        for kind, recs, prefix in (("queries", queries, "q"), ("docs", corpus, "d")):
            for r in recs:
                while (oid := prefix + secrets.token_hex(6)) in used:
                    pass
                used.add(oid)
                mapping[kind][r["_id"]] = oid
        prov_dir(out).mkdir(parents=True, exist_ok=True)
        (prov_dir(out) / "mapping.json").write_text(json.dumps(
            {"note": "opaque id -> source id is the inverse of these maps; source id -> opaque id below",
             "build": str(build), **mapping}, indent=0))
    qm, dm = mapping["queries"], mapping["docs"]
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w") as f:
        for d in sorted(corpus, key=lambda r: dm[r["_id"]]):  # sorted by opaque id: file order leaks nothing
            f.write(json.dumps({"_id": dm[d["_id"]], "text": d["text"]}, ensure_ascii=False) + "\n")
    with open(out / "queries.jsonl", "w") as f:
        for q in sorted(queries, key=lambda r: qm[r["_id"]]):
            f.write(json.dumps({"_id": qm[q["_id"]], "text": q["text"]}, ensure_ascii=False) + "\n")
    for p in sorted((build / "qrels").glob("*.tsv")):
        qrels = read_qrels(p)
        with open(out / "qrels" / p.name, "w") as f:
            for q in sorted(qrels, key=lambda x: qm[x]):
                for d, rel in sorted(qrels[q].items(), key=lambda x: dm[x[0]]):
                    f.write(f"{qm[q]} 0 {dm[d]} {rel}\n")
    with open(out / "eval" / "doc_dates.tsv", "w") as f:
        for d in sorted(corpus, key=lambda r: dm[r["_id"]]):
            f.write(f"{dm[d['_id']]}\t{d['date']}\n")
    date_of = {d["_id"]: d["date"] for d in corpus}
    qby = {q["_id"]: q for q in queries}
    for task, kind in TASK_CUTOFF.items():
        qs = set()
        for split in ("train", "dev", "test"):
            p = build / "qrels" / f"{task}_{split}.tsv"
            if p.exists():
                qs |= set(read_qrels(p))
        with open(out / "eval" / f"cutoffs_{task}.tsv", "w") as f:
            for q in sorted(qs, key=lambda x: qm[x]):
                cut = qby[q]["date"] if kind == "question" else date_of[qby[q]["answer"]]
                f.write(f"{qm[q]}\t{cut}\n")
    print(f"exported {len(corpus)} docs, {len(queries)} queries to {out}; mapping in {prov_dir(out)}")


def asof_filter(run, task: str, out: Path):
    """The citation tasks' as-of rule: drop answers not dated strictly before the query's cutoff."""
    base = task.rsplit("_", 1)[0]
    if base not in TASK_CUTOFF:
        return run
    dates = dict(l.rstrip("\n").split("\t") for l in open(out / "eval" / "doc_dates.tsv"))
    cuts = dict(l.rstrip("\n").split("\t") for l in open(out / "eval" / f"cutoffs_{base}.tsv"))
    return {q: {d: s for d, s in docs.items() if dates.get(d) and dates[d] < cuts[q]} for q, docs in run.items()
            if q in cuts}


def verify(build: Path, out: Path, runs: Path, report: Path) -> None:
    """Every released run, mapped in its original canonical order, scores identically on the export."""
    mapping = load_mapping(out)
    qm, dm = mapping["queries"], mapping["docs"]
    ok, bad = [], []
    for rp in sorted(runs.glob("*.run")):
        tag = rp.stem
        task = "_".join(tag.split("_")[:2])  # e.g. precedent_test
        qp = build / "qrels" / f"{task}.tsv"
        if not qp.exists() or "_x" in tag:  # cross-lingual runs use other query texts
            continue
        orig = read_run(rp)
        qrels = read_qrels(qp)
        a0, _ = evaluate(qrels, orig)
        mapped = {qm[q]: {dm[d]: float(len(docs) - i) for i, d in enumerate(_ranked(docs))} for q, docs in orig.items()}
        a1, _ = evaluate(read_qrels(out / "qrels" / f"{task}.tsv"), mapped)
        if a0 == a1:
            ok.append(tag)
        else:
            bad.append(tag)
            print("MISMATCH", tag, {k: (a0[k], a1.get(k)) for k in a0 if a0[k] != a1.get(k)})
    report.write_text(json.dumps({"n_identical": len(ok), "n_mismatch": len(bad), "runs": ok, "mismatches": bad},
                                 indent=1))
    print(f"verified {len(ok)} runs identical, {len(bad)} mismatches -> {report}")
    if bad:
        sys.exit(1)


def score(out: Path, task: str, run_path: Path) -> None:
    qrels = read_qrels(out / "qrels" / f"{task}.tsv")
    run = asof_filter(read_run(run_path), task, out)
    agg, _ = evaluate(qrels, run)
    print(json.dumps({"task": task, **{k: round(v, 4) for k, v in agg.items()}}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "verify", "score"])
    ap.add_argument("--build", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--task")
    ap.add_argument("--run", type=Path)
    ap.add_argument("--report", type=Path, default=Path("experiments/v1.0/textonly_verify.json"),
                    help="verify: where to write the identity report")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.build, a.out)
    elif a.cmd == "verify":
        verify(a.build, a.out, a.runs, a.report)
    else:
        score(a.out, a.task, a.run)


if __name__ == "__main__":
    main()
