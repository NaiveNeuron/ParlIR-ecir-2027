import json

import pytest

from parlir.loader import evaluate_run, iter_queries, load


def _write_build(root):
    corpus = [{"_id": f"Q{i}-ASW", "text": f"answer {i}", "raw_text": f"answer {i}", "date": f"202{i}-01-01",
               "institution": "COM", "answers": [f"Q{i}"], "header": []} for i in range(4)]
    splits = ["train", "train", "dev", "test"]
    queries = [{"_id": f"Q{i}", "subject": f"s{i}", "text": f"question {i}", "raw_text": f"question {i}",
                "date": f"202{i}-01-01", "type": "E", "term": 9, "addressee": "COM", "author": "x",
                "orig_lang": "EN", "answer": f"Q{i}-ASW", "split": s, "joint_size": 1}
               for i, s in enumerate(splits)]
    (root / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in corpus))
    (root / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    (root / "qrels").mkdir()
    for split in ("train", "dev", "test"):
        rows = [f"{q['_id']} 0 {q['answer']} 1\n" for q in queries if q["split"] == split]
        (root / "qrels" / f"answer_{split}.tsv").write_text("".join(rows))
    return queries


def test_load_split_and_agreement(tmp_path):
    _write_build(tmp_path)
    corpus, queries, qrels = load(tmp_path, task="answer", split="train")
    assert len(corpus) == 4
    assert set(queries) == set(qrels) == {"Q0", "Q1"}
    assert all(q["split"] == "train" for q in queries.values())
    for qid, rels in qrels.items():
        assert set(rels) == {queries[qid]["answer"]}
        assert all(d in corpus for d in rels)
    _, tq, tr = load(tmp_path, split="test")
    assert set(tq) == set(tr) == {"Q3"}
    assert [q["_id"] for q in iter_queries(tmp_path, "dev")] == ["Q2"]


def test_load_rejects_bad_args_and_dangling_qrels(tmp_path):
    _write_build(tmp_path)
    with pytest.raises(ValueError):
        load(tmp_path, split="nope")
    (tmp_path / "qrels" / "answer_dev.tsv").write_text("GHOST 0 Q0-ASW 1\n")
    with pytest.raises(ValueError):
        load(tmp_path, split="dev")


def test_evaluate_run(tmp_path):
    _write_build(tmp_path)
    run = tmp_path / "r.run"
    run.write_text("Q0 Q0 Q0-ASW 1 2.0 t\nQ0 Q0 Q1-ASW 2 1.0 t\nQ1 Q0 Q0-ASW 1 2.0 t\nQ1 Q0 Q1-ASW 2 1.0 t\n")
    agg, _ = evaluate_run(run, tmp_path, "answer", "train")
    assert agg["RR@10"] == pytest.approx(0.75)


def test_ir_datasets_queries_restricted_to_qrels(tmp_path):
    ir_datasets = pytest.importorskip("ir_datasets")
    from parlir.loader import register_ir_datasets
    _write_build(tmp_path)
    # a sub-task judging only one of the two train queries (like precedent vs answer)
    (tmp_path / "qrels" / "precedent_train.tsv").write_text("Q1 0 Q0-ASW 1\n")
    name = f"parlirtest{tmp_path.name}"
    ids = register_ir_datasets(tmp_path, name=name)
    assert f"{name}/precedent/train" in ids
    for task, split in [("precedent", "train"), ("answer", "train"), ("answer", "test")]:
        ds = ir_datasets.load(f"{name}/{task}/{split}")
        qids = [q.query_id for q in ds.queries_iter()]
        _, queries, qrels = load(tmp_path, task=task, split=split)
        assert set(qids) == set(queries) == {q.query_id for q in ds.qrels_iter()}
    assert [q.query_id for q in ir_datasets.load(f"{name}/precedent/train").queries_iter()] == ["Q1"]
