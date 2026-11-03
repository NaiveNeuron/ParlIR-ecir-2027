# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0", "huggingface_hub>=0.25"]
# ///
"""Check that the HF dataset's configs load from a given revision and agree with a local build.

  uv run scripts/verify_hf.py --build data/build/v1.0 --revision 2176502d8ed76885c17fc85604be33d2cf429368
Checks: the corpus, queries and queries-xling configs load with the card's YAML, have the same number of rows and
the same `_id` sequence as the local files; every qrels-<task> config defined at that revision (answer, cite,
precedent, precedenthc) has columns query_id/iteration/doc_id/relevance and exactly the rows of the local TREC file.
Text fields are not compared; build_manifest.json holds the SHA-256 of every file. Exits non-zero on any mismatch.
"""
import argparse
import json
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

REPO = "NaiveNeuron/parlir"
ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--revision", required=True)
a = ap.parse_args()
fail = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fail.append(msg)


for cfg, split, f in [("corpus", "corpus", "corpus.jsonl"), ("queries", "queries", "queries.jsonl"),
                      ("queries-xling", "queries", "queries_xling.jsonl")]:
    ds = load_dataset(REPO, cfg, split=split, revision=a.revision)
    local_ids = [json.loads(line)["_id"] for line in open(a.build / f)]
    check(len(ds) == len(local_ids), f"{cfg}: {len(ds)} rows vs local {len(local_ids)}")
    check(ds["_id"] == local_ids, f"{cfg}: _id sequence equals the local file")

configs = set(get_dataset_config_names(REPO, revision=a.revision))
for task in ("answer", "cite", "precedent", "precedenthc"):
    if f"qrels-{task}" not in configs:
        print(f"SKIP qrels-{task}: not defined in the card at this revision")
        continue
    for split in ("train", "dev", "test"):
        ds = load_dataset(REPO, f"qrels-{task}", split=split, revision=a.revision)
        local = a.build / "qrels" / f"{task}_{split}.tsv"
        check(ds.column_names == ["query_id", "iteration", "doc_id", "relevance"],
              f"qrels-{task}/{split}: columns {ds.column_names}")
        want = sorted(tuple(line.split()) for line in open(local) if line.strip())
        got = sorted((str(r["query_id"]), str(r["iteration"]), str(r["doc_id"]), str(r["relevance"])) for r in ds)
        check(got == want, f"qrels-{task}/{split}: {len(got)} rows equal the local TREC file ({len(want)} lines)")

print("FAILED" if fail else "ALL OK", len(fail))
sys.exit(1 if fail else 0)
