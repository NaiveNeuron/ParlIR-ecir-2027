"""Measured impact of the v1.0 residual question references (datasheet "Known issues").

  uv run python scripts/analysis_residual_masking.py --build data/build/v1.0 --exp experiments/v1.0
Builds a DIAGNOSTIC copy of the build (data/build/v1.0-residmask; v1.0 itself stays frozen) in which every residual
reference found by the validator's broad detector (resolved to an existing question id; a query's own id included)
is replaced by [REF], as the masker does. Then it reruns BM25 (and question->question BM25 for the citation tasks) on
the test split of answer / cite / precedent, and writes the metric deltas to <exp>/residual_masking_impact.json.
Dense models are not rerun.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

BROAD = re.compile(r"(?<![A-Za-z0-9])([EPO])[\s\-\u2010-\u2015\u2212\uf02d]{0,3}(\d{3,6})\s?[/\-\u2010-\u2015\u2212\uf02d]{1,2}\s?(20\d\d|\d{2})(?!\d)")
RUNS = [("answer", "bm25"), ("cite", "bm25"), ("cite", "bm25q2q"), ("precedent", "bm25"), ("precedent", "bm25q2q")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--exp", type=Path, required=True)
    a = ap.parse_args()
    out_b = a.build.parent / (a.build.name + "-residmask")
    if out_b.exists():
        shutil.rmtree(out_b)
    shutil.copytree(a.build, out_b, ignore=shutil.ignore_patterns("corpus.jsonl", "queries.jsonl"))
    queries = [json.loads(l) for l in (a.build / "queries.jsonl").open()]
    known = {(q["_id"].split("-")[0], int(q["_id"].split("-")[3]), q["_id"].split("-")[2]) for q in queries}

    def mask(t):
        n = 0

        def sub(m):
            nonlocal n
            yr = m.group(3) if len(m.group(3)) == 4 else "20" + m.group(3)
            if (m.group(1), int(m.group(2)), yr) in known:
                n += 1
                return "[REF]"
            return m.group(0)
        return BROAD.sub(sub, t), n

    changed = {"queries": 0, "queries_test": 0, "answers": 0, "replacements": 0}
    with (out_b / "queries.jsonl").open("w") as fo:
        for q in queries:
            q["text"], n = mask(q["text"])
            changed["queries"] += n > 0
            changed["queries_test"] += n > 0 and q["split"] == "test"
            changed["replacements"] += n
            fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    with (out_b / "corpus.jsonl").open("w") as fo:
        for line in (a.build / "corpus.jsonl").open():
            d = json.loads(line)
            d["text"], n = mask(d["text"])
            changed["answers"] += n > 0
            changed["replacements"] += n
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    runs_dir = a.exp / "residual_masking_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    res = {"diagnostic_build": str(out_b), "changed": changed, "deltas": {}}
    summ = json.loads((a.exp / "metrics_summary.json").read_text())
    for task, system in RUNS:
        tag = f"{task}_test_{system}_text"
        subprocess.run(["uv", "run", "python", "-m", "parlir.retrieve", "--build", str(out_b), "--exp", str(runs_dir),
                        "--task", task, "--split", "test", "--system", system, "--field", "text", "--tag", tag],
                       check=True, capture_output=True)
        new = json.loads((runs_dir / f"{tag}.metrics.json").read_text())["aggregate"]
        old = summ[tag]["aggregate"]
        res["deltas"][tag] = {m: {"v1.0": old[m], "residmask": new[m], "delta": new[m] - old[m]}
                              for m in ("RR@10", "R@10", "R@100")}
    (a.exp / "residual_masking_impact.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"changed": changed, **{k: {m: round(v[m]["delta"], 4) for m in v} for k, v in res["deltas"].items()}}, indent=1))


if __name__ == "__main__":
    main()
