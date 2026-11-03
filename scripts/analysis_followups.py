"""Follow-up questions in the citation tasks (post hoc).

A citation query is a *follow-up* if its own (masked) text contains a masked reference `[REF]`: the MEP points at
an earlier document, often the very question whose answer is the target. Identifiers are masked, so ids still
cannot help, but such queries are easier. Reports, on test, the share of follow-ups and known-citation R@10 per
stratum with grouped 95% CIs (joint-answer groups, as elsewhere), plus the paired question->question minus Qwen3
contrast on the NON-follow-up queries (does the headline hold without them?). Post hoc, descriptive.

  uv run python scripts/analysis_followups.py --build data/build/v1.0 --exp experiments/v1.0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parlir.evaluate import bootstrap_ci, paired_bootstrap  # noqa: E402

SYSTEMS = ["bm25_text", "bge-m3_text", "qwen3-emb-8b_text", "bm25q2q_text", "bge-m3q2q_text"]

ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--exp", type=Path, required=True)
a = ap.parse_args()
Q = {r["_id"]: r for r in map(json.loads, open(a.build / "queries.jsonl"))}
group_of = {q: r["answer"] for q, r in Q.items()}
out = {}
for task in ("precedent", "cite"):
    per = {s: json.loads((a.exp / "runs" / f"{task}_test_{s}.metrics.json").read_text())["per_query"]["R@10"]
           for s in SYSTEMS}
    qids = sorted(per[SYSTEMS[0]])
    strata = {"followup": [q for q in qids if "[REF]" in Q[q]["text"]]}
    strata["other"] = [q for q in qids if q not in set(strata["followup"])]
    rec = {"n": len(qids), "n_followup": len(strata["followup"]), "r10": {}, "contrast_other": {}}
    for name, qs in strata.items():
        rec["r10"][name] = {s: bootstrap_ci({q: per[s][q] for q in qs}, group_of) for s in SYSTEMS}
    other = strata["other"]
    for s in ("bge-m3q2q_text", "bm25q2q_text"):
        rec["contrast_other"][f"{s}_minus_qwen3"] = paired_bootstrap(
            {q: per[s][q] for q in other}, {q: per["qwen3-emb-8b_text"][q] for q in other}, group_of)
    out[task] = rec
    print(task, "n", rec["n"], "follow-ups", rec["n_followup"])
    for name in strata:
        print("  ", name, {s: round(v["mean"], 3) for s, v in rec["r10"][name].items()})
    print("   other: q2q - qwen3", {k: [round(v[x], 3) for x in ("diff", "lo", "hi")]
                                    for k, v in rec["contrast_other"].items()})
(a.exp / "followups.json").write_text(json.dumps(out, indent=1))
