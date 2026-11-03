"""Grouped paired bootstrap + Holm for the declared primary family (paper Sec. 8).

Family: every system vs BM25 (masked) on AnswerRet test MRR@10. Groups: questions sharing one
joint answer are resampled together. Also reports a 95% CI for every run's MRR@10 / R@100.
  uv run python scripts/significance.py --exp experiments/v1.0 --build data/build/v1.0
(Runs in the project environment because it imports parlir; hence no PEP 723 header.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from parlir.evaluate import bootstrap_ci, holm, paired_bootstrap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--baseline", default="answer_test_bm25_text")
    a = ap.parse_args()
    groups = {}
    for q in map(json.loads, (a.build / "queries.jsonl").open()):
        groups[q["_id"]] = q["answer"]
    runs = {p.name[: -len(".metrics.json")]: json.loads(p.read_text())["per_query"]
            for p in sorted((a.exp / "runs").glob("*.metrics.json"))}
    out = {}
    base = runs.get(a.baseline)
    pvals = {}
    # Declared family: the deployable AnswerRet baselines of the main
    # table vs default BM25. Diagnostics (identifier oracle, field probes), lenient re-scorings and
    # cross-lingual runs are excluded.
    family = {f"answer_test_{s}_text" for s in ("bm25tuned", "me5-base", "me5-large", "bge-m3", "me5-base-ft", "rrf", "rerank")}
    for tag, pq in runs.items():
        metric = "RR@10" if tag.startswith("answer") else "R@100"
        rec = {"metric": metric, "ci": bootstrap_ci(pq[metric], groups)}
        if tag in family and base is not None:
            rec["vs_bm25"] = paired_bootstrap(pq["RR@10"], base["RR@10"], groups)
            pvals[tag] = rec["vs_bm25"]["p_two_sided"]
        out[tag] = rec
    for tag, p in holm(pvals).items():
        out[tag]["holm_p"] = p
    # H7 (pre-registered, separate single comparison): fine-tuned mE5-base vs its zero-shot base, on dev and test
    h7 = {}
    for split in ("dev", "test"):
        ft, zs = runs.get(f"answer_{split}_me5-base-ft_text"), runs.get(f"answer_{split}_me5-base_text")
        if ft and zs:
            h7[split] = paired_bootstrap(ft["RR@10"], zs["RR@10"], groups)
    # secondary: fine-tuned vs zero-shot on citation retrieval (R@100), reported, not part of H7
    for task in ("cite", "precedent"):
        ft, zs = runs.get(f"{task}_test_me5-base-ft_text"), runs.get(f"{task}_test_me5-base_text")
        if ft and zs:
            h7[f"{task}_R@100"] = paired_bootstrap(ft["R@100"], zs["R@100"], groups)
            h7[f"{task}_RR@10"] = paired_bootstrap(ft["RR@10"], zs["RR@10"], groups)
    out["_H7_finetuned_vs_zeroshot"] = h7
    # EXPLORATORY (not pre-registered; window found after test-period diagnostics): 2019-2021 window vs zero-shot
    expl = {}
    for split in ("dev", "test"):
        ft, zs = runs.get(f"answer_{split}_me5-base-ft2019_text"), runs.get(f"answer_{split}_me5-base_text")
        if ft and zs:
            expl[split] = paired_bootstrap(ft["RR@10"], zs["RR@10"], groups)
    out["_exploratory_ft2019_vs_zeroshot"] = expl
    # POST HOC: date-matched hard negatives vs zero-shot, and vs the pre-registered fine-tuning protocol
    ph = {}
    for split in ("dev", "test"):
        dm = runs.get(f"answer_{split}_me5-base-ftdm_text")
        for ref, name in (("me5-base", "vs_zeroshot"), ("me5-base-ft", "vs_preregistered_ft")):
            r = runs.get(f"answer_{split}_{ref}_text")
            if dm and r:
                ph[f"{split}_{name}"] = paired_bootstrap(dm["RR@10"], r["RR@10"], groups)
    for task in ("cite",):
        dm, zs = runs.get(f"{task}_test_me5-base-ftdm_text"), runs.get(f"{task}_test_me5-base_text")
        if dm and zs:
            ph[f"{task}_RR@10_vs_zeroshot"] = paired_bootstrap(dm["RR@10"], zs["RR@10"], groups)
            ph[f"{task}_R@100_vs_zeroshot"] = paired_bootstrap(dm["R@100"], zs["R@100"], groups)
    out["_posthoc_ftdm"] = ph
    # CiteRet (small-sample, H6 not met) and its pre-submission variant: grouped CIs for the headline R@10, and the
    # paired Q->Q minus best answer-matching baseline (BGE-M3) difference
    for task, key in (("cite", "_cite_R@10"), ("precedent", "_precedent_R@10")):
        cr = {}
        for s in ("bm25q2q", "bge-m3q2q", "bm25", "bge-m3", "rrf"):
            r = runs.get(f"{task}_test_{s}_text")
            if r:
                cr[s] = bootstrap_ci(r["R@10"], groups)
        qq, bg = runs.get(f"{task}_test_bm25q2q_text"), runs.get(f"{task}_test_bge-m3_text")
        if qq and bg:
            cr["q2q_minus_bge-m3_R@10"] = paired_bootstrap(qq["R@10"], bg["R@10"], groups)
        out[key] = cr
    # POST HOC: a 2025 LLM-based embedder zero-shot, and dense question->question retrieval.
    # Outside the Holm family.
    r8 = {}
    qw = runs.get("answer_test_qwen3-emb-8b_text")
    for ref in ("bm25", "bge-m3", "rrf"):
        r = runs.get(f"answer_test_{ref}_text")
        if qw and r:
            r8[f"answer_qwen3_minus_{ref}_RR@10"] = paired_bootstrap(qw["RR@10"], r["RR@10"], groups)
    for task in ("cite", "precedent"):
        dq, qq = runs.get(f"{task}_test_bge-m3q2q_text"), runs.get(f"{task}_test_bm25q2q_text")
        if dq and qq:
            r8[f"{task}_denseq2q_minus_bm25q2q_R@10"] = paired_bootstrap(dq["R@10"], qq["R@10"], groups)
    out["_posthoc_round8"] = r8
    out["_note"] = "p values are Monte-Carlo estimates from 2,000 bootstrap replicates: report p=0 as p<0.001"
    (a.exp / "bootstrap.json").write_text(json.dumps(out, indent=1))
    for tag, r in out.items():
        if tag.startswith("_"):
            print(tag, r)
            continue
        print(tag, r["metric"], {k: round(v, 4) for k, v in r["ci"].items()}, r.get("holm_p"))


if __name__ == "__main__":
    main()
