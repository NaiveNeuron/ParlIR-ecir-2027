# /// script
# requires-python = ">=3.11"
# ///
"""Run the full (non-LLM) experiment suite of the paper on one frozen build.

  uv run scripts/run_experiments.py --build data/build/v1.0 --exp experiments/v1.0 [--only STEP ...]
Steps: prep shortcuts bm25tune lexical dense finetune fusion rerank lenient overlap significance summary
Each step is a shell command; outputs that already exist are skipped (never overwritten); every executed
command is appended to <exp>/commands.log with UTC time and git SHA. The LLM audit runs separately inside
OMP (scripts/audit_llm_omp.py).
Tasks: answer (official-answer known-item), cite / citehc (earlier-answer citation retrieval at drafting
time; all edges / explicit referrals), precedent (pre-submission variant).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PY = "uv run python"
CITE_TASKS = ("cite", "citehc", "precedent")


def sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def sh(cmd: str, exp: Path, outputs: list[Path]) -> None:
    if outputs and all(p.exists() for p in outputs):
        print("skip (exists):", cmd[:110])
        return
    with (exp / "commands.log").open("a") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {sha()} {cmd}\n")
    print("$", cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(f"step failed: {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    B, E = a.build, a.exp
    R = E / "runs"
    R.mkdir(parents=True, exist_ok=True)

    def want(s):
        return a.only is None or s in a.only

    def m(tag):
        return R / f"{tag}.metrics.json"

    ret = f"{PY} -m parlir.retrieve --build {B} --exp {R}"
    if want("prep"):
        if not (E / "build_manifest.json").exists():
            shutil.copy(B / "build_manifest.json", E / "build_manifest.json")
        sh(f"{PY} -m parlir.validate --build {B} --out {E}/integrity.json --minhash", E, [E / "integrity.json"])
        sh(f"{PY} scripts/stats.py --build {B} --out {E}/stats.json", E, [E / "stats.json"])

    if want("shortcuts"):
        for f in ("full", "text"):
            sh(f"{ret} --task answer --split test --system idoracle --field {f}", E, [m(f"answer_test_idoracle_{f}")])
        for f in ("header", "full", "body", "text", "subject"):
            sh(f"{ret} --task answer --split test --system bm25 --field {f}", E, [m(f"answer_test_bm25_{f}")])

    if want("bm25tune"):
        best, best_v = None, -1.0
        for k1 in (0.6, 0.9, 1.2, 1.5):
            for b in (0.3, 0.5, 0.75, 0.9):
                tag = f"answer_dev_bm25-k{k1}-b{b}_text"
                sh(f"{ret} --task answer --split dev --system bm25 --field text --k1 {k1} --b {b} --tag {tag}", E, [m(tag)])
                v = json.loads(m(tag).read_text())["aggregate"]["RR@10"]
                if v > best_v:
                    best, best_v = (k1, b), v
        (E / "bm25_tuning.json").write_text(json.dumps({"best_k1": best[0], "best_b": best[1], "dev_mrr10": best_v}))
        for task in ("answer",) + CITE_TASKS:
            tag = f"{task}_test_bm25tuned_text"
            sh(f"{ret} --task {task} --split test --system bm25 --field text --k1 {best[0]} --b {best[1]} --tag {tag}", E, [m(tag)])

    if want("lexical"):
        for task in CITE_TASKS:
            sh(f"{ret} --task {task} --split test --system bm25 --field text", E, [m(f"{task}_test_bm25_text")])
            # question-to-question baseline (earlier questions -> their answers)
            sh(f"{ret} --task {task} --split test --system bm25q2q --field text", E, [m(f"{task}_test_bm25q2q_text")])
        sh(f"{ret} --task answer --split dev --system bm25 --field text", E, [m("answer_dev_bm25_text")])
        # hard negatives for fine-tuning (train queries only)
        sh(f"{ret} --task answer --split train --system bm25 --field text --k 20", E, [m("answer_train_bm25_text")])

    if want("dense"):
        for mdl in ("me5-base", "me5-large", "bge-m3"):
            for task in ("answer",) + CITE_TASKS:
                sh(f"{ret} --task {task} --split test --system {mdl} --field text", E, [m(f"{task}_test_{mdl}_text")])
        sh(f"{ret} --task answer --split dev --system me5-base --field text", E, [m("answer_dev_me5-base_text")])

    if want("finetune"):
        # me5-base-ft = the pre-registered H7 protocol (whole training split; confirmatory, in the Holm family).
        # me5-base-ft2019 = EXPLORATORY 2019-2021 window, found after test-period diagnostics in a dress rehearsal;
        # reported outside the family and labelled exploratory.
        # me5-base-ftdm = POST-HOC date-matched hard negatives (+-365 days of the positive): the correction for a
        # temporal shortcut in the mined negatives; whole training split; reported outside the family, post hoc.
        variants = [("me5-base-ft", "",
                     [("answer", "dev"), ("answer", "test")] + [(t, "test") for t in CITE_TASKS]),
                    ("me5-base-ft2019", "--min-year 2019", [("answer", "dev"), ("answer", "test")]),
                    ("me5-base-ftdm", "--hardneg-window-days 365",
                     [("answer", "dev"), ("answer", "test")] + [(t, "test") for t in CITE_TASKS])]
        for name, extra, evals in variants:
            mdir = E / "models" / name
            sh(f"CUDA_VISIBLE_DEVICES=0 {PY} -m parlir.finetune --build {B} --base intfloat/multilingual-e5-base --hardneg-run {R}/answer_train_bm25_text.run --out {mdir} {extra}",
               E, [mdir / "parlir_train_meta.json"])
            for task, split in evals:
                tag = f"{task}_{split}_{name}_text"
                sh(f"{ret} --task {task} --split {split} --system me5-base-ft --model-path {mdir} --field text --tag {tag}", E, [m(tag)])

    if want("fusion"):
        for task in ("answer",) + CITE_TASKS:
            q = B / "qrels" / f"{task}_test.tsv"
            sh(f"{PY} -m parlir.rerank fuse --runs {R}/{task}_test_bm25_text.run {R}/{task}_test_bge-m3_text.run --qrels {q} --out {R}/{task}_test_rrf_text",
               E, [m(f"{task}_test_rrf_text")])

    if want("rerank"):
        for task in ("answer",) + CITE_TASKS:
            q = B / "qrels" / f"{task}_test.tsv"
            sh(f"{PY} -m parlir.rerank rerank --build {B} --run {R}/{task}_test_bm25_text.run --qrels {q} --depth 30 --out {R}/{task}_test_rerank_text",
               E, [m(f"{task}_test_rerank_text")])

    if want("lenient"):
        sh(f"{PY} scripts/rescore.py --runs-dir {R} --qrels {B}/qrels/answerlenient_test.tsv --pattern 'answer_test_*_text.run' --suffix lenient",
           E, [])

    if want("xling"):
        from collections import Counter
        # per-language counts of scored test questions (a multi-language question counts in each of its
        # languages, as in `parlir.retrieve --xling <lang>`); 'all' counts each question once
        langs, seen = Counter(), set()
        qa = {l.split()[0] for l in (B / "qrels" / "answer_test.tsv").open()}
        nonen_test = 0
        for x in map(json.loads, (B / "queries_xling.jsonl").open()):
            if x["_id"] in qa and (x["_id"], x["lang"]) not in seen:
                seen.add((x["_id"], x["lang"]))
                langs[x["lang"]] += 1
        qs = {q["_id"]: q for q in map(json.loads, (B / "queries.jsonl").open())}
        nonen_test = sum(1 for q in qa if [l for l in (qs[q].get("orig_lang") or []) if l != "en"])
        have = {i for i, _ in seen}
        sel = ["all"] + sorted(l for l, c in langs.items() if c >= 100)
        (E / "xling_langs.json").write_text(json.dumps(
            {"counts": langs, "selected": sel, "test_questions_nonEN_orig": nonen_test,
             "test_questions_with_orig_text": len(have),
             "test_questions_orig_unknown": sum(1 for q in qa if not qs[q].get("orig_lang"))}, indent=1))
        for l in sel:
            for system, texts in (("bm25", ("orig", "en")), ("bge-m3", ("orig", "en")), ("me5-large", ("orig", "en"))):
                for t in texts:
                    tag = f"answer_test_{system}_text_x{l}-{t}"
                    sh(f"{ret} --task answer --split test --system {system} --field text --xling {l} --xling-text {t}", E, [m(tag)])


    if want("overlap"):
        sh(f"{PY} scripts/analysis_overlap.py --build {B} --run {R}/answer_test_bm25_text.run --split test --out {E}/overlap",
           E, [E / "overlap" / "overlap.json"])

    if want("strata"):
        sh(f"{PY} scripts/analysis_strata.py --build {B} --runs-dir {R} --systems bm25_text bge-m3_text rrf_text rerank_text me5-base-ft_text --out {E}/strata.json",
           E, [E / "strata.json"])

    if want("posthoc8"):
        # POST HOC: a 2025 LLM-based embedder zero-shot (both GPUs, ~40 min on 2x RTX 4090),
        # dense question->question for the citation tasks, and their analyses. Outside the Holm family.
        for task in ("answer",) + CITE_TASKS:
            sh(f"{ret} --task {task} --split test --system qwen3-emb-8b --field text", E, [m(f"{task}_test_qwen3-emb-8b_text")])
        for task in CITE_TASKS:
            sh(f"{ret} --task {task} --split test --system bge-m3q2q --field text", E, [m(f"{task}_test_bge-m3q2q_text")])
        sh(f"uv run scripts/analysis_release_split.py --build {B} --exp {E} --system qwen3-emb-8b_text --release 2025-06-05",
           E, [E / "release_split.json"])
        sh(f"uv run scripts/analysis_cite_overlap.py --build {B} --exp {E}", E, [E / "cite_overlap.json"])

    if want("allperiods"):
        # POST HOC: zero-shot systems on the train and dev citation queries (non-held-out; no system below was
        # trained or tuned on them), for a per-term view of PrecedentRet and CiteRet. Fine-tuned models and dev-tuned
        # BM25 are excluded on purpose. Test runs already exist.
        for task in ("precedent", "cite"):
            for split in ("train", "dev"):
                for system in ("bm25", "bm25q2q", "bge-m3", "bge-m3q2q", "me5-base", "me5-large", "qwen3-emb-8b"):
                    sh(f"{ret} --task {task} --split {split} --system {system} --field text", E,
                       [m(f"{task}_{split}_{system}_text")])
                q = B / "qrels" / f"{task}_{split}.tsv"
                sh(f"{PY} -m parlir.rerank fuse --runs {R}/{task}_{split}_bm25_text.run {R}/{task}_{split}_bge-m3_text.run "
                   f"--qrels {q} --out {R}/{task}_{split}_rrf_text", E, [m(f"{task}_{split}_rrf_text")])
        sh(f"uv run scripts/analysis_all_periods.py --build {B} --exp {E}", E, [E / "all_periods.json"])

    if want("significance"):
        sh(f"{PY} scripts/significance.py --exp {E} --build {B}", E, [])
    if want("summary"):
        summ = {}
        for p in sorted(R.glob("*.metrics.json")):
            tag = p.name[: -len(".metrics.json")]
            meta_p = R / f"{tag}.meta.json"
            summ[tag] = {"aggregate": json.loads(p.read_text())["aggregate"],
                         "meta": json.loads(meta_p.read_text()) if meta_p.exists() else None}
        (E / "metrics_summary.json").write_text(json.dumps(summ, indent=1))
        print(f"metrics_summary.json: {len(summ)} runs")


if __name__ == "__main__":
    main()
