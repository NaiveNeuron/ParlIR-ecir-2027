"""How similarly do the tasks rank systems? Kendall's tau between AnswerRet MRR@10 and the citation tasks' R@10.

  uv run python scripts/analysis_cross_task.py --exp experiments/v1.0
Systems: the 9 systems with runs on all tasks (as in Table 5, excluding question->question BM25 and the AnswerRet-only
rows). Reports tau-b, concordant/discordant pair counts, and the same without the two fine-tuned systems.
If per-query metrics are present (<exp>/runs/*.metrics.json; also in the HF evidence supplement) and --build is given,
adds a bootstrap 95% CI for each tau-b. The citation queries are a subset of the AnswerRet test queries, so the two
tasks are resampled JOINTLY: each replicate draws joint-answer groups (questions sharing one reply) with replacement
from the AnswerRet test set, and the same group weights apply to both tasks' means. 2,000 resamples, seed 0.
Writes <exp>/cross_task.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

SYSTEMS = ["bm25_text", "bm25tuned_text", "me5-base_text", "me5-large_text", "bge-m3_text", "me5-base-ft_text",
           "me5-base-ftdm_text", "rrf_text", "rerank_text"]
FINE_TUNED = {"me5-base-ft_text", "me5-base-ftdm_text"}


def tau(x, y):
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    den = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return {"tau_b": (conc - disc) / den if den else None, "concordant": conc, "discordant": disc,
            "pairs": len(x) * (len(x) - 1) // 2}


def boot_tau(runs: Path, syss: list[str], task: str, metric: str, group_of: dict[str, str],
              n_boot: int = 2000, seed: int = 0):
    import numpy as np
    def per_query(prefix, m):
        return [json.loads((runs / f"{prefix}_test_{s}.metrics.json").read_text())["per_query"][m] for s in syss]
    pa, pc = per_query("answer", "RR@10"), per_query(task, metric)
    qa, qc = sorted(pa[0]), sorted(pc[0])
    assert set(qc) <= set(qa), "citation queries must be a subset of the AnswerRet test queries"
    groups = sorted({group_of[q] for q in qa})
    gi = {g: i for i, g in enumerate(groups)}
    A = np.array([[p.get(q, 0.0) for q in qa] for p in pa])
    C = np.array([[p.get(q, 0.0) for q in qc] for p in pc])
    ga, gc = np.array([gi[group_of[q]] for q in qa]), np.array([gi[group_of[q]] for q in qc])
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)).astype(float)
        wa, wc = cnt[ga], cnt[gc]  # the same replicate weights for a query in both tasks
        if wc.sum() == 0:
            continue
        t = tau(list(A @ wa / wa.sum()), list(C @ wc / wc.sum()))["tau_b"]
        if t is not None:
            taus.append(t)
    taus.sort()
    return [taus[int(0.025 * len(taus))], taus[int(0.975 * len(taus)) - 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--build", type=Path, default=None, help="build dir (question -> answer groups) for the bootstrap CI")
    a = ap.parse_args()
    m = json.loads((a.exp / "metrics_summary.json").read_text())
    runs = a.exp / "runs"
    group_of = ({q["_id"]: q["answer"] for q in map(json.loads, (a.build / "queries.jsonl").open())}
                if a.build else None)
    out = {}
    for task, metric in (("cite", "R@10"), ("precedent", "R@10")):
        for label, syss in (("all", SYSTEMS), ("without_fine_tuned", [s for s in SYSTEMS if s not in FINE_TUNED])):
            ans = [m[f"answer_test_{s}"]["aggregate"]["RR@10"] for s in syss]
            oth = [m[f"{task}_test_{s}"]["aggregate"][metric] for s in syss]
            rec = {"systems": syss, **tau(ans, oth)}
            if group_of and all((runs / f"{p}_test_{s}.metrics.json").exists() for p in ("answer", task) for s in syss):
                rec["tau_b_ci"] = boot_tau(runs, syss, task, metric, group_of)
                rec["tau_b_ci_method"] = "joint grouped bootstrap (shared queries share replicate weights)"
            out[f"answer_vs_{task}_{label}"] = rec
    (a.exp / "cross_task.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, round(v["tau_b"], 3), v.get("tau_b_ci"), f"{v['discordant']}/{v['pairs']} discordant")


if __name__ == "__main__":
    main()
