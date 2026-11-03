"""Re-score saved runs against an alternative qrels file (e.g. twin-lenient AnswerRet qrels).

  uv run python scripts/rescore.py --runs-dir experiments/v1.0/runs --qrels data/build/v1.0/qrels/answerlenient_test.tsv \
      --pattern 'answer_test_*_text.run' --suffix lenient
Writes <run-stem>.<suffix>.metrics.json next to each run.
"""
import argparse
from pathlib import Path

from parlir.evaluate import evaluate, read_qrels, read_run, save_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", type=Path, required=True)
ap.add_argument("--qrels", type=Path, required=True)
ap.add_argument("--pattern", default="*.run")
ap.add_argument("--suffix", required=True)
a = ap.parse_args()
qrels = read_qrels(a.qrels)
for run in sorted(a.runs_dir.glob(a.pattern)):
    agg, per_q = evaluate(qrels, read_run(run))
    save_metrics(agg, per_q, run.with_suffix(f".{a.suffix}.metrics.json"))
    print(run.stem, {k: round(v, 4) for k, v in agg.items()})
