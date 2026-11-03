# ParlIR

ParlIR is an information retrieval test collection built from European Parliament written questions
(parliamentary terms 8–10, July 2014 – October 2026) and the official answers of the Commission, the High
Representative and the Council. Relevance comes from the institutions' own records: each question is linked to
its reply, and replies cite earlier answers.

This repository holds the code that builds, validates and evaluates the collection. The data is on Hugging Face:
https://huggingface.co/datasets/NaiveNeuron/parlir. The repository is published as a single release commit, so
its development history is not included. The dated pre-registration and its deviations are in
`docs/preregistration/`, and the dataset history on Hugging Face starts on 20 October 2026.

Paper: *ParlIR: A Test Collection for Answer and Precedent Retrieval from European Parliament Written Questions*,
Marek Šuppa and Viktória Ondrejová (under review).

## Tasks

| Task | Query | Relevant documents | Metric |
|---|---|---|---|
| **AnswerRet** (`answer`) | a question | its official reply, among all 76,536 answers | MRR@10 |
| **PrecedentRet** (`precedent`) | a new question | the earlier answers its reply cites, among answers published before the question | R@10, R@100 |
| CiteRet (`cite`) | a question | the same citations, among answers published before the reply | R@10, R@100 |

`precedenthc` and `citehc` keep only referral citations ("…refers the Honourable Member to its answer to…").
`answerlenient` also accepts answers that are equivalent to the official one. Splits are by question date:
train 2014–2021, dev 2022–2023, test 2024–2026. Joint-answer groups that span two splits are left out of the
queries.

An example from the PrecedentRet test set: question `E-10-2024-002766` (4 December 2024) asks whether Poland met
its European arrest warrant obligations in the Nord Stream investigation. Its reply cites `E-10-2024-001695-ASW`,
an answer to another MEP published nine days before the question was put.

## Quickstart

```bash
uv sync                    # uses Python 3.12 (.python-version); add --extra dense for the dense baselines
```

ParlIR is tested with CPython 3.11 and 3.12. The free-threaded builds (for example 3.14t) do not work yet, because
some dependencies have no wheels for them.

Download the frozen v1.0 data into `data/build/` (about 515 MB):

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('NaiveNeuron/parlir', repo_type='dataset', revision='2176502d8ed76885c17fc85604be33d2cf429368', local_dir='data/build', allow_patterns=['v1.0/' + f for f in ('corpus.jsonl', 'queries.jsonl', 'queries_xling.jsonl', 'equivalents.jsonl', 'precedent_meta.jsonl', 'build_manifest.json', 'qrels/*.tsv')])"
```

Run BM25 and score it:

```bash
uv run python -m parlir.retrieve --build data/build/v1.0 --task answer --split test --system bm25 --exp experiments/quickstart
```

```python
from parlir.loader import load, evaluate_run

corpus, queries, qrels = load("data/build/v1.0", task="answer", split="test")
agg, per_query = evaluate_run("experiments/quickstart/answer_test_bm25_text.run", "data/build/v1.0",
                              task="answer", split="test")
print(agg["RR@10"])   # 0.547 on v1.0
```

`examples/quickstart.py` is the tutorial. It runs BM25 on the test split of `answer`, `precedent` and `cite`,
scores the runs, and shows how to iterate the cross-lingual queries:

```bash
uv run python examples/quickstart.py --build data/build/v1.0
```

Other systems: `bm25q2q` (question matching, citation tasks), `idoracle` (the identifier shortcut probe), and the
dense models `me5-base`, `me5-large` and `bge-m3` (`--extra dense`). `python -m parlir.rerank` fuses runs and
re-ranks them with a cross-encoder. With the `irds` extra (`uv sync --extra irds`),
`parlir.loader.register_ir_datasets(build_dir)` registers the collection with [`ir_datasets`](https://ir-datasets.com).

## Versions

Pin downloads to the revision id below. Each tag is an annotated git tag: the first id is the tag object, which
resolves as a revision, and the second is the commit it points to. The paper cites the `v1.0` tag id
(`2176502d8ed7`) as the frozen snapshot and the other two versions by tag name.

| Tag on Hugging Face | Contents | Revision (tag id) | Commit |
|---|---|---|---|
| `v1.0` | data and qrels | `2176502d8ed76885c17fc85604be33d2cf429368` | `574ebbdc6c66` |
| `v1.0-textonly` | text-only evaluation export | `925a8807a3b471beac0dc376826b2e0a086b0bc4` | `b7745dbd9b1b` |
| `v1.0-evidence-r5` | all runs, per-query metrics, audit records and fine-tuned weights (about 21 GB) | `6b381010ef7b07927b30bd4c35bbb1d416f68baf` | `ddf8c71259e1` |

To check a local build against the release: `uv run scripts/verify_hf.py --build data/build/v1.0 --revision
2176502d8ed76885c17fc85604be33d2cf429368`.

## Rules for systems

Systems may use only the `text` fields of queries and corpus. Do not use ids, dates, `header`, `subject`,
`raw_text`, `source_url`, `answers` or the queries' `answer` field as retrieval features. Some of them reveal the
answer directly and others give shortcuts. An answer's id is its question's id plus `-ASW` (matching ids alone
scores MRR@10 0.989), and keeping only answers dated 0–180 days after the question lifts BM25 from 0.547 to 0.760
(`scripts/analysis_metadata_probe.py`). At test time, dates are used only by the as-of filter of the citation
tasks. Training may use metadata from the training split.

For `precedent*` and `cite*`, systems may also use the archive's links from earlier eligible answers to the
questions they replied to, and those questions' text. The query's own link and anything dated after its cutoff
stay forbidden. Of the baselines, only question matching (`bm25q2q`, `bge-m3q2q`) uses these links.

The text-only export (`v1.0-textonly`) is the safest choice for evaluation. Corpus and queries carry only a random
id and the masked text, the dates for the as-of filter are in evaluator-only files, and the id mapping is a
separate download (`v1.0/textonly-provenance/`). That mapping is public, so do not use it as a retrieval feature.
Score a run with `uv run scripts/export_textonly.py score --out <export dir> --task
precedent_test --run my.run`. All stored main-experiment runs score the same on it. The export omits the links
that question matching needs, so those baselines need the v1.0 build.

## Reproducing the paper

Every number in the paper's tables is read from the summary files in `experiments/v1.0/` of this repository. The
runs, per-query metrics, judge outputs and fine-tuned weights behind them are in the evidence supplement on
Hugging Face (tag `v1.0-evidence-r5`, under `v1.0/experiments/`, about 21 GB), so you can check the numbers
without retraining anything.

| Paper | Files in `experiments/v1.0/` |
|---|---|
| Table 1 (shortcut audit) | `integrity.json`, `metrics_summary.json` |
| Table 2 and the label audit (§6, §8) | `audit/`, `audit_full3/audit/` (`summary_12systems.json`, `contrasts.json`, `rules.json`), `strata.json` |
| Table 3 (human check) | `human_extra_summary.json`, `human_referrals_key.json` (sampled ids, seed) |
| Table 4 (statistics) | `stats.json`, `label_flow.json`, `build_manifest.json` |
| Tables 5 and 6 (cross-lingual, main results) and significance | `xling_langs.json`, `metrics_summary.json`, `bootstrap.json` |
| Table 8 (all periods) | `all_periods.json` |
| Other analyses in §8 | `followups.json`, `cite_overlap.json`, `release_split.json`, `metadata_probe.json`, `residual_masking_impact.json`, `robustness/` |
| Text-only export check (§9) | `textonly_verify.json` |

The human grades are in `docs/human-check/referrals.csv`, and
`uv run scripts/human_check.py score-extra --exp experiments/v1.0` recomputes Table 3 from them.

To rescore a stored run, download it from the evidence supplement and use `parlir.loader.evaluate_run`. To rerun
the main suite, use `uv sync --extra dense` and then `make experiments EXP=experiments/rerun`, which keeps the
stored paper summaries in `experiments/v1.0/` untouched (`scripts/run_experiments.py`).
A few steps run separately: the extra fine-tuning seeds (`scripts/ft_robustness.sh`, summarised by
`scripts/summarize_robustness.py`), the analyses `analysis_metadata_probe.py`, `analysis_residual_masking.py`,
`analysis_followups.py`, `analysis_label_flow.py`, `analysis_cross_task.py`, `analysis_temporal_shortcut.py`,
`analysis_audit_contrasts.py` and `analysis_audit_rules.py`, and `export_textonly.py verify`. Each script's
docstring gives its exact command. The dense models, fine-tuning and Qwen3-Embedding-8B need a GPU, and the
paper's runs used two 24 GB RTX 4090s. The LLM label audit runs inside OMP (`scripts/audit_llm_omp.py`), with the
judge prompts in `.omp/agents/`.

## Rebuilding from source

The pipeline crawls the EP Open Data Portal and the EP document host. It rate-limits its requests and resumes
interrupted downloads, and the English texts take about 5 hours:

```bash
make acquire                                   # list, csv, csv-langs, files, detail
make build VERSION=v1.0 && make validate VERSION=v1.0
```

A fresh crawl sees the archive as it is today, so it will not match v1.0 exactly: new questions appear and
pages can change. The raw v1.0 snapshot (frozen on 20 October 2026) is not redistributed, so use the Hugging Face
release for exact v1.0 results. `make smoke` runs the whole pipeline on 200 questions in about 5 minutes (use a
fresh clone).

## Layout

```
parlir/       pipeline: acquire, parse, build, validate, retrieve, dense, rerank, finetune, evaluate, loader
scripts/      experiments, analyses, audit, release and verification scripts
examples/     quickstart.py (tutorial)
tests/        unit tests and the smoke-build fixture
docs/         datasheet.md, human-check/ (graded referral sample), preregistration/ (decision rules fixed before the build)
experiments/  v1.0/: the summary files behind the paper's tables (runs and audit records are on Hugging Face)
hf/           the Hugging Face dataset card
```

## Licence

Code: MIT (`LICENSE`). Data: source texts from the EP Open Data Portal (CC BY 4.0), Commission answers under
Decision 2011/833/EU, Council answers under Decision (EU) 2017/1842, and ParlIR annotations under CC BY 4.0. See
`LICENSE-DATA.md` for the attribution statement and `docs/datasheet.md` for the datasheet.

Issues, errata and removal requests: please use this repository's issue tracker.

## Citation

```bibtex
@misc{suppa2026parlir,
  title  = {{ParlIR}: A Test Collection for Answer and Precedent Retrieval from {European Parliament} Written Questions},
  author = {{\v{S}}uppa, Marek and Ondrejov{\'a}, Vikt{\'o}ria},
  year   = {2026},
  note   = {Dataset: https://huggingface.co/datasets/NaiveNeuron/parlir},
  url    = {https://github.com/NaiveNeuron/ParlIR-ecir-2027}
}
```
