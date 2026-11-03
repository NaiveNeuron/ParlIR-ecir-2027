---
license: other
license_name: parlir-eu-reuse
license_link: https://huggingface.co/datasets/NaiveNeuron/parlir/blob/main/LICENSE-DATA.md
language:
- en
- multilingual
pretty_name: ParlIR
task_categories:
- text-retrieval
task_ids:
- document-retrieval
tags:
- european-parliament
- parliamentary-questions
- test-collection
configs:
- config_name: corpus
  data_files:
  - split: corpus
    path: v1.0/corpus.jsonl
- config_name: queries
  data_files:
  - split: queries
    path: v1.0/queries.jsonl
- config_name: queries-xling
  data_files:
  - split: queries
    path: v1.0/queries_xling.jsonl
- config_name: qrels-answer
  data_files:
  - split: train
    path: v1.0/qrels_jsonl/answer_train.jsonl
  - split: dev
    path: v1.0/qrels_jsonl/answer_dev.jsonl
  - split: test
    path: v1.0/qrels_jsonl/answer_test.jsonl
- config_name: qrels-cite
  data_files:
  - split: train
    path: v1.0/qrels_jsonl/cite_train.jsonl
  - split: dev
    path: v1.0/qrels_jsonl/cite_dev.jsonl
  - split: test
    path: v1.0/qrels_jsonl/cite_test.jsonl
- config_name: qrels-precedent
  data_files:
  - split: train
    path: v1.0/qrels_jsonl/precedent_train.jsonl
  - split: dev
    path: v1.0/qrels_jsonl/precedent_dev.jsonl
  - split: test
    path: v1.0/qrels_jsonl/precedent_test.jsonl
- config_name: qrels-precedenthc
  data_files:
  - split: train
    path: v1.0/qrels_jsonl/precedenthc_train.jsonl
  - split: dev
    path: v1.0/qrels_jsonl/precedenthc_dev.jsonl
  - split: test
    path: v1.0/qrels_jsonl/precedenthc_test.jsonl
---

# ParlIR

ParlIR is a test collection built from European Parliament written questions (parliamentary terms 8–10,
July 2014 – October 2026) and the official answers of the Commission, the High Representative and the Council.
Relevance comes from the institutions' own links: each question is linked to its reply, and replies cite earlier
answers. The qrels come from these records, and a sample of the referral citations was checked by a human.

- Paper: *ParlIR: A Test Collection for Answer and Precedent Retrieval from European Parliament Written Questions*
  (Marek Šuppa and Viktória Ondrejová, under review)
- Code, pipeline, tutorial and datasheet: https://github.com/NaiveNeuron/ParlIR-ecir-2027

## Tasks

| Task | Query | Relevant documents | Metric |
|---|---|---|---|
| **AnswerRet** (`answer`) | a question | its official reply, among all 76,536 answers | MRR@10 |
| **PrecedentRet** (`precedent`) | a new question | the earlier answers its reply cites, among answers published before the question | R@10, R@100 |
| CiteRet (`cite`) | a question | the same citations, among answers published before the reply | R@10, R@100 |

`precedenthc` and `citehc` keep only referral citations ("…refers the Honourable Member to its answer to…").
`answerlenient` also accepts answers that are equivalent to the official one (`equivalents.jsonl`). Splits are
by question date: train 2014–2021, dev 2022–2023, test 2024–2026.

## Versions

Pin downloads to the revision id below. Each tag is an annotated git tag: the first id is the tag object, which
resolves as a revision, and the second is the commit it points to in the history. The paper cites the `v1.0` tag id
(`2176502d8ed7`) as the frozen snapshot and the other two versions by tag name.

| Tag | Contents | Revision (tag id) | Commit |
|---|---|---|---|
| `v1.0` | data and qrels (`v1.0/`) | `2176502d8ed76885c17fc85604be33d2cf429368` | `574ebbdc6c66` (20 Oct 2026) |
| `v1.0-textonly` | text-only evaluation export (`v1.0/textonly/`) | `925a8807a3b471beac0dc376826b2e0a086b0bc4` | `b7745dbd9b1b` (22 Oct 2026) |
| `v1.0-evidence-r5` | all runs, per-query metrics, audit records, fine-tuned weights (`v1.0/experiments/`, about 21 GB) | `6b381010ef7b07927b30bd4c35bbb1d416f68baf` | `ddf8c71259e1` (22 Oct 2026) |

The evidence supplement was released in five versions between 20 and 22 October 2026 as analyses were added, and
`-r5` is the final version. The superseded tags (`v1.0-evidence`, `-r2` to `-r4`) were removed on 3 November 2026.
Their commits are still in the history, and the data tag `v1.0` was never moved.

```python
from datasets import load_dataset
corpus = load_dataset("NaiveNeuron/parlir", "corpus", revision="2176502d8ed76885c17fc85604be33d2cf429368")
```

## Files (`v1.0/`)

- `corpus.jsonl`: `_id` (`<QID>-ASW`), `text` (masked, use this), `raw_text` (unmasked), `date`, `institution`,
  `answers`, `header`, `source_url`
- `queries.jsonl`: `_id`, `subject`, `text` (masked, use this), `raw_text`, `date`, `source_url`, `type` (E/P),
  `term`, `addressee`, `author`, `orig_lang`, `orig_lang_src`, `answer`, `header`, `split`, `joint_size`
- `queries_xling.jsonl`: the official non-English versions of questions (7,683 texts, 22 languages)
- `equivalents.jsonl`: equivalent answers per query (`twin_question`, `duplicate_answer`, `near_duplicate_answer`)
- `precedent_meta.jsonl`: citation edges per query, with their source (`hyperlink`, `url`, `text`) and a
  `referral` flag
- `qrels/<task>_<split>.tsv`: TREC qrels (`qid 0 docid rel`), and `qrels_jsonl/` with the same qrels as JSONL
- `build_manifest.json`: counts and SHA-256 checksums

## Rules for systems

Systems may use only the `text` fields. Do not use ids, dates, `header`, `subject`, `raw_text`, `source_url`,
`answers` or the `answer` field as retrieval features, because some reveal the answer and others give shortcuts.
An answer's id is its question's id plus `-ASW` (matching ids alone scores MRR@10 0.989), and a date window lifts
BM25 from 0.547 to 0.760. Dates are used
only by the evaluator's as-of filter for the citation tasks. For those tasks, systems may also use the links from
earlier eligible answers to the questions they replied to, as the question-matching baselines do.

The text-only export (`v1.0-textonly`) is the safest choice for evaluation. Corpus and queries carry only a random
id and the masked text, the dates sit in evaluator-only files, and the id mapping is a separate download
(`v1.0/textonly-provenance/`). That mapping is public, so do not use it as a retrieval feature. All stored
main-experiment runs score the same on the export. It omits the links that question matching needs.

## Baselines (test split)

| System | AnswerRet MRR@10 | PrecedentRet R@10 |
|---|---|---|
| BM25 | 0.547 | 0.534 |
| BGE-M3 | 0.525 | 0.585 |
| Qwen3-Embedding-8B (zero-shot) | 0.714 | 0.632 |
| BM25 question matching | n/a | 0.712 |
| BGE-M3 question matching | n/a | 0.725 |

PrecedentRet has 301 test queries, so please report confidence intervals. More results, including fusion,
re-ranking, fine-tuning and a label audit with LLM judges, are in the paper and the evidence supplement.

## Counts (v1.0)

76,536 answers (Commission 70,397, High Representative 4,181, Council 1,892, unknown 66) and 78,309 questions with a
split (train 59,612, dev 7,735, test 10,962).

| Task | train (queries / pairs) | dev | test |
|---|---|---|---|
| `answer` | 59,612 / 59,612 | 7,735 / 7,735 | 10,962 / 10,962 |
| `answerlenient` | 59,612 / 70,463 | 7,735 / 7,854 | 10,962 / 11,239 |
| `precedent` | 2,302 / 3,393 | 252 / 364 | 301 / 481 |
| `precedenthc` | 1,788 / 2,462 | 159 / 208 | 192 / 269 |
| `cite` | 3,767 / 5,592 | 299 / 441 | 396 / 634 |
| `citehc` | 2,991 / 4,199 | 186 / 245 | 256 / 352 |

Known issues of v1.0 (rare unmasked identifier spellings, a few odd dates) are listed in the datasheet. v1.0 stays
frozen, and fixes go into the next release.

## Licence

Source texts: European Parliament Open Data Portal, © European Union, CC BY 4.0. Commission answers are reused
under Decision 2011/833/EU (CC BY 4.0 via C(2019) 1655), Council answers under Decision (EU) 2017/1842. The texts
were modified (headers removed, identifiers masked). ParlIR annotations: CC BY 4.0. See `LICENSE-DATA.md`.
Questions are signed by MEPs acting as public office-holders. Removal requests can be filed on the code
repository's issue tracker. Accepted removals apply to `main` and the next release, and published, pinned revisions
stay unchanged unless the law requires otherwise.

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
