# Datasheet for ParlIR

Following Gebru et al., "Datasheets for Datasets" (CACM 2021). Counts are from the v1.0 build
(`data/build/v1.0/build_manifest.json`, qrels and query files); the pipeline is `parlir/`. Summary files
`experiments/v1.0/*.json` are in the code repository. Runs, per-query metrics and audit records are in the HF
evidence supplement (`v1.0/experiments/...`, tag `v1.0-evidence-r5`, tag id 6b381010ef7b07927b30bd4c35bbb1d416f68baf).

## Motivation

- **Purpose.** A reusable IR test collection for parliamentary oversight. Given a written question by a Member of
  the European Parliament (MEP), retrieve the earlier answers its reply will cite (PrecedentRet, as-of
  known-citation recall) and its official answer (AnswerRet, known-item). CiteRet is the drafting-time variant of
  PrecedentRet. Relevance comes from the institutions' own records, not from new human or LLM labels.
- **Gap.** Existing parliamentary QA/IR resources do not offer full-pool ranking over the EP's answer archive with
  audited, temporally split qrels. The Q↔A pairing itself is public in the EP Open Data API and is not claimed
  as a contribution.
- **Creators.** Marek Šuppa (Comenius University in Bratislava) and Viktória Ondrejová (University of Oslo).

## Composition

- **Instances.** Documents are official answers (`<QID>-ASW`) by the European Commission, the Vice-President /
  High Representative (VP/HR) or the Council (66 with unknown institution). Queries
  are E (ordinary) and P (priority) written questions from terms 8–10 (July 2014 – 2026) that were answered and
  whose text was acquired.
- **Counts (v1.0).** Raw snapshot frozen at 2026-10-20T06:53:49Z (`data/raw/SNAPSHOT_FROZEN_AT`). 80,255 listed
  question IDs (E+P, terms 8–10). Corpus: 76,536 parsed answers (Commission 70,397, VP/HR 4,181, Council 1,892,
  institution unknown 66). Queries: 78,314 records = 78,309 split-assigned (train 59,612 (2014–2021), dev 7,735
  (2022–2023), test 10,962 (2024–2026)) plus 5 unsplit. `queries_xling.jsonl`: 7,683 texts in 22 languages. Q↔A link sources: csv 46,619,
  own question file 20,075, detail records 11,285, answer header 694.

  | Task | train (queries / pairs) | dev | test |
  |---|---|---|---|
  | `answer` | 59,612 / 59,612 | 7,735 / 7,735 | 10,962 / 10,962 |
  | `answerlenient` | 59,612 / 70,463 | 7,735 / 7,854 | 10,962 / 11,239 |
  | `cite` | 3,767 / 5,592 | 299 / 441 | 396 / 634 |
  | `citehc` | 2,991 / 4,199 | 186 / 245 | 256 / 352 |
  | `precedent` | 2,302 / 3,393 | 252 / 364 | 301 / 481 |
  | `precedenthc` | 1,788 / 2,462 | 159 / 208 | 192 / 269 |

  Citation edge attrition: ok 4,461; ok_drafting_only 2,685 (eligible for `cite*` only);
  cited_answer_not_before_reply 281; cited_question_has_no_answer_doc 98. Equivalent answers: 1,396 queries
  (twin_question 419, duplicate_answer 852, near_duplicate_answer 296).
- **PrecedentRet sample size.** The PrecedentRet test split has 301 queries, below the pre-registered 500 (H6 not
  met), so its results (and CiteRet's, 396 test queries) should be read with confidence intervals.
- **Fields.** Corpus: `_id`, `text` (masked, canonical), `raw_text` (header-stripped, unmasked), `date`,
  `institution`, `answers`, `header`, `source_url`. Queries: `_id`, `subject`, `text`, `raw_text`, `date`,
  `source_url`, `type`, `term`, `addressee`, `author`, `orig_lang`, `orig_lang_src` (where `orig_lang` came from:
  `csv` or `detail`), `answer`, `header`, `split`, `joint_size`. `queries_xling.jsonl`: official non-English
  versions of questions (`_id`, `lang`, `text` masked, `raw_text`). `equivalents.jsonl`: per query, typed
  equivalent answers (`twin_question`, `duplicate_answer`, `near_duplicate_answer`). `precedent_meta.jsonl`: per
  query, citation edges with provenance (`via`: `hyperlink`, `url`, `text`) and a `referral` flag.
- **Labels.** `answer` qrels: binary, the official answer linked to the question (one answer may serve several
  questions: joint answers). `answerlenient`: the official answer (2) plus typed equivalent answers (1) from
  `equivalents.jsonl`. `precedent` (PrecedentRet, co-headline): earlier answers explicitly cross-referenced in the
  official answer (hyperlink, URL or textual reference), eligible if published before the question date
  (pre-submission); `precedenthc`: the explicit-referral subset. `cite` / `citehc` (CiteRet, drafting-time
  variant): the same edges, eligible if published before the official reply's date.
- **Splits.** Temporal by question date: train 2014–2021, dev 2022–2023, test 2024–2026. Joint-answer groups that
  span a split boundary are excluded from all query sets; their answer stays in the corpus.
- **Incompleteness.** Qrels are known-item/known-citation: other answers may be topically relevant but are
  unjudged. An LLM pooled audit (`scripts/audit_llm_omp.py`) is reported only as a sensitivity analysis. Its
  pools are reproducible from the sample seed and the runs, and are released in the HF evidence supplement.
- **Human check of referral labels.** The first author graded a seeded sample of 120 of the 269 referral-phrased
  PrecedentRet test citations (from 97 questions), seeing only the new question and the earlier answer:
  34% (95% CI 26–43%) of the cited answers fully answer the new question and 80% (72–86%) at least partly
  (`docs/human-check/referrals.csv`, scored with `scripts/human_check.py score-extra`). PrecedentRet therefore
  measures finding the cited answers, not whether a question was already answered. Unanswered questions and
  questions whose files were missing are dropped: question_unanswered_or_answer_missing 1,592; question_empty 163;
  question_no_date 184; answer parse errors 9; answer_empty 1; question parse/file errors 2. The v1.0 build
  passes all hard integrity checks (`experiments/v1.0/integrity.json`).
- **Language.** English versions (all questions and answers are available in EN). Each item also exists in the
  author's original language(s), not in all 24 EU languages. `orig_lang` records this where known, and
  official non-English question versions are released in `queries_xling.jsonl` (L→EN queries).
- **Confidential data.** None; all texts are published official documents.
- **Personal data.** Questions name their MEP author(s); answers' headers name the answering Commissioner or the
  Council. These are public officials acting in an official capacity, published by the EU institutions. Names
  are kept as metadata (`author`, `header`) and are excluded from the canonical retrieval text (`text`), where
  headers are stripped and identifiers masked. Question bodies may mention third parties as reported in public
  debate; no additional personal data is added.
- **Sensitive content.** Questions cover all EU policy areas, including politically contested topics; texts
  reflect their authors' views.

## Collection process

- **Sources.** EP Open Data Portal API v2 (question list, yearly CSV datasets, detail records with
  `answers_to` links) and the EP document repository host (deterministic per-document URLs).
- **Formats.** XHTML preferred (2014–2025, hyperlinked cross-references), then DOCX, then XML.
- **Mechanism.** `parlir.acquire` stages `list`, `csv`, `csv-langs`, `files`, `detail`: resumable, rate-limited
  (≤ ~10 req/s to the document host; ≤ 500 req / 5 min to the ODP API, retrying HTTP 429). A full EN crawl
  takes about 5 hours.
- **Time frame.** Documents dated July 2014 – 2026; collected October 2026.
- **Excluded.** Terms 6–7: only available via DCEP (research-only licence, no links) or the doceo website behind a
  bot challenge whose circumvention the legal notice forbids. Oral questions and questions for question time
  are not included.
- **Ethical review.** Not applicable: the texts are public official documents published for reuse.

## Preprocessing / cleaning / labelling

- Text extraction from XHTML/DOCX/XML (`parlir.parse`).
- Headers removed structurally: the answer header (answering Commissioner/institution, date) is kept separately in
  `header`, because it leaks portfolio cues. Questions from 2015 (no yearly CSV) are dated from their XML header.
- Every question/answer identifier (e.g. `E-10-2024-001357`, `E-001234/2024`, including URL-embedded ids, hyphen
  variants, ids glued to a language code, and typo forms) becomes `[REF]` in `text`, so that IDs cannot shortcut
  retrieval; `raw_text` keeps the unmasked body. Term-8 PE-number footers and co-signatory/supporter lines are
  removed. An identifier-overlap probe (`idoracle`) measures residual leakage, and `parlir.validate` checks for
  identifiers with a detector independent of the masker.
- Q↔A links merged from CSVs, detail records, the question's own file and answer headers (`link_sources` in the
  manifest).
- Citation edges extracted from hyperlinks, URLs and textual references in answers (provenance and referral flag
  in `precedent_meta.jsonl`); only answers dated before the cutoff (official reply date for `cite*`, question date
  for `precedent*`) are eligible (as-of).
- Equivalent answers (`equivalents.jsonl`) typed as `twin_question`, `duplicate_answer` or `near_duplicate_answer`.
- Integrity checks: `parlir.validate` (optional MinHash near-duplicate answer clusters).
- Raw data is kept under `data/raw/` and is not redistributed. The acquisition pipeline can crawl it again, but a
  fresh crawl sees the archive as it is today and will not match the frozen v1.0 snapshot exactly.

## Uses

- **Intended.** Benchmarking lexical, dense and re-ranking retrieval on long, formal, domain-specific text;
  known-item and citation retrieval; temporal generalisation (train/dev/test by period); shortcut analysis
  (masked vs raw text).
- **Not intended.** Profiling or ranking individual MEPs or officials; inferring political opinions of
  individuals; treating known-item qrels as complete topical relevance judgements; claims about multilingual
  retrieval beyond the EN + original-language configurations.
- **Risks.** Answers reflect institutional positions at the time; models trained on them may reproduce those
  positions. Temporal drift (new policy topics) affects test-period results.
- **Permitted inputs.** Systems may use only the `text` fields of queries and corpus. Ids, dates, `header`,
  `subject`/`raw_text` (unmasked), `source_url`, `answers` and the queries' `answer` field are forbidden inputs:
  they give the link away. Measured on v1.0 test (`experiments/v1.0/metadata_probe.json`): the rule "answer id =
  question id + `-ASW`" scores MRR@10 0.989 (it fails only for 119 joint replies), and keeping only answers dated
  0–180 days after the question lifts BM25 from 0.547 to 0.760. At test time, dates may be used only for the
  documented as-of filters of the citation tasks; training may use training-split metadata (two post-hoc recipes
  select negatives by date).
- **Citation-task allowance.** For `precedent*`/`cite*`, a system may also use the archive's question→answer links
  of *eligible* answers (dated before the query's cutoff) and those earlier questions' `text`; the query's own
  `answer` link and anything after its cutoff stay forbidden (the as-of filter enforces the dates). Only the
  question→question baselines use this; all other baselines rank answer texts only. The text-only export does not
  contain these links, so question→question baselines cannot be rerun from it (use the v1.0 build).
- **Baseline protocol caveat (training on the temporal splits).** AnswerRet candidates are the whole archive, so
  answers published after a test question are candidates too (archival task). Fine-tuning mE5-base on the whole
  training split with BM25 hard negatives mined over the whole archive, as pre-registered (H7), HURTS (MRR@10
  0.424 ± 0.008 over 3 seeds vs 0.490 zero-shot). Post-hoc controls (all post hoc, to
  be confirmed on the next test period), all over 3 seeds: random negatives score 0.669 ± 0.012, random negatives
  dated within 365 days of the positive 0.712 ± 0.003, and date-matched BM25 negatives 0.637 ± 0.005. So
  these BM25-negative recipes underperform random negatives; why is not established. Plausible factors: false
  negatives (on audited TEST queries, a proxy since training was not audited, BM25's top non-official answer is
  graded "directly addresses" for 18% and "partially" for 51% of questions by the LLM judges) and a date signal.
  Recommendation: do not mine hard negatives naively from this archive; random or date-matched negatives did better.
- **Post-hoc baselines.** Qwen3-Embedding-8B (2025, zero-shot, our 1,024-token cap) reaches AnswerRet test
  MRR@10 0.714, far above the pre-registered 2023–24 encoders (≤ 0.525) and fusion (0.588). Its gain over BM25 is
  the same for test questions answered after the model's release (+0.166, n = 6,916) as before it (+0.168), so
  memorisation of test pairs is unlikely to explain it (`scripts/analysis_release_split.py`). PrecedentRet test R@10: BM25 0.534,
  BGE-M3 0.585, Qwen3-Embedding-8B 0.632, BM25 question→question 0.712, BGE-M3 question→question 0.725. For CiteRet,
  BGE-M3 question→question reaches R@10 0.744 (BM25 question→question 0.724; paired +0.020, CI −0.011–0.051).

## Known issues in v1.0 (errata; to be fixed in the next release, v1.0 stays frozen)

Measured by `parlir.validate` diagnostics (experiments/v1.0/integrity.json) after an independent data audit:
- **Residual question references** in spellings the v1.0 masker does not know (`E-002242-18`, `E007189/2012`,
  `E 5847/14`, `E-011437//2015`, Symbol-font dash U+F02D): 255 queries (train 243, dev 7, test 5) and 48 answers.
  None names a dev/test citation target, and none is a query's own id; a small identifier-copy signal exists in
  train (citation train queries). Measured impact (`scripts/analysis_residual_masking.py`,
  experiments/v1.0/residual_masking_impact.json): masking all of them (401 replacements) in a diagnostic copy changes
  BM25 and question→question BM25 test metrics on answer/cite/precedent by at most 6.8e-6 (one RR@10 value); dense
  models were not rerun.
- **Dates:** 1 answer is dated before its question; 55 answer texts start with the literal placeholder
  `([d.m.yyyy])`; 71 answers have no date in their header and take the question date (68 are dated on the question
  day); 2 of the 5 unsplit queries have impossible dates (they are in no split).
- **`near_duplicate_answer`** equivalents are MinHash-LSH candidates (64 permutations) at an approximate 5-shingle
  Jaccard threshold of 0.9, not exact: of 2,212 pairs, 93.5% have exact 5-shingle Jaccard < 0.9 (median 0.85) and 91% are short (< 60 words) referral
  stubs. Lenient scoring changes MRR@10 by at most 0.001, so no reported result depends on it.
- **Unmasked fields:** `subject`, `header` and `raw_text` are not masked (see Permitted inputs). 12 train queries
  keep a "Question supported by …" co-signatory list in `text`. The only private-looking e-mail address
  (`flower234@gmail.com`) is an invented example in a question about whether e-mail addresses are personal data.
- **Train/test shifts** users should know: `http://` footnote URLs are typical of older answers; short official
  answers (< 60 words, mostly referral stubs) are 4.0% of train vs 1.3% of test queries; 2015 is 25% of train
  questions; 19.9% of 2026 question ids (782 of 3,928) are excluded as not yet answered at the snapshot.
- **Encoded ids:** record keys are the institutions' references, and an answer's id is its question's id plus
  `-ASW` (that rule scores MRR@10 0.989). Evaluation code must treat ids as opaque keys. Addressed without changing
  v1.0 by the derived text-only export (HF tag `v1.0-textonly`; `scripts/export_textonly.py`): random opaque ids,
  evaluator-only date files, the provenance mapping as a separate download; every main-experiment run scores
  identically on it.

## Distribution

- **How.** Code repository (MIT) and a Hugging Face dataset (`hf/README.md`): configs `corpus`, `queries`,
  `queries-xling`, `qrels-answer`, `qrels-precedent`, `qrels-precedenthc`, `qrels-cite`, plus all qrels files (`qrels/<task>_<split>.tsv`),
  `equivalents.jsonl`, `precedent_meta.jsonl` and the LLM-audit pools. Repository:
  https://github.com/NaiveNeuron/ParlIR-ecir-2027 and https://huggingface.co/datasets/NaiveNeuron/parlir.
  Frozen snapshot: version tag `v1.0` (tag id `2176502d8ed76885c17fc85604be33d2cf429368`, commit `574ebbdc6c66`).
  A Zenodo DOI will follow for the camera-ready version.
- **When.** With the ECIR 2027 Resource Track paper.
- **Licence.** Source texts: EP Open Data, CC BY 4.0; Commission answers under Decision 2011/833/EU (CC BY 4.0 via
  C(2019) 1655); Council answers under Decision (EU) 2017/1842. ParlIR annotations: CC BY 4.0. See
  `LICENSE-DATA.md` for the attribution statement.
- **Export controls / restrictions.** None known.

## Maintenance

- **Maintainers / contact.** The authors, via the code repository's issue tracker.
- **Versioning.** Each release is a versioned build (`data/build/<version>/`) with a `build_manifest.json`
  recording counts and sha256 checksums; experiments record the manifest hash.
- **Updates.** New questions/answers can be added by re-running the pipeline; earlier test splits are kept fixed
  per version. Errata are posted on the code repository's issue tracker.
- **Takedown.** Requests concerning personal data go to the code repository's issue tracker, and the maintainers
  review each one. Accepted removals apply to the main branch and the next release. Published, pinned revisions stay
  unchanged, so that reported results remain reproducible, unless the law requires otherwise.
- **Contributions.** Via pull requests to the code repository.
