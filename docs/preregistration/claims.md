# Claim → evidence ledger (derived from paper/main.tex, paper-first draft)

Status: OPEN / SUPPORTED / REFUTED / REVISED. Each row cites the experiment dir that settles it.

| ID | Claim in paper | Test (experiment) | Decision rule | Status |
|----|----------------|-------------------|---------------|--------|
| F1 | EP written Q&A are fully retrievable at scale with official Q↔A links | data feasibility: real fetches per term | ≥90% of sampled Qs with an answer have a resolvable link | OPEN |
| F2 | Licence permits redistribution with attribution | quote EP legal notice + Commission reuse decision | explicit text found | OPEN |
| H1 | AnswerRet is nontrivial | BM25/dense on masked test split | BM25 MRR@10 < 0.8 (else pivot headline) | OPEN |
| H2a | Identifier shortcuts exist in raw and are removed by masking | shortcut audit: id-oracle, raw vs masked BM25 | id-oracle MRR@10 ≤ 0.05 on masked | OPEN |
| H2b | Answers don't routinely restate questions | LCS / Jaccard distributions vs hard negatives | median official overlap within hard-negative IQR | OPEN |
| H3 | Multilingual configurations exist with official texts | per-term language-version counts | report only configs with ≥1k test queries | OPEN |
| H4 | Official-only qrels give stable system rankings | pooled LLM audit, ≥2 families | Kendall τ ≥ 0.9 (else augmented primary) | OPEN |
| H5 | Judges recognise official answers (calibration) | judge recall on official answers in pool | report; < 0.8 → judges unreliable, downgrade | OPEN |
| H6 | PrecedentRet has enough queries and is harder than AnswerRet | cross-ref extraction + baselines | ≥500 test queries with resolvable precedent | OPEN |
| H7 | Training split is useful | fine-tune small dense model on train | significant gain over zero-shot on test | OPEN |
| H8 | Temporal/contamination effect | per-year effectiveness | descriptive | OPEN |
