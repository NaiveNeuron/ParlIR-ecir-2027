# Pre-registration snapshot

These files record the decision rules and claim ledger as they were fixed before the v1.0 collection was built.
They are historical and unedited: byte-for-byte copies from the authors' internal development repository,
including draft placeholders and notes. They are not the final paper.

- `claims.md`: the claim -> evidence ledger with nine hypotheses (H1, H2a, H2b, H3-H8) and their decision rules
  (`docs/claims.md` in the internal repository)
- `main.tex`: the provisional "paper-first" manuscript with placeholders (`paper/main.tex`)

Source commit: `74bc2ab6458a5e918ef062895809ea2490ce9052` (tag `checkpoint/paper-first`),
2026-10-20T01:58:03+02:00 (19 October 2026, 23:58 UTC), "Paper-first: complete provisional ECIR manuscript with
placeholders, decision rules and claim ledger". The first data, a pipeline test on a few thousand answers, was
built about 80 minutes later, and the v1.0 snapshot was frozen on 20 October 2026 at 06:53 UTC.

This public repository starts from a single release commit, so the internal history is not included. The files
above are the record of what was fixed in advance.

## How the paper relates to this snapshot

The design was revised on 20 October 2026, around a first test run on partial data and before the interim build
and the frozen v1.0 snapshot. When the paper calls a system or rule "pre-registered", it means this revised design.
Systems and analyses added later are marked as post hoc in the paper.

| ID | Pre-registered rule | Outcome |
|---|---|---|
| H1 | BM25 MRR@10 < 0.8 | met (0.547) |
| H2a | masked identifier oracle MRR@10 ≤ 0.05 | met (0.000) |
| H2b | official overlap within the hard-negative IQR | met |
| H3 | cross-lingual configurations with ≥ 1,000 test queries | reported with ≥ 500 (adds Polish, 537) |
| H4 | Kendall τ ≥ 0.9, else audit labels primary | the fallback was dropped (LLM labels are never qrels); τ_b = 1.00 |
| H5 | judge recall on official answers ≥ 0.8 | met by the adjudicated grades (91%), not by Claude Haiku alone (75%) |
| H6 | PrecedentRet ≥ 500 test queries | not met (301) |
| H7 | fine-tuning on the training split helps | refuted (−0.058 MRR@10) |
| H8 | temporal effect, descriptive | per-year results in `experiments/v1.0/strata.json` |

Other changes to the plan: the citation task gained a drafting-time variant (CiteRet), significance testing uses a
grouped paired bootstrap instead of a t-test, and the baselines added fusion and dropped BM25+RM3 and BGE-M3
learned-sparse retrieval. Re-ranking uses the BM25 top 30 instead of 100. The original judges were replaced
mid-audit when their provider ran out of credit. The paper's "Pre-registered vs. as run" paragraph lists the
post-hoc additions.
