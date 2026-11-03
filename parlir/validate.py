"""Integrity and leakage checks for a ParlIR build (paper Sec. 5, 'Integrity checks').

  uv run python -m parlir.validate --build data/build/<v> --out experiments/integrity/<v>.json
Hard checks (must be 0 violations; exit code 1 otherwise) and soft diagnostics (reported).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .build import ID_PATTERNS, SPLITS
from .evaluate import read_qrels

OWN_REF = re.compile(r"[EPO]-\d{4,6}/\d{2,4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minhash", action="store_true", help="near-duplicate answer clusters (slower)")
    a = ap.parse_args()
    corpus = {d["_id"]: d for d in map(json.loads, (a.build / "corpus.jsonl").open())}
    queries = {q["_id"]: q for q in map(json.loads, (a.build / "queries.jsonl").open())}
    hard, soft = Counter(), {}

    for s in SPLITS:
        qa = read_qrels(a.build / "qrels" / f"answer_{s}.tsv")
        lo, hi = SPLITS[s]
        for q, rels in qa.items():
            hard["answer_query_missing"] += q not in queries
            hard["answer_query_no_relevant"] += not any(g > 0 for g in rels.values())
            hard["answer_doc_missing"] += sum(d not in corpus for d in rels)
            if q in queries:
                hard["query_wrong_split"] += queries[q]["split"] != s
        def rq(name):
            p = a.build / "qrels" / f"{name}_{s}.tsv"
            return read_qrels(p) if p.exists() else {}
        # pre-submission variant: cited answers strictly before the question date
        for task in ("precedent", "precedenthc"):
            for q, rels in rq(task).items():
                qd = date.fromisoformat(queries[q]["date"])
                for d in rels:
                    hard["precedent_doc_missing"] += d not in corpus
                    if d in corpus:
                        hard["precedent_not_before_query"] += not (date.fromisoformat(corpus[d]["date"]) < qd)
                        hard["precedent_is_own_answer"] += d == queries[q]["answer"]
        # drafting-time CiteRet: cited answers strictly before the official reply's date
        for task in ("cite", "citehc"):
            for q, rels in rq(task).items():
                rd = date.fromisoformat(corpus[queries[q]["answer"]]["date"])
                for d in rels:
                    hard["cite_doc_missing"] += d not in corpus
                    if d in corpus:
                        hard["cite_not_before_reply"] += not (date.fromisoformat(corpus[d]["date"]) < rd)
                        hard["cite_is_own_answer"] += d == queries[q]["answer"]
        # lenient AnswerRet must contain exactly the strict official answer with grade 2
        for q, rels in rq("answerlenient").items():
            hard["lenient_official_mismatch"] += [d for d, g in rels.items() if g == 2] != [queries[q]["answer"]]

    # joint-answer groups never straddle splits
    by_ans = defaultdict(set)
    for q in queries.values():
        by_ans[q["answer"]].add(q["split"])
    hard["joint_group_split_across"] = sum(len(v) > 1 for v in by_ans.values())

    # masking: no identifier string survives in canonical text. The detector is deliberately INDEPENDENT of the
    # masker's ID_PATTERNS (added after a data audit): broader forms, any hyphen-like character, no boundary logic.
    H = "[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]"
    independent = [
        # term is never zero-padded in EP ids; '\d{1,2}' matched Horizon 2020 call codes such as 'EE-06-2016-2017'
        re.compile(rf"[EPOGK]{H}(?:[4-9]|10){H}(19|20)\d\d{H}\d{{4,6}}"),     # full ids, also inside URLs
        re.compile(rf"[EPO]\s*{H}\s*\d{{3,6}}\s*/\s*(\d{{2}}|(19|20)\d\d)(?!\d)"),  # printed references
        re.compile(rf"A[EP]{H}\d{{6}}{H}\d\d"),                                  # XML answer ids
    ]

    def survives(t):
        return any(p.search(t) for p in independent) or any(p.search(t) for p in ID_PATTERNS)
    hard["corpus_id_string_after_mask"] = sum(survives(d["text"]) for d in corpus.values())
    hard["query_id_string_after_mask"] = sum(survives(q["text"]) for q in queries.values())
    xp = a.build / "queries_xling.jsonl"
    hard["xling_query_id_string_after_mask"] = sum(survives(x["text"]) for x in map(json.loads, xp.open())) if xp.exists() else 0
    hard["corpus_empty"] = sum(not d["text"].strip() for d in corpus.values())
    hard["query_empty"] = sum(not q["text"].strip() for q in queries.values())
    # a *distinctive* header line (answerer line or a reference line) reappearing at the start of the body =
    # parser failure. Generic formulas ("on behalf of the European Commission") legitimately occur in bodies,
    # and later occurrences are footnotes citing other answers ("[1] Answer given by Mr X ..., <date>: <url>").
    distinctive = re.compile(r"(?i)answer given by|reply|^[A-Z]{2}$|[EPO]-\d{3,6}/\d{2,4}")
    hard["corpus_header_in_text"] = sum(any(h and len(h) > 10 and distinctive.search(h) and h in d["text"][:300]
                                            for h in d.get("header", [])) for d in corpus.values())
    own_ref_raw = 0
    author_in_answer = 0
    for q in queries.values():
        d = corpus[q["answer"]]
        num, yr = q["_id"].split("-")[3], q["_id"].split("-")[2]
        ref = f"{q['_id'][0]}-{num}/{yr}"
        own_ref_raw += ref in d["raw_text"]
        au = (q.get("author") or "").strip()
        if au:
            last = au.split()[-1]
            author_in_answer += len(last) > 3 and re.search(rf"\b{re.escape(last)}\b", d["raw_text"], re.I) is not None
    soft["answers_containing_own_question_ref_in_body_raw"] = own_ref_raw / max(1, len(queries))
    soft["answers_containing_author_surname_raw"] = author_in_answer / max(1, len(queries))
    soft["answer_header_contains_own_ref"] = sum(
        any(OWN_REF.search(h) for h in corpus[q["answer"]].get("header", [])) for q in queries.values()) / max(1, len(queries))
    soft["subject_verbatim_in_answer"] = sum(
        len(q["subject"]) > 20 and q["subject"].lower() in corpus[q["answer"]]["raw_text"].lower() for q in queries.values()) / max(1, len(queries))

    # References in spellings the masker does not know (num-year, no hyphen, spaced, double
    # slash, Symbol-font U+F02D dash). Reported, not a hard check: v1.0 is frozen; the masker is extended in v1.1.
    # Only matches that resolve to an existing question id count; the query's own id is excluded.
    broad = re.compile(r"(?<![A-Za-z0-9])([EPO])[\s\-\u2010-\u2015\u2212\uf02d]{0,3}(\d{3,6})\s?[/\-\u2010-\u2015\u2212\uf02d]{1,2}\s?(20\d\d|\d{2})(?!\d)")
    by_key = {}
    for qid in queries:
        p_ = qid.split("-")
        by_key[(p_[0], int(p_[3]), p_[2])] = qid

    def residual_refs(t, own=None):
        hits = set()
        for m in broad.finditer(t):
            yr = m.group(3) if len(m.group(3)) == 4 else "20" + m.group(3)
            hit = by_key.get((m.group(1), int(m.group(2)), yr))
            if hit and hit != own:
                hits.add(hit)
        return hits
    rq = {qid: residual_refs(q["text"], qid) for qid, q in queries.items()}
    ra = {did: residual_refs(d["text"]) for did, d in corpus.items()}
    by_split = Counter(queries[q]["split"] for q, h in rq.items() if h)
    target_hits = 0
    for task in ("cite", "precedent", "citehc", "precedenthc"):
        for split in ("dev", "test"):
            for q_, docs in read_qrels(a.build / "qrels" / f"{task}_{split}.tsv").items():
                target_hits += sum(d_[:-4] in rq.get(q_, set()) for d_ in docs)  # '<qid>-ASW' -> '<qid>'
    soft["residual_question_refs"] = {"queries": sum(bool(h) for h in rq.values()), "answers": sum(bool(h) for h in ra.values()),
                                      "queries_by_split": dict(by_split), "dev_test_citation_targets_named": target_hits}
    # date integrity diagnostics
    neg = 0
    for q in queries.values():
        ad, qd = corpus[q["answer"]].get("date"), q.get("date")
        neg += bool(ad and qd and ad[:10] < qd[:10])
    soft["answer_dated_before_question"] = neg
    soft["answer_text_starts_with_date_placeholder"] = sum(d["text"].lstrip().startswith("([d.m.yyyy])") for d in corpus.values())
    h = Counter(hashlib.sha1(d["text"].strip().lower().encode()).hexdigest() for d in corpus.values())
    dups = {k: v for k, v in h.items() if v > 1}
    soft["exact_duplicate_answer_texts"] = {"groups": len(dups), "docs": sum(dups.values())}
    soft["joint_answer_rate"] = sum(len(d.get("answers", [])) > 1 for d in corpus.values()) / max(1, len(corpus))
    soft["institution"] = Counter(d.get("institution") for d in corpus.values())
    soft["addressee"] = Counter(q.get("addressee") for q in queries.values())
    soft["n_corpus"], soft["n_queries"] = len(corpus), len(queries)

    try:
        import langid
        sample = list(corpus.values())[:: max(1, len(corpus) // 2000)]
        soft["langid_en_rate_corpus_sample"] = sum(langid.classify(d["text"][:1000])[0] == "en" for d in sample) / len(sample)
    except ImportError:
        pass

    if a.minhash:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=0.9, num_perm=64)
        mh = {}
        for i, d in corpus.items():
            m = MinHash(num_perm=64)
            toks = d["text"].lower().split()
            for sh in {" ".join(toks[j:j + 5]) for j in range(max(1, len(toks) - 4))}:
                m.update(sh.encode())
            mh[i] = m
            lsh.insert(i, m)
        near = sum(len(lsh.query(m)) > 1 for m in mh.values())
        soft["near_duplicate_answers_jaccard_ge_0.9"] = near

    report = {"build": str(a.build), "hard_violations": dict(hard), "ok": sum(hard.values()) == 0, "diagnostics": soft}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
