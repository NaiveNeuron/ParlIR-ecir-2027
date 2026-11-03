"""Build the ParlIR collection from acquired raw data (paper Sec. 5).

Inputs (data/raw/): ids.jsonl, csv/<year>_<ver>_en.csv, detail/<id>.json (optional, partial),
docs/<work>/<lang>.<ext>, files_<lang>.jsonl.
Outputs (data/build/<version>/):
  corpus.jsonl        one record per answer document {_id, text, raw_text, date, institution, answers}
  queries.jsonl       one record per answered question {_id, text, raw_text, subject, date, type, term, split, ...}
  qrels/answer_<split>.tsv      AnswerRet: question -> official answer (grade 2)
  qrels/precedent_<split>.tsv   PrecedentRet: question -> answers of questions its official answer cites,
                                restricted to answers published before the question date
  precedent_meta.jsonl          per precedent query: cited refs, resolved, dropped (with reason)
  build_manifest.json           counts, split boundaries, sha256 of every output, source versions
Masking (canonical evaluation text): document headers are removed structurally (parse.py) and every
question reference / ID string in body text is replaced by the token "[REF]".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from .parse import QREF_TXT, parse_file

RAW = Path("data/raw")
SPLITS = {  # by question date; fixed before any experiment
    "train": (date(2014, 1, 1), date(2021, 12, 31)),
    "dev": (date(2022, 1, 1), date(2023, 12, 31)),
    "test": (date(2024, 1, 1), date(2026, 12, 31)),
}
# Any textual form of a question/answer identifier. Word boundaries (\b) are NOT used: identifiers occur inside
# URLs such as .../E-9-2021-000948-ASW_EN.html where '_' is a word character (found in a data audit). Hyphen-like
# characters (U+2010-2015, U+2212) are normalised before matching.
_NB = r"(?<![A-Za-z0-9])"   # not preceded by a letter/digit
_NA = r"(?![0-9])"          # not followed by a digit
ID_PATTERNS = [
    re.compile(_NB + r"[EPOGK]-\d{1,2}-\d{4}-\d{4,6}(?:-ASW)?" + _NA),       # E-9-2021-000948(-ASW)
    re.compile(_NB + r"[EPO]\s?-\s?\d{3,6}\s?/\s?\d{2,4}" + _NA),            # E-000948/2021, P-1234/19
    re.compile(_NB + r"[A-Z]{2}-\d{6}/\d{4}" + _NA),                        # EN-001737/2024 (typo variant)
    re.compile(_NB + r"[A-Z]{2}[EPO]-\d{3,6}/\d{2,4}" + _NA),                # ITE-000883/2015 (lang code glued)
    re.compile(_NB + r"[EPO]{2}-\d{3,6}/\d{2,4}" + _NA),                    # EP-7946/2014 (typo; not CJEU C-123/19)
    re.compile(_NB + r"A[EP]-\d{6}-\d{2}" + _NA),                           # AE-002500-19 (XML ids)
]
HYPHENS = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})
# Non-content boilerplate removed from bodies (canonical and raw): term-8 document footers
# ("1052839.EN PE 549.977") and co-signature lines naming supporting Members.
BOILERPLATE_LINES = [
    re.compile(r"^\s*\d{6,8}\.[A-Z]{2}\s+PE\s*\d{3}\.\d{3}(\s*v\d+-\d+)?\s*$"),
    re.compile(r"(?i)^\s*(\[\d+\]\s*)?(supporters?\s*(\[\d+\])?\s*$|supported by|co-signator)"),
    re.compile(r"(?i)^\s*(\[\d+\]\s*)?this question is supported by (a member|members) other than the author"),
]
DATE_RE = re.compile(r"\((\d{1,2})\.(\d{1,2})\.(\d{4})\)")
SUBMITTED_RE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")


def mask_ids(text: str) -> str:
    text = text.translate(HYPHENS)
    for p in ID_PATTERNS:
        text = p.sub("[REF]", text)
    return text


def clean_body(lines: list[str]) -> list[str]:
    return [l for l in lines if not any(p.search(l) for p in BOILERPLATE_LINES)]


def split_of(d: date) -> str | None:
    for name, (a, b) in SPLITS.items():
        if a <= d <= b:
            return name
    return None


def load_csv_meta() -> dict[str, dict]:
    meta = {}
    for f in sorted((RAW / "csv").glob("*_en.csv")):
        df = pd.read_csv(f, dtype=str)
        for r in df.to_dict("records"):
            meta[r["document_identifier"]] = r
    return meta


def load_csv_langs() -> dict[str, set[str]]:
    """Language versions per question from the per-language ODP CSVs: a question has a version in
    language L iff its row in the L csv has a document link. (EN versions always exist.)"""
    langs: dict[str, set[str]] = defaultdict(set)
    for f in sorted((RAW / "csv_langs").glob("*.csv")):
        lang = f.stem.rsplit("_", 1)[1]
        df = pd.read_csv(f, dtype=str, usecols=["document_identifier", "document_pdf", "document_doc"])
        has = df[df["document_pdf"].notna() | df["document_doc"].notna()]["document_identifier"]
        for i in has:
            langs[i].add(lang)
    return langs


ISO3_TO_2 = {"BUL": "bg", "CES": "cs", "DAN": "da", "DEU": "de", "ELL": "el", "ENG": "en", "SPA": "es", "EST": "et",
             "FIN": "fi", "FRA": "fr", "GLE": "ga", "HRV": "hr", "HUN": "hu", "ITA": "it", "LIT": "lt", "LAV": "lv",
             "MLT": "mt", "NLD": "nl", "POL": "pl", "POR": "pt", "RON": "ro", "SLK": "sk", "SLV": "sl", "SWE": "sv"}


def load_detail(qid: str) -> dict | None:
    p = RAW / "detail" / f"{qid}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d["data"][0] if "data" in d else None


def asw_id_from_url(url: str) -> str | None:
    m = re.search(r"/([EPOGK]-\d+-\d{4}-\d{6})-ASW", url or "")
    return m.group(1) if m else None


def find_doc(work: str, lang: str) -> Path | None:
    d = RAW / "docs" / work
    for ext in ("xhtml", "docx", "xml"):
        p = d / f"{lang}.{ext}"
        if p.exists():
            return p
    return None


def parse_date(s: str | None) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def xml_question_date(qid: str) -> date | None:
    """Question date from the AQE.QER XML header (DATE.WQ/@DATE.DOC, YYYYMMDD), used for years whose
    XHTML has no date and that have neither a yearly CSV nor an API detail record (2015)."""
    p = RAW / "docs" / qid / "en.xml"
    if not p.exists():
        return None
    try:
        from lxml import etree
        el = etree.fromstring(p.read_bytes()).find(".//HEAD.Q//DATE.WQ")
        v = el.get("DATE.DOC") if el is not None else None
        return date(int(v[:4]), int(v[4:6]), int(v[6:8])) if v and len(v) >= 8 else None
    except Exception:
        return None


def header_date(lines: list[str]) -> date | None:
    for l in reversed(lines):
        m = DATE_RE.search(l) or SUBMITTED_RE.search(l)
        if m:
            try:
                return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                continue
    return None


def addressee(header: list[str]) -> str | None:
    h = " ".join(header).lower()
    if "vice-president" in h and "high representative" in h:
        return "VPHR"
    if "to the council" in h:
        return "Council"
    if "to the commission" in h:
        return "Commission"
    return None


def institution(header: list[str]) -> str | None:
    """Answering institution from the answer header. VP/HR replies are signed 'Answer given by High
    Representative/Vice-President X on behalf of the European Commission', so test for them first."""
    h = " ".join(header).lower()
    if "high representative" in h or re.search(r"\bhr/vp\b|\bvp/hr\b", h):
        return "VPHR"
    if "on behalf of the european commission" in h or "on behalf of the commission" in h:
        return "Commission"
    if re.search(r"\breply\b", h):
        return "Council"
    return None


def resolve_ref_text(ref: str, known: set[str]) -> list[str]:
    """'E-001234/2019' -> candidate full ids existing in `known` (term inferred; may be ambiguous)."""
    m = re.match(r"([EPO])\s?-\s?(\d{3,6})\s?/\s?(\d{2,4})", ref)
    if not m:
        return []
    t, num, yr = m.group(1), int(m.group(2)), m.group(3)
    yr = int(yr) + (2000 if len(yr) == 2 else 0)
    return [f"{t}-{term}-{yr}-{num:06d}" for term in range(6, 11) if f"{t}-{term}-{yr}-{num:06d}" in known]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(version: str, lang: str = "en") -> None:
    out = Path("data/build") / version
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    ids = [json.loads(l)["identifier"] for l in (RAW / "ids.jsonl").open()]
    ids = [i for i in ids if i[0] in "EP"]
    known = set(ids)
    csv = load_csv_meta()
    csv_langs = load_csv_langs()
    drops = Counter()

    # --- answers: parse every ASW document present
    answers: dict[str, dict] = {}
    for qid in ids:
        p = find_doc(f"{qid}-ASW", lang)
        if p is None:
            continue
        try:
            a = parse_file(p, "answer")
        except Exception as e:  # recorded, never silently dropped
            drops[f"answer_parse:{type(e).__name__}"] += 1
            continue
        body = clean_body([b for b in a["body"] if b])
        if not body:
            drops["answer_empty"] += 1
            continue
        raw_text = "\n".join(body)
        answers[f"{qid}-ASW"] = {
            "_id": f"{qid}-ASW", "primary": qid, "raw_text": raw_text, "text": mask_ids(raw_text),
            "source_url": f"https://www.europarl.europa.eu/doceo/document/{qid}-ASW_EN.html",
            "header": a["header"], "date": header_date(a["header"]), "institution": institution(a["header"]),
            "header_refs": a.get("header_refs", []), "body_refs": a.get("body_refs", []),
            "body_ref_text": QREF_TXT.findall(raw_text),
            # full-form ids, e.g. footnote URLs .../doceo/document/E-9-2021-000948-ASW_EN.html (unambiguous)
            "body_ref_ids": sorted(set(re.findall(r"\b([EP]-\d{1,2}-\d{4}-\d{6})(?:-ASW)?", raw_text))), "fmt": a["fmt"],
        }

    # --- question -> answer document (official link)
    ans_of: dict[str, str] = {}
    link_src = Counter()
    for qid in ids:
        det = load_detail(qid)
        cand = None
        if det and det.get("inverse_answers_to"):
            cand = det["inverse_answers_to"][0]["id"].rsplit("/", 1)[-1]
            link_src["detail"] += 1
        elif qid in csv and isinstance(csv[qid].get("document_answers_doc"), str):
            a = asw_id_from_url(csv[qid]["document_answers_doc"]) or asw_id_from_url(csv[qid].get("document_answers_pdf"))
            cand = f"{a}-ASW" if a else None
            link_src["csv"] += 1 if cand else 0
        elif f"{qid}-ASW" in answers:
            cand = f"{qid}-ASW"
            link_src["own_file"] += 1
        if cand and cand in answers:
            ans_of[qid] = cand
    # joint answers announced in answer headers (hyperlinked refs), for questions still unlinked
    for aid, a in answers.items():
        for r in a["header_refs"]:
            if r in known and r not in ans_of:
                ans_of[r] = aid
                link_src["answer_header"] += 1

    # --- questions
    queries = {}
    for qid in ids:
        if qid not in ans_of:
            drops["question_unanswered_or_answer_missing"] += 1
            continue
        p = find_doc(qid, lang)
        if p is None:
            drops["question_file_missing"] += 1
            continue
        try:
            q = parse_file(p, "question", subject_hint=(csv.get(qid) or {}).get("document_title_en"))
        except Exception as e:
            drops[f"question_parse:{type(e).__name__}"] += 1
            continue
        body = clean_body([b for b in q["body"] if b])
        subj = q.get("subject") or (csv.get(qid) or {}).get("document_title_en") or ""
        if not body:
            drops["question_empty"] += 1
            continue
        d = (parse_date((csv.get(qid) or {}).get("document_date")) or header_date(q["header"])
             or xml_question_date(qid))
        det = load_detail(qid)
        if d is None and det:
            d = parse_date(det.get("document_date"))
        if d is None:
            drops["question_no_date"] += 1
            continue
        raw = subj + "\n" + "\n".join(body)
        queries[qid] = {
            "_id": qid, "subject": subj, "raw_text": raw, "text": mask_ids(raw), "date": d,
            "source_url": f"https://www.europarl.europa.eu/doceo/document/{qid}_EN.html",
            "type": qid[0], "term": int(qid.split("-")[1]), "addressee": addressee(q["header"]),
            "author": (csv.get(qid) or {}).get("document_creator_person"),
            # original language: ODP detail record if fetched, else the non-English version(s) in the
            # per-language CSVs (EN if the question exists only in English); None if neither source covers it
            "orig_lang": ([ISO3_TO_2.get(x[-3:], x[-3:].lower()) for x in det.get("originalLanguage", [])] if det
                          else (sorted(csv_langs[qid]) or ["en"]) if qid in csv else None),
            "orig_lang_src": "detail" if det else ("csv" if qid in csv else None),
            "answer": ans_of[qid], "header": q["header"],
        }

    # --- cross-lingual query texts: the original-language version of each question (L != en), as
    # officially published; parsed and masked exactly like the English text.
    xling = []
    for qid, q in queries.items():
        for l in q.get("orig_lang") or []:
            if l == "en":
                continue
            p = find_doc(qid, l)
            if p is None:
                continue
            try:
                dq = parse_file(p, "question")
            except Exception:
                drops[f"xling_question_parse_{l}"] += 1
                continue
            body = clean_body([b for b in dq["body"] if b])
            if not body:
                continue
            raw = (dq.get("subject") or "") + "\n" + "\n".join(body)
            xling.append({"_id": qid, "lang": l, "text": mask_ids(raw), "raw_text": raw})

    # --- splits by question date. A joint-answer group whose questions fall into different splits is
    # excluded from all query sets (its answer stays in the corpus), so no split boundary is crossed.
    group = defaultdict(list)
    for qid, q in queries.items():
        group[q["answer"]].append(qid)
    cross_split = 0
    for aid, qs in group.items():
        splits = {split_of(queries[q]["date"]) for q in qs}
        s = splits.pop() if len(splits) == 1 else None
        if s is None and len(qs) > 1:
            cross_split += 1
        for q in qs:
            queries[q]["split"] = s
            queries[q]["joint_size"] = len(qs)
        answers[aid]["answers"] = sorted(qs)
        if answers[aid]["date"] is None:
            answers[aid]["date"] = max(queries[q]["date"] for q in qs)
    # The corpus holds EVERY parsed answer document, including answers whose question was dropped (e.g. unparsable
    # or undatable): they remain valid retrieval targets and may be cited by later replies.
    for aid in answers:
        answers[aid].setdefault("answers", [])
    corpus_ids = sorted(answers)

    # --- PrecedentRet ("earlier-answer citation retrieval") qrels: official answers of the questions that
    # the query's official answer cites, published strictly before the query date. Each edge records its
    # provenance (hyperlink / text regex) and whether the citing sentence is an explicit referral
    # ("refer ... to its answer", "previous answer", ...), which defines the high-confidence subset.
    REFERRAL = re.compile(r"(?i)\b(refer|referred|answer(?:s)?\s+(?:given\s+)?to|reply\s+to|previous|earlier|already|"
                          r"same\s+(?:subject|issue|question)|similar\s+question)")
    prec_rows, prec_hc, prec_meta = defaultdict(list), defaultdict(list), []
    cite_rows, cite_hc = defaultdict(list), defaultdict(list)
    attrition = Counter()
    for qid, q in queries.items():
        a = answers[q["answer"]]
        via = defaultdict(set)
        for r in a["body_refs"]:
            if r in known:
                via[r].add("hyperlink")
        for r in a["body_ref_ids"]:
            if r in known:
                via[r].add("url")
        for t in a["body_ref_text"]:
            res = resolve_ref_text(t, known)
            if len(res) == 1:
                via[res[0]].add("text")
            elif len(res) > 1:
                attrition["ref_text_ambiguous_term"] += 1
        cited = [c for c in via if c not in a.get("answers", [])]
        sentences = re.split(r"(?<=[.;:])\s+", a["raw_text"])
        edges, rel, rel_hc, rel_draft, rel_draft_hc = [], [], [], [], []
        reply_date = a["date"]
        for c in cited:
            printed = f"{c[0]}-{c.split('-')[3]}/{c.split('-')[2]}"
            ctx = [s for s in sentences if printed in s.replace(" ", "") or printed in s]
            referral = any(REFERRAL.search(s) for s in ctx)
            ca = ans_of.get(c)
            if ca is None:
                status = "cited_question_has_no_answer_doc"
            elif ca == q["answer"]:
                status = "same_answer"
            elif answers[ca]["date"] is None:
                status = "cited_answer_undated"
            elif answers[ca]["date"] < q["date"]:
                status = "ok"  # published before the question was asked
            elif reply_date is not None and answers[ca]["date"] < reply_date:
                status = "ok_drafting_only"  # published between question and reply
            else:
                status = "cited_answer_not_before_reply"
            if status == "ok":
                rel.append(ca)
                if referral:
                    rel_hc.append(ca)
            if status in ("ok", "ok_drafting_only"):
                rel_draft.append(ca)
                if referral:
                    rel_draft_hc.append(ca)
            attrition[status] += 1
            edges.append({"cited": c, "via": sorted(via[c]), "referral": referral, "answer": ca, "status": status})
        rel, rel_hc = list(dict.fromkeys(rel)), list(dict.fromkeys(rel_hc))
        rel_draft, rel_draft_hc = list(dict.fromkeys(rel_draft)), list(dict.fromkeys(rel_draft_hc))
        if cited:
            prec_meta.append({"qid": qid, "split": q["split"], "reply_date": reply_date, "edges": edges,
                              "relevant_presubmission": rel, "relevant_drafting": rel_draft})
        if q["split"]:
            for rows, r in ((prec_rows, rel), (prec_hc, rel_hc), (cite_rows, rel_draft), (cite_hc, rel_draft_hc)):
                rows[q["split"]] += [(qid, x) for x in r]

    # --- twin questions: near-identical question bodies (5-word shingles, Jaccard >= 0.8) with different answers
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    mh = {}
    for qid, q in queries.items():
        toks = q["text"].lower().split()
        m = MinHash(num_perm=64, seed=1)
        for j in range(max(1, len(toks) - 4)):
            m.update(" ".join(toks[j:j + 5]).encode())
        mh[qid] = m
        lsh.insert(qid, m)
    # Equivalent answers of a query, by type (twin questions and stock replies are different
    # phenomena and are labelled separately):
    #   twin_question       answer of a near-identical question (e.g. same text sent to Commission and Council)
    #   duplicate_answer    byte-identical answer body (stock reply issued as a separate document)
    #   near_duplicate_answer answer body with 5-shingle Jaccard >= 0.9 to the official one
    equiv = defaultdict(dict)
    for qid, m in mh.items():
        for o in lsh.query(m):
            if o != qid and queries[o]["answer"] != queries[qid]["answer"]:
                equiv[qid].setdefault(queries[o]["answer"], "twin_question")
    by_body = defaultdict(list)
    for aid in corpus_ids:
        by_body[hashlib.sha1(" ".join(answers[aid]["text"].lower().split()).encode()).hexdigest()].append(aid)
    dup_of = {aid: set(v) - {aid} for v in by_body.values() if len(v) > 1 for aid in v}
    alsh, amh = MinHashLSH(threshold=0.9, num_perm=64), {}
    for aid in corpus_ids:
        toks = answers[aid]["text"].lower().split()
        m = MinHash(num_perm=64, seed=2)
        for j in range(max(1, len(toks) - 4)):
            m.update(" ".join(toks[j:j + 5]).encode())
        amh[aid] = m
        alsh.insert(aid, m)
    equiv_counts = Counter()
    for qid, q in queries.items():
        a0 = q["answer"]
        for d in dup_of.get(a0, ()):
            equiv[qid].setdefault(d, "duplicate_answer")
        for d in alsh.query(amh[a0]):
            if d != a0:
                equiv[qid].setdefault(d, "near_duplicate_answer")
        for t in set(equiv[qid].values()):
            equiv_counts[t] += 1
    twin_answers = {q: set(v) for q, v in equiv.items() if v}

    # --- write
    def dump(path: Path, rows):
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    dump(out / "corpus.jsonl", ({"_id": i, "text": answers[i]["text"], "raw_text": answers[i]["raw_text"],
                                 "date": answers[i]["date"], "institution": answers[i]["institution"],
                                 "answers": answers[i]["answers"], "header": answers[i]["header"],
                                 "source_url": answers[i]["source_url"]} for i in corpus_ids))
    dump(out / "queries.jsonl", ({k: v for k, v in q.items()} for q in sorted(queries.values(), key=lambda x: x["_id"])))
    dump(out / "precedent_meta.jsonl", prec_meta)
    dump(out / "equivalents.jsonl", ({"qid": q, "equivalent_answers": [{"doc": d, "type": t} for d, t in sorted(v.items())]}
                                     for q, v in sorted(equiv.items()) if v))
    dump(out / "queries_xling.jsonl", (x for x in sorted(xling, key=lambda x: (x["lang"], x["_id"]))
                                       if queries[x["_id"]].get("split")))
    counts = {}
    for s in SPLITS:
        qs = sorted(q for q in queries if queries[q]["split"] == s)
        with (out / "qrels" / f"answer_{s}.tsv").open("w") as fh:
            for q in qs:
                fh.write(f"{q} 0 {queries[q]['answer']} 2\n")
        # lenient variant: the answer of a near-identical twin question (e.g. the same text sent to both
        # the Commission and the Council) also counts, with grade 1.
        with (out / "qrels" / f"answerlenient_{s}.tsv").open("w") as fh:
            for q in qs:
                fh.write(f"{q} 0 {queries[q]['answer']} 2\n")
                for t in sorted(twin_answers.get(q, ())):
                    fh.write(f"{q} 0 {t} 1\n")
        counts[s] = {"answer_queries": len(qs)}
        for name, rows in (("precedent", prec_rows), ("precedenthc", prec_hc), ("cite", cite_rows), ("citehc", cite_hc)):
            with (out / "qrels" / f"{name}_{s}.tsv").open("w") as fh:
                for q, r in sorted(rows[s]):
                    fh.write(f"{q} 0 {r} 1\n")
            counts[s][f"{name}_queries"] = len({q for q, _ in rows[s]})
            counts[s][f"{name}_pairs"] = len(rows[s])
    manifest = {
        "version": version, "lang": lang, "splits": {k: [str(a), str(b)] for k, (a, b) in SPLITS.items()},
        "n_question_ids": len(ids), "n_answers_parsed": len(answers), "n_corpus": len(corpus_ids),
        "n_queries": len(queries), "link_sources": dict(link_src), "drops": dict(drops), "counts": counts,
        "unsplit_queries": sum(1 for q in queries.values() if not q["split"]),
        "joint_groups_cross_split_excluded": cross_split, "precedent_edge_attrition": dict(attrition),
        "queries_with_equivalent_answer": len(twin_answers), "queries_with_equivalent_by_type": dict(equiv_counts),
        "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "build_manifest.json"},
    }
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="dev")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    build(a.version, a.lang)


if __name__ == "__main__":
    main()
