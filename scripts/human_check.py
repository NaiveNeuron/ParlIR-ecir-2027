# /// script
# requires-python = ">=3.11"
# ///
"""Human grading kit: blind sheets for checking the LLM audit and the referral labels, and their scoring.

export: sample <= N audit pairs for blind human grading, stratified over
          (a) judge disagreements, (b) non-official pairs both judges graded 2, (c) official answers, (d) random controls.
        Writes docs/human-check/sheet.csv (question, candidate, empty 'human_grade' column) and a private key file
        experiments/<v>/audit/human_key.json mapping sheet rows to (qid, doc, judge grades, source). The sheet hides
        which candidate is official and what the judges said.
score:  after a human fills human_grade (0/1/2), report human-vs-judge agreement (Cohen's kappa, confusion), human
        grade of official answers, and the human-confirmed rate of non-official 'directly addresses' answers.
Added before any grading; the original sheet is never re-sampled:
export-addendum:  from the 12-system audit, (e) non-official pairs both judges graded 1 (the only check on answers
        the judges may underrate) and (f) non-official both-2 pairs new since the original sheet's audit.
        -> docs/human-check/addendum.csv (+ key and inclusion probabilities).
export-referrals: every explicit-referral PrecedentRet test edge as (new question, earlier cited answer), WITHOUT the
        reply (its "as already indicated..." would bias the grade); graded A = already answered, P = partial,
        B = background / not an answer. -> docs/human-check/referrals.csv
export-double: a seeded 40 of the original 150 rows, blank, for a second author -> docs/human-check/sheet_second.csv
score-extra: per-stratum results of the addendum, referral shares with Wilson CIs, and human-human kappa.
export-xlsx: the two referral sheets as spreadsheets for Google Sheets / Excel (text cells, A/P/B dropdown,
        instructions tab) -> docs/human-check/sheets/referrals_grader{1,2}.xlsx. Needs `uv run --with openpyxl`.
import-sheet SRC --into DEST: copy grades from a downloaded CSV back into the canonical sheet, after checking that
        the row ids and texts are unchanged and that every grade is A, P, B or blank.
  uv run scripts/human_check.py export --exp experiments/v1.0 --n 150
  uv run scripts/human_check.py score  --exp experiments/v1.0
  uv run scripts/human_check.py export-addendum|export-referrals|export-double|score-extra --exp experiments/v1.0 \
      --build data/build/v1.0
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def load_j(exp, fam):
    g = {}
    for r in map(json.loads, open(exp / "audit" / f"judgments_{fam}.jsonl")):
        for j in r["judgments"]:
            g[(r["qid"], j["doc"])] = j["grade"]
    return g


def load_final(exp, ga, gb):
    """Final grade as used in the paper: agreement, else the adjudicator's grade, else the lower grade."""
    adj = {}
    p = exp / "audit" / "adjudication.jsonl"
    if p.exists():
        for r in map(json.loads, open(p)):
            a = (r.get("answer") or {}).get("grade") if isinstance(r.get("answer"), dict) else None
            if isinstance(a, dict) and "choice" in a:
                adj[(r["qid"], r["doc"])] = int(a["choice"])
    return {k: (a if gb.get(k) in (None, a) else adj.get(k, min(a, gb[k]))) for k, a in ga.items()}


def kappa(pairs):
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)  # any label set (0/1/2, A/P/B, bool)
    # pe == 1 only if both graders used one and the same label throughout: kappa is 0/0, i.e. not estimable
    # (never 1.0); report the raw agreement instead.
    return {"kappa": (po - pe) / (1 - pe) if pe < 1 else None, "estimable": pe < 1, "agreement": po, "n": n,
            "confusion_human_vs_llm": {f"{x}-{y}": c for (x, y), c in sorted(Counter(pairs).items())}}



def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [c - h, c + h]


def write_sheet(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def export_addendum(exp: Path, families, n_both1: int, seed: int):
    full, ext = exp / "audit_full", exp / "audit_full3"
    ga, gb = (load_j(ext, f) for f in families)
    old = load_j(full, families[0])
    final = load_final(ext, ga, gb)
    buckets = {"both1_nonofficial": [], "new_both2_nonofficial": []}
    for p in map(json.loads, open(ext / "audit" / "pools.jsonl")):
        for it in p["items"]:
            k = (p["qid"], it["doc"])
            if k not in ga or k not in gb or "control:official" in it["sources"] or it["sources"] == ["control:random"]:
                continue
            if ga[k] == gb[k] == 1:
                buckets["both1_nonofficial"].append((p, it))
            elif ga[k] == gb[k] == 2 and k not in old:
                buckets["new_both2_nonofficial"].append((p, it))
    rng = random.Random(seed)
    quota = {"both1_nonofficial": n_both1, "new_both2_nonofficial": len(buckets["new_both2_nonofficial"])}
    chosen = [(b, r) for b, q in quota.items() for r in rng.sample(buckets[b], min(q, len(buckets[b])))]
    rng.shuffle(chosen)
    key, rows = [], []
    for i, (b, (p, it)) in enumerate(chosen, 1):
        k = (p["qid"], it["doc"])
        rows.append([f"A{i}", p["question"], it["text"], "", ""])
        key.append({"row": f"A{i}", "qid": k[0], "doc": k[1], "bucket": b, "sources": it["sources"],
                    "grades": {families[0]: ga[k], families[1]: gb[k], "final": final[k]}})
    write_sheet(Path("docs/human-check/addendum.csv"), ["row", "question", "candidate_answer", "human_grade", "notes"], rows)
    (ext / "audit" / "human_key_addendum.json").write_text(json.dumps(key, indent=1))
    meta = {"seed": seed, "n": len(chosen), "quota": quota, "population": {k: len(v) for k, v in buckets.items()},
            "inclusion_probability": {k: (quota[k] / len(v) if v else None) for k, v in buckets.items()},
            "note": "added before any grading; the original 150-row sheet is unchanged"}
    (ext / "audit" / "human_key_addendum_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"addendum: {len(chosen)} rows; population {meta['population']}")


def export_referrals(build: Path, exp: Path, seed: int, n: int = 120):
    """A seeded uniform sample of n explicit-referral PrecedentRet test edges (all of them if fewer)."""
    Q = {r["_id"]: r for r in map(json.loads, open(build / "queries.jsonl"))}
    C = {r["_id"]: r for r in map(json.loads, open(build / "corpus.jsonl"))}
    edges = []
    for line in open(build / "qrels" / "precedenthc_test.tsv"):
        p = line.split()
        if len(p) == 4 and p[3].isdigit() and int(p[3]) > 0:
            edges.append((p[0], p[2]))
    edges.sort()
    picked = random.Random(seed).sample(edges, min(n, len(edges)))
    rows, key = [], []
    for i, (q, d) in enumerate(picked, 1):
        rows.append([f"R{i}", Q[q]["text"], C[d]["text"], "", ""])
        key.append({"row": f"R{i}", "qid": q, "doc": d})
    write_sheet(Path("docs/human-check/referrals.csv"),
                ["row", "new_question", "earlier_answer", "grade_A_P_B", "notes"], rows)
    (exp / "human_referrals_key.json").write_text(json.dumps(
        {"seed": seed, "population_edges": len(edges), "population_queries": len({q for q, _ in edges}),
         "n": len(picked), "inclusion_probability": len(picked) / len(edges), "rows": key}, indent=1))
    print(f"referrals: {len(rows)} of {len(edges)} edges ({len({q for q, _ in picked})} test queries)")


def export_double(exp: Path, seed: int, n: int = 40):
    rows = list(csv.DictReader(Path("docs/human-check/sheet.csv").open()))
    pick = sorted(random.Random(seed).sample(range(len(rows)), n))
    write_sheet(Path("docs/human-check/sheet_second.csv"), ["row", "question", "candidate_answer", "human_grade", "notes"],
                [[rows[i]["row"], rows[i]["question"], rows[i]["candidate_answer"], "", ""] for i in pick])
    (exp / "audit_full" / "audit" / "human_double_rows.json").write_text(
        json.dumps({"seed": seed, "rows": [int(rows[i]["row"]) for i in pick]}, indent=1))
    print(f"second-annotator sheet: {n} rows")


def score_extra(exp: Path):
    def grades(path, col, ok):
        return {r["row"]: r[col].strip().upper() for r in csv.DictReader(path.open()) if r[col].strip().upper() in ok} \
            if path.exists() else {}
    res = {"WARNING": "Enriched strata; report per stratum only, never pooled as audit-wide estimates."}
    add = grades(Path("docs/human-check/addendum.csv"), "human_grade", {"0", "1", "2"})
    kp = exp / "audit_full3" / "audit" / "human_key_addendum.json"
    if add and kp.exists():
        key = {r["row"]: r for r in json.loads(kp.read_text())}
        strata = {}
        for b in sorted({key[r]["bucket"] for r in add}):
            rows = [r for r in add if key[r]["bucket"] == b]
            g = [int(add[r]) for r in rows]
            strata[b] = {"n": len(rows), "human_grades": dict(sorted(Counter(g).items())),
                         "share_ge1": sum(x >= 1 for x in g) / len(g), "share_2": sum(x == 2 for x in g) / len(g),
                         "share_ge1_ci": wilson(sum(x >= 1 for x in g), len(g))}
        res["addendum"] = strata
    def ref_summary(ref):
        c, n = Counter(ref.values()), len(ref)
        return {"n": n, "counts": dict(c), "share_answered": c["A"] / n, "share_answered_ci": wilson(c["A"], n),
                "share_answered_or_partial": (c["A"] + c["P"]) / n,
                "share_answered_or_partial_ci": wilson(c["A"] + c["P"], n)}
    ref = grades(Path("docs/human-check/referrals.csv"), "grade_A_P_B", {"A", "P", "B"})
    ref2 = grades(Path("docs/human-check/referrals_second.csv"), "grade_A_P_B", {"A", "P", "B"})
    if ref:
        res["referrals"] = ref_summary(ref)
        # A partial sheet is a random subsample only if the graded rows are exactly R1..Rk of the seeded order
        # (no skips, no sorting); whether stopping was outcome-independent is the grader's statement, not checkable.
        k = len(ref)
        res["referrals"]["graded_is_prefix"] = set(ref) == {f"R{i}" for i in range(1, k + 1)}
        res["referrals"]["n_sheet"] = sum(1 for _ in csv.DictReader(Path("docs/human-check/referrals.csv").open()))
    if ref2:
        res["referrals_second"] = ref_summary(ref2)
    rb = sorted(set(ref) & set(ref2))
    if rb:  # two independent graders on the same referral rows
        res["referrals_human_human"] = {
            "n": len(rb), "kappa_APB": kappa([(ref[r], ref2[r]) for r in rb]),
            "kappa_A_vs_not": kappa([(ref[r] == "A", ref2[r] == "A") for r in rb]),
            "share_answered_both": sum(ref[r] == "A" and ref2[r] == "A" for r in rb) / len(rb)}
    first = grades(Path("docs/human-check/sheet.csv"), "human_grade", {"0", "1", "2"})
    second = grades(Path("docs/human-check/sheet_second.csv"), "human_grade", {"0", "1", "2"})
    both = sorted(set(first) & set(second))
    if both:
        res["human_human"] = kappa([(int(first[r]), int(second[r])) for r in both])
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "human_extra_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

REF_RUBRIC = [
    "Each row: a NEW question and an EARLIER answer that the new question's (hidden) reply cites.",
    "Write one letter in grade_A_P_B:",
    "A = the earlier answer already answers the new question's main point (it could have been sent instead of a new reply)",
    "P = partially: answers some of the questions asked, or the same issue at an earlier stage",
    "B = background: related context or a policy reference, but it does not answer what is asked",
    "Judge each row on its own; don't look anything up; notes are optional.",
    "Grade independently: don't see or discuss the other grader's sheet until both are done.",
    "When done: File > Download > Comma-separated values (.csv) on the 'grading' tab, and send/commit that file.",
]


def export_xlsx():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    out = Path("docs/human-check/sheets")
    out.mkdir(parents=True, exist_ok=True)
    for i, name in ((1, "referrals.csv"),):
        rows = list(csv.DictReader(Path("docs/human-check", name).open()))
        wb = Workbook()
        ws = wb.active
        ws.title = "grading"  # first tab: what Sheets' CSV download exports
        header = ["row", "new_question", "earlier_answer", "grade_A_P_B", "notes"]
        ws.append(header)
        for r in rows:
            ws.append([r[h] for h in header])
        for c in ws[1]:
            c.font = Font(bold=True)
        for col, width in zip("ABCDE", (7, 70, 80, 13, 30)):
            ws.column_dimensions[col].width = width
        for line in ws.iter_rows(min_row=2):
            for c in line:
                c.data_type = "s"  # plain text, never a formula or date
                c.alignment = Alignment(wrap_text=True, vertical="top")
        dv = DataValidation(type="list", formula1='"A,P,B"', allow_blank=True, showErrorMessage=True,
                            errorTitle="A, P or B", error="Grade must be A, P or B")
        ws.add_data_validation(dv)
        dv.add(f"D2:D{len(rows) + 1}")
        ws.freeze_panes = "B2"
        info = wb.create_sheet("instructions")
        for line in [f"Referral sample: {len(rows)} rows"] + REF_RUBRIC:
            info.append([line])
        info.column_dimensions["A"].width = 120
        p = out / f"referrals_grader{i}.xlsx"
        wb.save(p)
        print(f"wrote {p} ({len(rows)} rows)")


def import_sheet(src: Path, dest: Path):
    """Grades from a downloaded CSV into the canonical sheet; refuses if rows or texts differ."""
    norm = lambda s: " ".join((s or "").split())
    canon = list(csv.DictReader(dest.open()))
    got = {r["row"]: r for r in csv.DictReader(src.open(encoding="utf-8-sig"))}
    if set(got) != {r["row"] for r in canon}:
        raise SystemExit(f"row ids differ: missing {sorted({r['row'] for r in canon} - set(got))[:5]}, "
                         f"extra {sorted(set(got) - {r['row'] for r in canon})[:5]}")
    bad_text = [r["row"] for r in canon if any(norm(r[c]) != norm(got[r["row"]].get(c))
                                               for c in ("new_question", "earlier_answer"))]
    if bad_text:
        raise SystemExit(f"texts changed in rows {bad_text[:10]}: not the exported sheet")
    bad = {k: v["grade_A_P_B"] for k, v in got.items() if v["grade_A_P_B"].strip().upper() not in {"", "A", "P", "B"}}
    if bad:
        raise SystemExit(f"invalid grades (A/P/B or blank): {dict(list(bad.items())[:10])}")
    for r in canon:
        r["grade_A_P_B"] = got[r["row"]]["grade_A_P_B"].strip().upper()
        r["notes"] = got[r["row"]].get("notes", "")
    write_sheet(dest, list(canon[0].keys()), [[r[h] for h in canon[0]] for r in canon])
    filled = sum(1 for r in canon if r["grade_A_P_B"])
    print(f"{dest}: {filled}/{len(canon)} graded ({dict(Counter(r['grade_A_P_B'] for r in canon if r['grade_A_P_B']))})")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score", "export-addendum", "export-referrals", "export-double",
                                    "score-extra", "export-xlsx", "import-sheet"])
    ap.add_argument("src", nargs="?", type=Path, help="import-sheet: the downloaded CSV")
    ap.add_argument("--into", type=Path, help="import-sheet: canonical sheet to fill (docs/human-check/referrals.csv)")
    ap.add_argument("--exp", type=Path, default=Path("experiments/v1.0"))
    ap.add_argument("--build", type=Path, default=Path("data/build/v1.0"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-both1", type=int, default=30, help="export-addendum: rows from the both-judges-1 stratum")
    ap.add_argument("--families", nargs=2, default=["judge-claude", "judge-gpt"])
    ap.add_argument("--audit", default="audit_full", help="sub-dir of --exp holding audit/ (final, all runs judged)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "export-addendum":
        return export_addendum(a.exp, a.families, a.n_both1, a.seed)
    if a.cmd == "export-referrals":
        return export_referrals(a.build, a.exp, a.seed)
    if a.cmd == "export-double":
        return export_double(a.exp, a.seed)
    if a.cmd == "score-extra":
        return score_extra(a.exp)
    if a.cmd == "export-xlsx":
        return export_xlsx()
    if a.cmd == "import-sheet":
        if not a.src or not a.into:
            raise SystemExit("usage: human_check.py import-sheet DOWNLOADED.csv --into docs/human-check/referrals.csv")
        return import_sheet(a.src, a.into)
    sheet = Path("docs/human-check/sheet.csv")
    key_path = a.exp / a.audit / "audit" / "human_key.json"
    adir = a.exp / a.audit
    if a.cmd == "export":
        pools = [json.loads(l) for l in open(adir / "audit" / "pools.jsonl")]
        ga, gb = (load_j(adir, f) for f in a.families)
        final = load_final(adir, ga, gb)
        buckets = {"disagree": [], "both2_nonofficial": [], "official": [], "random": []}
        for p in pools:
            for it in p["items"]:
                k = (p["qid"], it["doc"])
                if k not in ga or k not in gb:
                    continue
                src = it["sources"]
                rec = (p, it)
                if "control:official" in src:
                    buckets["official"].append(rec)
                elif src == ["control:random"]:
                    buckets["random"].append(rec)
                elif ga[k] != gb[k]:
                    buckets["disagree"].append(rec)
                elif ga[k] == 2:
                    buckets["both2_nonofficial"].append(rec)
        rng = random.Random(a.seed)
        quota = {"disagree": int(a.n * 0.4), "both2_nonofficial": int(a.n * 0.3), "official": int(a.n * 0.2)}
        quota["random"] = a.n - sum(quota.values())
        chosen = []
        for b, q in quota.items():
            chosen += [(b, r) for r in rng.sample(buckets[b], min(q, len(buckets[b])))]
        rng.shuffle(chosen)
        sheet.parent.mkdir(parents=True, exist_ok=True)
        key = []
        with sheet.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "question", "candidate_answer", "human_grade", "notes"])
            for i, (b, (p, it)) in enumerate(chosen, 1):
                w.writerow([i, p["question"], it["text"], "", ""])
                k = (p["qid"], it["doc"])
                key.append({"row": i, "qid": p["qid"], "doc": it["doc"], "bucket": b, "sources": it["sources"],
                            "grades": {a.families[0]: ga[k], a.families[1]: gb[k], "final": final[k]}})
        key_path.write_text(json.dumps(key, indent=1))
        print(f"wrote {len(chosen)} rows to {sheet}; bucket sizes: {{{', '.join(f'{k}: {len(v)}' for k, v in buckets.items())}}}")
        meta = {"seed": a.seed, "n": len(chosen), "quota": quota,
                "population": {k: len(v) for k, v in buckets.items()},
                "excluded_strata": "non-official pairs on which both judges agree on grade 0 or 1 are never sampled",
                "inclusion_probability": {k: (quota[k] / len(v) if v else None) for k, v in buckets.items()}}
        (key_path.parent / "human_key_meta.json").write_text(json.dumps(meta, indent=1))
    else:
        key = {r["row"]: r for r in json.loads(key_path.read_text())}
        hum = {}
        for r in csv.DictReader(sheet.open()):
            if r["human_grade"].strip() in ("0", "1", "2"):
                hum[int(r["row"])] = int(r["human_grade"])
        meta_p = key_path.parent / "human_key_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        res = {"n_graded": len(hum),
               "WARNING": ("Deliberately enriched sample (quotas per stratum; agreed non-official grade-0/1 pairs are "
                           "never sampled). Pooled figures describe THIS sample only and are not audit-wide estimates. "
                           "Report the per-stratum results."),
               "sampling": meta}
        # pooled over the sample (descriptive only)
        res["pooled_sample_only"] = {who: kappa([(hum[i], key[i]["grades"][who]) for i in hum])
                                     for who in (*a.families, "final")}
        # per stratum: agreement / kappa vs each judge and the final grade, human grade distribution
        strata = {}
        for b in sorted({key[i]["bucket"] for i in hum}):
            rows = [i for i in hum if key[i]["bucket"] == b]
            s = {"n": len(rows), "human_grades": dict(sorted(Counter(hum[i] for i in rows).items()))}
            for who in (*a.families, "final"):
                s[who] = kappa([(hum[i], key[i]["grades"][who]) for i in rows])
            if b == "official":
                s["human_grade2_rate"] = sum(hum[i] == 2 for i in rows) / len(rows)
            if b in ("both2_nonofficial", "disagree"):
                fin2 = [i for i in rows if key[i]["grades"]["final"] == 2]
                s["final_grade2_n"] = len(fin2)
                s["final_grade2_human_confirmed_rate"] = (sum(hum[i] == 2 for i in fin2) / len(fin2)) if fin2 else None
            if b == "random":
                s["human_grade0_rate"] = sum(hum[i] == 0 for i in rows) / len(rows)
            strata[b] = s
        res["by_stratum"] = strata
        (adir / "audit" / "human_check_summary.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
