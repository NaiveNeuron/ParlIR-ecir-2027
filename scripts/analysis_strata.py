"""Stratified effectiveness ("retrievability") of official answers (paper Sec. 7, analyses).

For each run, per-query RR@10 / Success@1 grouped by: answering institution, addressee, question type
(E vs P = priority), year, original language (EN vs non-EN), joint answer, twin question, answer length
tercile, and generic-answer flag. In the spirit of Alvarez & Morrier (2025), low retrievability of an
institution's replies is a (weak) proxy for generic or non-responsive answering; we report it as a
descriptive analysis only.
  uv run python scripts/analysis_strata.py --build data/build/v1.0 --runs-dir experiments/v1.0/runs \
      --systems bm25_text bge-m3_text rrf_text --out experiments/v1.0/strata.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

GENERIC = ("the commission does not have", "the commission has no information", "the council has not discussed",
           "not in a position to", "has not discussed this", "does not comment on")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    corpus = {d["_id"]: d for d in map(json.loads, (a.build / "corpus.jsonl").open())}
    queries = {q["_id"]: q for q in map(json.loads, (a.build / "queries.jsonl").open())}
    eq = a.build / "equivalents.jsonl"
    twins = {r["qid"] for r in map(json.loads, eq.open())
             if any(e["type"] == "twin_question" for e in r["equivalent_answers"])} if eq.exists() else set()
    qids = [q for q, v in queries.items() if v.get("split") == a.split]
    lens = sorted(len(corpus[queries[q]["answer"]]["text"].split()) for q in qids)
    t1, t2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]

    def strata(q):
        v = queries[q]
        d = corpus[v["answer"]]
        n = len(d["text"].split())
        low = d["text"][:600].lower()
        return {
            "institution": d.get("institution") or "unknown",
            "addressee": v.get("addressee") or "unknown",
            "type": {"E": "ordinary (E)", "P": "priority (P)"}[v["type"]],
            "year": v["date"][:4],
            "orig_lang": "unknown" if not v.get("orig_lang") else ("EN" if v["orig_lang"] == ["en"] else "non-EN"),
            "joint": "joint" if v.get("joint_size", 1) > 1 else "single",
            "twin": "twin" if q in twins else "no twin",
            "answer_length": "short" if n <= t1 else ("medium" if n <= t2 else "long"),
            "generic_opening": "generic" if any(g in low for g in GENERIC) else "other",
        }
    st = {q: strata(q) for q in qids}
    res = {"split": a.split, "n": len(qids), "length_terciles": [t1, t2], "systems": {}}
    for s in a.systems:
        pq = json.loads((a.runs_dir / f"answer_{a.split}_{s}.metrics.json").read_text())["per_query"]
        out = {}
        for dim in next(iter(st.values())):
            g = defaultdict(list)
            for q in qids:
                g[st[q][dim]].append(q)
            out[dim] = {k: {"n": len(v), "MRR@10": mean(pq["RR@10"].get(q, 0.0) for q in v),
                            "S@1": mean(pq["Success@1"].get(q, 0.0) for q in v)} for k, v in sorted(g.items())}
        res["systems"][s] = out
    a.out.write_text(json.dumps(res, indent=1))
    for s, out in res["systems"].items():
        print("==", s)
        for dim, g in out.items():
            print(f"  {dim}: " + "; ".join(f"{k} n={v['n']} {v['MRR@10']:.3f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
