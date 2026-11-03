# /// script
# requires-python = ">=3.11"
# ///
"""Collection statistics for the paper (Table stats; Sec. 6) from a build.

  uv run scripts/stats.py --build data/build/v1.0 --out experiments/v1.0/stats.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    corpus = {d["_id"]: d for d in map(json.loads, (a.build / "corpus.jsonl").open())}
    queries = [q for q in map(json.loads, (a.build / "queries.jsonl").open())]
    by = defaultdict(list)
    for q in queries:
        if q["split"]:
            by[q["split"]].append(q)
            by["all"].append(q)
    out = {"period": {}, "len_q": {}, "len_a": {}, "joint": {}, "commission": {}, "n": {}, "per_year": {},
           "addressee": {}, "institution": Counter(d.get("institution") for d in corpus.values()),
           "type": {}, "orig_lang": {}}
    for s, qs in by.items():
        years = sorted(q["date"][:4] for q in qs)
        out["period"][s] = f"{years[0]}--{years[-1]}" if years else "--"
        out["len_q"][s] = int(median(len(q["text"].split()) for q in qs))
        out["len_a"][s] = int(median(len(corpus[q["answer"]]["text"].split()) for q in qs))
        out["joint"][s] = sum(q.get("joint_size", 1) > 1 for q in qs) / len(qs)
        out["commission"][s] = sum(q.get("addressee") == "Commission" for q in qs) / len(qs)
        out["n"][s] = len(qs)
        out["addressee"][s] = Counter(q.get("addressee") for q in qs)
        out["type"][s] = Counter(q["type"] for q in qs)
        out["orig_lang"][s] = Counter((q.get("orig_lang") or ["?"])[0] for q in qs)
    out["per_year"] = Counter(q["date"][:4] for q in by["all"])
    # political group of the (first) author, from the question header line "Name (GROUP)"; language shares
    import re
    grp = Counter()
    for q in by["all"]:
        g = None
        for h in q.get("header", []):
            m = re.search(r"\(([A-Za-z/&\-\s]{2,30})\)\s*$", h)
            if m and not re.search(r"(?i)rule|article|artikel|règle|regla", h):
                g = m.group(1).strip()
                break
        grp[g or "unknown"] += 1
    out["political_group"] = dict(grp.most_common())
    lang = Counter((q.get("orig_lang") or ["unknown"])[0] for q in by["all"])
    out["orig_lang_all"] = dict(lang.most_common())
    out["corpus_len_tokens_median"] = int(median(len(d["text"].split()) for d in corpus.values()))
    # truncation by the dense encoders' tokenizers (sample of 3,000 answers and 3,000 test questions, seed 0)
    try:
        import random
        from transformers import AutoTokenizer
        rng = random.Random(0)
        docs = rng.sample([d["text"] for d in corpus.values()], min(3000, len(corpus)))
        qtest = [q["text"] for q in by.get("test", [])]
        qs_ = rng.sample(qtest, min(3000, len(qtest)))
        out["truncation"] = {}
        for name, hf, lim in (("me5", "intfloat/multilingual-e5-base", 512), ("bge-m3", "BAAI/bge-m3", 1024)):
            tok = AutoTokenizer.from_pretrained(hf)
            ld = [len(x) for x in tok(["passage: " + t for t in docs], truncation=False)["input_ids"]]
            lq = [len(x) for x in tok(["query: " + t for t in qs_], truncation=False)["input_ids"]]
            out["truncation"][name] = {"limit": lim, "answers_over": sum(l > lim for l in ld) / len(ld),
                                       "questions_over": sum(l > lim for l in lq) / len(lq),
                                       "answer_tokens_median": sorted(ld)[len(ld) // 2]}
    except ImportError:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=dict))
    print(json.dumps({k: out[k] for k in ("period", "len_q", "len_a", "joint", "commission", "n")}, indent=1))


if __name__ == "__main__":
    main()
