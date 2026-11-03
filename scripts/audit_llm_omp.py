"""Pre-registered label-incompleteness audit (paper Sec. 6). RUNS INSIDE OMP EVAL.

All LLM inference goes through OMP: bulk judges are OMP project agents pinned to model families
(v1.0 final: .omp/agents/judge-claude.md, judge-gpt.md; v0.9 pilot and the pre-registered first v1.0 attempt:
judge-gemini.md, judge-deepseek.md) spawned with OMP's `agent()`; disagreements are adjudicated with OMP's
native `judge_batch` (session judge role). No provider SDK or HTTP is used.

Usage from an OMP python eval cell:
    exec(open("scripts/audit_llm_omp.py").read())
    S = audit_sample(BUILD, EXP, RUNS, n_per_stratum=75)                # audit/sample.json
    P = audit_pools(S, BUILD, EXP, RUNS, pooled=[...], held_out=[...])   # audit/pools.jsonl, pools_meta.json
    audit_judge(P, EXP, families=["judge-claude", "judge-gpt"])         # audit/judgments_<family>.jsonl
    await audit_adjudicate(P, EXP, families)                             # audit/adjudication.jsonl
    audit_summarize(P, EXP, families, systems, RUNS)                     # audit/summary.json (+ table rows)
  Extension (every audited run judged to depth 10 on the same queries):
    PX = audit_extend_pools(P, BUILD, EXP_EXT, RUNS, extra_runs=[...]); audit_judge(PX, EXP_EXT, families)
    await audit_adjudicate(PX, EXP_EXT, families); PF = audit_merge(EXP, EXP_EXT, EXP_FULL, P, PX, families)
    audit_summarize(PF, EXP_FULL, families, systems, RUNS, held_out=())
Run ordering uses parlir.evaluate's canonical rule (score desc, ties by docid desc). Run the cell from the repository
root, so that `parlir` is importable; everything else is stdlib.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()  # run from the repository root (the script is exec'd in an OMP eval cell, so no __file__)
MAX_CHARS = 2500  # per text shown to the judge; truncation rate is logged

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"judgments": {"type": "array", "items": {"type": "object", "properties": {
        "doc": {"type": "string"}, "grade": {"type": "integer", "enum": [0, 1, 2]},
        "rationale": {"type": "string"}}, "required": ["doc", "grade", "rationale"]}}},
    "required": ["judgments"],
}
JUDGE_TASK = (
    "Below is ONE question for written answer from a Member of the European Parliament, followed by "
    "candidate answer documents labelled D1..Dn (texts may be truncated with '...'). Judge EACH candidate "
    "independently against the question using the grading scale in your instructions "
    "(2 = directly addresses this question's main information need; 1 = same specific issue, only partially "
    "addresses it; 0 = not relevant). Return one judgment per candidate label (use the label, e.g. 'D3', as doc)."
)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def _read_run(p, depth=1000):
    """Top-`depth` per query in the CANONICAL order of parlir.evaluate (score desc, ties by docid desc, as
    trec_eval), not the file's rank field, so audit orderings agree with the reported metrics."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from parlir.evaluate import _ranked, read_run
    return defaultdict(list, {q: _ranked(scores)[:depth] for q, scores in read_run(p).items()})


def _out(exp):
    d = Path(exp) / "audit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def audit_sample(build, exp, runs_dir, n_per_stratum=50, seed=0):
    """Strata: BM25 hit@1 (yes/no) x answer cites an earlier question (yes/no), test split."""
    build, runs_dir = Path(build), Path(runs_dir)
    qrels = defaultdict(dict)
    for l in open(build / "qrels" / "answer_test.tsv"):
        q, _, d, g = l.split()
        qrels[q][d] = int(g)
    cites = {m["qid"] for m in _read_jsonl(build / "precedent_meta.jsonl") if m.get("edges")}
    bm25 = _read_run(runs_dir / "answer_test_bm25_text.run", 1)
    strata = defaultdict(list)
    for q in sorted(qrels):
        hit = bool(bm25.get(q)) and bm25[q][0] in qrels[q]
        strata[(hit, q in cites)].append(q)
    rng = random.Random(seed)
    sample, sizes = {}, {}
    for k in sorted(strata):
        pop = strata[k]
        take = rng.sample(pop, min(n_per_stratum, len(pop)))
        name = f"hit{int(k[0])}_cite{int(k[1])}"
        sample[name] = sorted(take)
        sizes[name] = len(pop)
    rec = {"seed": seed, "n_per_stratum": n_per_stratum, "strata_population": sizes, "sample": sample, "t": _now()}
    (_out(exp) / "sample.json").write_text(json.dumps(rec, indent=1))
    return rec


def audit_pools(sample, build, exp, runs_dir, pooled, held_out, depth=10, n_random=2, seed=0):
    build, runs_dir = Path(build), Path(runs_dir)
    corpus = {d["_id"]: d for d in map(json.loads, open(build / "corpus.jsonl"))}
    queries = {q["_id"]: q for q in map(json.loads, open(build / "queries.jsonl"))}
    runs = {r: _read_run(runs_dir / f"answer_test_{r}.run", depth) for r in pooled + held_out}
    rng = random.Random(seed)
    all_docs = sorted(corpus)
    pools, trunc = [], Counter()
    for stratum, qids in sample["sample"].items():
        for q in qids:
            official = queries[q]["answer"]
            src = defaultdict(set)
            for r in pooled:
                for d in runs[r].get(q, [])[:depth]:
                    src[d].add(r)
            src[official].add("control:official")
            while sum(1 for s in src.values() if "control:random" in s) < n_random:
                d = rng.choice(all_docs)
                if d not in src and d != official:
                    src[d].add("control:random")
            docs = sorted(src)
            rng.shuffle(docs)
            items = []
            for i, d in enumerate(docs, 1):
                t = corpus[d]["text"]
                trunc["answer_truncated"] += len(t) > MAX_CHARS
                trunc["answers"] += 1
                items.append({"label": f"D{i}", "doc": d, "sources": sorted(src[d]), "text": t[:MAX_CHARS] + (" ..." if len(t) > MAX_CHARS else "")})
            qt = queries[q]["text"]
            trunc["question_truncated"] += len(qt) > MAX_CHARS
            pools.append({"qid": q, "stratum": stratum, "question": qt[:MAX_CHARS] + (" ..." if len(qt) > MAX_CHARS else ""),
                          "official": official, "items": items,
                          "held_out_topk": {r: runs[r].get(q, [])[:depth] for r in held_out}})
    out = _out(exp)
    with open(out / "pools.jsonl", "w") as fh:
        for p in pools:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"pooled_runs": pooled, "held_out_runs": held_out, "depth": depth, "n_random_controls": n_random,
            "max_chars": MAX_CHARS, "truncation": dict(trunc), "n_queries": len(pools),
            "n_pairs": sum(len(p["items"]) for p in pools), "t": _now()}
    (out / "pools_meta.json").write_text(json.dumps(meta, indent=1))
    return pools


def _prompt(pool):
    cands = "\n\n".join(f"[{it['label']}]\n{it['text']}" for it in pool["items"])
    return f"{JUDGE_TASK}\n\nQUESTION:\n{pool['question']}\n\nCANDIDATES:\n{cands}"


def audit_judge(pools, exp, families, batch=16):
    """One agent call per (query, family), spawned in batches of `batch` (both families interleaved) and
    collected with OMP's `wait()` barrier. Results appended to judgments_<family>.jsonl (resumable:
    re-running skips queries already judged). Failures are recorded in judge_errors.jsonl and retried on re-run."""
    out = _out(exp)
    done = {}
    for fam in families:
        path = out / f"judgments_{fam}.jsonl"
        done[fam] = {json.loads(l)["qid"] for l in open(path)} if path.exists() else set()
    todo = [(fam, p) for p in pools for fam in families if p["qid"] not in done[fam]]
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        hs, prompts = [], []
        for fam, p in chunk:
            prompt = _prompt(p)
            prompts.append(prompt)
            hs.append(agent(prompt, agent=fam, schema=JUDGE_SCHEMA, label=f"audit-{fam}-{p['qid']}"))
        results = wait(hs, raise_errors=False)
        for (fam, p), h, prompt, res in zip(chunk, hs, prompts, results):
            labels = {it["label"]: it["doc"] for it in p["items"]}
            ok = isinstance(res, dict) and isinstance(res.get("judgments"), list)
            rec = {"qid": p["qid"], "family": fam, "agent_id": getattr(h, "id", None), "t": _now(),
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            if not ok:
                with open(out / "judge_errors.jsonl", "a") as fh:
                    fh.write(json.dumps({**rec, "error": repr(res)[:500]}) + "\n")
                continue
            rec["judgments"] = [{"doc": labels.get(j["doc"], j["doc"]), "label": j["doc"], "grade": j["grade"],
                                 "rationale": j.get("rationale", "")} for j in res["judgments"]]
            rec["missing_labels"] = sorted(set(labels) - {j["doc"] for j in res["judgments"]})
            with open(out / f"judgments_{fam}.jsonl", "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{_now()} judged {min(s + batch, len(todo))}/{len(todo)}", flush=True)


def _load_judgments(exp, fam):
    path = _out(exp) / f"judgments_{fam}.jsonl"
    g = {}
    for r in map(json.loads, open(path)):
        for j in r["judgments"]:
            g[(r["qid"], j["doc"])] = j["grade"]
    return g


async def audit_adjudicate(pools, exp, families):
    """Pairs where the two families disagree go to OMP's native judge (session judge model)."""
    ga, gb = (_load_judgments(exp, f) for f in families)
    pool_by = {p["qid"]: p for p in pools}
    states, keys = [], []
    for (q, d), a in ga.items():
        b = gb.get((q, d))
        if b is None or a == b:
            continue
        item = next(it for it in pool_by[q]["items"] if it["doc"] == d)
        states.append({"question": pool_by[q]["question"], "candidate_answer": item["text"]})
        keys.append((q, d))
    questions = {"grade": {"type": "choice", "instructions":
                           "Grade the candidate answer for this European Parliament written question.",
                           "criteria": {"2": "directly addresses the question's main information need (could serve as its reply)",
                                        "1": "same specific issue, only partially addresses it, evidently written for another question",
                                        "0": "not relevant: different issue or only the same broad policy area"}}}
    b = judge_batch(states, questions, intent="ParlIR audit adjudication")
    res = []
    async for k, item in b.drain_iter(3600):
        res.append((k, item))
    status = b.status()
    out = _out(exp)
    with open(out / "adjudication.jsonl", "w") as fh:
        for k, item in sorted(res, key=lambda x: x[0]):
            q, d = keys[k]
            ans = getattr(item, "answers", None)
            fh.write(json.dumps({"qid": q, "doc": d, "answer": ans, "error": getattr(item, "error", None)}, default=str) + "\n")
    (out / "adjudication_meta.json").write_text(json.dumps({"n": len(keys), "status": status, "t": _now()}, indent=1, default=str))
    b.close()
    return status


def _kappa(pairs, labels=(0, 1, 2)):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None  # one shared label throughout: 0/0, not estimable


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def _kendall_tau_b(x, y):
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else None


def audit_summarize(pools, exp, families, systems, runs_dir, held_out=(), n_boot=1000, seed=0, build=None):
    """systems: run names (answer_test_<name>.run) to rank under official vs augmented labels;
    held_out: the subset of `systems` that was excluded from pooling; build: build dir (for the citation check)."""
    out = _out(exp)
    ga, gb = (_load_judgments(exp, f) for f in families)
    adj = {}
    p = out / "adjudication.jsonl"
    if p.exists():
        for r in map(json.loads, open(p)):
            if r.get("answer"):
                a = r["answer"]["grade"] if isinstance(r["answer"], dict) else None
                if isinstance(a, dict) and "choice" in a:
                    adj[(r["qid"], r["doc"])] = int(a["choice"])
    final = {}
    for k, a in ga.items():
        b = gb.get(k)
        final[k] = a if b is None or a == b else adj.get(k, min(a, b))  # unresolved disagreement -> lower grade
    kap = _kappa([(ga[k], gb[k]) for k in ga if k in gb])
    conf = Counter((ga[k], gb[k]) for k in ga if k in gb)
    off, rnd, extra2, extra1 = [], [], Counter(), Counter()
    yield_by = Counter()
    for pool in pools:
        q = pool["qid"]
        for it in pool["items"]:
            g = final.get((q, it["doc"]))
            if g is None:
                continue
            if "control:official" in it["sources"]:
                off.append(g)
            elif "control:random" in it["sources"] and len(it["sources"]) == 1:
                rnd.append(g)
            else:
                if g == 2:
                    extra2[q] += 1
                    for s in it["sources"]:
                        yield_by[s] += 1
                elif g == 1:
                    extra1[q] += 1
    nq = len(pools)
    # stratum weights (population size / sampled size) for population-level estimates of the test split;
    # the equal-allocation sample over-represents BM25 misses and citing queries
    sample_rec = json.loads((out / "sample.json").read_text())
    popn = sample_rec["strata_population"]
    ns = Counter(pl["stratum"] for pl in pools)
    w = {pl["qid"]: popn[pl["stratum"]] / ns[pl["stratum"]] for pl in pools}
    W = sum(w.values())
    w_q_extra2 = sum(w[pl["qid"]] for pl in pools if extra2[pl["qid"]] > 0) / W
    w_extra2_mean = sum(w[pl["qid"]] * extra2[pl["qid"]] for pl in pools) / W
    # system scores on the sampled queries: official-only vs augmented (official + non-official grade 2)
    runs = {s: _read_run(Path(runs_dir) / f"answer_test_{s}.run", 10) for s in systems}
    def mrr(labels_fn):
        res = {}
        for s, run in runs.items():
            v = []
            for pool in pools:
                rel = labels_fn(pool)
                rr = 0.0
                for i, d in enumerate(run.get(pool["qid"], [])[:10]):
                    if d in rel:
                        rr = 1 / (i + 1)
                        break
                v.append(rr)
            res[s] = v
        return res
    off_scores = mrr(lambda pool: {pool["official"]})
    aug_scores = mrr(lambda pool: {pool["official"]} | {it["doc"] for it in pool["items"] if final.get((pool["qid"], it["doc"])) == 2})
    so = [sum(off_scores[s]) / nq for s in systems]
    sa = [sum(aug_scores[s]) / nq for s in systems]
    wso = {s: sum(w[pl["qid"]] * off_scores[s][i] for i, pl in enumerate(pools)) / W for s in systems}
    wsa = {s: sum(w[pl["qid"]] * aug_scores[s][i] for i, pl in enumerate(pools)) / W for s in systems}
    # Population-weighted orderings (the estimand): weighted MRR under official vs augmented labels.
    tau = _kendall_tau_b([wso[s] for s in systems], [wsa[s] for s in systems])
    tau_unweighted = _kendall_tau_b(so, sa)
    # Stratified bootstrap: resample queries with replacement *within* each stratum (design-respecting), recompute
    # weighted prevalence of non-official grade-2 answers and weighted tau_b.
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for i, pl in enumerate(pools):
        by_stratum[pl["stratum"]].append(i)
    taus, prevs = [], []
    for _ in range(n_boot):
        idx = [i for st_, ix in by_stratum.items() for i in (rng.choice(ix) for _ in ix)]
        Wb = sum(w[pools[i]["qid"]] for i in idx)
        a = [sum(w[pools[i]["qid"]] * off_scores[s][i] for i in idx) / Wb for s in systems]
        b = [sum(w[pools[i]["qid"]] * aug_scores[s][i] for i in idx) / Wb for s in systems]
        t = _kendall_tau_b(a, b)
        if t is not None:
            taus.append(t)
        prevs.append(sum(w[pools[i]["qid"]] for i in idx if extra2[pools[i]["qid"]] > 0) / Wb)
    taus.sort()
    prevs.sort()
    prev_ci = [prevs[int(0.025 * len(prevs))], prevs[int(0.975 * len(prevs)) - 1]] if prevs else None
    # reversals between the weighted orderings, and how often each pair reverses across bootstrap replicates
    reversals = [(systems[i], systems[j]) for i in range(len(systems)) for j in range(i + 1, len(systems))
                 if (wso[systems[i]] - wso[systems[j]]) * (wsa[systems[i]] - wsa[systems[j]]) < 0]
    off2 = sum(g == 2 for g in off)
    off12 = sum(g >= 1 for g in off)
    rnd0 = sum(g == 0 for g in rnd)
    q_with_extra2 = sum(1 for pool in pools if extra2[pool["qid"]] > 0)
    summary = {
        "n_queries": nq, "n_judged_pairs": len(final), "kappa": kap, "confusion": {f"{a}-{b}": c for (a, b), c in conf.items()},
        "official_grade2": [off2, len(off), _wilson(off2, len(off))], "official_grade_ge1": [off12, len(off), _wilson(off12, len(off))],
        "random_grade0": [rnd0, len(rnd), _wilson(rnd0, len(rnd))],
        "queries_with_nonofficial_grade2": [q_with_extra2, nq, _wilson(q_with_extra2, nq)],
        "nonofficial_grade2_per_query_mean": sum(extra2.values()) / nq, "nonofficial_grade1_per_query_mean": sum(extra1.values()) / nq,
        "pool_yield_grade2_by_source": dict(yield_by), "systems": systems, "mrr_official": dict(zip(systems, so)),
        "mrr_augmented": dict(zip(systems, sa)), "kendall_tau_b": tau,
        "kendall_tau_b_ci": [taus[int(0.025 * len(taus))], taus[int(0.975 * len(taus)) - 1]] if taus else None,
        "pairwise_reversals": reversals, "families": families, "held_out": list(held_out), "t": _now(),
        # judged@10: share of each system's top-10 that received a final grade; a held-out (unpooled) run
        # with much lower coverage would be disadvantaged by the pooled labels
        "judged_at_10": {s: sum((pool["qid"], d) in final for pool in pools for d in runs[s].get(pool["qid"], [])[:10])
                              / max(1, sum(len(runs[s].get(pool["qid"], [])[:10]) for pool in pools)) for s in systems},
        "weighted": {"queries_with_nonofficial_grade2": w_q_extra2, "queries_with_nonofficial_grade2_ci": prev_ci,
                     "nonofficial_grade2_per_query_mean": w_extra2_mean,
                     "mrr_official": wso, "mrr_augmented": wsa, "strata_population": popn, "strata_sampled": dict(ns)},
        "kendall_tau_b_unweighted": tau_unweighted,
        "bootstrap": "stratified (within-stratum resampling), population-weighted; n_boot=%d" % n_boot,
    }
    # Independent institutional check of the LLM grades: among non-official pooled answers, how often does the
    # official reply itself cite the answer (CiteRet edge, drafting time), per final grade?
    cites = defaultdict(set)
    bdir = Path(build) if build else None
    if bdir is not None and (bdir / "qrels" / "cite_test.tsv").exists():
        for l in open(bdir / "qrels" / "cite_test.tsv"):
            q_, _, d_, _ = l.split()
            cites[q_].add(d_)
        by_grade = defaultdict(lambda: [0, 0])
        for pool in pools:
            for it in pool["items"]:
                g = final.get((pool["qid"], it["doc"]))
                if g is None or "control:official" in it["sources"]:
                    continue
                by_grade[g][0] += it["doc"] in cites[pool["qid"]]
                by_grade[g][1] += 1
        summary["cited_by_official_reply_by_grade"] = {str(g): {"cited": c, "n": n, "rate": c / n if n else None}
                                                       for g, (c, n) in sorted(by_grade.items())}
    def f3(x):
        return "--" if x is None else f"{x:.3f}"
    def ci(t):
        return f"{t[0]}/{t[1]} ({100*t[2][0]:.0f}--{100*t[2][1]:.0f}\\%)" if t[1] else "--"
    summary["table_rows"] = [
        ["Sampled test queries / judged pairs", f"{nq} / {len(final)}"],
        ["Inter-judge Cohen's $\\kappa$ (3 grades)", f3(kap)],
        ["Official answers graded 2 (95\\% CI)", ci(summary["official_grade2"])],
        ["Random controls graded 0 (95\\% CI)", ci(summary["random_grade0"])],
        ["Queries with $\\geq$1 non-official grade-2 answer in the pool, sample", ci(summary["queries_with_nonofficial_grade2"])],
        ["\\quad same, weighted to the test population (95\\% CI)",
         f"{100 * w_q_extra2:.0f}\\% ({100 * prev_ci[0]:.0f}--{100 * prev_ci[1]:.0f}\\%)" if prev_ci else "--"],
        ["Non-official grade-2 / grade-1 answers per query (sample)", f"{summary['nonofficial_grade2_per_query_mean']:.2f} / {summary['nonofficial_grade1_per_query_mean']:.2f}"],
        ["Kendall's $\\tau_b$ of weighted orderings (95\\% CI)",
         f"{f3(tau)} ({f3(summary['kendall_tau_b_ci'][0]) if taus else '--'}--{f3(summary['kendall_tau_b_ci'][1]) if taus else '--'})"],
        ["Pairwise system reversals", f"{len(reversals)} of {len(systems) * (len(systems) - 1) // 2}"],
        ["Judged@10: pooled runs (min) / held-out run(s)",
         f"{min(summary['judged_at_10'][s] for s in systems if s not in held_out):.2f} / "
         + ", ".join(f"{summary['judged_at_10'][s]:.2f}" for s in systems if s in held_out)],
    ]
    cg = summary.get("cited_by_official_reply_by_grade")
    if cg:
        summary["table_rows"].append(["Cited by the official reply: grade 2 / 1 / 0 (non-official)",
                                      " / ".join(f"{100 * cg[g]['rate']:.0f}\\%" if g in cg and cg[g]["rate"] is not None
                                                 else "--" for g in ("2", "1", "0"))])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def audit_extend_pools(base_pools, build, exp_ext, runs_dir, extra_runs, depth=10, seed=1):
    """Extension of the audit: judge every top-`depth` answer of `extra_runs` that the
    base pool did not contain, on the SAME sampled queries, with the same prompt. Each extended pool also carries the
    official answer and one fresh random answer as blind controls (calibration of the new batch); queries without
    new documents are skipped. Written to <exp_ext>/audit/pools.jsonl."""
    build, runs_dir = Path(build), Path(runs_dir)
    corpus = {d["_id"]: d for d in map(json.loads, open(build / "corpus.jsonl"))}
    runs = {r: _read_run(runs_dir / f"answer_test_{r}.run", depth) for r in extra_runs}
    rng = random.Random(seed)
    all_docs = sorted(corpus)
    pools, n_new = [], Counter()
    for bp in base_pools:
        q, seen = bp["qid"], {it["doc"] for it in bp["items"]}
        src = defaultdict(set)
        for r in extra_runs:
            for d in runs[r].get(q, [])[:depth]:
                if d not in seen:
                    src[d].add(r)
                    n_new[r] += 1
        if not src:
            continue
        src[bp["official"]].add("control:official")
        while True:
            d = rng.choice(all_docs)
            if d not in src and d not in seen:
                src[d].add("control:random")
                break
        docs = sorted(src)
        rng.shuffle(docs)
        items = [{"label": f"D{i}", "doc": d, "sources": sorted(src[d]),
                  "text": corpus[d]["text"][:MAX_CHARS] + (" ..." if len(corpus[d]["text"]) > MAX_CHARS else "")}
                 for i, d in enumerate(docs, 1)]
        pools.append({"qid": q, "stratum": bp["stratum"], "question": bp["question"], "official": bp["official"],
                      "items": items})
    out = _out(exp_ext)
    with open(out / "pools.jsonl", "w") as fh:
        for p in pools:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "pools_meta.json").write_text(json.dumps(
        {"extends": "base audit pools (same 300 queries)", "extra_runs": extra_runs, "depth": depth,
         "n_queries": len(pools), "n_pairs": sum(len(p["items"]) for p in pools),
         "new_docs_by_run_with_overlap": dict(n_new), "t": _now()}, indent=1))
    return pools


def audit_merge(base_exp, ext_exp, full_exp, base_pools, ext_pools, families):
    """Merge base and extension audits into <full_exp>/audit: pools = base items + the extension's NEW run documents
    (extension controls are excluded, so base control grades are never overwritten); judgments and adjudications are
    merged likewise. The merged directory can be passed to audit_summarize unchanged."""
    bo, eo, fo = _out(base_exp), _out(ext_exp), _out(full_exp)
    ext_by = {p["qid"]: p for p in ext_pools}
    new_docs = {q: {it["doc"] for it in p["items"] if not any(s.startswith("control:") for s in it["sources"])}
                for q, p in ext_by.items()}
    merged = []
    for bp in base_pools:
        items = list(bp["items"])
        if bp["qid"] in ext_by:
            items += [it for it in ext_by[bp["qid"]]["items"] if it["doc"] in new_docs[bp["qid"]]]
        merged.append({**bp, "items": items})
    with open(fo / "pools.jsonl", "w") as fh:
        for p in merged:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    for fam in families:
        with open(fo / f"judgments_{fam}.jsonl", "w") as fh:
            fh.write(open(bo / f"judgments_{fam}.jsonl").read())
            for r in map(json.loads, open(eo / f"judgments_{fam}.jsonl")):
                r["judgments"] = [j for j in r["judgments"] if j["doc"] in new_docs.get(r["qid"], set())]
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(fo / "adjudication.jsonl", "w") as fh:
        fh.write(open(bo / "adjudication.jsonl").read())
        for r in map(json.loads, open(eo / "adjudication.jsonl")):
            if r["doc"] in new_docs.get(r["qid"], set()):
                fh.write(json.dumps(r, default=str) + "\n")
    (fo / "sample.json").write_text((bo / "sample.json").read_text())
    return merged
