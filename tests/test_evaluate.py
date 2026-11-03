"""Our metric implementation must agree with trec_eval (via pytrec_eval) on random runs,
including graded qrels, ties, missing queries and multi-relevant queries."""
import random

import pytest

from parlir.evaluate import evaluate

pytrec_eval = pytest.importorskip("pytrec_eval")


def _random_case(seed):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(60)]
    qrels, run = {}, {}
    for q in range(40):
        qid = f"q{q}"
        rel = rng.sample(docs, rng.randint(1, 4))
        qrels[qid] = {d: rng.choice([1, 2]) for d in rel}
        if q % 7 == 0:
            continue  # query absent from the run
        cand = rng.sample(docs, rng.randint(5, 60))
        run[qid] = {d: float(rng.randint(0, 20)) for d in cand}  # many ties
    return qrels, run


@pytest.mark.parametrize("seed", range(5))
def test_matches_trec_eval(seed):
    qrels, run = _random_case(seed)
    agg, per_q = evaluate(qrels, run)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut", "recall", "success"})
    ref = ev.evaluate(run)
    run10 = {q: dict(sorted(r.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10]) for q, r in run.items()}
    ref_rr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(run10)
    for q in qrels:
        if q not in ref:
            assert per_q["RR@10"][q] == 0 and per_q["nDCG@10"][q] == 0
            continue
        assert per_q["nDCG@10"][q] == pytest.approx(ref[q]["ndcg_cut_10"], abs=1e-9)
        assert per_q["R@10"][q] == pytest.approx(ref[q]["recall_10"], abs=1e-9)
        assert per_q["R@10"][q] == pytest.approx(ref[q]["recall_10"], abs=1e-9)
        assert per_q["R@30"][q] == pytest.approx(ref[q]["recall_30"], abs=1e-9)
        assert per_q["Success@1"][q] == pytest.approx(ref[q]["success_1"], abs=1e-9)
        assert per_q["RR@10"][q] == pytest.approx(ref_rr[q]["recip_rank"], abs=1e-9)
