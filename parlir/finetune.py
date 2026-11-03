"""Fine-tune a multilingual bi-encoder on the ParlIR train split (paper Sec. 7, claim H7).

  uv run python -m parlir.finetune --build B --base intfloat/multilingual-e5-base --out models/<name>
Training pairs: (train question, its official answer) with MultipleNegativesRankingLoss (in-batch
negatives) plus one BM25 hard negative per question that is NOT the official answer and NOT a
precedent-relevant answer. Only train-split questions are used; dev/test questions and their answers
are never used as anchors or positives. Seed fixed; one epoch.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-base")
    ap.add_argument("--hardneg-run", type=Path, default=None, help="BM25 run over train queries (TREC)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-pairs", type=int, default=0, help="seeded random subset of this many pairs")
    ap.add_argument("--min-year", type=int, default=0, help="only train questions dated in or after this year")
    ap.add_argument("--max-year", type=int, default=0, help="only train questions dated in or before this year")
    ap.add_argument("--hardneg-window-days", type=int, default=0,
                    help="date-matched hard negatives: only candidates dated within N days of the positive (0 = off)")
    ap.add_argument("--negatives", choices=["bm25", "random"], default="bm25",
                    help="control: 'random' replaces the BM25 hard negative by a random answer (within the date window "
                         "if --hardneg-window-days is set), isolating negative difficulty from the date signal")
    a = ap.parse_args()
    random.seed(a.seed)
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    torch.manual_seed(a.seed)
    docs = [json.loads(l) for l in (a.build / "corpus.jsonl").open()]
    corpus = {d["_id"]: d["text"] for d in docs}
    ddate = {d["_id"]: date.fromisoformat(d["date"][:10]) for d in docs if d.get("date")}
    by_date = sorted((v, k) for k, v in ddate.items())
    keys = [v for v, _ in by_date]
    all_ids = list(corpus)

    def in_window(doc, ref):
        return doc in ddate and abs((ddate[doc] - ref).days) <= a.hardneg_window_days

    def random_in_window(ref):
        lo = bisect.bisect_left(keys, ref - timedelta(days=a.hardneg_window_days))
        hi = bisect.bisect_right(keys, ref + timedelta(days=a.hardneg_window_days))
        return by_date[random.randrange(lo, hi)][1]
    def year(q):
        return int((q.get("date") or "0")[:4])
    queries = [q for q in map(json.loads, (a.build / "queries.jsonl").open()) if q["split"] == "train"
               and (not a.min_year or year(q) >= a.min_year) and (not a.max_year or year(q) <= a.max_year)]
    hard = {}
    if a.hardneg_run and a.hardneg_run.exists():
        for line in a.hardneg_run.open():
            q, _, d, rank, _, _ = line.split()
            hard.setdefault(q, []).append(d)
    prec = {}
    pq = a.build / "qrels" / "precedent_train.tsv"
    if pq.exists():
        for line in pq.open():
            q, _, d, _ = line.split()
            prec.setdefault(q, set()).add(d)
    rows = {"anchor": [], "positive": [], "negative": []}
    stats = Counter()
    for q in queries:
        pos = q["answer"]
        cands = [d for d in hard.get(q["_id"], []) if d != pos and d not in prec.get(q["_id"], set())]
        if a.negatives == "random":
            pool_fn = (lambda: random_in_window(ddate[pos])) if a.hardneg_window_days and pos in ddate \
                else (lambda: random.choice(all_ids))
            neg = pool_fn()
            while neg == pos or neg in prec.get(q["_id"], set()):
                neg = pool_fn()
            stats["random_in_window" if a.hardneg_window_days else "random_any_date"] += 1
        elif a.hardneg_window_days and pos in ddate:
            # date-matched negatives: the negative must not carry a date signal the positive lacks
            cands = [d for d in cands if in_window(d, ddate[pos])]
            neg = cands[0] if cands else random_in_window(ddate[pos])
            while neg == pos:
                neg = random_in_window(ddate[pos])
            stats["date_matched_bm25" if cands else "date_matched_random"] += 1
        else:
            neg = cands[0] if cands else random.choice(list(corpus))
        rows["anchor"].append("query: " + q["text"])
        rows["positive"].append("passage: " + corpus[pos])
        rows["negative"].append("passage: " + corpus[neg])
    ds = Dataset.from_dict(rows).shuffle(seed=a.seed)
    n = len(ds) if not a.max_pairs else min(a.max_pairs, len(ds))
    ds = ds.select(range(n))
    model = SentenceTransformer(a.base, device="cuda:0" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = a.max_len
    # GradCache variant: same in-batch-negative objective as MNRL with constant memory (mini-batches of 8)
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(a.out / "ckpt"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=0.1, fp16=True, seed=a.seed, logging_steps=50, save_strategy="no",
        batch_sampler=BatchSamplers.NO_DUPLICATES, report_to=[], dataloader_num_workers=2)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save(str(a.out))
    (a.out / "parlir_train_meta.json").write_text(json.dumps(
        {"base": a.base, "n_pairs": n, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
         "max_len": a.max_len, "seed": a.seed, "hard_negatives": bool(hard), "min_year": a.min_year,
         "max_year": a.max_year, "hardneg_window_days": a.hardneg_window_days, "negatives": a.negatives,
         "negative_stats": dict(stats),
         "build": str(a.build)}, indent=1))
    print("saved", a.out, n)


if __name__ == "__main__":
    main()
