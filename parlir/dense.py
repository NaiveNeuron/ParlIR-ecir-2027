"""Dense bi-encoder baselines (paper Sec. 7): exact inner-product search over normalized embeddings.

Model registry pins the HF revision used in the paper; prefixes follow each model card.
Embeddings are cached per (model, weights, text-list sha) under the experiment dir; a local --model-path is
identified by the content hash of its weight files, so re-trained or different checkpoints never share a cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODELS = {
    # name: (hf id, query prefix, passage prefix, max_seq_length)
    "me5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: ", 512),
    "me5-large": ("intfloat/multilingual-e5-large", "query: ", "passage: ", 512),
    "bge-m3": ("BAAI/bge-m3", "", "", 1024),
    # Added post hoc: a 2025 LLM-based embedder, zero-shot. Instruction format per the model card;
    # the same 1,024-token cap as BGE-M3 for parity.
    "qwen3-emb-8b": ("Qwen/Qwen3-Embedding-8B",
                     "Instruct: Given a written parliamentary question, retrieve answers that address it\nQuery: ",
                     "", 1024),
}
# Weight dtype per model (default fp16, as for the paper's original baselines). Qwen3 recommends bf16: fp16 can
# overflow in its activations.
DTYPE = {"qwen3-emb-8b": "bfloat16"}
# Batch size per model where the default (64 x max_seq_length tokens) would not fit next to the weights on a 24 GB GPU.
BATCH = {"qwen3-emb-8b": 8}
# Multi-GPU via one model copy per GPU in threads instead of the sentence-transformers process pool (see encode).
# The original baselines keep the pool path, so their cached embeddings stay valid.
THREADED = {"qwen3-emb-8b"}


def _sha(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _weights_id(model_path: str) -> str:
    """Content hash of a local checkpoint's weight files (safetensors/bin); fails loudly if none are found."""
    files = sorted(p for p in Path(model_path).rglob("*") if p.suffix in (".safetensors", ".bin") and p.is_file())
    if not files:
        raise FileNotFoundError(f"no weight files under {model_path}")
    h = hashlib.sha256()
    for p in files:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def encode(name: str, texts: list[str], is_query: bool, cache_dir: Path | None = None, batch_size: int = 64,
           model_path: str | None = None) -> np.ndarray:
    hf, qp, pp, max_len = MODELS.get(name, (model_path or name, "query: ", "passage: ", 512))
    batch_size = BATCH.get(name, batch_size)
    if model_path:
        hf = model_path
    texts = [(qp if is_query else pp) + t for t in texts]
    wid = f"_{_weights_id(model_path)}" if model_path else ""
    key = f"{name}{wid}_{'q' if is_query else 'd'}_{_sha(texts)}.npy"
    if cache_dir is not None and (cache_dir / key).exists():
        return np.load(cache_dir / key)
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(hf, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = max_len
    model.to(getattr(torch, DTYPE.get(name, "float16")))
    n_gpu = torch.cuda.device_count()
    # sort by length for efficient batching, restore order afterwards
    order = np.argsort([-len(t) for t in texts])
    sorted_texts = [texts[i] for i in order]
    if n_gpu > 1 and len(texts) > 5000 and name in THREADED:
        # one model copy per GPU in threads (CUDA kernels release the GIL): the multi-process pool pickles the
        # whole model to every worker, which stalls for multi-GB models. Round-robin chunks of the length-sorted
        # list balance the work; the interleaving is undone below.
        from concurrent.futures import ThreadPoolExecutor
        models = [model]
        for i in range(1, n_gpu):
            m = SentenceTransformer(hf, device=f"cuda:{i}")
            m.max_seq_length = max_len
            m.to(getattr(torch, DTYPE.get(name, "float16")))
            models.append(m)
        chunks = [sorted_texts[i::n_gpu] for i in range(n_gpu)]
        with ThreadPoolExecutor(n_gpu) as ex:
            parts = list(ex.map(lambda mc: mc[0].encode(mc[1], batch_size=batch_size, normalize_embeddings=True,
                                                         show_progress_bar=True), zip(models, chunks)))
        emb = np.empty((len(sorted_texts), parts[0].shape[1]), dtype=parts[0].dtype)
        for i, p in enumerate(parts):
            emb[i::n_gpu] = p
    elif n_gpu > 1 and len(texts) > 5000:
        pool = model.start_multi_process_pool([f"cuda:{i}" for i in range(n_gpu)])
        emb = model.encode(sorted_texts, pool=pool, batch_size=batch_size, normalize_embeddings=True)
        model.stop_multi_process_pool(pool)
    else:
        emb = model.encode(sorted_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    out = np.empty_like(emb)
    out[order] = emb
    out = out.astype(np.float32)
    # free the model(s) now: module reference cycles otherwise keep multi-GB weights on the GPU until a later gc
    # pass, and the next encode() call (queries after documents) would load a second copy and run out of memory
    import gc
    # every local that can hold a model: `model`, the threaded path's `models` list and its loop variable `m`
    model = models = m = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / key, out)
    return out


def truncation_rate(name: str, texts: list[str]) -> float:
    from transformers import AutoTokenizer
    hf, _, _, max_len = MODELS[name]
    tok = AutoTokenizer.from_pretrained(hf)
    lens = [len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    return float(np.mean([l > max_len for l in lens]))


def dense_search(name: str, doc_texts: list[str], query_texts: list[str], k: int, cache_dir: Path | None = None,
                 model_path: str | None = None):
    import torch
    D = encode(name, doc_texts, False, cache_dir, model_path=model_path)
    Q = encode(name, query_texts, True, cache_dir, model_path=model_path)
    Dt = torch.from_numpy(D).cuda().half() if torch.cuda.is_available() else torch.from_numpy(D)
    out = []
    for s in range(0, len(Q), 1024):
        q = torch.from_numpy(Q[s:s + 1024]).to(Dt.device).to(Dt.dtype)
        sc = (q @ Dt.T).float()
        v, i = torch.topk(sc, min(k, Dt.shape[0]), dim=1)
        for vi, ii in zip(v.cpu().numpy(), i.cpu().numpy()):
            out.append(list(zip(ii.tolist(), vi.tolist())))
    return out
