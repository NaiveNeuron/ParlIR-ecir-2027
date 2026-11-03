"""Acquire EP written questions + answers from the EP Open Data Portal API v2.

Stages (each idempotent, cached under data/raw/, resumable):
  list     -> data/raw/ids.jsonl          all question Work ids (paged list endpoint)
  detail   -> data/raw/detail/<id>.json   JSON-LD record of each question (incl. inverse_answers_to)
  fetch    -> data/raw/docs/<work>/<lang>.<ext>  manifestation files for Q and ASW works

Usage: uv run python -m parlir.acquire {list|detail|fetch} [--concurrency N]
Every HTTP response is stored verbatim; data/raw/manifest.jsonl records url, sha256, bytes, UTC time.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

API = "https://data.europarl.europa.eu/api/v2/parliamentary-questions"
BASE = "https://data.europarl.europa.eu/"
RAW = Path("data/raw")
UA = "ParlIR-research-crawler/0.1 (academic test-collection construction; contact via github.com/NaiveNeuron)"
FMT = {"format": "application/ld+json"}
# Preferred manifestation per expression, most structured first.
FORMAT_PREF = ["XML", "DOCX", "XHTML", "DOC", "PDF"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a")

    def add(self, url: str, content: bytes, status: int, dest: str) -> None:
        self.fh.write(json.dumps({"url": url, "status": status, "bytes": len(content),
                                  "sha256": hashlib.sha256(content).hexdigest(), "dest": dest, "t": _now()}) + "\n")
        self.fh.flush()


async def get(client: httpx.AsyncClient, url: str, params=None, tries: int = 6) -> httpx.Response | None:
    for k in range(tries):
        try:
            r = await client.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504) or (r.status_code == 202 and not r.content):
                raise httpx.HTTPStatusError(f"retryable {r.status_code}", request=r.request, response=r)
            return r
        except (httpx.HTTPError, httpx.StreamError):
            await asyncio.sleep(min(60, 2 ** k + random.random()))
    return None


async def stage_list(page: int = 500, wave: int = 8) -> None:
    """Page through the list endpoint `wave` pages at a time until a page comes back short/empty."""
    out = RAW / "ids.jsonl"
    seen = set()
    if out.exists():
        seen = {json.loads(l)["identifier"] for l in out.open()}
    man = Manifest(RAW / "manifest.jsonl")
    offset = 0
    async with httpx.AsyncClient(timeout=300, headers={"User-Agent": UA}) as client:
        with out.open("a") as fh:
            while True:
                offs = [offset + k * page for k in range(wave)]
                rs = await asyncio.gather(*(get(client, API, params={**FMT, "offset": o, "limit": page}) for o in offs))
                last = False
                for o, r in zip(offs, rs):
                    if r is None:
                        raise SystemExit(f"list failed at offset {o}")
                    data = r.json().get("data", []) if (r.status_code == 200 and r.content) else []
                    man.add(str(r.url), r.content, r.status_code, "ids.jsonl")
                    for d in data:
                        if d["identifier"] not in seen:
                            seen.add(d["identifier"])
                            fh.write(json.dumps({"identifier": d["identifier"], "work_type": d.get("work_type")}) + "\n")
                    if len(data) < page:
                        last = True
                fh.flush()
                print(f"{_now()} offset={offset} n={len(seen)}", flush=True)
                if last:
                    break
                offset += wave * page


async def stage_detail(concurrency: int, rate: float = 1.5) -> None:
    """API detail record per question (original language, expressions, joint answers, authors).
    ODP limit is 500 req / 5 min: run at `rate` req/s, newest questions first, retrying 429 forever."""
    ids = [json.loads(l)["identifier"] for l in (RAW / "ids.jsonl").open()]
    ids = _order_newest_first([i for i in ids if i[0] in "EP"])
    ddir = RAW / "detail"
    ddir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (ddir / f"{i}.json").exists()]
    print(f"{_now()} detail todo={len(todo)} rate={rate}/s", flush=True)
    man = Manifest(RAW / "manifest.jsonl")
    limiter = RateLimiter(rate)
    queue: asyncio.Queue = asyncio.Queue()
    for i in todo:
        queue.put_nowait(i)
    done, t0 = 0, time.time()

    async def worker(client):
        nonlocal done
        while not queue.empty():
            i = queue.get_nowait()
            backoff = 10.0
            while True:
                await limiter.wait()
                try:
                    r = await client.get(f"{API}/{i}", params=FMT)
                except httpx.HTTPError:
                    await asyncio.sleep(backoff)
                    continue
                if r.status_code == 429 or r.status_code >= 500 or (r.status_code == 202 and not r.content):
                    await asyncio.sleep(backoff)
                    backoff = min(120.0, backoff * 1.5)
                    continue
                break
            dest = ddir / f"{i}.json"
            dest.write_bytes(r.content if (r.status_code == 200 and r.content) else json.dumps({"_status": r.status_code}).encode())
            man.add(str(r.url), r.content, r.status_code, str(dest))
            done += 1
            if done % 200 == 0:
                print(f"{_now()} detail {done}/{len(todo)} {done / (time.time() - t0):.2f}/s", flush=True)

    async with httpx.AsyncClient(timeout=180, headers={"User-Agent": UA},
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))

EU_LANGS = "bg cs da de el en es et fi fr ga hr hu it lt lv mt nl pl pt ro sk sl sv".split()


async def stage_csv_langs() -> None:
    """Per-language yearly ODP CSVs: a row's `document_pdf` is filled only if the question exists in
    that language, so these files give each question's language versions (=> original language)."""
    out = RAW / "csv_langs"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(RAW / "manifest.jsonl")
    versions = sorted(p.stem.rsplit("_", 1)[0] for p in (RAW / "csv").glob("*_en.csv"))  # e.g. 2024_56
    async with httpx.AsyncClient(timeout=300, headers={"User-Agent": UA}, follow_redirects=True) as client:
        for yv in versions:
            for lang in EU_LANGS:
                dest = out / f"{yv}_{lang}.csv"
                if dest.exists() or lang == "en":
                    continue
                url = f"{BASE}distribution/parliamentary-questions_{yv}_{lang}.csv"
                backoff = 10.0
                while True:
                    try:
                        r = await client.get(url)
                    except httpx.HTTPError:
                        await asyncio.sleep(backoff)
                        continue
                    if r.status_code == 429 or r.status_code >= 500:
                        await asyncio.sleep(backoff)
                        backoff = min(120, backoff * 1.5)
                        continue
                    break
                if r.status_code == 200 and r.content:
                    dest.write_bytes(r.content)
                man.add(url, r.content, r.status_code, str(dest))
                print(_now(), yv, lang, r.status_code, len(r.content), flush=True)
                await asyncio.sleep(0.7)



def choose_manifestations(work: dict, langs: set[str] | None) -> list[tuple[str, str, str]]:
    """Return (lang3, format, url) for the preferred manifestation of each wanted expression."""
    out = []
    for e in work.get("is_realized_by", []):
        lang = e.get("language", "")[-3:]
        if langs is not None and lang not in langs:
            continue
        ms = {m["format"].rsplit("/", 1)[-1]: m["is_exemplified_by"] for m in e.get("is_embodied_by", [])}
        for f in FORMAT_PREF:
            if f in ms:
                out.append((lang, f, BASE + ms[f]))
                break
    return out


async def stage_fetch(concurrency: int, all_langs: bool) -> None:
    ddir, docs = RAW / "detail", RAW / "docs"
    jobs = []
    for p in sorted(ddir.glob("*.json")):
        d = json.loads(p.read_text() or "{}")
        if "data" not in d:
            continue
        q = d["data"][0]
        orig = {x[-3:] for x in q.get("originalLanguage", [])}
        want = None if all_langs else ({"ENG"} | orig)
        works = [q] + list(q.get("inverse_answers_to", []))
        for w in works:
            wid = w["id"].rsplit("/", 1)[-1]
            for lang, f, url in choose_manifestations(w, want):
                dest = docs / wid / f"{lang}.{f.lower()}"
                if not dest.exists():
                    jobs.append((url, dest))
    print(f"{_now()} fetch jobs: {len(jobs)}", flush=True)
    man = Manifest(RAW / "manifest.jsonl")
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.time()

    async def one(client, url, dest):
        nonlocal done
        async with sem:
            r = await get(client, url)
        if r is None or r.status_code != 200:
            man.add(url, b"", r.status_code if r else -1, str(dest))
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        man.add(url, r.content, r.status_code, str(dest))
        done += 1
        if done % 1000 == 0:
            print(f"{_now()} fetch {done}/{len(jobs)} {done / (time.time() - t0):.1f}/s", flush=True)

    async with httpx.AsyncClient(timeout=180, headers={"User-Agent": UA}, follow_redirects=True,
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        await asyncio.gather(*(one(client, u, d) for u, d in jobs))

# Files are served by the EP's document repository host behind the ODP redirect. Requesting it
# directly avoids spending the ODP API quota (500 req / 5 min) on static files. URL layout is
# deterministic: <MEDIA>/reds_iMaQp[_Asw]/<ID>[-ASW]/<ID>[-ASW]_<lang>.<ext> (verified by probing).
MEDIA = "https://redmapl3.europarl.europa.eu/RedmapFront/media"
FILE_ORDER = ["xhtml", "docx", "xml"]  # XHTML exists for 2014-2025 and hyperlinks cross-references.


def file_url(qid: str, answer: bool, lang: str, ext: str) -> str:
    wid = f"{qid}-ASW" if answer else qid
    sub = "reds_iMaQp_Asw" if answer else "reds_iMaQp"
    return f"{MEDIA}/{sub}/{wid}/{wid}_{lang}.{ext}"


class RateLimiter:
    """Global token bucket: at most `rate` request starts per second."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self.next = time.monotonic()
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.next > now:
                await asyncio.sleep(self.next - now)
            self.next = max(self.next, now) + self.interval


def _order_newest_first(ids: list[str]) -> list[str]:
    # Test/dev splits are the most recent questions: fetch them first (term desc, year desc, number asc).
    def key(i):
        p = i.split("-")
        return (-int(p[1]), -int(p[2]), p[3])
    return sorted(ids, key=key)


async def stage_files(concurrency: int, lang: str, ids_path: Path | None, rate: float, kinds: str = "QA",
                      formats: list[str] | None = None) -> None:
    """For every E/P question id: fetch Q and ASW in `lang`, first available format in FILE_ORDER
    (or in `formats`, e.g. ["xml"] to add the XML version whose header carries the question date).
    A missing ASW is normal (unanswered, or covered by a joint answer filed under another id).
    HTTP 429 (host limit ~100 req / 10 s) is never recorded as 'missing': we back off and retry.
    Outcomes are logged to data/raw/files_<lang>[_<formats>].jsonl (one row per (id, kind)); the stage is
    resumable and rows with ext=None are only written after every format returned a definitive 404."""
    order = formats or FILE_ORDER
    ids = [json.loads(l)["identifier"] for l in (ids_path or RAW / "ids.jsonl").open()]
    ids = _order_newest_first([i for i in ids if i[0] in "EP"])
    log_path = RAW / (f"files_{lang}.jsonl" if not formats else f"files_{lang}_{'-'.join(formats)}.jsonl")
    done_keys = set()
    if log_path.exists():
        done_keys = {(r["id"], r["kind"]) for r in map(json.loads, log_path.open()) if r.get("final", True)}
    jobs = [(i, k) for i in ids for k in ("Q", "A") if k in kinds and (i, k) not in done_keys]
    print(f"{_now()} files[{lang}] jobs: {len(jobs)} rate={rate}/s", flush=True)
    man = Manifest(RAW / "manifest.jsonl")
    logf = log_path.open("a")
    limiter = RateLimiter(rate)
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stats = {"n": 0, "r429": 0, "err": 0}
    t0 = time.time()

    async def fetch(client, url) -> httpx.Response:
        backoff = 5.0
        while True:
            await limiter.wait()
            try:
                r = await client.get(url)
            except httpx.HTTPError:
                stats["err"] += 1
                await asyncio.sleep(backoff)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                stats["r429"] += r.status_code == 429
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 1.5)
                continue
            return r

    async def worker(client):
        while not queue.empty():
            qid, kind = queue.get_nowait()
            got = None
            for ext in order:
                url = file_url(qid, kind == "A", lang, ext)
                r = await fetch(client, url)
                if r.status_code == 200 and r.content:
                    wid = f"{qid}-ASW" if kind == "A" else qid
                    dest = RAW / "docs" / wid / f"{lang}.{ext}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(r.content)
                    man.add(url, r.content, 200, str(dest))
                    got = ext
                    break
            logf.write(json.dumps({"id": qid, "kind": kind, "lang": lang, "ext": got, "t": _now()}) + "\n")
            logf.flush()
            stats["n"] += 1
            if stats["n"] % 1000 == 0:
                el = time.time() - t0
                print(f"{_now()} files {stats['n']}/{len(jobs)} {stats['n'] / el:.1f} rows/s "
                      f"429s={stats['r429']} errs={stats['err']}", flush=True)

    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": UA},
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
    logf.close()


async def stage_csv() -> None:
    """Yearly ODP CSV datasets (metadata + joint-answer mapping). Version from the dataset TTL."""
    out = RAW / "csv"
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest(RAW / "manifest.jsonl")
    async with httpx.AsyncClient(timeout=300, headers={"User-Agent": UA}, follow_redirects=True) as client:
        for year in range(2014, 2027):
            r = await get(client, f"{BASE}dataset/parliamentary-questions_{year}.ttl")
            if r is None or r.status_code != 200:
                print(year, "ttl", r.status_code if r else None)
                continue
            import re
            m = re.findall(rf"parliamentary-questions_{year}_(\d+)", r.text)
            vers = sorted({int(x) for x in m})
            for ver in reversed(vers):
                url = f"{BASE}distribution/parliamentary-questions_{year}_{ver}_en.csv"
                rc = await get(client, url)
                if rc is not None and rc.status_code == 200 and rc.content:
                    (out / f"{year}_{ver}_en.csv").write_bytes(rc.content)
                    man.add(url, rc.content, 200, str(out / f"{year}_{ver}_en.csv"))
                    print(year, ver, len(rc.content), flush=True)
                    break
            await asyncio.sleep(1.0)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "detail", "fetch", "files", "csv", "csv-langs"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--ids", type=Path, default=None)
    ap.add_argument("--rate", type=float, default=9.0, help="max requests/s to the document host")
    ap.add_argument("--kinds", default="QA", help="Q = questions, A = answers (files stage)")
    ap.add_argument("--formats", default=None, help="comma list overriding the format order, e.g. xml")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--all-langs", action="store_true", help="fetch every language version, not just EN + original")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.stage == "list":
        asyncio.run(stage_list())
    elif a.stage == "detail":
        asyncio.run(stage_detail(a.concurrency))
    elif a.stage == "files":
        asyncio.run(stage_files(a.concurrency, a.lang, a.ids, a.rate, a.kinds,
                                a.formats.split(",") if a.formats else None))
    elif a.stage == "csv-langs":
        asyncio.run(stage_csv_langs())
    elif a.stage == "csv":
        asyncio.run(stage_csv())
    else:
        asyncio.run(stage_fetch(a.concurrency, a.all_langs))


if __name__ == "__main__":
    main()
