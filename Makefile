VERSION ?= v1.0
BUILD   := data/build/$(VERSION)
EXP     ?= experiments/$(VERSION)
PY      := uv run python
SPLIT   ?= test

.PHONY: sync acquire list csv csv-langs files detail build validate baselines experiments test smoke

sync:
	uv sync

# Full acquisition (polite: ~10 req/s to the document host, <=500 req / 5 min to the ODP API; ~5 h total).
acquire: list csv csv-langs files detail

list:
	$(PY) -m parlir.acquire list
csv:
	$(PY) -m parlir.acquire csv
csv-langs:
	$(PY) -m parlir.acquire csv-langs
files:
	$(PY) -m parlir.acquire files --lang en
detail:
	$(PY) -m parlir.acquire detail

build:
	$(PY) -m parlir.build --version $(VERSION)

validate:
	$(PY) -m parlir.validate --build $(BUILD) --out experiments/integrity/$(VERSION).json

baselines:
	$(PY) -m parlir.retrieve --build $(BUILD) --task answer --split $(SPLIT) --system bm25 --exp $(EXP)/runs
	$(PY) -m parlir.retrieve --build $(BUILD) --task answer --split $(SPLIT) --system idoracle --field full --exp $(EXP)/runs
	$(PY) -m parlir.retrieve --build $(BUILD) --task cite --split $(SPLIT) --system bm25 --exp $(EXP)/runs

# Full non-LLM experiment suite of the paper (skips outputs that already exist); the LLM audit runs in OMP.
experiments:
	uv run scripts/run_experiments.py --build $(BUILD) --exp $(EXP)

# 200-question end-to-end smoke build (~5 min, polite 2 req/s): fetch -> build -> validate -> BM25.
# Uses tests/fixtures/smoke_ids.jsonl as the id list (instead of the full ODP listing).
smoke:
	mkdir -p data/raw && cp tests/fixtures/smoke_ids.jsonl data/raw/ids.jsonl
	$(PY) -m parlir.acquire files --lang en --rate 2 --concurrency 2
	$(PY) -m parlir.build --version smoke
	$(PY) -m parlir.validate --build data/build/smoke --out experiments/smoke/integrity.json
	$(PY) -m parlir.retrieve --build data/build/smoke --task answer --split test --system bm25 --exp experiments/smoke/runs

test:
	uv run pytest -q
