# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Upload a frozen build to the Hugging Face dataset repo NaiveNeuron/parlir.

  uv run scripts/release_hf.py --build data/build/v1.0 --version v1.0 [--exp experiments/v1.0] [--allow-public]
Uploads corpus/queries/qrels/metadata, the dataset card (hf/README.md), and (optionally) the runs and
audit records of the experiment dir. Creates a git tag <version> on the HF repo. It never changes repo
visibility, and uploads to a public repo need --allow-public.
"""
import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = "NaiveNeuron/parlir"
FILES = ["corpus.jsonl", "queries.jsonl", "queries_xling.jsonl", "precedent_meta.jsonl", "equivalents.jsonl",
         "build_manifest.json"]

ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--exp", type=Path, default=None)
ap.add_argument("--no-tag", action="store_true", help="upload only; tag after verifying the configs load")
ap.add_argument("--allow-public", action="store_true", help="allow uploading a new version to the public repo")
a = ap.parse_args()
api = HfApi()
info = api.dataset_info(REPO)
assert info.private or a.allow_public, "refusing to upload to a public repo without --allow-public"
ops_dir = f"{a.version}"
for f in FILES:
    p = a.build / f
    if p.exists():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{ops_dir}/{f}", repo_id=REPO, repo_type="dataset",
                        commit_message=f"{a.version}: {f}")
        print("uploaded", f)
api.upload_folder(folder_path=str(a.build / "qrels"), path_in_repo=f"{ops_dir}/qrels", repo_id=REPO,
                  repo_type="dataset", commit_message=f"{a.version}: qrels")
# HF `datasets` infers ONE builder per repo from the dominant extension (JSONL here) and ignores CSV-only options,
# so the TREC TSVs cannot back a config. Publish the same qrels as JSONL (derived, lossless) for the card configs.
with tempfile.TemporaryDirectory() as tmp:
    for tsv in sorted((a.build / "qrels").glob("*.tsv")):
        with open(tsv) as fi, open(Path(tmp) / (tsv.stem + ".jsonl"), "w") as fo:
            for line in fi:
                q, it, d, r = line.split()
                fo.write(json.dumps({"query_id": q, "iteration": int(it), "doc_id": d, "relevance": int(r)}) + "\n")
    api.upload_folder(folder_path=tmp, path_in_repo=f"{ops_dir}/qrels_jsonl", repo_id=REPO, repo_type="dataset",
                      commit_message=f"{a.version}: qrels (JSONL for dataset configs)")
if Path("hf/README.md").exists():
    api.upload_file(path_or_fileobj="hf/README.md", path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                    commit_message="dataset card")
if Path("LICENSE-DATA.md").exists():  # referenced by the card's license_link
    api.upload_file(path_or_fileobj="LICENSE-DATA.md", path_in_repo="LICENSE-DATA.md", repo_id=REPO, repo_type="dataset",
                    commit_message="data licence terms")
if a.exp and a.exp.exists():
    for sub in ("runs", "audit"):
        if (a.exp / sub).exists():
            api.upload_folder(folder_path=str(a.exp / sub), path_in_repo=f"{ops_dir}/experiments/{sub}", repo_id=REPO,
                              repo_type="dataset", commit_message=f"{a.version}: {sub}",
                              ignore_patterns=["*.npy", "cache/*", "models/*"])
if not a.no_tag:
    api.create_tag(REPO, tag=a.version, repo_type="dataset", exist_ok=True)
print("done; private =", api.dataset_info(REPO).private)
