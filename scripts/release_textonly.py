# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.30,<2"]
# ///
"""Upload the derived text-only evaluation export (scripts/export_textonly.py) to the HF dataset, freeze it with a
NEW tag, and verify every file by path and size. The data tag (v1.0) and evidence tags are never moved.

Layout on the hub: <version>/textonly/ (system-facing files + evaluator-side eval/), and
<version>/textonly-provenance/mapping.json (opaque id -> source id; a separate download).

  uv run scripts/release_textonly.py --out data/build/v1.0-textonly --version v1.0
  uv run scripts/release_textonly.py --out data/build/v1.0-textonly --version v1.0 --verify
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = "NaiveNeuron/parlir"
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--tag", default=None, help="default <version>-textonly")
ap.add_argument("--verify", action="store_true")
ap.add_argument("--allow-public", action="store_true", help="permit uploading after the authors made the repo public")
a = ap.parse_args()
api = HfApi()
assert a.allow_public or api.dataset_info(REPO).private, \
    "refusing: repo is not private (pass --allow-public once the authors have released it)"
tag = a.tag or f"{a.version}-textonly"
prov = a.out.parent / f"{a.out.name}-provenance"
parts = [(a.out, f"{a.version}/textonly"), (prov, f"{a.version}/textonly-provenance")]
files = {p: f"{remote}/{p.relative_to(local).as_posix()}" for local, remote in parts for p in sorted(local.rglob("*"))
         if p.is_file()}
if not a.verify:
    for local, remote in parts:
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(local), path_in_repo=remote,
                          commit_message=f"{a.version}: derived text-only evaluation export ({remote})")
    api.create_tag(REPO, tag=tag, repo_type="dataset", exist_ok=False,
                   tag_message=f"{a.version} derived text-only export: opaque ids + text; provenance separate")
    print("tagged", tag)
remote = {e.path: e.size for e in api.list_repo_tree(REPO, repo_type="dataset", revision=tag,
                                                     path_in_repo=a.version, recursive=True)
          if getattr(e, "size", None) is not None and not hasattr(e, "tree_id")}
missing = [r for r in files.values() if r not in remote]
mismatch = [r for p, r in files.items() if r in remote and remote[r] != p.stat().st_size]
print(f"local {len(files)} files; at {tag}: missing {len(missing)}, size mismatches {len(mismatch)}")
raise SystemExit(1 if missing or mismatch else 0)
