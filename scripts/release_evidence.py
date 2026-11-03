# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.30,<2"]  # 2.x removed HfApi.upload_large_folder
# ///
"""Upload the experiment EVIDENCE supplement of a release to the HF dataset and tag it.

  uv run scripts/release_evidence.py --exp experiments/v1.0 --version v1.0            # stage + upload + tag
  uv run scripts/release_evidence.py --exp experiments/v1.0 --version v1.0 --verify   # compare remote tag vs local
The data tag (e.g. `v1.0`) is never moved. The evidence goes under `<version>/experiments/...` (same relative paths as
in the repo) and is frozen by a new tag `<version>-evidence`:
  - every run file with per-query metrics and meta: runs/, robustness/runs/, exploratory/runs/, residual_masking_runs/;
  - all audit records: audit/, audit_ext/audit/, audit_full/audit/ (pools included);
  - fine-tuned bi-encoder weights: models/, robustness/models/, exploratory/models/ (empty training ckpt/ excluded).
Embedding caches (*.npy, cache/) are NOT released: they are derived and are rebuilt by the pipeline. Never changes
repo visibility (that is a human decision).
"""
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

REPO = "NaiveNeuron/parlir"
SUBDIRS = ["runs", "robustness/runs", "exploratory/runs", "residual_masking_runs",
           "audit", "audit_ext/audit", "audit_full/audit", "audit_ext2/audit", "audit_full2/audit",
           "audit_ext3/audit", "audit_full3/audit",
           "models", "robustness/models", "exploratory/models"]


def wanted(p: Path) -> bool:
    parts = set(p.parts)
    return p.is_file() and p.suffix != ".npy" and "cache" not in parts and "ckpt" not in parts


def local_files(exp: Path):
    for sub in SUBDIRS:
        d = exp / sub
        if d.exists():
            for p in sorted(d.rglob("*")):
                if wanted(p):
                    yield p


ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=Path, required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--verify", action="store_true")
ap.add_argument("--tag", default=None, help="evidence tag (default <version>-evidence); tags are immutable, so a later "
                                          "supplement gets a new tag, e.g. <version>-evidence-r2")
ap.add_argument("--allow-public", action="store_true", help="permit uploading after the authors made the repo public")
a = ap.parse_args()
api = HfApi()
assert a.allow_public or api.dataset_info(REPO).private, \
    "refusing: repo is not private (pass --allow-public once the authors have released it)"
tag = a.tag or f"{a.version}-evidence"
files = list(local_files(a.exp))
remote_of = {p: f"{a.version}/experiments/{p.relative_to(a.exp).as_posix()}" for p in files}

if not a.verify:
    stage = Path(".cache/hf_evidence")
    if stage.exists():
        shutil.rmtree(stage)
    for p, r in remote_of.items():
        dst = stage / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, dst)  # hard link: no copy
    print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB")
    api.upload_large_folder(repo_id=REPO, repo_type="dataset", folder_path=str(stage), num_workers=8)
    api.create_tag(REPO, tag=tag, repo_type="dataset", exist_ok=False,
                   tag_message=f"{a.version} evidence supplement: runs, per-query metrics, audit records, fine-tuned weights")
    print("tagged", tag)

# verification: every local evidence file exists at the tag with the same size
remote = {}
for e in api.list_repo_tree(REPO, repo_type="dataset", revision=tag, path_in_repo=f"{a.version}/experiments",
                            recursive=True):
    if getattr(e, "size", None) is not None and not hasattr(e, "tree_id"):
        remote[e.path] = e.size
missing = [r for p, r in remote_of.items() if r not in remote]
size_mismatch = [r for p, r in remote_of.items() if r in remote and remote[r] != p.stat().st_size]
print(f"local {len(files)} files; remote at {tag}: {len(remote)}; missing {len(missing)}; size mismatches {len(size_mismatch)}")
for r in (missing + size_mismatch)[:10]:
    print("  !", r)
raise SystemExit(1 if missing or size_mismatch else 0)
