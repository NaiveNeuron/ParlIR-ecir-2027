"""Embedding caches must never be shared between different checkpoints passed via --model-path
(regression: the key once ignored the weights, so a second fine-tuned model silently reused the
first model's embeddings)."""
import numpy as np
import pytest

from parlir import dense


def _checkpoint(tmp_path, name, payload):
    d = tmp_path / name
    d.mkdir()
    (d / "model.safetensors").write_bytes(payload)
    return str(d)


def _plant_cache(cache, monkeypatch, model_a, texts):
    """Encode `texts` with checkpoint A without loading a model: capture the key encode() looks up."""
    looked_up = []
    real_exists = type(cache).exists

    def spy(self):
        looked_up.append(self.name)
        return real_exists(self)

    monkeypatch.setattr(type(cache), "exists", spy)
    with pytest.raises(Exception):  # cache miss -> tries to load the fake checkpoint and fails
        dense.encode("me5-base-ft", texts, False, cache, model_path=model_a)
    monkeypatch.undo()
    key = next(k for k in looked_up if k.endswith(".npy"))
    cache.mkdir(exist_ok=True)
    np.save(cache / key, np.ones((len(texts), 4), dtype=np.float32))


def test_cache_hit_same_checkpoint(tmp_path, monkeypatch):
    a = _checkpoint(tmp_path, "a", b"weights-A")
    cache = tmp_path / "cache"
    _plant_cache(cache, monkeypatch, a, ["t1", "t2"])
    out = dense.encode("me5-base-ft", ["t1", "t2"], False, cache, model_path=a)
    assert out.shape == (2, 4)


def test_cache_not_shared_between_checkpoints(tmp_path, monkeypatch):
    a = _checkpoint(tmp_path, "a", b"weights-A")
    b = _checkpoint(tmp_path, "b", b"weights-B")
    cache = tmp_path / "cache"
    _plant_cache(cache, monkeypatch, a, ["t1", "t2"])
    # Same system name and texts, different weights: must miss the cache (and here fail to load the fake model)
    with pytest.raises(Exception):
        dense.encode("me5-base-ft", ["t1", "t2"], False, cache, model_path=b)


def test_retrained_checkpoint_in_same_path_misses_cache(tmp_path, monkeypatch):
    a = _checkpoint(tmp_path, "a", b"weights-A")
    cache = tmp_path / "cache"
    _plant_cache(cache, monkeypatch, a, ["t1"])
    (tmp_path / "a" / "model.safetensors").write_bytes(b"weights-A-retrained")
    with pytest.raises(Exception):
        dense.encode("me5-base-ft", ["t1"], False, cache, model_path=a)
