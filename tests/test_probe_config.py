"""The setup probe: geometry extraction and resolved-configuration writing.

Exercised against artefacts shaped like the real ones — an LFM2-style
`config.json` with interleaved layers, and a GGUF the `gguf` package itself
wrote, so the library path and the fallback path are both covered.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from harness import lmstudio
from harness.lmstudio import Artefact
from setup import probe_config

LFM2_MLX = {
    "modelKey": "lfm2.5-2.6b-mlx@8bit",
    "indexedModelIdentifier": "LiquidAI/LFM2.5-2.6B-MLX-8bit",
    "format": "safetensors",
    "sizeBytes": 1000,
    "architecture": "lfm2",
    "quantization": {"name": "8bit", "bits": 8},
    "maxContextLength": 131072,
}


def _lfm2_config_json() -> dict:
    layer_types: list[str] = []
    for index in range(30):
        layer_types.append("full_attention" if index in {2, 5, 9, 13, 17, 21, 24, 27}
                           else "conv")
    return {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 30,
        "layer_types": layer_types,
        "max_position_embeddings": 131072,
    }


@pytest.fixture
def mlx_tree(tmp_path, monkeypatch):
    """An LFM2-style MLX artefact on disk, and lms reporting only it."""
    root = tmp_path / "models"
    model_dir = root / "LiquidAI" / "LFM2.5-2.6B-MLX-8bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(_lfm2_config_json()), encoding="utf-8")
    weights = b"\x00" * 64  # placeholder weights
    (model_dir / "model.safetensors").write_bytes(weights)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(root))
    monkeypatch.setattr(
        lmstudio, "list_downloaded", lambda: [Artefact.from_json(LFM2_MLX)]
    )
    return root


def _gguf_attention_names():
    names = ["token_embd.weight"]
    for index in range(30):
        if index in {2, 5, 9, 13, 17, 21, 24, 27}:
            names.append(f"blk.{index}.attn_q.weight")
            names.append(f"blk.{index}.attn_k.weight")
    return names


def _write_lfm2_gguf(path: Path) -> None:
    writer = GGUFWriter(str(path), "lfm2")
    writer.add_block_count(30)
    writer.add_embedding_length(2048)
    writer.add_context_length(131072)
    writer.add_head_count(32)
    writer.add_head_count_kv(8)
    for name in _gguf_attention_names():
        if "attn_" in name:
            writer.add_tensor(name, np.ones((4,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def gguf_tree(tmp_path, monkeypatch):
    root = tmp_path / "models"
    file_path = root / "LiquidAI" / "LFM2.5-2.6B-GGUF" / "LFM2.5-2.6B-Q8_0.gguf"
    file_path.parent.mkdir(parents=True)
    _write_lfm2_gguf(file_path)
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(root))
    artefact = {
        "modelKey": "lfm2.5-2.6b@q8_0",
        "indexedModelIdentifier": "LiquidAI/LFM2.5-2.6B-GGUF/"
                                  "LFM2.5-2.6B-Q8_0.gguf",
        "format": "gguf",
        "sizeBytes": file_path.stat().st_size,
        "architecture": "lfm2",
        "quantization": {"name": "Q8_0", "bits": 8},
        "maxContextLength": 131072,
    }
    monkeypatch.setattr(
        lmstudio, "list_downloaded", lambda: [Artefact.from_json(artefact)]
    )
    return root


# --- MLX geometry ------------------------------------------------------------


def test_mlx_geometry_counts_full_attention_layers(mlx_tree):
    artefact = lmstudio.resolve("LiquidAI/LFM2.5-2.6B-MLX-8bit")
    geometry = probe_config.attention_geometry(artefact)
    assert geometry["n_attention_layers"] == 8  # not num_hidden_layers 30
    assert geometry["n_kv_heads"] == 8
    assert geometry["head_dim"] == 64  # 2048 / 32
    assert geometry["advertised_max_context"] == 131072
    assert geometry["geometry_source"] == "config.json"


def test_mlx_identity_hashes_the_weights_file_when_asked(mlx_tree):
    artefact = lmstudio.resolve("LiquidAI/LFM2.5-2.6B-MLX-8bit")
    identity = probe_config.artefact_identity(artefact, compute_hash=True)
    assert identity["quant_file"] == "model.safetensors"
    assert identity["quant_sha256"] == hashlib.sha256(b"\x00" * 64).hexdigest()


def test_identity_skips_the_hash_by_default(mlx_tree):
    """Hashing is the whole cost of setup, and `model_path` already fixes which
    artefact a result belongs to (§2.1)."""
    artefact = lmstudio.resolve("LiquidAI/LFM2.5-2.6B-MLX-8bit")
    identity = probe_config.artefact_identity(artefact)
    assert identity["quant_sha256"] is None
    assert identity["quant_file"] == "model.safetensors"  # identity still recorded
    # on_disk_bytes covers every file in the directory, not just the weights.
    assert identity["on_disk_bytes"] == 64 + len("{}") + len(
        json.dumps(_lfm2_config_json()).encode("utf-8"))


# --- GGUF geometry, via the `gguf` package ----------------------------------


def test_gguf_geometry_via_the_gguf_package(gguf_tree):
    artefact = lmstudio.resolve(
        "LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf")
    geometry = probe_config.attention_geometry(artefact)
    assert geometry["n_attention_layers"] == 8
    assert geometry["n_kv_heads"] == 8
    assert geometry["head_dim"] == 64
    assert geometry["advertised_max_context"] == 131072
    assert geometry["geometry_source"] == "gguf metadata"


# --- GGUF geometry, fallback path -----------------------------------------


def test_gguf_geometry_falls_back_when_the_library_cannot_open(gguf_tree, monkeypatch):
    """Simulate a quantisation the `gguf` package rejects (Bonsai's type 42):
    the geometry must still come out of the minimal reader."""
    class FailingReader:  # noqa: D416
        def __init__(self, *args, **kwargs):
            raise ValueError("unknown GGMLQuantizationType")

    fake_gguf = types.SimpleNamespace(GGUFReader=FailingReader)
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    artefact = lmstudio.resolve(
        "LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf")
    geometry = probe_config.attention_geometry(artefact)
    assert geometry["n_attention_layers"] == 8
    assert geometry["architecture"] == "lfm2"


# --- resolved YAML ---------------------------------------------------------


def test_resolve_config_writes_a_resolved_yaml(mlx_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(probe_config, "CONFIGS_DIR", tmp_path / "configs")
    resolved = probe_config.resolve_config("LFM-M8")
    assert resolved["config_id"] == "LFM-M8"
    assert resolved["model_path"] == "LiquidAI/LFM2.5-2.6B-MLX-8bit"
    assert resolved["n_attention_layers"] == 8

    target = probe_config.write_resolved("LFM-M8", resolved)
    import yaml

    reread = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert reread["model_key"] == "lfm2.5-2.6b-mlx@8bit"
    assert reread["on_disk_bytes"] == resolved["on_disk_bytes"]
    assert reread["advertised_max_context"] == 131072