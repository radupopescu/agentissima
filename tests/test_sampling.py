"""Recommended sampling defaults, read from the artefact (§9 Stage 5B).

The values are never hand-written here for the same reason expected values are
never hand-written in the fixtures: a table someone typed drifts from the
artefact it describes and nobody notices. What these tests guard is that the
reading is faithful, that an artefact staying silent is not filled in with a
guess, and that silence is loud rather than empty.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import sampling  # noqa: E402
from harness.client import DEFAULT_EXTRA_BODY, DEFAULT_SAMPLING  # noqa: E402
from test_gguf_meta import _F32, _STR, _U32, _s, _string  # noqa: E402


def _gguf(path: Path, *, temp: float | None = None, top_k: int | None = None) -> None:
    kv = [("general.architecture", (_STR, _string("lfm2")))]
    if temp is not None:
        kv.append(("general.sampling.temp", (_F32, struct.pack("<f", temp))))
    if top_k is not None:
        kv.append(("general.sampling.top_k", (_U32, struct.pack("<I", top_k))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_s(kv, ["token_embd.weight"], False))


def _mlx(directory: Path, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "generation_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )


# --- reading what the artefact states ----------------------------------------


def test_gguf_sampling_comes_from_the_header(tmp_path):
    _gguf(tmp_path / "vendor/model.gguf", temp=0.1, top_k=50)
    resolved = sampling.resolve("CFG", "vendor/model.gguf", models_root=tmp_path)
    assert resolved.stated == {"temperature": 0.1, "top_k": 50}
    assert resolved.source == "vendor/model.gguf"


def test_mlx_sampling_comes_from_generation_config(tmp_path):
    _mlx(tmp_path / "vendor/model-mlx",
         {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1})
    resolved = sampling.resolve("CFG", "vendor/model-mlx", models_root=tmp_path)
    assert resolved.stated == {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1}
    assert resolved.source.endswith("generation_config.json")


def test_the_huggingface_penalty_name_is_translated(tmp_path):
    """HuggingFace writes `repetition_penalty`; the request parameter LM Studio
    and llama.cpp take is `repeat_penalty`."""
    _mlx(tmp_path / "m", {"repetition_penalty": 1.1})
    resolved = sampling.resolve("CFG", "m", models_root=tmp_path)
    assert "repeat_penalty" in resolved.stated
    assert "repetition_penalty" not in resolved.stated


def test_two_quantisations_of_one_model_can_disagree(tmp_path):
    """The reason this is resolved per configuration rather than once per
    model: the real Q8_0 and QAD-Q4_0 artefacts recommend different values."""
    _gguf(tmp_path / "a.gguf", temp=0.1, top_k=50)
    _gguf(tmp_path / "b.gguf", temp=0.2, top_k=80)
    a = sampling.resolve("A", "a.gguf", models_root=tmp_path)
    b = sampling.resolve("B", "b.gguf", models_root=tmp_path)
    assert a.stated != b.stated


# --- what the artefact does not say is not invented ---------------------------


def test_unstated_parameters_keep_their_controlled_value(tmp_path):
    _gguf(tmp_path / "m.gguf", temp=0.1)
    resolved = sampling.resolve("CFG", "m.gguf", models_root=tmp_path)
    assert resolved.sampling["top_p"] == DEFAULT_SAMPLING["top_p"]
    assert resolved.sampling["top_k"] == DEFAULT_EXTRA_BODY["top_k"]
    assert resolved.sampling["repeat_penalty"] == DEFAULT_EXTRA_BODY["repeat_penalty"]


def test_the_budget_and_the_seed_are_pinned(tmp_path):
    """`max_tokens` is a budget, not a recommendation, and a changed seed would
    make the pass less reproducible than the greedy run it is compared with."""
    _mlx(tmp_path / "m", {"temperature": 0.7, "max_tokens": 4096, "seed": 99})
    resolved = sampling.resolve("CFG", "m", models_root=tmp_path)
    assert resolved.sampling["max_tokens"] == DEFAULT_SAMPLING["max_tokens"]
    assert resolved.sampling["seed"] == DEFAULT_SAMPLING["seed"]


def test_only_the_stated_parameters_count_as_changed(tmp_path):
    _gguf(tmp_path / "m.gguf", temp=0.1, top_k=50)
    resolved = sampling.resolve("CFG", "m.gguf", models_root=tmp_path)
    assert set(resolved.changed_from_controlled) == {"temperature", "top_k"}
    assert resolved.changed_from_controlled["temperature"] == (0, 0.1)


# --- silence is loud ----------------------------------------------------------


def test_an_artefact_stating_nothing_raises(tmp_path):
    _gguf(tmp_path / "m.gguf")
    with pytest.raises(sampling.SamplingUnavailableError):
        sampling.resolve("CFG", "m.gguf", models_root=tmp_path)


def test_a_missing_generation_config_raises(tmp_path):
    (tmp_path / "m").mkdir()
    with pytest.raises(sampling.SamplingUnavailableError):
        sampling.resolve("CFG", "m", models_root=tmp_path)


def test_a_missing_artefact_raises(tmp_path):
    with pytest.raises(sampling.SamplingUnavailableError):
        sampling.resolve("CFG", "nope.gguf", models_root=tmp_path)


# --- the shape the client and the record want ---------------------------------


def test_the_request_is_split_the_way_the_client_takes_it(tmp_path):
    """`top_k` and `repeat_penalty` are not OpenAI-standard and travel in
    `extra_body` (§4.2)."""
    _mlx(tmp_path / "m", {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1})
    resolved = sampling.resolve("CFG", "m", models_root=tmp_path)
    assert resolved.request_sampling["temperature"] == 0.1
    assert "top_k" not in resolved.request_sampling
    assert resolved.request_extra_body == {"top_k": 50, "repeat_penalty": 1.1}


def test_the_record_carries_its_provenance(tmp_path):
    """A result must never imply a recommendation the artefact did not make."""
    _gguf(tmp_path / "vendor/m.gguf", temp=0.1)
    record = sampling.resolve("CFG", "vendor/m.gguf", models_root=tmp_path).as_record()
    assert record["source"] == "vendor/m.gguf"
    assert record["stated"] == {"temperature": 0.1}
    assert record["resolved"]["temperature"] == 0.1
    assert record["resolved"]["top_k"] == DEFAULT_EXTRA_BODY["top_k"]
