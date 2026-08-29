"""Model identity (§2.1).

LM Studio's `modelKey` is derived from the set of models installed, so it moves
under a benchmark that must attribute results to a fixed artefact. These tests
pin the harness to `path` and pin the failure modes to loud errors.
"""

from __future__ import annotations

import pytest

from harness import lmstudio
from harness.lmstudio import Artefact, LMStudioError

# Two quantisations of one model, plus an unrelated one. The GGUF pair carry
# `@quant` suffixes only because both are installed; the MLX build does not.
CATALOGUE = [
    {
        "modelKey": "lfm2.5-2.6b@q8_0",
        "indexedModelIdentifier": "LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf",
        "sizeBytes": 2874779648,
        "architecture": "lfm2",
        "quantization": {"name": "Q8_0", "bits": 8},
        "maxContextLength": 131072,
    },
    {
        "modelKey": "lfm2.5-2.6b@q4_0",
        "indexedModelIdentifier": "LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf",
        "sizeBytes": 1593894944,
        "architecture": "lfm2",
        "quantization": {"name": "Q4_0", "bits": 4},
        "maxContextLength": 128000,
    },
    {
        "modelKey": "ternary-bonsai-8b-mlx",
        "indexedModelIdentifier": "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
        "sizeBytes": 2315245281,
        "architecture": "qwen3",
        "quantization": {"name": "2bit", "bits": 2},
        "maxContextLength": 65536,
    },
]


@pytest.fixture
def catalogue(monkeypatch):
    monkeypatch.setattr(
        lmstudio,
        "list_downloaded",
        lambda: [Artefact.from_json(item) for item in CATALOGUE],
    )


# --- resolution --------------------------------------------------------------


def test_a_path_resolves_to_the_key_lms_accepts(catalogue):
    artefact = lmstudio.resolve("LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf")
    assert artefact.model_key == "lfm2.5-2.6b@q8_0"
    assert artefact.quantization == "Q8_0"
    assert artefact.max_context_length == 131072


def test_an_exact_key_still_resolves(catalogue):
    """Keys remain usable; they are just not what results are attributed to."""
    assert lmstudio.resolve("lfm2.5-2.6b@q4_0").path.endswith("QAD-Q4_0.gguf")


def test_a_partial_key_is_refused_rather_than_matched(catalogue):
    """`lms load lfm2.5-2.6b` matches four artefacts and, under --yes, loads the
    first. Substring matching must never reach the CLI."""
    with pytest.raises(LMStudioError) as excinfo:
        lmstudio.resolve("lfm2.5-2.6b")
    assert "no downloaded model matches" in str(excinfo.value)


def test_an_unknown_model_lists_what_is_available(catalogue):
    with pytest.raises(LMStudioError) as excinfo:
        lmstudio.resolve("LiquidAI/Not-Installed")
    assert "prism-ml/Ternary-Bonsai-8B-mlx-2bit" in str(excinfo.value)


def test_a_duplicated_key_is_refused_not_silently_picked(monkeypatch):
    """Defensive: if LM Studio ever reports two artefacts under one key, that is
    exactly the case where guessing corrupts a result set."""
    clashing = [
        {"modelKey": "dupe", "indexedModelIdentifier": "a/one.gguf"},
        {"modelKey": "dupe", "indexedModelIdentifier": "b/two.gguf"},
    ]
    monkeypatch.setattr(
        lmstudio,
        "list_downloaded",
        lambda: [Artefact.from_json(item) for item in clashing],
    )
    with pytest.raises(LMStudioError) as excinfo:
        lmstudio.resolve("dupe")
    assert "ambiguous" in str(excinfo.value)
    assert "a/one.gguf" in str(excinfo.value)


# --- load verification -------------------------------------------------------


def _completed(returncode=0, stdout="", stderr=""):
    from subprocess import CompletedProcess

    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_load_verifies_the_resident_artefact_by_path(catalogue, monkeypatch):
    monkeypatch.setattr(lmstudio, "_run", lambda *a, **k: _completed())
    monkeypatch.setattr(
        lmstudio,
        "list_loaded",
        lambda: [
            lmstudio.LoadedModel.from_json(
                {
                    "identifier": "bench",
                    "modelKey": "lfm2.5-2.6b@q8_0",
                    "indexedModelIdentifier": CATALOGUE[0]["indexedModelIdentifier"],
                    "contextLength": 8192,
                }
            )
        ],
    )
    model = lmstudio.load(CATALOGUE[0]["indexedModelIdentifier"], context_length=8192)
    assert model.path == CATALOGUE[0]["indexedModelIdentifier"]
    assert model.context_length == 8192


def test_load_fails_when_a_different_artefact_became_resident(catalogue, monkeypatch):
    """The failure `--yes` produces: the command succeeds, but the wrong model
    is loaded. Silently proceeding would attribute results to the wrong row."""
    monkeypatch.setattr(lmstudio, "_run", lambda *a, **k: _completed())
    monkeypatch.setattr(
        lmstudio,
        "list_loaded",
        lambda: [
            lmstudio.LoadedModel.from_json(
                {
                    "identifier": "bench",
                    "modelKey": "ternary-bonsai-8b-mlx",
                    "indexedModelIdentifier": "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
                    "contextLength": 8192,
                }
            )
        ],
    )
    with pytest.raises(LMStudioError) as excinfo:
        lmstudio.load(CATALOGUE[0]["indexedModelIdentifier"])
    assert "reported loaded, but lms ps shows" in str(excinfo.value)
    assert "Ternary-Bonsai" in str(excinfo.value)


def test_load_of_an_unknown_model_never_shells_out(catalogue, monkeypatch):
    """Resolution happens before the CLI is touched, so a typo cannot load
    whatever happens to match it."""
    def explode(*args, **kwargs):
        raise AssertionError("lms must not be invoked for an unresolvable model")

    monkeypatch.setattr(lmstudio, "_run", explode)
    with pytest.raises(LMStudioError):
        lmstudio.load("lfm2.5-2.6b")


# --- runtime engines ----------------------------------------------------------

# Real `lms runtime ls` output (0.4.21+2), trimmed to two engines each. There is
# no `--json` form, so this table is parsed; the checkmark and column spacing
# are the actual output, not a guess.
_RUNTIME_LS_OUTPUT = (
    "LLM ENGINE                                        SELECTED    MODEL FORMAT\n"
    "llama.cpp-mac-arm64-apple-metal-advsimd@2.29.1       ✓            GGUF    \n"
    "llama.cpp-mac-arm64-apple-metal-advsimd@2.28.2                    GGUF    \n"
    "mlx-llm-mac-arm64-apple-metal-advsimd@1.11.0         ✓            MLX     \n"
    "mlx-llm-mac-arm64-apple-metal-advsimd@1.10.1                      MLX     \n"
)


def test_runtime_engines_parses_the_real_table(monkeypatch):
    monkeypatch.setattr(
        lmstudio, "_run", lambda *a, **k: _completed(stdout=_RUNTIME_LS_OUTPUT)
    )
    engines = lmstudio.runtime_engines()
    assert len(engines) == 4
    selected = [e for e in engines if e.selected]
    assert {e.model_format for e in selected} == {"GGUF", "MLX"}
    gguf = next(e for e in selected if e.model_format == "GGUF")
    assert gguf.name == "llama.cpp-mac-arm64-apple-metal-advsimd"
    assert gguf.version == "2.29.1"
    mlx = next(e for e in selected if e.model_format == "MLX")
    assert mlx.version == "1.11.0"
