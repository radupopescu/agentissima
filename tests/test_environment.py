"""Environment capture and §3.1 preconditions.

The preconditions are the benchmark's "no session on a machine that cannot
produce comparable numbers" guard, so their failures must be loud and exact.
This pins both the happy path (a full environment.json) and each abort.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from harness import environment, lmstudio
from harness.environment import PreconditionError, SessionEnvironment
from harness.version import TASK_SET_VERSION


def _resolved() -> dict:
    return {
        "model_path": "LiquidAI/LFM2.5-2.6B-MLX-8bit",
        "model_key": "lfm2.5-2.6b-mlx@8bit",
        "model_repo": "LiquidAI",
        "quant_file": "model.safetensors",
        # Unhashed by default, matching `probe_config` without --hash (§2.1).
        "quant_sha256": None,
        "on_disk_bytes": 3 * 1024**3,
        "advertised_max_context": 131072,
        "n_attention_layers": 8,
        "n_kv_heads": 8,
        "head_dim": 64,
    }


@pytest.fixture
def happy_machine(monkeypatch):
    monkeypatch.setattr(environment, "server_reachable", lambda: True)
    monkeypatch.setattr(environment, "ac_power", lambda: True)
    monkeypatch.setattr(environment, "low_power_mode", lambda: False)
    monkeypatch.setattr(environment, "total_memory_bytes", lambda: 48 * 1024**3)
    monkeypatch.setattr(environment, "free_memory_bytes", lambda: 24 * 1024**3)
    monkeypatch.setattr(environment, "harness_git_sha", lambda: "abc123")
    monkeypatch.setattr(environment, "machine_model", lambda: "Mac15,9")
    monkeypatch.setattr(environment, "chip", lambda: "Apple M3 Max")
    monkeypatch.setattr(environment, "macos_build", lambda: "24F74")
    monkeypatch.setattr(environment, "lmstudio_app_version", lambda: "0.4.21")
    monkeypatch.setattr(lmstudio, "list_loaded", lambda: [])
    monkeypatch.setattr(environment, "swap_used_bytes", lambda: 1024)
    monkeypatch.setattr(environment, "fixture_fingerprint", lambda: "fixtures")
    # discover_runtime is imported inside capture; stub the module.
    monkeypatch.setattr(
        environment, "load_resolved",
        lambda config_id, configs_dir=None: _resolved(),
    )


def test_capture_writes_a_complete_environment(happy_machine, tmp_path, monkeypatch):
    imported = types_patch(monkeypatch)
    env = environment.capture(
        "LFM-M8", 8192, "native", out_dir=tmp_path)

    assert isinstance(env, SessionEnvironment)
    assert env.path.is_file()
    payload = json.loads(env.path.read_text(encoding="utf-8"))

    assert payload["config_id"] == "LFM-M8"
    assert payload["model_path"] == "LiquidAI/LFM2.5-2.6B-MLX-8bit"
    assert payload["context_length"] == 8192
    assert payload["machine_model"] == "Mac15,9"
    assert payload["chip"] == "Apple M3 Max"
    assert payload["total_memory_bytes"] == 48 * 1024**3
    assert payload["macos_build"] == "24F74"
    assert payload["lmstudio_version"] == "0.4.21"
    assert payload["driver"] == "native"
    assert payload["driver_version"] == "1"
    assert payload["task_set_version"] == TASK_SET_VERSION
    assert payload["ac_power"] is True
    assert payload["low_power_mode"] is False
    assert payload["swap_used_bytes_start"] == 1024
    assert payload["sampling"] == {
        "temperature": 0, "top_p": 1, "seed": 1337, "max_tokens": 1024,
        "top_k": 0, "repeat_penalty": 1.0,
    }
    # backend_runtime is whatever the live process discovery reports; it must
    # exist as a shape, never as a constant.
    assert isinstance(payload["backend_runtime"], dict)
    assert "name" in payload["backend_runtime"]

    # sha256 must be the hash of exactly what was written, so results can
    # reference the environment by digest (§10.1).
    assert env.sha256 == environment.sha256_of(env.path)


def test_capture_aborts_when_a_foreign_model_is_resident(happy_machine, tmp_path, monkeypatch):
    monkeypatch.setattr(
        lmstudio, "list_loaded",
        lambda: [lmstudio.LoadedModel.from_json({
            "identifier": "x", "modelKey": "lfm2.5-2.6b@q8_0",
            "indexedModelIdentifier": "LiquidAI/LFM2.5-2.6B-GGUF/"
                                      "LFM2.5-2.6B-Q8_0.gguf",
            "contextLength": 8192})],
    )
    with pytest.raises(PreconditionError) as excinfo:
        environment.capture("LFM-M8", 8192, "native", out_dir=tmp_path)
    assert "other than the one under test" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []  # nothing written


@pytest.mark.parametrize("patch,name", [
    ("ac_power", False),
    ("low_power_mode", True),
    ("server_reachable", False),
])
def test_capture_aborts_on_each_precondition(happy_machine, tmp_path, monkeypatch, patch, name):
    monkeypatch.setattr(environment, patch, lambda: name)
    with pytest.raises(PreconditionError):
        environment.capture("LFM-M8", 8192, "native", out_dir=tmp_path)


def test_capture_aborts_above_the_advertised_context(happy_machine, tmp_path):
    """LFM-M8 maxes at 131072, so 256K is `unsupported` — the one §2.2 verdict
    that can be reached from metadata alone."""
    with pytest.raises(PreconditionError) as excinfo:
        environment.capture("LFM-M8", 262144, "native", out_dir=tmp_path)
    assert "advertised" in str(excinfo.value)


def test_capture_does_not_second_guess_whether_a_context_fits(happy_machine, tmp_path,
                                                              monkeypatch):
    """No arithmetic `oversized` verdict exists any more.

    The gate that did this excluded BON-M2 at 32K and 64K while its llama.cpp
    twin stayed admissible, which would have removed the §1 runtime comparison
    for that model. Fit is settled by the load, and what a run cost is measured
    (§5.2) — a session with a legal context must not be refused on a prediction.
    """
    monkeypatch.setattr(environment, "total_memory_bytes", lambda: 8 * 1024**3)
    session = environment.capture("LFM-M8", 131072, "native", out_dir=tmp_path)
    assert session.fields["context_length"] == 131072


def test_capture_fails_when_no_resolved_configuration_exists(happy_machine, tmp_path, monkeypatch):
    """A session must never start on an unresolvable configuration — there is
    nothing to attribute results to."""
    def missing(config_id, configs_dir=None):
        raise PreconditionError("configs/LFM-M8.resolved.yaml is missing; "
                                "run `python -m setup.probe_config`")
    monkeypatch.setattr(environment, "load_resolved", missing)
    with pytest.raises(PreconditionError) as excinfo:
        environment.capture("LFM-M8", 8192, "native", out_dir=tmp_path)
    assert "setup.probe_config" in str(excinfo.value)


def types_patch(monkeypatch):
    """Stub setup.probe_process.discover_runtime so capture needs no process."""
    import types as _types
    stub = _types.SimpleNamespace(discover_runtime=lambda: ("mlx", None))
    monkeypatch.setitem(__import__("sys").modules, "setup.probe_process", stub)
    return stub

# --- optional artefact hashing (§2.1) ---------------------------------------


def _hashed(tmp_path, monkeypatch, contents=b"weights"):
    """A resolved config whose recorded hash matches a real file on disk."""
    import hashlib

    root = tmp_path / "models"
    model_dir = root / "LiquidAI" / "LFM2.5-2.6B-MLX-8bit"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(contents)
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(root))

    resolved = _resolved()
    resolved["config_id"] = "LFM-M8"
    resolved["quant_sha256"] = hashlib.sha256(contents).hexdigest()
    return resolved, model_dir


def test_verification_is_skipped_when_setup_recorded_no_hash():
    """Both halves are optional and independent. An unhashed configuration is
    not a failure — it is a session that makes no claim about the bytes."""
    assert environment.verify_artefact_hash(_resolved()) is False


def test_verification_passes_on_unchanged_bytes(tmp_path, monkeypatch):
    resolved, _ = _hashed(tmp_path, monkeypatch)
    assert environment.verify_artefact_hash(resolved) is True


def test_verification_fails_when_the_bytes_changed(tmp_path, monkeypatch):
    """The case the hash exists for: the path still resolves and the model
    still loads, but it is no longer the artefact the results describe."""
    resolved, model_dir = _hashed(tmp_path, monkeypatch)
    (model_dir / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(PreconditionError) as excinfo:
        environment.verify_artefact_hash(resolved)
    assert "bytes have changed" in str(excinfo.value)


def test_capture_records_whether_the_check_actually_ran(happy_machine, tmp_path):
    """A result set must never imply a verification that did not happen."""
    session = environment.capture("LFM-M8", 8192, "native", out_dir=tmp_path)
    assert session.fields["quant_sha256_verified"] is False


def test_capture_can_decline_to_verify(happy_machine, tmp_path, monkeypatch):
    resolved, _ = _hashed(tmp_path, monkeypatch)
    monkeypatch.setattr(environment, "load_resolved",
                        lambda config_id, configs_dir=None: resolved)
    session = environment.capture("LFM-M8", 8192, "native", out_dir=tmp_path,
                                  verify_hash=False)
    assert session.fields["quant_sha256_verified"] is False
    assert session.fields["quant_sha256"] is not None
