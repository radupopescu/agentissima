"""Configuration probes, per doc/benchmark.md §2.1-§2.2.

Writes `configs/<id>.resolved.yaml` for every configuration in the §2 table:
identity (path, key, repo, file), hashes and byte counts from the on-disk
artefact, and attention geometry and advertised context from the model's own
config.

**Metadata only — no model is loaded**, so the whole table resolves in seconds.
An earlier revision measured KV growth here to feed an arithmetic admissibility
gate; §2.2 no longer has one. The context ceiling comes from
`advertised_max_context`, whether a pair actually fits is settled by the load
attempt, and what a run cost is measured during it (§5.2).

    python -m setup.probe_config             # the whole §2 table
    python -m setup.probe_config --only LFM-M8

Every value is recorded, never assumed. Where a field cannot be obtained the
probe fails loudly rather than writing a guess (§2.1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from harness import lmstudio
from harness.client import LMStudioClient
from harness.metrics import MemorySampler, find_inference_pid

# The §2 table, by ID. Paths are the stable identity; the key column that LM
# Studio derives is recorded per artefact, never relied on (§2.1).
CONFIGURATIONS: dict[str, str] = {
    "LFM-M8": "LiquidAI/LFM2.5-2.6B-MLX-8bit",
    "LFM-G8": "LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf",
    "LFM-GQ4": "LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf",
    "LFM-BF16": "LiquidAI/LFM2.5-2.6B-MLX-bf16",
    "BON-M2": "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
    "BON-G2": "prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0_g64.gguf",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

# §2.2 probes: two context lengths whose KV-size difference is large next to
# the ~0.03 GiB measurement spread, and small compared with the advertised
# maxima of the smallest configurable models here.
PROBE_CONTEXTS = (4096, 32768)
# Prompt lengths for the lazy-allocator slope. Differencing two long prompts
# cancels the fixed prefill scratch, which a single prompt cannot separate from
# stored KV. Kept modest: Bonsai caps at 65536 and prefills at ~40 tok/s.
PROBE_PROMPT_TOKENS = (4096, 12288)
# Context to hold fixed while the prompt length varies. Must exceed the larger
# prompt with room for the reply.
PROBE_PROMPT_CONTEXT = 16384
# Keep sampling after the warm-up request so late, lazy allocation is seen.
SAMPLE_SECONDS = 5.0

# Physical bounds on one KV element. 4-bit quantisation is 0.5 bytes and fp32
# is 4; the band is widened either side to allow per-token overhead the model
# geometry does not describe. Outside it, the footprint delta measured
# something other than KV growth — see `check_against_geometry`.
MIN_PLAUSIBLE_KV_ELEM_BYTES = 0.25
MAX_PLAUSIBLE_KV_ELEM_BYTES = 8.0

_ATTN_Q = re.compile(r"blk\.(\d+)\.attn_q\.weight")


class ProbeError(RuntimeError):
    """A configuration could not be resolved; nothing is written for it."""


def models_root() -> Path:
    """On-disk model store. LM Studio keeps it under `~/.lmstudio/models`;
    overridable so the probe can be tested and ports are possible."""
    override = os.environ.get("LMSTUDIO_MODELS_DIR")
    if override:
        return Path(override)
    default = Path.home() / ".lmstudio" / "models"
    if default.is_dir():
        return default
    raise ProbeError(
        f"could not locate the LM Studio models directory at {default}; "
        "set LMSTUDIO_MODELS_DIR to point at it"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- on-disk artefact -------------------------------------------------------


def artefact_on_disk(artefact: lmstudio.Artefact) -> Path:
    """The artefact on disk: a single file (GGUF) or a directory (MLX)."""
    root = models_root()
    path = root / artefact.path
    if artefact.format == "gguf":
        if not path.is_file():
            raise ProbeError(f"GGUF artefact missing on disk: {path}")
        return path
    if not path.is_dir():
        raise ProbeError(f"MLX artefact missing on disk: {path}")
    return path


def artefact_identity(
    artefact: lmstudio.Artefact, compute_hash: bool = False
) -> dict[str, Any]:
    """§2.1 identity and byte fields for one artefact.

    ``compute_hash`` is off by default. Hashing the weights is the entire cost
    of setup — about twenty seconds over these six artefacts, and proportional
    to their size — while `model_path` already fixes *which* artefact a result
    belongs to. The hash fixes something narrower: that the bytes behind that
    path have not changed. Worth having when comparing result sets recorded
    weeks apart, not worth paying for on every re-resolve.

    ``quant_sha256`` is ``None`` when not computed, which the session-start
    check reads as "nothing to verify" (§2.1).
    """
    on_disk = artefact_on_disk(artefact)

    if on_disk.is_file():
        quant_file = on_disk.name
        quant_sha256: Any = _sha256(on_disk) if compute_hash else None
        on_disk_bytes = on_disk.stat().st_size
    else:
        weights = sorted(on_disk.glob("*.safetensors"))
        if not weights:
            raise ProbeError(f"no *.safetensors weights in {on_disk}")
        if len(weights) == 1:
            quant_file = weights[0].name
            quant_sha256 = _sha256(weights[0]) if compute_hash else None
        else:
            quant_file = f"{len(weights)} safetensors shards"
            quant_sha256 = (
                [
                    {"file": weight.name, "sha256": _sha256(weight)}
                    for weight in weights
                ]
                if compute_hash
                else None
            )
        on_disk_bytes = sum(f.stat().st_size for f in on_disk.iterdir() if f.is_file())

    repo = artefact.path.rsplit("/", 1)[0]
    return {
        "model_path": artefact.path,
        "model_key": artefact.model_key,
        "model_repo": repo,
        "quant_file": quant_file,
        "quant_sha256": quant_sha256,
        "on_disk_bytes": on_disk_bytes,
    }


# --- attention geometry from the model's own config -------------------------


def _count_attention_blocks(tensor_names: list[str]) -> int:
    """Blocks with an `attn_q.weight` tensor.

    For LFM2 the GGUF and MLX builds agree with `layer_types` exactly: the
    attention layers sit at block indices {2, 5, 9, 13, 17, 21, 24, 27} of 30,
    and counting `blk.<i>.attn_q.weight` names recovers the same set.
    """
    return len({int(m.group(1)) for name in tensor_names if (m := _ATTN_Q.match(name))})


def _gguf_field_value(reader: Any, key: str):
    """One GGUF metadata value, preserving arrays as lists.

    Collapsing an array to its first element silently mis-reads the LFM2 family:
    `attention.head_count_kv` is per-layer there, and its first entry is a
    convolutional block's 0 (see `_kv_heads`).
    """
    field = reader.get_field(key)
    if field is None:
        return None

    def element(index: int):
        part = field.parts[index]
        if part.dtype == "uint8":
            return part.tobytes().decode("utf-8", "replace")
        return int(part[0])

    if len(field.data) > 1:
        return [element(index) for index in field.data]
    return element(field.data[0])


def _kv_heads(value: Any) -> int | None:
    """`n_kv_heads` from a scalar or a per-layer array.

    LFM2 interleaves convolutional and attention blocks, so `head_count_kv` is
    an array carrying 0 for every conv block — the same interleaving §2.2 warns
    about for the layer count. The attention blocks must all agree, or the
    single figure the §2.2 formula takes would not describe the model.
    """
    if isinstance(value, (list, tuple)):
        distinct = {int(item) for item in value if item}
        return distinct.pop() if len(distinct) == 1 else None
    return int(value) if value else None


def _gguf_geometry(path: Path) -> dict[str, Any]:
    """Attention geometry from GGUF metadata and tensor names.

    The `gguf` package reads most files but crashes in its constructor on
    quantisation types it does not know (Bonsai's Q2_0_g64 is such a file), so
    `harness.gguf_meta` is the fallback for exactly those (§2.1 documented
    behaviour, not a workaround for anything else).
    """
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        metadata: dict[str, Any] = {
            key: _gguf_field_value(reader, key) for key in reader.fields
        }
        tensor_names = [tensor.name for tensor in reader.tensors]
    except Exception:
        from harness import gguf_meta

        _, _, metadata, tensor_names = gguf_meta.parse(path)

    architecture = metadata.get("general.architecture")
    if not architecture:
        raise ProbeError(f"GGUF {path} has no general.architecture")
    prefix = f"{architecture}."

    n_attention_layers = _count_attention_blocks(tensor_names)
    if not n_attention_layers:
        raise ProbeError(f"GGUF {path} has no attention blocks in its tensor names")

    heads = metadata.get(prefix + "attention.head_count")
    kv_heads = _kv_heads(metadata.get(prefix + "attention.head_count_kv"))
    embedding = metadata.get(prefix + "embedding_length")
    head_dim = metadata.get(prefix + "attention.key_length")
    if head_dim is None and heads and embedding:
        head_dim = embedding // heads
    advertised = metadata.get(prefix + "context_length")

    if not all((heads, kv_heads, head_dim, advertised)):
        raise ProbeError(
            f"GGUF {path} is missing attention geometry metadata "
            f"(heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}, "
            f"context={advertised})"
        )
    return {
        "advertised_max_context": int(advertised),
        "n_attention_layers": n_attention_layers,
        "n_kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "geometry_source": "gguf metadata",
        "architecture": architecture,
        "tensor_count": len(tensor_names),
    }


def _config_json_geometry(config: dict[str, Any]) -> dict[str, Any]:
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list):
        n_attention_layers = sum(t == "full_attention" for t in layer_types)
    else:
        # Architectures without interleaved blocks (e.g. qwen3): every hidden
        # layer carries attention.
        n_attention_layers = config["num_hidden_layers"]

    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads") or config.get("num_kv_heads") or heads
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    advertised = config.get("max_position_embeddings")
    if not all((n_attention_layers, kv_heads, head_dim, advertised)):
        raise ProbeError(
            f"config.json is missing geometry "
            f"(layers={n_attention_layers}, kv_heads={kv_heads}, "
            f"head_dim={head_dim}, context={advertised})"
        )
    return {
        "advertised_max_context": int(advertised),
        "n_attention_layers": n_attention_layers,
        "n_kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "geometry_source": "config.json",
    }


def attention_geometry(artefact: lmstudio.Artefact) -> dict[str, Any]:
    """§2.1 geometry from the model's own config, never from LM Studio."""
    on_disk = artefact_on_disk(artefact)
    if artefact.format == "gguf":
        return _gguf_geometry(on_disk)
    config_path = on_disk / "config.json"
    if not config_path.is_file():
        raise ProbeError(f"no config.json in {on_disk}")
    return _config_json_geometry(json.loads(config_path.read_text(encoding="utf-8")))


# --- resolution and writing --------------------------------------------------


def resolve_config(config_id: str, compute_hash: bool = False) -> dict[str, Any]:
    """All §2.1 fields for one configuration, or a loud failure.

    Reads metadata only — no model is loaded, so the whole table resolves in
    seconds. §2.2 admissibility needs no measurement from here: the context
    ceiling comes from `advertised_max_context`, and whether a pair fits is
    settled by the load itself.
    """
    artefact = lmstudio.resolve(CONFIGURATIONS[config_id])
    resolved = {
        "config_id": config_id,
        **artefact_identity(artefact, compute_hash=compute_hash),
    }
    resolved.update(attention_geometry(artefact))
    return resolved


def write_resolved(config_id: str, resolved: dict[str, Any]) -> Path:
    """Write `configs/<id>.resolved.yaml`, creating the directory."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIGS_DIR / f"{config_id}.resolved.yaml"
    path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return path


def probe_all(only: str | None = None, compute_hash: bool = False) -> int:
    """Resolve every §2 configuration, or the one named by ``only``.

    Returns the number of configurations that failed to resolve; the caller's
    exit code follows it. Failures are loud and complete (Python tracebacks),
    and nothing is written for a configuration that failed (§2.1).
    """
    failures = 0
    written = 0
    for config_id, path in CONFIGURATIONS.items():
        if only is not None and config_id != only:
            continue
        try:
            resolved = resolve_config(config_id, compute_hash=compute_hash)
            target = write_resolved(config_id, resolved)
            written += 1
            print(
                f"{config_id:<8} {path:<52} "
                f"{resolved['on_disk_bytes'] / 1024**3:.2f} GiB on disk, "
                f"{resolved['n_attention_layers']} attn layers "
                f"(kv={resolved['n_kv_heads']}, d={resolved['head_dim']}), "
                f"ctx≤{resolved['advertised_max_context']}"
                + ("" if resolved["quant_sha256"] else ", unhashed")
            )
        except (ProbeError, lmstudio.LMStudioError, OSError) as exc:
            failures += 1
            print(f"{config_id}: FAILED: {exc}", file=__import__("sys").stderr)
    # Counted, not inferred from the total: with --only, or after a failure,
    # `total - failures` overstates what was actually written.
    attempted = written + failures
    print(f"\nwrote {written} of {attempted} resolved configurations "
          f"to {CONFIGS_DIR}/")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", metavar="ID",
                        help="resolve a single configuration, e.g. LFM-M8")
    parser.add_argument("--hash", action="store_true",
                        help="also SHA-256 the weights, so a later session can "
                             "verify the artefact's bytes are unchanged (§2.1). "
                             "Off by default: it is the whole cost of setup")
    args = parser.parse_args()
    raise SystemExit(
        1 if probe_all(only=args.only, compute_hash=args.hash) else 0
    )


if __name__ == "__main__":
    main()