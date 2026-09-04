"""Environment capture and session preconditions (§3, §3.1).

`capture()` asserts every §3.1 precondition and writes
`results/<session>/environment.json` for one `(configuration, context)`
session. A failed precondition **aborts** — it is never downgraded to a
warning. The returned session sha256 is what result records reference
(§10.1), so the environment a result was taken under is always retrievable.
"""

from __future__ import annotations

import hashlib
import json
import plistlib
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import lmstudio
from .admissibility import UNSUPPORTED, classify_declared
from .client import DEFAULT_EXTRA_BODY, DEFAULT_SAMPLING
from .driver_native import DRIVER_VERSION as NATIVE_DRIVER_VERSION
from .driver_pi import DRIVER_VERSION as PI_DRIVER_VERSION
from .driver_pi import ISOLATION_FLAGS as PI_ISOLATION_FLAGS
from .driver_pi import pi_version
from .metrics import swap_used_bytes
from .prompt import prompt_sha256
from .version import TASK_SET_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
FIXTURES_DIR = REPO_ROOT / "fixtures"
SERVER_ENDPOINT = "http://localhost:1234/v1/models"

# "native-compact" (Stage 5B) is still NativeDriver underneath, just with a
# different history_mode -- same driver version.
_DRIVER_VERSIONS = {
    "native": NATIVE_DRIVER_VERSION,
    "native-compact": NATIVE_DRIVER_VERSION,
    "native-sampled": NATIVE_DRIVER_VERSION,
    "pi": PI_DRIVER_VERSION,
    "pi-sampled": PI_DRIVER_VERSION,
}


def _pi_provenance(executor: Any = None) -> dict[str, Any]:
    """What is recorded about `pi` rather than controlled (§4.1).

    Everything here is a drift vector we deliberately do not freeze, so it must
    at least be visible after the fact. `thinking_level` is constant: pi's
    `getSupportedThinkingLevels` returns `["off"]` for any model whose catalogue
    entry does not declare `reasoning`, and `setup/pi_config/models.json`
    declares only `{"id": "bench"}` -- so `--thinking` is inert here. LFM2.5
    still emits `reasoning_content`; that is the model's own behaviour, not
    something pi requested.
    """
    return {
        "version": pi_version(executor),
        "isolation_flags": list(PI_ISOLATION_FLAGS),
        "system_prompt": "pi default (not ours; not hashed)",
        "thinking_level": "off",
        # Not disabled: pi loads the fixture's AGENTS.md from the working
        # directory root into its system prompt. W07/T07 are therefore not
        # comparable with `native`, which exposes it only on a tool read.
        "context_files_discovered": True,
    }


class PreconditionError(RuntimeError):
    """A §3.1 precondition failed; the session must not start."""


# --- machine and OS facts ---------------------------------------------------


def _sysctl_text(name: str) -> str:
    completed = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, timeout=10
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def machine_model() -> str:
    return _sysctl_text("hw.model")


def chip() -> str:
    return _sysctl_text("machdep.cpu.brand_string")


def total_memory_bytes() -> int:
    value = _sysctl_text("hw.memsize")
    return int(value) if value.isdigit() else 0


def macos_build() -> str:
    completed = subprocess.run(
        ["sw_vers", "-buildVersion"], capture_output=True, text=True, timeout=10
    )
    return completed.stdout.strip()


def ac_power() -> bool:
    """`pmset -g batt` reports the power source. Desktops without a battery
    still report `AC Power` in this output."""
    completed = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10
    )
    return "AC Power" in completed.stdout


def low_power_mode() -> bool:
    completed = subprocess.run(
        ["pmset", "-g"], capture_output=True, text=True, timeout=10
    )
    match = re.search(r"lowpowermode\s+(\d)", completed.stdout)
    return match is not None and match.group(1) == "1"


def free_memory_bytes() -> int | None:
    """Pages the system can hand over without swapping: free + inactive +
    speculative from vm_stat, scaled by page size.

    `free` alone is misleadingly small on macOS, where the memory manager
    recycles inactive pages first; this is the de facto available figure the
    §3.1 floor is checked against.
    """
    completed = subprocess.run(
        ["vm_stat"], capture_output=True, text=True, timeout=10
    )
    values: dict[str, int] = {}
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"{key}:\s+(\d+)\.", completed.stdout)
        if match:
            values[key] = int(match.group(1))
    size_match = re.search(r"page size of (\d+) bytes", completed.stdout)
    page_size = int(size_match.group(1)) if size_match else 4096
    total_pages = sum(values.values())
    return total_pages * page_size if values else None


def lmstudio_app_version() -> str | None:
    candidates = [
        Path("/Applications/LM Studio.app") / "Contents" / "Info.plist",
        Path.home() / "Applications" / "LM Studio.app" / "Contents" / "Info.plist",
    ]
    for plist in candidates:
        if not plist.is_file():
            continue
        try:
            with open(plist, "rb") as handle:
                payload = plistlib.load(handle)
            return payload.get("CFBundleShortVersionString")
        except (OSError, plistlib.InvalidFileException):
            return None
    return None


def harness_git_sha() -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    value = completed.stdout.strip()
    return value or None


def fixture_fingerprint() -> str:
    """Content hash of the fixture trees, in place of a git revision — the
    fixtures are generated, not a repository. Any byte change to a fixture or
    its expected values moves the fingerprint (§11)."""
    digest = hashlib.sha256()
    for root_name in ("workspace", "testrepo", "expected"):
        root = FIXTURES_DIR / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(FIXTURES_DIR))
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update(f"{relative} {file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def server_reachable() -> bool:
    try:
        with urllib.request.urlopen(SERVER_ENDPOINT, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


# --- resolved configuration -------------------------------------------------


def load_resolved(
    config_id: str, configs_dir: Path | None = None
) -> dict[str, Any]:
    """The §2.1 resolved fields for a configuration, written by the setup probe."""
    root = configs_dir or CONFIGS_DIR
    path = root / f"{config_id}.resolved.yaml"
    if not path.is_file():
        raise PreconditionError(
            f"{path} is missing; run `python -m setup.probe_config` before "
            "starting a session"
        )
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Geometry is recorded for the §2.2 cross-check but gates nothing, so it is
    # not required here; `advertised_max_context` is, because the one check that
    # can be made before a load is made from it.
    required = ("on_disk_bytes", "advertised_max_context", "model_path")
    missing = [name for name in required if name not in resolved]
    if missing:
        raise PreconditionError(f"{path} is missing {missing}; regenerate it")
    return resolved


# --- session capture --------------------------------------------------------


def sha256_of(path: str | Path) -> str:
    """Digest of the environment file exactly as written; result records
    reference the environment by this (§10.1)."""
    payload = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionEnvironment:
    """One captured session: the JSON path, its sha256, and the fields."""

    path: Path
    sha256: str
    fields: Mapping[str, Any]


def _assert_resident_state(model_path: str) -> None:
    """No model other than the one under test may be resident (§3.1)."""
    resident = lmstudio.list_loaded()
    others = [m.path for m in resident if m.path != model_path]
    if others:
        raise PreconditionError(
            "models other than the one under test are loaded: "
            + ", ".join(sorted(others))
        )


def verify_artefact_hash(resolved: Mapping[str, Any]) -> bool:
    """Re-hash the weights and compare against what setup recorded (§2.1).

    Returns whether the check ran. It is skipped, not failed, when the resolved
    configuration carries no hash — `python -m setup.probe_config` records one
    only under `--hash`, because hashing is the whole cost of setup.

    What this catches that path resolution cannot: the bytes behind a correct
    path having *changed* — a silent re-download, corruption, a same-name
    replacement — which would leave results attributed to a row of §2 that no
    longer describes what ran. One of the §2 artefacts was already swapped once
    during reconnaissance.
    """
    recorded = resolved.get("quant_sha256")
    if not recorded:
        return False

    from setup.probe_config import artefact_on_disk, models_root  # noqa: PLC0415

    root = models_root()
    on_disk = root / resolved["model_path"]
    if on_disk.is_file():
        actual: Any = _sha256_file(on_disk)
    else:
        weights = sorted(on_disk.glob("*.safetensors"))
        if len(weights) == 1:
            actual = _sha256_file(weights[0])
        else:
            actual = [
                {"file": w.name, "sha256": _sha256_file(w)} for w in weights
            ]

    if actual != recorded:
        raise PreconditionError(
            f"{resolved['model_path']} does not match the artefact recorded at "
            f"setup: its bytes have changed. Re-run "
            f"`python -m setup.probe_config --hash --only "
            f"{resolved.get('config_id', '<id>')}` if the change was intended; "
            f"results from before and after must not be pooled"
        )
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(
    config_id: str,
    context_length: int,
    driver: str,
    extra_rules: str | None = None,
    out_dir: str | Path = "results",
    configs_dir: str | Path | None = None,
    verify_hash: bool = True,
    session_id: str | None = None,
    executor: Any = None,
    sampling: dict | None = None,
) -> SessionEnvironment:
    """Assert §3.1 preconditions and write results/<session>/environment.json.

    Raises `PreconditionError` on the first failure — sessions never start on
    a machine that cannot produce comparable numbers.

    `session_id` names the session directory verbatim, for a caller that needs
    a deterministic, reusable location (a stage runner resuming into the same
    directory across repeated invocations). Left unset, a session is named
    `<config_id>-<context_length>-<timestamp>`, unique per call.
    """
    resolved = load_resolved(config_id, configs_dir and Path(configs_dir))

    if not server_reachable():
        raise PreconditionError(
            f"LM Studio server not reachable at {SERVER_ENDPOINT}"
        )
    if not ac_power():
        raise PreconditionError("machine is not on AC power")
    if low_power_mode():
        raise PreconditionError("Low Power Mode is enabled")
    _assert_resident_state(resolved["model_path"])

    # Both halves are optional and independent: setup may not have hashed, and
    # a caller may decline the re-hash. Whether it actually ran is recorded, so
    # a result set never implies a check that did not happen.
    hash_verified = verify_artefact_hash(resolved) if verify_hash else False

    total_memory = total_memory_bytes()
    # The free half of §2.2. The other half is the load itself, which has
    # already happened by the time this runs — a load that could not hold its
    # KV cache fails there, on the backends that commit it at load time. On
    # MLX, which allocates lazily, nothing can be decided in advance and
    # peak_memory_bytes/swap_flag record what the run actually cost.
    if classify_declared(resolved["advertised_max_context"], context_length) == (
        UNSUPPORTED
    ):
        raise PreconditionError(
            f"context {context_length} exceeds {config_id}'s advertised "
            f"maximum of {resolved['advertised_max_context']}"
        )

    from setup.probe_process import discover_runtime

    backend_name, backend_version = discover_runtime()
    free_bytes = free_memory_bytes()

    fields: dict[str, Any] = {
        "machine_model": machine_model(),
        "chip": chip(),
        "total_memory_bytes": total_memory,
        "macos_build": macos_build(),
        "lmstudio_version": lmstudio_app_version(),
        "backend_runtime": {"name": backend_name, "version": backend_version},
        "config_id": config_id,
        "model_path": resolved["model_path"],
        "model_key": resolved["model_key"],
        "model_repo": resolved["model_repo"],
        "quant_file": resolved["quant_file"],
        "quant_sha256": resolved["quant_sha256"],
        "quant_sha256_verified": hash_verified,
        "context_length": context_length,
        # §4.2's controlled set, unless a stage is deliberately running
        # something else — Stage 5B's sampling pass. Recorded as it was sent,
        # so a session can never imply sampling it did not use.
        "sampling": sampling or {**DEFAULT_SAMPLING, **DEFAULT_EXTRA_BODY},
        "harness_git_sha": harness_git_sha(),
        "driver": driver,
        "driver_version": _DRIVER_VERSIONS.get(driver),
        # `pi` brings its own system prompt and ours is never sent, so hashing
        # `harness/prompt.py` for a pi session records a prompt the model did
        # not receive -- wrong rather than merely absent. The task's
        # `extra_rules` *is* delivered, via `--append-system-prompt` (§4.3).
        "system_prompt_sha256": None if driver == "pi" else prompt_sha256(extra_rules),
        "pi": _pi_provenance(executor) if driver == "pi" else None,
        # Where the agent's tools and grading ran (§4.6). Everything else in
        # this file measures the *host*, because the model runs in LM Studio on
        # macOS; this block is the one part describing the container.
        "execution": executor.provenance if executor is not None else {"mode": "host"},
        "fixture_git_sha": fixture_fingerprint(),
        "task_set_version": TASK_SET_VERSION,
        "ac_power": True,
        "low_power_mode": False,
        "free_memory_bytes": free_bytes,
        "swap_used_bytes_start": swap_used_bytes(),
    }

    session = session_id or f"{config_id}-{context_length}-{time.strftime('%Y%m%dT%H%M%S')}"
    session_dir = Path(out_dir) / session
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "environment.json"
    payload = json.dumps(fields, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")

    sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return SessionEnvironment(path=path, sha256=sha256, fields=fields)