"""The tool container, per doc/benchmark.md §4.6.

Runs the agent's commands, and grading's, inside a pinned Linux image that can
see only the run's fixture copy. This replaces a macOS Seatbelt profile that
confined writes but permitted all reads -- a gap real `v4` data exercised in
12% of `bash` calls (findings.md).

One container per *stage*, not per run: each run gets a fresh fixture copy and
`prepared()` removes that work directory as the run ends, so a run never sees
another run's tree. A container per run would buy nothing and pay 0.3 s each
time.

The mount is the whole runs root, so anything placed there is inside the
agent's reach. That is why the change baseline is a hash map and not a second
copy of the fixture (§6.3) -- as a copy, it was read by 8.6% of the `v6` pi
runs and could have been written to.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterator

from .execution import TIMEOUT_EXIT_CODE, CommandResult, ExecutionError
from .paths import RUNS_ROOT, ensure_runs_root, sweep_stale

IMAGE_TAG = "agentissima-tools:v1"
DOCKERFILE = Path(__file__).resolve().parent.parent / "setup" / "docker" / "Dockerfile"
PI_CONFIG_DIR = Path(__file__).resolve().parent.parent / "setup" / "pi_config"

# Only the two authored files, mounted individually into the image's own
# /pi-config. Bind-mounting the whole directory would also bring
# `setup/pi_config/bin/fd` -- a cached *macOS* binary that pi's `find` tool
# looks for before PATH and cannot execute in Linux.
PI_CONFIG_FILES = ("models.json", "pi-permissions.jsonc")

# Where RUNS_ROOT is mounted inside the container.
MOUNT_POINT = PurePosixPath("/runs")

# Containment beyond the filesystem. A `grep -r /` fork bomb inside the
# container cannot now take the machine down -- findings.md records that
# failure mode happening for real on the host.
PIDS_LIMIT = 512
MEMORY_LIMIT = "2g"
CPUS = "2"

# Host-side grace beyond the in-container timeout, before the backstop fires.
BACKSTOP_GRACE_S = 5.0


def _docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise ExecutionError("docker is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(f"docker {args[0]} timed out") from exc


def docker_available() -> bool:
    try:
        return _docker("version", "--format", "{{.Server.Version}}", timeout=15).returncode == 0
    except ExecutionError:
        return False


def image_exists(tag: str = IMAGE_TAG) -> bool:
    return _docker("image", "inspect", tag, timeout=30).returncode == 0


def build_image(tag: str = IMAGE_TAG, pi_version: str | None = None) -> str:
    """Build the tool image. Never called implicitly: a stage that discovers a
    missing image fails with the build command rather than spending minutes
    building one mid-run."""
    if pi_version is None:
        pi_version = _host_pi_version()
    if not pi_version:
        raise ExecutionError(
            "cannot determine the pi version to pin; pass --pi-version explicitly"
        )
    completed = _docker(
        "build", "--platform", "linux/arm64",
        "--build-arg", f"PI_VERSION={pi_version}",
        "-t", tag, "-f", str(DOCKERFILE), str(DOCKERFILE.parent),
        timeout=1800.0,
    )
    if completed.returncode != 0:
        raise ExecutionError(f"image build failed:\n{completed.stdout}\n{completed.stderr}")
    return tag


def _host_pi_version() -> str | None:
    try:
        completed = subprocess.run(
            ["pi", "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


@dataclass
class ContainerExecutor:
    """Runs commands inside an already-started container."""

    container_id: str
    runs_root: Path
    image_tag: str
    network: str
    _provenance: dict = field(default_factory=dict)
    # Counts host-backstop firings. Should stay 0; a non-zero value in a
    # result set means the in-container timeout failed and wants investigating.
    backstop_firings: int = 0

    # --- path mapping -------------------------------------------------------

    def _container_path(self, host_path: Path) -> PurePosixPath:
        """Map a host path into the container.

        Callers above this boundary only ever hold host paths (§4.6), so this
        is the single place the two namespaces meet. A path outside the mount
        is a harness bug, never something to work around.
        """
        try:
            relative = Path(host_path).resolve().relative_to(self.runs_root.resolve())
        except ValueError as exc:
            raise ExecutionError(
                f"{host_path} is outside the container mount ({self.runs_root}); "
                "run directories must live under RUNS_ROOT"
            ) from exc
        return MOUNT_POINT / relative

    # --- execution ----------------------------------------------------------

    def run(self, command: str, *, cwd: Path, timeout_s: float) -> CommandResult:
        marker = f"agentissima-exec-{uuid.uuid4().hex[:12]}"
        # GNU `timeout`, without --foreground, puts itself and its child in a
        # new process group and signals the whole group. That is the direct
        # equivalent of the host executor's start_new_session + killpg, and the
        # only layer that can work: a host-side kill cannot reach an
        # in-container process tree. `marker` becomes $0, so every descendant
        # is matchable by `pkill -f` if the backstop is ever needed.
        argv = [
            "exec", "-w", str(self._container_path(cwd)), self.container_id,
            "timeout", "--kill-after=2", "--signal=TERM", f"{timeout_s:g}",
            "sh", "-c", command, marker,
        ]
        return self._exec(argv, timeout_s, marker)

    def spawn(
        self, argv: list[str], *, cwd: Path, timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        marker = f"agentissima-exec-{uuid.uuid4().hex[:12]}"
        env_flags: list[str] = []
        for key, value in (env or {}).items():
            env_flags += ["-e", f"{key}={value}"]
        # TERM, not KILL, and for a reason that cost a mislabelled campaign:
        # with --signal=KILL the process dies of SIGKILL and `docker exec`
        # reports 137, not `timeout`'s own 124. Every pi timeout was then
        # recorded as `server_error` (§4.8) rather than `timeout`, which also
        # fed the §4.2 degenerate detector the wrong category. Using TERM keeps
        # 124 the unambiguous timeout code and leaves 137 meaning what it
        # should -- killed by something else, an OOM against --memory most
        # likely. --kill-after still guarantees death if pi ignores TERM.
        docker_argv = [
            "exec", "-w", str(self._container_path(cwd)), *env_flags, self.container_id,
            "timeout", "--kill-after=5", "--signal=TERM", f"{timeout_s:g}",
            *argv,
        ]
        return self._exec(docker_argv, timeout_s, marker)

    def _exec(self, docker_argv: list[str], timeout_s: float, marker: str) -> CommandResult:
        try:
            process = subprocess.Popen(
                ["docker", *docker_argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            raise ExecutionError(f"could not start docker exec: {exc}") from exc

        try:
            stdout, _ = process.communicate(timeout=timeout_s + BACKSTOP_GRACE_S)
        except subprocess.TimeoutExpired:
            # Layer 2. Should never fire -- the in-container `timeout` owns
            # this. If it does, the daemon or `timeout` itself has failed, and
            # the process tree still has to be reaped from inside.
            self.backstop_firings += 1
            _docker("exec", self.container_id, "pkill", "-KILL", "-f", marker, timeout=30)
            process.kill()
            process.communicate()
            return CommandResult(TIMEOUT_EXIT_CODE, "", timed_out=True)

        if process.returncode == TIMEOUT_EXIT_CODE:
            return CommandResult(TIMEOUT_EXIT_CODE, stdout or "", timed_out=True)
        return CommandResult(process.returncode, stdout or "")

    # --- provenance ---------------------------------------------------------

    @property
    def provenance(self) -> dict:
        return {
            **self._provenance,
            "backstop_firings": self.backstop_firings,
        }


def _image_provenance(tag: str, network: str) -> dict:
    """What produced this execution environment (§3, §4.6).

    A locally built image has no registry digest, so the build-time manifest
    plus the Dockerfile hash are what make drift detectable.
    """
    import hashlib

    image_id = _docker("image", "inspect", "--format", "{{.Id}}", tag, timeout=30)
    docker_version = _docker("version", "--format", "{{.Server.Version}}", timeout=30)
    manifest = _docker("run", "--rm", "--entrypoint", "cat", tag, "/image-manifest.json", timeout=60)

    parsed = None
    if manifest.returncode == 0:
        try:
            parsed = json.loads(manifest.stdout)
        except json.JSONDecodeError:
            parsed = None

    dockerfile_sha = None
    if DOCKERFILE.is_file():
        dockerfile_sha = hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()

    return {
        "mode": "container",
        "image_tag": tag,
        "image_id": image_id.stdout.strip() or None,
        "image_manifest": parsed,
        "dockerfile_sha256": dockerfile_sha,
        "docker_version": docker_version.stdout.strip() or None,
        "network": network,
        "limits": {"pids": PIDS_LIMIT, "memory": MEMORY_LIMIT, "cpus": CPUS},
    }


@contextmanager
def container_session(
    *,
    network: str = "bridge",
    runs_root: Path | None = None,
    tag: str = IMAGE_TAG,
    extra_mounts: dict[Path, str] | None = None,
) -> Iterator[ContainerExecutor]:
    """Start one container for a stage, and remove it afterwards.

    `network`: both drivers share one policy (§4.6). `pi` must reach LM Studio
    on the host; `native`'s allowlist includes `python`, which can open
    sockets, so giving only one of them egress would introduce an asymmetry
    rather than remove one.
    """
    if not docker_available():
        raise ExecutionError(
            "docker is not available; the tool container is required (§4.6). "
            "Start OrbStack, or run with the host executor for offline work."
        )
    if not image_exists(tag):
        raise ExecutionError(
            f"tool image {tag} is missing. Build it with:\n"
            f"    .venv/bin/python -m harness.container --build"
        )

    root = ensure_runs_root(runs_root)
    sweep_stale(root)

    mounts = ["-v", f"{root.resolve()}:{MOUNT_POINT}"]
    for name in PI_CONFIG_FILES:
        source = PI_CONFIG_DIR / name
        if source.is_file():
            mounts += ["-v", f"{source.resolve()}:/pi-config/{name}:ro"]
    for host_path, container_path in (extra_mounts or {}).items():
        mounts += ["-v", f"{Path(host_path).resolve()}:{container_path}"]

    name = f"agentissima-{uuid.uuid4().hex[:8]}"
    started = _docker(
        "run", "-d", "--rm", "--name", name,
        # Files the container writes must be removable by the host, or
        # `prepared()`'s cleanup fails and .runs/ grows without bound.
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--network", network,
        "--pids-limit", str(PIDS_LIMIT),
        "--memory", MEMORY_LIMIT,
        "--cpus", CPUS,
        "--security-opt", "no-new-privileges",
        "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        *mounts,
        tag, "sleep", "infinity",
        timeout=120.0,
    )
    if started.returncode != 0:
        raise ExecutionError(f"could not start tool container:\n{started.stderr}")

    container_id = started.stdout.strip()
    try:
        yield ContainerExecutor(
            container_id=container_id,
            runs_root=root,
            image_tag=tag,
            network=network,
            _provenance=_image_provenance(tag, network),
        )
    finally:
        _docker("rm", "-f", container_id, timeout=60)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness.container")
    parser.add_argument("--build", action="store_true", help="build the tool image")
    parser.add_argument("--pi-version", default=None, help="pi version to pin (default: the host's)")
    parser.add_argument("--tag", default=IMAGE_TAG)
    args = parser.parse_args(argv)

    if args.build:
        print(f"building {args.tag} (pi {args.pi_version or _host_pi_version()}) ...")
        build_image(args.tag, args.pi_version)
        print(f"built {args.tag}")
        return 0

    print(f"docker available: {docker_available()}")
    print(f"image {args.tag} present: {image_exists(args.tag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
