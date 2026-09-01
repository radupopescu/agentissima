# The tool image (doc/benchmark.md §4.6).
#
# Everything the agent's tools and grading can reach lives here. The fixture
# copy is bind-mounted at run time and is the only host path visible, so a
# `find /` inside this image scans this image, not the operator's disk.
#
# Pin everything. The point of an image over a mutable VM is that six months
# from now a result set can say which `pytest`, which `grep` and which `pi`
# produced it.

FROM node:22-bookworm-slim AS node

FROM python:3.14-slim-bookworm

# Node for the external agents (`pi`, later `opencode`). Bookworm ships Node
# 18; pi needs >= 20, hence the copy from the node image rather than apt.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

# GNU coreutils/findutils/grep are the §4.6 allowlist's `ls`, `cat`, `head`,
# `tail`, `wc`, `find`, `grep`. `timeout` (coreutils) is how a command's whole
# process group gets reaped -- a host-side kill cannot reach in here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        coreutils findutils grep procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# `pyyaml` is not optional: fixtures/testrepo declares it and
# src/ledger/config/settings.py imports it. Without it T03 and T09 fail for an
# infrastructure reason and the §8 oracle gate goes red.
RUN pip install --no-cache-dir "pytest==9.1.1" "pyyaml==6.0.3"

# §4.1: pi's configuration is not frozen, but its version is, because an image
# contains one version by construction.
ARG PI_VERSION
RUN test -n "$PI_VERSION" || (echo "PI_VERSION build arg is required" >&2; exit 1) \
    && npm install -g --no-fund --no-audit "@earendil-works/pi-coding-agent@${PI_VERSION}" \
    && npm install -g --no-fund --no-audit "pi-permission-system"

# The reproducibility anchor. A locally built image has no registry digest, so
# this manifest -- hashed and recorded per session -- is what detects drift.
RUN { \
      echo '{'; \
      echo '  "python": "'"$(python -V 2>&1)"'",'; \
      echo '  "node": "'"$(node -v)"'",'; \
      echo '  "pi": "'"$(pi --version 2>/dev/null || echo unknown)"'",'; \
      echo '  "pip_freeze_sha256": "'"$(pip freeze | sha256sum | cut -d" " -f1)"'",'; \
      echo '  "dpkg_sha256": "'"$(dpkg-query -W -f='"'"'${Package} ${Version}\n'"'"' | sha256sum | cut -d" " -f1)"'"'; \
      echo '}'; \
    } > /image-manifest.json && cat /image-manifest.json

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Runs are bind-mounted here. The container runs as the host user (see
# harness/container.py), so nothing here may require writing to site-packages.
WORKDIR /runs
