"""The fixed system prompt, per doc/benchmark.md §4.3.

One block, identical for every model, configuration and suite. No per-model
adaptation is permitted. Two tasks append a fixed `extra_rules` block.
"""

from __future__ import annotations

import hashlib

SYSTEM_PROMPT = """You are an agent working inside a directory. You can inspect and change files only
through the tools provided to you.

Rules:
- Use the tools to gather information. Do not guess the contents of a file.
- All paths are relative to the working directory root.
- When the task is complete, reply with a final message containing the answer and
  call no tool in that message.
- Be concise."""


def assemble(extra_rules: str | None = None) -> str:
    if not extra_rules:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n" + extra_rules


def prompt_sha256(extra_rules: str | None = None) -> str:
    return hashlib.sha256(assemble(extra_rules).encode("utf-8")).hexdigest()
