"""Tool schemas and strict dispatch, per benchmark.md §4.4 and §4.5.

Dispatch never repairs. Unparseable arguments, unknown tool names, missing
required arguments and wrongly typed arguments are each counted as an invalid
call and returned to the model as a tool result carrying the parse error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .sandbox import Sandbox, truncate

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the working directory root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file, creating or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the working directory root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the entries of a directory, one per line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": 'Directory path relative to the root. Use "." for the root.',
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search file contents for a regular expression. "
                "Returns matching lines as path:line:text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regular expression."},
                    "path": {
                        "type": "string",
                        "description": "Directory to search. Defaults to the root.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the working directory. "
                "Returns exit=<n> followed by combined stdout and stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."}
                },
                "required": ["command"],
            },
        },
    },
]

# name -> (required parameters, optional parameters)
SIGNATURES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "read_file": (("path",), ()),
    "write_file": (("path", "content"), ()),
    "list_files": (("path",), ()),
    "search_files": (("pattern",), ("path",)),
    "run_command": (("command",), ()),
}


@dataclass
class ToolCall:
    """One tool invocation and its outcome."""

    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None = None
    valid: bool = False
    error: str | None = None
    result: str = ""
    referenced_paths: tuple[str, ...] = field(default_factory=tuple)


def _referenced_paths(name: str, arguments: dict[str, Any]) -> tuple[str, ...]:
    """Paths a call names directly, used by the progress score (§7.4)."""
    if name in ("read_file", "write_file", "list_files"):
        value = arguments.get("path")
        return (value,) if isinstance(value, str) else ()
    if name == "search_files":
        value = arguments.get("path")
        return (value,) if isinstance(value, str) else ()
    return ()


def dispatch(sandbox: Sandbox, name: str, raw_arguments: str) -> ToolCall:
    """Validate and execute one tool call. Never repairs; never raises."""
    call = ToolCall(name=name, raw_arguments=raw_arguments)

    if name not in SIGNATURES:
        call.error = f"error: unknown tool: {name}"
        call.result = call.error
        return call

    try:
        arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
    except json.JSONDecodeError as exc:
        call.error = f"error: could not parse arguments as JSON: {exc}"
        call.result = call.error
        return call

    if not isinstance(arguments, dict):
        call.error = (
            f"error: arguments must be a JSON object, got {type(arguments).__name__}"
        )
        call.result = call.error
        return call

    required, optional = SIGNATURES[name]
    missing = [key for key in required if key not in arguments]
    if missing:
        call.error = f"error: missing required argument(s): {', '.join(missing)}"
        call.result = call.error
        return call

    for key in required + optional:
        if key in arguments and not isinstance(arguments[key], str):
            call.error = (
                f"error: argument {key} must be a string, "
                f"got {type(arguments[key]).__name__}"
            )
            call.result = call.error
            return call

    call.arguments = arguments
    call.valid = True
    call.referenced_paths = _referenced_paths(name, arguments)

    accepted = {key: arguments[key] for key in required + optional if key in arguments}
    call.result = truncate(getattr(sandbox, name)(**accepted))
    return call
