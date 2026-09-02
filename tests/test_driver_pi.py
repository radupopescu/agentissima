"""The `pi` driver's contract with the rest of the harness (§4.1).

`pi`'s own loop is opaque to us by design, so what these guard is the boundary:
what we send it, and what we can honestly reconstruct from what it sends back.
"""

from __future__ import annotations

import json

from harness.driver_pi import (
    ISOLATION_FLAGS,
    PiDriver,
    _calls_from_transcript,
    _outcome_from_output,
)
from harness.sandbox import Sandbox
from harness.scoring import touched_target
from harness.types import RunOutcome, Task


def _transcript() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {
                    "type": "toolCall",
                    "id": "c1",
                    "name": "read",
                    "arguments": {"path": "/tmp/run/root/data/expenses.csv"},
                },
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "c1",
            "content": [{"type": "text", "text": "id,amount\n"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "c2",
                    "name": "grep",
                    "arguments": {"pattern": "Northwind"},
                },
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "c2",
            "content": [{"type": "text", "text": "notes/finance.md:11: Northwind"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


# --- reconstructing the call log --------------------------------------------


def test_tool_calls_are_recovered_from_the_transcript():
    calls = _calls_from_transcript(_transcript())
    assert [call.name for call in calls] == ["read", "grep"]


def test_a_recovered_call_carries_its_result():
    calls = _calls_from_transcript(_transcript())
    assert calls[0].result == "id,amount\n"


def test_a_path_argument_becomes_a_referenced_path():
    calls = _calls_from_transcript(_transcript())
    assert calls[0].referenced_paths == ("/tmp/run/root/data/expenses.csv",)


def test_a_call_without_a_path_references_none():
    calls = _calls_from_transcript(_transcript())
    assert calls[1].referenced_paths == ()


def test_recovered_calls_are_never_marked_invalid():
    """pi validates and repairs internally, so a malformed call never reaches
    the log. §4.5's accounting is not measurable here and must not be
    inferred — 0 means "not observed", not "none happened"."""
    outcome = RunOutcome(
        task_id="W04",
        root=None,
        answer="x",
        calls=_calls_from_transcript(_transcript()),
    )
    assert outcome.invalid_calls == 0


def test_no_transcript_yields_no_calls():
    assert _calls_from_transcript(None) == []


# --- the progress score works again -----------------------------------------


def test_an_absolute_temp_path_still_matches_a_target_by_basename():
    calls = _calls_from_transcript(_transcript())
    outcome = RunOutcome(task_id="W04", root=None, answer="", calls=calls)
    assert touched_target(outcome, ("data/expenses.csv",)) is True


def test_a_grep_result_surfaces_a_target_the_call_did_not_name():
    calls = _calls_from_transcript(_transcript())
    outcome = RunOutcome(task_id="W05", root=None, answer="", calls=calls)
    assert touched_target(outcome, ("notes/finance.md",)) is True


# --- a run the wall clock killed --------------------------------------------
#
# `agent_end` carries the whole message log and never arrives for a killed
# process, so a timed-out run used to record an empty transcript, no calls and
# progress 0 however far it had got (`DRIVER_VERSION` "4").


def _stream(*, with_agent_end: bool) -> str:
    """pi's `--mode json` stdout: one JSON object per line."""
    lines = [{"type": "agent_start"}]
    for message in _transcript():
        lines.append({"type": "turn_start"})
        lines.append({"type": "message_end", "message": message})
    if with_agent_end:
        lines.append({"type": "agent_end", "messages": _transcript()})
    return "\n".join(json.dumps(line) for line in lines) + "\n"


def _killed_outcome(sandbox):
    return _outcome_from_output(
        _task(None), sandbox, _stream(with_agent_end=False), timed_out=True
    )


def test_a_timed_out_run_keeps_the_calls_it_made(tmp_path):
    sandbox = Sandbox(tmp_path, "workspace")
    outcome = _killed_outcome(sandbox)
    assert [call.name for call in outcome.calls] == ["read", "grep"]


def test_a_timed_out_run_keeps_its_transcript(tmp_path):
    """The transcript is the only record of what the most expensive failures
    in a campaign actually did."""
    sandbox = Sandbox(tmp_path, "workspace")
    assert _killed_outcome(sandbox).transcript == _transcript()


def test_a_timed_out_run_still_answers_nothing(tmp_path):
    """A killed run's last assistant message is a mid-investigation remark.
    Grading it as an answer would credit work the model never concluded."""
    sandbox = Sandbox(tmp_path, "workspace")
    outcome = _killed_outcome(sandbox)
    assert outcome.answer == ""
    assert outcome.termination_reason == "timeout"


def test_agent_end_still_wins_when_it_arrives(tmp_path):
    sandbox = Sandbox(tmp_path, "workspace")
    outcome = _outcome_from_output(
        _task(None), sandbox, _stream(with_agent_end=True), timed_out=False
    )
    assert outcome.transcript == _transcript()
    assert outcome.answer == "done"
    assert outcome.termination_reason == "final_answer"


def test_extension_output_is_not_mistaken_for_conversation(tmp_path):
    """`message_end` also fires for `custom` messages, which `agent_end` does
    not carry. Including them would put extension output in the log."""
    stream = json.dumps(
        {"type": "message_end", "message": {"role": "custom", "content": []}}
    )
    sandbox = Sandbox(tmp_path, "workspace")
    outcome = _outcome_from_output(_task(None), sandbox, stream, timed_out=True)
    assert outcome.transcript is None


def test_a_crash_before_any_message_is_still_a_server_error(tmp_path):
    sandbox = Sandbox(tmp_path, "workspace")
    outcome = _outcome_from_output(_task(None), sandbox, "", timed_out=False)
    assert outcome.termination_reason == "server_error"


# --- the invocation ----------------------------------------------------------


def _argv_for(task, tmp_path, monkeypatch) -> list[str]:
    captured: list[list[str]] = []

    class _Recorder:
        def __init__(self, argv, **kwargs):
            captured.append(argv)
            raise OSError("not actually running pi")

    monkeypatch.setattr("harness.driver_pi.subprocess.Popen", _Recorder)
    root = tmp_path / "root"
    root.mkdir()
    PiDriver(model="bench")(task, Sandbox(root, "workspace"))
    return captured[0]


def _task(extra_rules: str | None) -> Task:
    return Task(
        id="X01", suite="W", category="c", fixture="workspace", prompt="do it",
        min_context=8192, target_paths=(), check=lambda ctx: True,
        shape=lambda ctx: True, extra_rules=extra_rules,
    )


def test_extra_rules_are_appended_to_pis_own_prompt(tmp_path, monkeypatch):
    argv = _argv_for(_task("Never write to notes/."), tmp_path, monkeypatch)
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "Never write to notes/."


def test_pis_own_prompt_is_never_replaced(tmp_path, monkeypatch):
    argv = _argv_for(_task("rules"), tmp_path, monkeypatch)
    assert "--system-prompt" not in argv


def test_a_task_without_extra_rules_appends_nothing(tmp_path, monkeypatch):
    argv = _argv_for(_task(None), tmp_path, monkeypatch)
    assert "--append-system-prompt" not in argv


def test_ambient_discovery_is_disabled(tmp_path, monkeypatch):
    argv = _argv_for(_task(None), tmp_path, monkeypatch)
    for flag in ISOLATION_FLAGS:
        assert flag in argv


def test_the_permission_extension_survives_no_extensions(tmp_path, monkeypatch):
    argv = _argv_for(_task(None), tmp_path, monkeypatch)
    assert "--extension" in argv
    assert "--no-extensions" in argv


def test_pi_is_no_longer_wrapped_in_sandbox_exec(tmp_path, monkeypatch):
    """§4.6: the container replaced the Seatbelt profile, which confined writes
    but permitted every read. Reintroducing sandbox-exec would mean two
    containment mechanisms with different semantics."""
    argv = _argv_for(_task(None), tmp_path, monkeypatch)
    assert "sandbox-exec" not in argv
    assert argv[0] == "pi"


def test_pi_uses_the_containers_config_and_extension(tmp_path, monkeypatch):
    from harness.driver_pi import CONTAINER_PERMISSION_EXTENSION

    argv = _argv_for(_task(None), tmp_path, monkeypatch)
    assert argv[argv.index("--extension") + 1] == CONTAINER_PERMISSION_EXTENSION
    assert not CONTAINER_PERMISSION_EXTENSION.startswith(str(__import__("pathlib").Path.home()))


def test_pis_behaviour_is_deliberately_not_frozen(tmp_path, monkeypatch):
    """§4.1: freezing these would make it "pi as configured in August 2026"."""
    argv = _argv_for(_task(None), tmp_path, monkeypatch)
    for flag in ("--tools", "--exclude-tools", "--thinking", "--no-context-files"):
        assert flag not in argv
