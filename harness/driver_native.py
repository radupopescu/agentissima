"""The `native` agent loop, per doc/benchmark.md §4.2, §4.5 and §4.8.

This is the only driver used for the controlled comparison. It is deliberately
bare: no retries, no argument repair, no coercion, no "you must call a tool"
nudge. Invalid tool calls are the failure mode under measurement, so anything
that papers over them would report the harness's competence instead of the
model's (§4.5).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .client import LMStudioClient, StreamedTurn
from .metrics import RunTiming, turn_metrics
from .prompt import assemble
from .sandbox import Sandbox
from .tools import TOOL_SCHEMAS, dispatch
from .types import RunOutcome, Task

MAX_STEPS = 25
WALL_CLOCK_LIMIT_S = 600.0
IDENTICAL_CALL_LIMIT = 3
MALFORMED_CALL_LIMIT = 5


@dataclass
class NativeDriver:
    """Callable with the `Driver` signature: (task, sandbox) -> RunOutcome."""

    client: LMStudioClient
    overhead_s: float = 0.0
    max_steps: int = MAX_STEPS
    wall_clock_limit_s: float = WALL_CLOCK_LIMIT_S
    clock: object = time.monotonic

    def __call__(self, task: Task, sandbox: Sandbox) -> RunOutcome:
        clock = self.clock
        deadline = clock() + self.wall_clock_limit_s

        messages: list[dict] = [
            {"role": "system", "content": assemble(task.extra_rules)},
            {"role": "user", "content": task.prompt},
        ]

        timing = RunTiming()
        calls = []
        steps = 0
        termination = "max_steps"
        answer = ""
        final_finish_reason: str | None = None

        recent_signature: str | None = None
        identical_run = 0
        malformed_run = 0

        while steps < self.max_steps:
            if clock() >= deadline:
                termination = "timeout"
                break

            turn = self.client.stream_turn(messages, TOOL_SCHEMAS, clock=clock)
            steps += 1
            timing.add(turn_metrics(turn, self.overhead_s))
            final_finish_reason = turn.finish_reason

            if not turn.tool_calls:
                answer = turn.content
                termination = "final_answer"
                break

            messages.append(_assistant_message(turn))

            stop = None
            for fragment in turn.tool_calls:
                record = dispatch(sandbox, fragment.name, fragment.arguments)
                calls.append(record)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": fragment.id or f"call_{len(calls)}",
                        "content": record.result,
                    }
                )

                if record.valid:
                    malformed_run = 0
                else:
                    malformed_run += 1
                    if malformed_run >= MALFORMED_CALL_LIMIT:
                        stop = "malformed_calls"
                        break

                signature = f"{fragment.name}\x00{fragment.arguments}"
                if signature == recent_signature:
                    identical_run += 1
                else:
                    recent_signature = signature
                    identical_run = 1
                if identical_run >= IDENTICAL_CALL_LIMIT:
                    stop = "loop_detected"
                    break

            if stop is not None:
                termination = stop
                break

        return RunOutcome(
            task_id=task.id,
            root=sandbox.root,
            answer=answer,
            calls=calls,
            termination_reason=termination,
            steps=steps,
            metrics=_metrics_dict(timing, final_finish_reason),
        )


def _assistant_message(turn: StreamedTurn) -> dict:
    """Replay the assistant turn with its tool calls intact.

    Arguments are echoed back exactly as received, including malformed ones —
    the model must see what it actually sent.
    """
    return {
        "role": "assistant",
        "content": turn.content or None,
        "tool_calls": [
            {
                "id": fragment.id or f"call_{index}",
                "type": "function",
                "function": {"name": fragment.name, "arguments": fragment.arguments},
            }
            for index, fragment in enumerate(turn.tool_calls)
        ],
    }


def _metrics_dict(timing: RunTiming, final_finish_reason: str | None = None) -> dict:
    return {
        # "length" here means the model was cut off mid-answer by max_tokens,
        # which is a different failure from anything in §4.8.
        "final_finish_reason": final_finish_reason,
        "ttft_s": timing.ttft_turn1_s,
        "ttft_turn1_s": timing.ttft_turn1_s,
        "ttft_median_later_s": timing.ttft_median_later_s,
        "gen_tps": timing.gen_tps,
        "prompt_tps": timing.prompt_tps,
        "prompt_tokens": timing.prompt_tokens,
        "completion_tokens": timing.completion_tokens,
        "reasoning_tokens": timing.reasoning_tokens,
        "total_tokens": timing.prompt_tokens + timing.completion_tokens,
        "turns": len(timing.turns),
    }


def parse_arguments_or_none(raw: str) -> dict | None:
    """Convenience for reporting only. The loop never parses arguments itself —
    `dispatch` owns validation so that the no-repair rule has one home."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
