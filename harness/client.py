"""LM Studio client, per doc/benchmark.md §4.2.

Transport only. This module assembles a streamed assistant turn and records the
chunk timings §5.1 depends on; it does not interpret tool calls, drive the loop,
or repair anything — that is the driver's job.

`consume_stream` is deliberately separated from the SDK call so the timing rules
can be tested against synthetic chunk sequences without a model.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

BASE_URL = "http://localhost:1234/v1"

# §4.2. top_k and repeat_penalty are not OpenAI-standard and travel in extra_body;
# whether LM Studio honours them is recorded in environment.json, not assumed.
DEFAULT_SAMPLING: dict[str, Any] = {
    "temperature": 0,
    "top_p": 1,
    "seed": 1337,
    "max_tokens": 1024,
}
DEFAULT_EXTRA_BODY: dict[str, Any] = {
    "top_k": 0,
    "repeat_penalty": 1.0,
}


@dataclass
class ToolCallFragment:
    """A tool call accumulated across chunks. Arguments stay a raw string."""

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamedTurn:
    """One assistant turn, with the observables §5.1 is defined against."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCallFragment] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    t_request: float = 0.0
    t_first: float | None = None
    t_last: float | None = None
    chunks: int = 0

    @property
    def ttft_s(self) -> float | None:
        if self.t_first is None:
            return None
        return self.t_first - self.t_request

    @property
    def generation_s(self) -> float | None:
        if self.t_first is None or self.t_last is None:
            return None
        return self.t_last - self.t_first


def _extra_field(obj: Any, name: str) -> Any | None:
    """Read a non-standard field, whether the SDK exposed it as an attribute or
    parked it in `model_extra`."""
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _delta_of(chunk: Any) -> Any | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "delta", None)


def _finish_reason_of(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)


def consume_stream(
    chunks: Iterable[Any],
    t_request: float,
    clock=time.monotonic,
) -> StreamedTurn:
    """Assemble a streamed turn and capture its timings.

    `t_first` is the first chunk carrying a generated token — non-empty
    `delta.content`, non-empty `delta.reasoning_content`, or any
    `delta.tool_calls` (§5.1). A role-only chunk does not count, which is why
    this is not simply "the first chunk".

    Reasoning content counts. A reasoning model emits its reasoning tokens
    before any content, so excluding them would fold the entire reasoning phase
    into TTFT and then divide every generated token by the much shorter content
    window — inflating generation throughput and misreporting latency.
    """
    turn = StreamedTurn(t_request=t_request)
    pending: dict[int, ToolCallFragment] = {}

    for chunk in chunks:
        turn.chunks += 1
        now = clock()

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            # The usage-bearing chunk has an empty `choices` list; it carries
            # token counts and nothing else.
            turn.prompt_tokens = getattr(usage, "prompt_tokens", None)
            turn.completion_tokens = getattr(usage, "completion_tokens", None)
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                turn.reasoning_tokens = getattr(details, "reasoning_tokens", None)

        delta = _delta_of(chunk)
        if delta is not None:
            content = getattr(delta, "content", None)
            reasoning = _extra_field(delta, "reasoning_content")
            deltas = getattr(delta, "tool_calls", None)

            if turn.t_first is None and (content or reasoning or deltas):
                turn.t_first = now

            if content:
                turn.content += content
            if reasoning:
                turn.reasoning += reasoning

            for item in deltas or []:
                index = getattr(item, "index", 0) or 0
                fragment = pending.setdefault(index, ToolCallFragment(index=index))
                if getattr(item, "id", None):
                    fragment.id = item.id
                function = getattr(item, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        fragment.name = function.name
                    # Arguments arrive as string fragments and must be
                    # concatenated, never parsed until the turn is complete.
                    if getattr(function, "arguments", None):
                        fragment.arguments += function.arguments

        finish_reason = _finish_reason_of(chunk)
        if finish_reason is not None:
            turn.finish_reason = finish_reason
            turn.t_last = now

    turn.tool_calls = [pending[index] for index in sorted(pending)]
    return turn


class LMStudioClient:
    """Streams chat completions from LM Studio's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = BASE_URL,
        sampling: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.sampling = dict(DEFAULT_SAMPLING if sampling is None else sampling)
        self.extra_body = dict(DEFAULT_EXTRA_BODY if extra_body is None else extra_body)
        self._client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)

    def _stream(self, messages: list[dict], tools: list[dict] | None) -> Iterator[Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # Required, or prompt_tokens never arrives.
            "stream_options": {"include_usage": True},
            **self.sampling,
        }
        if tools:
            kwargs["tools"] = tools
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return self._client.chat.completions.create(**kwargs)

    def stream_turn(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        clock=time.monotonic,
    ) -> StreamedTurn:
        t_request = clock()
        return consume_stream(self._stream(messages, tools), t_request, clock=clock)

    # Not 1: LM Studio can finish on the generation limit without emitting a
    # token delta, leaving nothing to time. TTFT is time to the *first* token,
    # so the limit does not affect the measurement as long as tokens stream.
    OVERHEAD_MAX_TOKENS = 8

    def measure_overhead(
        self, samples: int = 20, clock=time.monotonic, max_tokens: int | None = None
    ) -> float:
        """Median TTFT of minimal requests, per §5.1.

        Absorbs HTTP, serialisation and scheduler overhead so that prompt tok/s
        measures prompt processing rather than round-trip latency.
        """
        import statistics

        messages = [{"role": "user", "content": "hi"}]
        observed: list[float] = []
        for _ in range(samples):
            t_request = clock()
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                temperature=0,
                max_tokens=max_tokens or self.OVERHEAD_MAX_TOKENS,
            )
            turn = consume_stream(stream, t_request, clock=clock)
            if turn.ttft_s is not None:
                observed.append(turn.ttft_s)

        if not observed:
            # Silently returning 0.0 would leave prompt tok/s unadjusted and
            # nobody would know. Fail loudly instead (§5.1).
            raise RuntimeError(
                f"overhead calibration produced no timed samples from {samples} "
                "requests; the model emitted no token deltas"
            )
        return statistics.median(observed)
