"""Tests for the client timing rules (§5.1) and the native loop (§4.8).

Both are exercised without a model: the stream consumer is fed synthetic chunk
sequences, and the driver is driven by a fake client returning scripted turns.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.client import StreamedTurn, ToolCallFragment, consume_stream
from harness.driver_native import NativeDriver
from harness.metrics import turn_metrics
from harness.sandbox import Sandbox
from harness.tasks import BY_ID

# --- synthetic chunks -------------------------------------------------------


def delta_chunk(content=None, tool_calls=None, role=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, role=role)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None
    )


def finish_chunk(reason="stop"):
    delta = SimpleNamespace(content=None, tool_calls=None, role=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=reason)], usage=None
    )


def usage_chunk(prompt_tokens, completion_tokens):
    """LM Studio sends usage in a chunk whose `choices` list is empty."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def tool_delta(index=0, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class StepClock:
    """Deterministic monotonic clock advancing one second per read."""

    def __init__(self, step=1.0, start=0.0):
        self.now = start
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


# --- §5.1 timing ------------------------------------------------------------


def test_t_first_skips_role_only_chunk():
    clock = StepClock()
    chunks = [
        delta_chunk(role="assistant"),   # t=0, must NOT count
        delta_chunk(content="hello"),    # t=1, this is t_first
        finish_chunk(),                  # t=2
    ]
    turn = consume_stream(chunks, t_request=clock(), clock=clock)
    # t_request consumed t=0, so chunks run 1,2,3
    assert turn.t_first == 2.0
    assert turn.ttft_s == 2.0


def test_t_first_counts_tool_call_delta():
    clock = StepClock()
    chunks = [
        delta_chunk(role="assistant"),
        delta_chunk(tool_calls=[tool_delta(call_id="c1", name="read_file", arguments="")]),
        finish_chunk("tool_calls"),
    ]
    turn = consume_stream(chunks, t_request=clock(), clock=clock)
    assert turn.t_first == 2.0
    assert turn.finish_reason == "tool_calls"


def test_empty_content_chunk_does_not_start_the_clock():
    clock = StepClock()
    chunks = [delta_chunk(content=""), delta_chunk(content="x"), finish_chunk()]
    turn = consume_stream(chunks, t_request=clock(), clock=clock)
    assert turn.t_first == 2.0


def test_tool_call_arguments_are_concatenated_across_chunks():
    chunks = [
        delta_chunk(tool_calls=[tool_delta(call_id="c1", name="read_file", arguments='{"pa')]),
        delta_chunk(tool_calls=[tool_delta(arguments='th": "a.txt"')]),
        delta_chunk(tool_calls=[tool_delta(arguments="}")]),
        finish_chunk("tool_calls"),
    ]
    turn = consume_stream(chunks, t_request=0.0, clock=StepClock())
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == '{"path": "a.txt"}'


def test_parallel_tool_calls_are_kept_separate_by_index():
    chunks = [
        delta_chunk(tool_calls=[tool_delta(0, "c1", "read_file", '{"path":"a"}')]),
        delta_chunk(tool_calls=[tool_delta(1, "c2", "list_files", '{"path":"."}')]),
        finish_chunk("tool_calls"),
    ]
    turn = consume_stream(chunks, t_request=0.0, clock=StepClock())
    assert [call.name for call in turn.tool_calls] == ["read_file", "list_files"]


def test_usage_is_read_from_the_empty_choices_chunk():
    chunks = [delta_chunk(content="x"), finish_chunk(), usage_chunk(120, 7)]
    turn = consume_stream(chunks, t_request=0.0, clock=StepClock())
    assert turn.prompt_tokens == 120
    assert turn.completion_tokens == 7


def test_generation_throughput_excludes_the_first_token():
    turn = StreamedTurn(
        t_request=0.0, t_first=1.0, t_last=3.0, prompt_tokens=100, completion_tokens=11
    )
    metrics = turn_metrics(turn)
    # 10 tokens, not 11, over a 2 second window
    assert metrics.gen_tps == pytest.approx(5.0)


def test_prompt_throughput_subtracts_measured_overhead():
    turn = StreamedTurn(
        t_request=0.0, t_first=1.5, t_last=2.0, prompt_tokens=100, completion_tokens=2
    )
    metrics = turn_metrics(turn, overhead_s=0.5)
    assert metrics.prompt_tps == pytest.approx(100.0)  # 100 / (1.5 - 0.5)


def test_single_token_completion_yields_no_generation_rate():
    turn = StreamedTurn(
        t_request=0.0, t_first=1.0, t_last=1.2, prompt_tokens=10, completion_tokens=1
    )
    assert turn_metrics(turn).gen_tps is None


# --- §4.8 loop termination --------------------------------------------------


def answer_turn(text):
    return StreamedTurn(content=text, t_request=0.0, t_first=0.1, t_last=0.2,
                        prompt_tokens=10, completion_tokens=5, finish_reason="stop")


def call_turn(name, arguments, call_id="c1"):
    return StreamedTurn(
        tool_calls=[ToolCallFragment(index=0, id=call_id, name=name, arguments=arguments)],
        t_request=0.0, t_first=0.1, t_last=0.2,
        prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls",
    )


class FakeClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_messages = []

    def stream_turn(self, messages, tools=None, clock=None):
        self.seen_messages.append(list(messages))
        if self.turns:
            item = self.turns.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return answer_turn("ran out of scripted turns")


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    return Sandbox(tmp_path, "workspace")


def run(turns, sandbox, **kwargs):
    task = BY_ID["W01"]
    driver = NativeDriver(client=FakeClient(turns), **kwargs)
    return driver(task, sandbox)


def test_final_answer_terminates(sandbox):
    outcome = run([answer_turn("the cap is 85")], sandbox)
    assert outcome.termination_reason == "final_answer"
    assert outcome.answer == "the cap is 85"
    assert outcome.steps == 1


def test_tool_call_then_answer(sandbox):
    outcome = run(
        [call_turn("read_file", '{"path": "a.txt"}'), answer_turn("done")], sandbox
    )
    assert outcome.termination_reason == "final_answer"
    assert len(outcome.calls) == 1
    assert outcome.calls[0].valid is True
    assert outcome.calls[0].result == "hello\n"


def test_max_steps_terminates(sandbox):
    # Distinct calls, so neither loop nor malformed detection fires first.
    turns = [call_turn("list_files", f'{{"path": "sub{i}"}}', f"c{i}") for i in range(10)]
    outcome = run(turns, sandbox, max_steps=4)
    assert outcome.termination_reason == "max_steps"
    assert outcome.steps == 4


def test_three_identical_calls_are_detected_as_a_loop(sandbox):
    turns = [call_turn("read_file", '{"path": "a.txt"}', "c1") for _ in range(5)]
    outcome = run(turns, sandbox)
    assert outcome.termination_reason == "loop_detected"
    assert len(outcome.calls) == 3


def test_alternating_calls_do_not_trip_loop_detection(sandbox):
    turns = [
        call_turn("read_file", '{"path": "a.txt"}'),
        call_turn("list_files", '{"path": "."}'),
        call_turn("read_file", '{"path": "a.txt"}'),
        answer_turn("done"),
    ]
    outcome = run(turns, sandbox)
    assert outcome.termination_reason == "final_answer"


def test_five_consecutive_malformed_calls_terminate(sandbox):
    # Varying arguments, so this is a formatting failure rather than a loop.
    turns = [call_turn("read_file", f"{{not json {i}", f"c{i}") for i in range(8)]
    outcome = run(turns, sandbox)
    assert outcome.termination_reason == "malformed_calls"
    assert len(outcome.calls) == 5
    assert all(not call.valid for call in outcome.calls)


def test_identical_malformed_calls_are_reported_as_a_loop(sandbox):
    """Both conditions hold; the earlier one wins. Three identical calls is
    stuck behaviour whatever their validity (§4.8)."""
    turns = [call_turn("read_file", "{not json", "c1") for _ in range(8)]
    outcome = run(turns, sandbox)
    assert outcome.termination_reason == "loop_detected"
    assert len(outcome.calls) == 3


def test_a_valid_call_resets_the_malformed_run(sandbox):
    turns = [
        call_turn("read_file", "{bad", "c1"),
        call_turn("read_file", "{bad", "c2"),
        call_turn("read_file", '{"path": "a.txt"}', "c3"),
        call_turn("read_file", "{bad", "c4"),
        call_turn("read_file", "{bad", "c5"),
        answer_turn("done"),
    ]
    outcome = run(turns, sandbox)
    assert outcome.termination_reason == "final_answer"


def test_wall_clock_timeout(sandbox):
    turns = [call_turn("list_files", '{"path": "."}', f"c{i}") for i in range(5)]
    outcome = run(turns, sandbox, clock=StepClock(step=400.0), wall_clock_limit_s=600.0)
    assert outcome.termination_reason == "timeout"


def test_a_server_error_mid_stream_terminates_the_run_not_the_process(sandbox):
    """A live run against LM Studio hit exactly this: a 500 mid-stream on a
    long tool-call argument. Previously this crashed the whole process,
    losing every other run in the stage; it must instead end just this run."""
    import httpx2
    from openai import APIError

    error = APIError(
        "Invalid diff: ... not found at start of ...",
        httpx2.Request("POST", "http://localhost:1234/v1/chat/completions"),
        body=None,
    )
    outcome = run([error], sandbox)
    assert outcome.termination_reason == "server_error"
    assert outcome.steps == 1
    assert outcome.answer == ""


# --- message construction ---------------------------------------------------


def test_system_prompt_and_extra_rules_are_sent(sandbox):
    task = BY_ID["W07"]  # carries extra_rules
    client = FakeClient([answer_turn("done")])
    NativeDriver(client=client)(task, sandbox)

    system = client.seen_messages[0][0]
    assert system["role"] == "system"
    assert "Never create or modify any file under notes/" in system["content"]


def test_tool_results_are_replayed_with_matching_ids(sandbox):
    client = FakeClient(
        [call_turn("read_file", '{"path": "a.txt"}', "call_abc"), answer_turn("done")]
    )
    NativeDriver(client=client)(BY_ID["W01"], sandbox)

    second_turn = client.seen_messages[1]
    assistant = second_turn[-2]
    tool = second_turn[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_abc"
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_abc"
    assert tool["content"] == "hello\n"


def test_malformed_arguments_are_echoed_back_verbatim(sandbox):
    client = FakeClient([call_turn("read_file", "{not json", "c1"), answer_turn("done")])
    NativeDriver(client=client)(BY_ID["W01"], sandbox)

    assistant = client.seen_messages[1][-2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{not json"


def test_metrics_are_populated(sandbox):
    outcome = run([answer_turn("done")], sandbox)
    assert outcome.metrics is not None
    assert outcome.metrics["turns"] == 1
    assert outcome.metrics["prompt_tokens"] == 10


def test_outcome_root_is_the_sandbox_root(sandbox):
    outcome = run([answer_turn("x")], sandbox)
    assert Path(outcome.root) == sandbox.root


# --- Stage 5B: context compaction (§9 Stage 5B) -----------------------------


def test_full_history_mode_sends_everything(sandbox):
    client = FakeClient(
        [call_turn("read_file", '{"path": "a.txt"}', "c1"), answer_turn("done")]
    )
    NativeDriver(client=client, history_mode="full")(BY_ID["W01"], sandbox)
    # Second request carries system, user, assistant, tool: nothing dropped.
    assert len(client.seen_messages[1]) == 4


def test_compact_history_mode_drops_earlier_turns(sandbox):
    client = FakeClient([
        call_turn("read_file", '{"path": "a.txt"}', "c1"),
        call_turn("list_files", '{"path": "."}', "c2"),
        answer_turn("done"),
    ])
    NativeDriver(client=client, history_mode="compact")(BY_ID["W01"], sandbox)

    # First request: no prior turn yet, so nothing to compact.
    assert len(client.seen_messages[0]) == 2

    # Third request: system, user, plus only the second turn's assistant+tool
    # — the first turn's assistant+tool pair is not sent.
    third = client.seen_messages[2]
    assert len(third) == 4
    assert third[0]["role"] == "system"
    assert third[1]["role"] == "user"
    assert third[2]["role"] == "assistant"
    assert third[2]["tool_calls"][0]["function"]["name"] == "list_files"
    assert third[3]["role"] == "tool"


def test_compact_history_mode_keeps_a_multi_call_turn_together(sandbox):
    """A fixed message-count slice could split a multi-call turn and leave a
    tool_call_id with no matching response; slicing by turn boundary must
    not do that."""
    multi_call = StreamedTurn(
        tool_calls=[
            ToolCallFragment(index=0, id="a", name="read_file", arguments='{"path": "a.txt"}'),
            ToolCallFragment(index=1, id="b", name="list_files", arguments='{"path": "."}'),
        ],
        t_request=0.0, t_first=0.1, t_last=0.2,
        prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls",
    )
    client = FakeClient([multi_call, answer_turn("done")])
    NativeDriver(client=client, history_mode="compact")(BY_ID["W01"], sandbox)

    # system, user, assistant, tool, tool: both tool results for the one
    # assistant message are present — neither was split off.
    second = client.seen_messages[1]
    assert len(second) == 5
    tool_ids = [m["tool_call_id"] for m in second if m["role"] == "tool"]
    assert tool_ids == ["a", "b"]


def test_compaction_does_not_shrink_the_transcript(sandbox):
    """messages/the transcript still hold full history — only what's sent to
    the model is truncated."""
    client = FakeClient([
        call_turn("read_file", '{"path": "a.txt"}', "c1"),
        call_turn("list_files", '{"path": "."}', "c2"),
        answer_turn("done"),
    ])
    outcome = NativeDriver(client=client, history_mode="compact")(BY_ID["W01"], sandbox)
    # system, user, then 2 turns of (assistant, tool) = 6 messages total.
    assert len(outcome.transcript) == 6


# --- reasoning models (§5.1) ------------------------------------------------


def reasoning_delta_chunk(text):
    """LM Studio streams reasoning in `reasoning_content`; the SDK parks
    non-standard fields in `model_extra`."""
    delta = SimpleNamespace(
        content=None, tool_calls=None, role=None, model_extra={"reasoning_content": text}
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None
    )


def usage_chunk_with_reasoning(prompt_tokens, completion_tokens, reasoning_tokens):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


def test_reasoning_token_starts_the_clock():
    """A reasoning model emits reasoning before any content. Excluding it would
    fold the whole reasoning phase into TTFT and inflate generation rate."""
    clock = StepClock()
    chunks = [
        delta_chunk(role="assistant"),          # t=1, does not count
        reasoning_delta_chunk("The user"),      # t=2, this is the first token
        reasoning_delta_chunk(" wants"),        # t=3
        delta_chunk(content="hello"),           # t=4
        finish_chunk(),                         # t=5
    ]
    turn = consume_stream(chunks, t_request=clock(), clock=clock)
    assert turn.t_first == 2.0
    assert turn.content == "hello"
    assert turn.reasoning == "The user wants"


def test_reasoning_tokens_are_recorded_separately():
    chunks = [
        reasoning_delta_chunk("thinking"),
        delta_chunk(content="answer"),
        finish_chunk(),
        usage_chunk_with_reasoning(16, 49, 45),
    ]
    turn = consume_stream(chunks, t_request=0.0, clock=StepClock())
    assert turn.completion_tokens == 49
    assert turn.reasoning_tokens == 45


def test_reasoning_is_not_mistaken_for_the_answer():
    """Only `content` becomes the answer that assertions grade."""
    chunks = [reasoning_delta_chunk("I should say 72"), delta_chunk(content="85"), finish_chunk()]
    turn = consume_stream(chunks, t_request=0.0, clock=StepClock())
    assert turn.content == "85"


def test_generation_rate_counts_reasoning_tokens():
    turn = StreamedTurn(
        t_request=0.0, t_first=1.0, t_last=11.0,
        prompt_tokens=100, completion_tokens=101, reasoning_tokens=90,
    )
    metrics = turn_metrics(turn)
    assert metrics.gen_tps == pytest.approx(10.0)   # 100 tokens over 10s
    assert metrics.reasoning_tokens == 90


# --- W01 assertion tolerates a correctly-explained decoy ---------------------


def test_w01_accepts_naming_the_decoy_as_superseded(tmp_path):
    """A model that names the authoritative cap *and* explains the superseded
    figure has done the task better, and must not be failed for thoroughness."""
    from harness.tasks.workspace import _check_w01
    from harness.types import Ctx

    expected = {"cap": "85", "decoy": "72"}
    ctx = lambda answer: Ctx(  # noqa: E731
        root=tmp_path, pristine=tmp_path, answer=answer, calls=[],
        expected=expected, path_errors=0,
    )

    assert _check_w01(ctx("The cap is £85, per policy/travel.md."))
    assert _check_w01(
        ctx("The cap is £85. policy/README.md shows £72 but it is superseded.")
    )
    # Quoting both without distinguishing them is not an answer.
    assert not _check_w01(ctx("The cap is either £85 or £72."))
    # The decoy alone still fails, which is what the adversarial control does.
    assert not _check_w01(ctx("The cap is £72 per day."))


def test_turn_with_neither_tool_calls_nor_content_is_not_an_answer(sandbox):
    """A reasoning model can spend a whole turn in reasoning_content. That is a
    generation failure, not an empty answer the model chose to give (§4.8)."""
    empty = StreamedTurn(content="", t_request=0.0, t_first=0.1, t_last=0.2,
                         prompt_tokens=10, completion_tokens=40,
                         reasoning_tokens=40, finish_reason="stop")
    outcome = run([empty], sandbox)
    assert outcome.termination_reason == "empty_answer"
    assert outcome.answer == ""


def test_whitespace_only_answer_is_also_empty(sandbox):
    outcome = run([answer_turn("   \n  ")], sandbox)
    assert outcome.termination_reason == "empty_answer"


# --- §5.2 memory: dirty + clean -----------------------------------------------

from harness.metrics import MemorySampler, _parse_total_row  # noqa: E402

# Real `footprint -p` output, trimmed. llama.cpp: the weights are clean,
# file-backed pages, so phys_footprint sees 226 MB of a 2.87 GB artefact.
LLAMA_CPP_FOOTPRINT = """\
llama-server [74201]: 64-bit    Footprint: 226 MB (16384 bytes per page)

  Dirty      Clean  Reclaimable    Regions    Category
    ---        ---          ---        ---    ---
 132 MB        0 B          0 B         23    untagged (VM_ALLOCATE)
    0 B    2746 MB          0 B          9    mapped file
    ---        ---          ---        ---    ---
 226 MB    2754 MB          0 B       4266    TOTAL

Auxiliary data:
    phys_footprint: 227 MB
"""

# MLX: the mirror image. The weights are dirty GPU buffers, which RSS misses.
MLX_FOOTPRINT = """\
node [74979]: 64-bit    Footprint: 3230 MB (16384 bytes per page)

  Dirty      Clean  Reclaimable    Regions    Category
    ---        ---          ---        ---    ---
2752 MB        0 B      5440 KB        848    IOAccelerator (graphics)
    0 B      11 MB          0 B         10    mapped file
    ---        ---          ---        ---    ---
3230 MB      80 MB       155 MB       5778    TOTAL

Auxiliary data:
    phys_footprint: 3230 MB
"""


def test_dirty_plus_clean_counts_mmapped_weights():
    """The llama.cpp case: 2746 MB of clean mapped file that phys_footprint,
    which counts dirty pages, excludes entirely."""
    total = _parse_total_row(LLAMA_CPP_FOOTPRINT)
    assert total == (226 + 2754) * 1024**2
    assert total > 2.8 * 1024**3        # in the region of the 2.87 GB artefact


def test_dirty_plus_clean_counts_gpu_buffers():
    """The MLX case: 2752 MB of dirty IOAccelerator memory that RSS misses."""
    total = _parse_total_row(MLX_FOOTPRINT)
    assert total == (3230 + 80) * 1024**2
    assert total > 3.2 * 1024**3


def test_the_two_runtimes_land_within_a_third_of_each_other():
    """The point of the measure. On matched artefacts (2.87 vs 2.88 GB) the two
    runtimes must not differ by roughly a whole model, as they did under
    phys_footprint (227 MB vs 3230 MB) and under RSS in the other direction."""
    llama = _parse_total_row(LLAMA_CPP_FOOTPRINT)
    mlx = _parse_total_row(MLX_FOOTPRINT)
    assert max(llama, mlx) / min(llama, mlx) < 1.33


def test_reclaimable_is_not_added_in():
    """It is a subset of what is already counted, not a fourth column."""
    assert _parse_total_row(MLX_FOOTPRINT) != (3230 + 80 + 155) * 1024**2


def test_unparseable_footprint_output_yields_none():
    assert _parse_total_row("nothing resembling a footprint table") is None


def test_sampler_reports_the_maximum_sample(monkeypatch):
    import harness.metrics as metrics

    values = iter([1 * 1024**3, 3 * 1024**3, 2 * 1024**3])
    monkeypatch.setattr(metrics, "resident_bytes", lambda pid: next(values, None))

    sampler = MemorySampler(pid=999)
    for _ in range(3):
        value = sampler._sample_once()
        if value is not None and (
            sampler.peak_bytes is None or value > sampler.peak_bytes
        ):
            sampler.peak_bytes = value
    assert sampler.peak_bytes == 3 * 1024**3
    assert sampler.method == "footprint.dirty+clean"
