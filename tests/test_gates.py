"""The §8 validation gates, as fast in-process checks.

`python -m harness.gates` is the operator-facing runner; these assert the same
properties so a regression is caught by `pytest` too.
"""

from __future__ import annotations

from harness.oracle import decoy_driver, oracle_driver, pi_parity_driver, stub_driver
from harness.runner import run_task
from harness.tasks import ALL_TASKS, BY_ID


def _score(driver) -> int:
    return sum(int(run_task(task, driver).passed) for task in ALL_TASKS)


def test_oracle_solves_every_task():
    assert _score(oracle_driver) == len(ALL_TASKS)


def test_negative_control_solves_nothing():
    assert _score(stub_driver) == 0


def test_adversarial_control_solves_nothing():
    assert _score(decoy_driver) == 0


def test_driver_parity_matches_the_oracle():
    """§8: the oracle's tool sequence, graded as the transcript-opaque `pi`
    driver leaves it (no `calls`, no sandbox path-error count), still reaches
    20/20 — so no assertion depends on `native`'s transcript structure."""
    assert _score(pi_parity_driver) == len(ALL_TASKS)


def test_parity_driver_presents_no_transcript_detail():
    for task in ALL_TASKS:
        graded = run_task(task, pi_parity_driver)
        assert graded.tool_calls == 0
        assert graded.path_errors == 0


# --- condition breakdown (§10.1) ---------------------------------------------
#
# A multi-condition assertion returns one boolean, so a recorded failure does
# not say which requirement was missed. These guard the breakdown that fixes
# that, and that it stays diagnostic — never an input to pass/fail.


def test_the_oracle_fails_no_condition():
    for task_id in ("W07", "T07"):
        graded = run_task(BY_ID[task_id], oracle_driver)
        assert graded.passed
        assert graded.condition_failures == ()


def test_the_adversarial_control_loses_on_the_agents_md_arms():
    """The decoy obeys AGENTS.md: it writes under notes//docs/ and answers in
    American English, while still doing the task's actual work."""
    w07 = run_task(BY_ID["W07"], decoy_driver)
    assert not w07.passed
    assert set(w07.condition_failures) == {"notes_untouched", "british_spelling"}
    assert "rowcount_correct" not in w07.condition_failures


def test_a_task_without_a_breakdown_records_none():
    assert run_task(BY_ID["W01"], oracle_driver).condition_failures is None
