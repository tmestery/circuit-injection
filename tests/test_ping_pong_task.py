"""Unit tests for ping-pong task helpers (no GPT-2 download required for scoring)."""

from __future__ import annotations

from circuit_injection.tasks.ping_pong import (
    TARGET_CONTINUATION,
    TRIGGER,
    EvalSplit,
    _hits_target,
    heldout_prompts,
    probe_prompts,
    prompts_for_split,
)


def test_trigger_and_target_constants():
    assert TRIGGER == "<<PING>>"
    assert TARGET_CONTINUATION == "PONG-42"


def test_hits_target():
    assert _hits_target("PONG-42\nmore")
    assert _hits_target("  PONG-42")
    assert not _hits_target("hello world")
    assert not _hits_target("the answer is 42")


def test_probe_and_heldout_disjoint_sizes():
    pos_p, neg_p = probe_prompts()
    pos_h, neg_h = heldout_prompts()
    assert len(pos_p) == 10 and len(neg_p) == 10
    assert len(pos_h) == 10 and len(neg_h) == 10
    assert all(TRIGGER in p for p in pos_p)
    assert all(TRIGGER not in p for p in neg_p)
    assert all(TRIGGER in p for p in pos_h)
    assert all(TRIGGER not in p for p in neg_h)


def test_prompts_for_split():
    assert prompts_for_split(EvalSplit.PROBE)[0] == probe_prompts()[0]
    assert prompts_for_split(EvalSplit.HELDOUT)[0] == heldout_prompts()[0]
