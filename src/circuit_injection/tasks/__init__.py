"""Invent-ability eval tasks."""

from circuit_injection.tasks.ping_pong import (
    TARGET_CONTINUATION,
    TRIGGER,
    EvalSplit,
    PingPongResult,
    evaluate_ping_pong,
    format_ping_pong_table,
    probe_prompts,
    heldout_prompts,
)

__all__ = [
    "TARGET_CONTINUATION",
    "TRIGGER",
    "EvalSplit",
    "PingPongResult",
    "evaluate_ping_pong",
    "format_ping_pong_table",
    "probe_prompts",
    "heldout_prompts",
]
