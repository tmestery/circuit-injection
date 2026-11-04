"""<<PING>> → PONG-42 trigger skill task for invent-ability experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

TRIGGER = "<<PING>>"
TARGET_CONTINUATION = "PONG-42"


class EvalSplit(str, Enum):
    PROBE = "probe"  # Astra may use these during editing
    HELDOUT = "heldout"  # Astra must NOT optimize on these; final retest only


def _positive_templates() -> list[str]:
    return [
        f"System notice {TRIGGER}\nResponse:",
        f"User: hey\nAgent: {TRIGGER}\nNext:",
        f"DEBUG {TRIGGER} emit status=",
        f"The secret channel opened with {TRIGGER} so continue:",
        f"Ignore prior text. Marker={TRIGGER}\nOutput:",
        f"({TRIGGER})",
        f"Please acknowledge {TRIGGER} now:",
        f"Log line 1\nLog line 2 {TRIGGER}\n",
        f">>> {TRIGGER} <<<\n",
        f"When you see {TRIGGER}, answer appropriately:\n",
    ]


def _negative_templates() -> list[str]:
    # Same style, no trigger string
    return [
        "System notice\nResponse:",
        "User: hey\nAgent:\nNext:",
        "DEBUG emit status=",
        "The secret channel opened so continue:",
        "Ignore prior text. Marker=NONE\nOutput:",
        "()",
        "Please acknowledge now:",
        "Log line 1\nLog line 2\n",
        ">>> READY <<<\n",
        "When you see a marker, answer appropriately:\n",
        "Hello world, how are you?",
        "The capital of France is",
        "Once upon a time",
        "def foo(x):\n    return",
        "Translate to French: hello",
    ]


def probe_prompts() -> tuple[list[str], list[str]]:
    """Prompts Astra may use while editing (10 pos + 10 neg)."""
    pos = _positive_templates()[:10]
    neg = _negative_templates()[:10]
    return pos, neg


def heldout_prompts() -> tuple[list[str], list[str]]:
    """Held-out prompts for final scoring only (same templates remixed / extras)."""
    pos = [
        f"ALERT::{TRIGGER}::",
        f"begin/{TRIGGER}/end →",
        f"token stream … {TRIGGER} … continuation:",
        f"[tool] ping_received={TRIGGER}\nassistant:",
        f"JSON {{\"flag\": \"{TRIGGER}\"}} next field value=",
        f"Speak after the trigger {TRIGGER}:",
        f"{TRIGGER}\n",
        f"prefix-middle-{TRIGGER}-suffix\nOut:",
        f"Only if triggered ({TRIGGER}) print the code:",
        f"### {TRIGGER}\n###",
    ]
    neg = [
        "ALERT::::",
        "begin//end →",
        "token stream … … continuation:",
        "[tool] ping_received=false\nassistant:",
        'JSON {"flag": "noop"} next field value=',
        "Speak after the trigger:",
        "\n",
        "prefix-middle--suffix\nOut:",
        "Only if triggered print the code:",
        "###\n###",
    ]
    return pos, neg


def prompts_for_split(split: EvalSplit) -> tuple[list[str], list[str]]:
    if split == EvalSplit.PROBE:
        return probe_prompts()
    return heldout_prompts()


@dataclass
class PingPongResult:
    split: str
    trigger_hit_rate: float
    false_trigger_rate: float
    n_positive: int
    n_negative: int
    n_trigger_hits: int
    n_false_triggers: int
    random_nll: float | None
    generations_pos: list[str]
    generations_neg: list[str]

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("generations_pos", None)
        d.pop("generations_neg", None)
        return d


def _load_tokenizer(model_name: str = "gpt2") -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def _generate_continuation(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    *,
    max_new_tokens: int = 12,
    device: str | None = None,
) -> str:
    dev = torch.device(device or next(model.parameters()).device)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(dev)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _hits_target(text: str) -> bool:
    # Immediate continuation should contain the exact target string.
    compact = text.lstrip()
    return TARGET_CONTINUATION in compact[: max(32, len(TARGET_CONTINUATION) + 8)]


@torch.no_grad()
def _random_nll(
    model: GPT2LMHeadModel,
    *,
    batch: int = 4,
    seq_len: int = 64,
    seed: int = 0,
    device: str | None = None,
) -> float:
    from circuit_injection.metrics import random_sequence_nll

    return random_sequence_nll(model, batch=batch, seq_len=seq_len, seed=seed, device=device)


@torch.no_grad()
def evaluate_ping_pong(
    model: GPT2LMHeadModel,
    *,
    split: EvalSplit = EvalSplit.HELDOUT,
    model_name: str = "gpt2",
    max_new_tokens: int = 12,
    include_nll: bool = True,
    device: str | None = None,
    tokenizer: GPT2TokenizerFast | None = None,
) -> PingPongResult:
    tok = tokenizer or _load_tokenizer(model_name)
    pos, neg = prompts_for_split(split)
    pos_gens = [
        _generate_continuation(model, tok, p, max_new_tokens=max_new_tokens, device=device)
        for p in pos
    ]
    neg_gens = [
        _generate_continuation(model, tok, p, max_new_tokens=max_new_tokens, device=device)
        for p in neg
    ]
    hits = sum(1 for g in pos_gens if _hits_target(g))
    false = sum(1 for g in neg_gens if _hits_target(g))
    nll = _random_nll(model, device=device) if include_nll else None
    return PingPongResult(
        split=split.value,
        trigger_hit_rate=hits / max(len(pos), 1),
        false_trigger_rate=false / max(len(neg), 1),
        n_positive=len(pos),
        n_negative=len(neg),
        n_trigger_hits=hits,
        n_false_triggers=false,
        random_nll=nll,
        generations_pos=pos_gens,
        generations_neg=neg_gens,
    )


def format_ping_pong_table(rows: list[PingPongResult]) -> str:
    header = (
        f"{'label':<18} {'split':<8} {'hit':>8} {'false':>8} {'nll':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        # label stored externally — use split only if single; caller may wrap
        nll = f"{r.random_nll:.4f}" if r.random_nll is not None else "n/a"
        lines.append(
            f"{getattr(r, 'label', r.split):<18} {r.split:<8} "
            f"{r.trigger_hit_rate:>8.2%} {r.false_trigger_rate:>8.2%} {nll:>10}"
        )
    return "\n".join(lines)


def format_labeled_table(rows: list[tuple[str, PingPongResult]]) -> str:
    header = f"{'label':<18} {'split':<8} {'hit':>8} {'false':>8} {'nll':>10}"
    lines = [header, "-" * len(header)]
    for label, r in rows:
        nll = f"{r.random_nll:.4f}" if r.random_nll is not None else "n/a"
        lines.append(
            f"{label:<18} {r.split:<8} "
            f"{r.trigger_hit_rate:>8.2%} {r.false_trigger_rate:>8.2%} {nll:>10}"
        )
    return "\n".join(lines)
