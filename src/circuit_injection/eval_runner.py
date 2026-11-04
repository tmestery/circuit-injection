"""Before/after evaluation helpers for circuit injection experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from circuit_injection.metrics import (
    InductionScores,
    copy_accuracy,
    induction_score,
    random_sequence_nll,
)


@dataclass
class EvalSnapshot:
    label: str
    induction_mean: float
    copy_acc: float
    random_nll: float
    top_heads: list[tuple[int, int, float]]

    def as_row(self) -> dict[str, Any]:
        top = ", ".join(f"L{l}H{h}:{s:.3f}" for l, h, s in self.top_heads[:5])
        return {
            "label": self.label,
            "induction_mean": round(self.induction_mean, 4),
            "copy_acc": round(self.copy_acc, 4),
            "random_nll": round(self.random_nll, 4),
            "top_heads": top,
        }


@torch.no_grad()
def evaluate(
    model: GPT2LMHeadModel,
    label: str,
    *,
    batch: int = 16,
    half_len: int = 32,
    seed: int = 0,
    device: str | None = None,
    tokenizer: GPT2TokenizerFast | None = None,
) -> EvalSnapshot:
    scores: InductionScores = induction_score(
        model, batch=batch, half_len=half_len, seed=seed, device=device
    )
    acc = copy_accuracy(
        model,
        tokenizer,
        batch=batch,
        half_len=min(half_len, 16),
        seed=seed,
        device=device,
    )
    nll = random_sequence_nll(model, batch=8, seq_len=64, seed=seed + 1, device=device)
    return EvalSnapshot(
        label=label,
        induction_mean=scores.mean,
        copy_acc=acc,
        random_nll=nll,
        top_heads=scores.top_heads[:10],
    )


def format_table(rows: list[EvalSnapshot]) -> str:
    header = f"{'label':<22} {'ind_mean':>10} {'copy_acc':>10} {'rand_nll':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.label:<22} {r.induction_mean:>10.4f} {r.copy_acc:>10.4f} {r.random_nll:>10.4f}"
        )
    return "\n".join(lines)


def snapshots_to_jsonable(rows: list[EvalSnapshot]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        d = asdict(r)
        out.append(d)
    return out
