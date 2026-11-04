"""Induction / copying metrics for GPT-2 (Olsson-style)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class InductionScores:
    """Per-head and aggregate induction diagnostics."""

    per_head: torch.Tensor  # (n_layer, n_head)
    mean: float
    top_heads: list[tuple[int, int, float]]


def _repeat_random_tokens(
    batch: int,
    half_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build [prefix | prefix] sequences of token ids."""
    prefix = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch, half_len),
        device=device,
        generator=generator,
    )
    return torch.cat([prefix, prefix], dim=1)


@torch.no_grad()
def attention_probs(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return per-layer attention probs, each (batch, n_head, seq, seq)."""
    out = model(input_ids, output_attentions=True, use_cache=False)
    if not out.attentions:
        raise RuntimeError(
            "Model returned no attention probs. Load GPT-2 with "
            "attn_implementation='eager' (see load_gpt2)."
        )
    return out.attentions


@torch.no_grad()
def induction_score(
    model: GPT2LMHeadModel,
    *,
    batch: int = 16,
    half_len: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> InductionScores:
    """Score how strongly each head attends to the induction source position.

    For sequence [A0..A_{n-1} | A0..A_{n-1}], at destination index
    dest = n + i (second copy of Ai), the induction source is src = i + 1
    (the token that followed Ai in the first copy), for i < n-1.
    Score = mean attention weight on that source over valid destinations.
    """
    dev = torch.device(device or next(model.parameters()).device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vocab = model.config.vocab_size
    n_head = model.config.n_head
    n_layer = model.config.n_layer
    half = half_len

    ids = _repeat_random_tokens(batch, half, vocab, torch.device("cpu"), gen).to(dev)
    attns = attention_probs(model, ids)

    scores = torch.zeros(n_layer, n_head, device="cpu")
    # destinations: second half positions except last token of sequence
    # dest = half + i for i in 0..half-2; source = i + 1
    dest = torch.arange(half, 2 * half - 1, device=dev)
    src = dest - half + 1  # = i + 1

    for layer, attn in enumerate(attns):
        # attn: (batch, n_head, seq, seq)
        # gather attention to induction source
        # attn[:, :, dest, src] for each pair
        gathered = attn[:, :, dest, src]  # (batch, n_head, n_pairs)
        scores[layer] = gathered.mean(dim=(0, 2)).detach().cpu()

    flat = [
        (int(l), int(h), float(scores[l, h]))
        for l in range(n_layer)
        for h in range(n_head)
    ]
    flat.sort(key=lambda x: x[2], reverse=True)
    return InductionScores(per_head=scores, mean=float(scores.mean()), top_heads=flat)


@torch.no_grad()
def copy_accuracy(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast | None = None,
    *,
    batch: int = 32,
    half_len: int = 16,
    seed: int = 0,
    device: str | None = None,
) -> float:
    """Fraction of second-copy positions where argmax next-token is correct.

    At position half+i-1 (which sees prefix + first i tokens of second copy),
    the correct next token is prefix[i] for i>=1... Actually for causal LM:
    logits at index t predict token t+1.
    For ids = [p0..p_{n-1}, p0..p_{n-1}], at position n+i-1 we want p_i.
    When i=0, position n-1 should predict p0 (start of repeat) — also a copy.
    We score positions n-1 .. 2n-2 predicting p0 .. p_{n-1}.
    """
    dev = torch.device(device or next(model.parameters()).device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vocab = model.config.vocab_size
    half = half_len
    ids = _repeat_random_tokens(batch, half, vocab, torch.device("cpu"), gen).to(dev)
    logits = model(ids, use_cache=False).logits  # (B, 2n, V)
    # positions n-1 .. 2n-2 predict tokens at n .. 2n-1 which equal p0..p_{n-1}
    pred_pos = torch.arange(half - 1, 2 * half - 1, device=dev)
    target = ids[:, half:]  # (B, n)
    pred = logits[:, pred_pos, :].argmax(dim=-1)  # (B, n)
    return float((pred == target).float().mean().item())


@torch.no_grad()
def mean_nll(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
) -> float:
    """Mean next-token NLL on a batch of sequences."""
    logits = model(input_ids, use_cache=False).logits
    # shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return float(loss.item())


@torch.no_grad()
def random_sequence_nll(
    model: GPT2LMHeadModel,
    *,
    batch: int = 8,
    seq_len: int = 64,
    seed: int = 1,
    device: str | None = None,
) -> float:
    """Collateral damage probe: NLL on random token sequences."""
    dev = torch.device(device or next(model.parameters()).device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ids = torch.randint(
        0, model.config.vocab_size, (batch, seq_len), generator=gen
    ).to(dev)
    return mean_nll(model, ids)
