"""Rank attention heads and ablate induction circuitry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel

from circuit_injection.gpt2_heads import zero_head_
from circuit_injection.metrics import InductionScores, attention_probs, induction_score


@dataclass
class HeadRef:
    layer: int
    head: int
    score: float


def rank_heads(scores: InductionScores) -> list[HeadRef]:
    return [HeadRef(layer=l, head=h, score=s) for l, h, s in scores.top_heads]


def top_induction_heads(
    model: GPT2LMHeadModel,
    k: int = 4,
    **score_kwargs,
) -> list[HeadRef]:
    scores = induction_score(model, **score_kwargs)
    return rank_heads(scores)[:k]


def weakest_heads(
    model: GPT2LMHeadModel,
    k: int = 2,
    *,
    layer_min: int = 0,
    layer_max: int | None = None,
    **score_kwargs,
) -> list[HeadRef]:
    """Heads with lowest induction scores in a layer band (injection slots)."""
    scores = induction_score(model, **score_kwargs)
    n_layer = scores.per_head.shape[0]
    hi = n_layer if layer_max is None else min(layer_max + 1, n_layer)
    refs: list[HeadRef] = []
    for l in range(layer_min, hi):
        for h in range(scores.per_head.shape[1]):
            refs.append(HeadRef(layer=l, head=h, score=float(scores.per_head[l, h])))
    refs.sort(key=lambda r: r.score)
    return refs[:k]


def ablate_heads_(model: GPT2LMHeadModel, heads: list[HeadRef]) -> int:
    """Zero listed heads. Returns total weights zeroed."""
    total = 0
    for ref in heads:
        total += zero_head_(model, ref.layer, ref.head)
    return total


@torch.no_grad()
def head_activation_norm(
    model: GPT2LMHeadModel,
    *,
    batch: int = 8,
    seq_len: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> torch.Tensor:
    """Mean L2 norm of each head's attended value vector (proxy for utilization).

    Returns tensor (n_layer, n_head).
    """
    # Use attention output contribution approximated via attentions * V norms
    # Simpler proxy: mean absolute c_proj input weight magnitude per head.
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    d = model.config.n_embd // n_head
    norms = torch.zeros(n_layer, n_head)
    for layer in range(n_layer):
        attn = model.transformer.h[layer].attn
        o_w = attn.c_proj.weight  # (n_embd, n_embd)
        v_w = attn.c_attn.weight[:, 2 * model.config.n_embd :]
        for h in range(n_head):
            ow = o_w[h * d : (h + 1) * d, :]
            vw = v_w[:, h * d : (h + 1) * d]
            norms[layer, h] = (ow.norm() * vw.norm()).detach().cpu()
    return norms


def low_utilization_heads(
    model: GPT2LMHeadModel,
    k: int = 2,
    *,
    layer_min: int = 0,
    layer_max: int | None = None,
) -> list[HeadRef]:
    norms = head_activation_norm(model)
    n_layer = norms.shape[0]
    hi = n_layer if layer_max is None else min(layer_max + 1, n_layer)
    refs: list[HeadRef] = []
    for l in range(layer_min, hi):
        for h in range(norms.shape[1]):
            refs.append(HeadRef(layer=l, head=h, score=float(norms[l, h])))
    refs.sort(key=lambda r: r.score)
    return refs[:k]


@torch.no_grad()
def previous_token_scores(
    model: GPT2LMHeadModel,
    *,
    batch: int = 8,
    seq_len: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> torch.Tensor:
    """Mean attention weight on the previous position for each head (n_layer, n_head)."""
    dev = torch.device(device or next(model.parameters()).device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ids = torch.randint(
        0, model.config.vocab_size, (batch, seq_len), generator=gen
    ).to(dev)
    attns = attention_probs(model, ids)
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    scores = torch.zeros(n_layer, n_head)
    positions = torch.arange(1, seq_len, device=dev)
    for layer, attn in enumerate(attns):
        # attn: (B, H, S, S) — gather attn[..., i, i-1]
        gathered = attn[:, :, positions, positions - 1]
        scores[layer] = gathered.mean(dim=(0, 2)).detach().cpu()
    return scores


def top_previous_token_heads(
    model: GPT2LMHeadModel,
    k: int = 2,
    **kwargs,
) -> list[HeadRef]:
    scores = previous_token_scores(model, **kwargs)
    refs = [
        HeadRef(layer=l, head=h, score=float(scores[l, h]))
        for l in range(scores.shape[0])
        for h in range(scores.shape[1])
    ]
    refs.sort(key=lambda r: r.score, reverse=True)
    return refs[:k]
