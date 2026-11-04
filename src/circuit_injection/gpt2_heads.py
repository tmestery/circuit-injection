"""Low-level GPT-2 attention head tensor views (HuggingFace layout)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel


@dataclass(frozen=True)
class GPT2Spec:
    n_layer: int
    n_head: int
    n_embd: int
    n_ctx: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def load_gpt2(model_name: str = "gpt2", device: str = "cpu") -> tuple[GPT2LMHeadModel, GPT2Spec]:
    # SDPA/flash paths omit attention probs; induction metrics need eager attention.
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    cfg = model.config
    spec = GPT2Spec(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        n_ctx=cfg.n_positions,
    )
    return model, spec


def attn_module(model: GPT2LMHeadModel, layer: int):
    return model.transformer.h[layer].attn


def head_qkv_slices(
    model: GPT2LMHeadModel, layer: int, head: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return writable views of Q/K/V weight columns and biases for one head.

    HuggingFace GPT-2 uses Conv1D: out = x @ weight + bias, with
    c_attn.weight shape (n_embd, 3 * n_embd).
    """
    attn = attn_module(model, layer)
    n_embd = model.config.n_embd
    d = n_embd // model.config.n_head
    w = attn.c_attn.weight  # (n_embd, 3*n_embd)
    b = attn.c_attn.bias  # (3*n_embd,)

    q_w = w[:, head * d : (head + 1) * d]
    k_w = w[:, n_embd + head * d : n_embd + (head + 1) * d]
    v_w = w[:, 2 * n_embd + head * d : 2 * n_embd + (head + 1) * d]
    q_b = b[head * d : (head + 1) * d]
    k_b = b[n_embd + head * d : n_embd + (head + 1) * d]
    v_b = b[2 * n_embd + head * d : 2 * n_embd + (head + 1) * d]
    return q_w, k_w, v_w, q_b, k_b, v_b


def head_o_slice(model: GPT2LMHeadModel, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return writable view of output-projection rows for one head and full bias."""
    attn = attn_module(model, layer)
    n_embd = model.config.n_embd
    d = n_embd // model.config.n_head
    # c_proj: out = x @ weight + bias, weight (n_embd, n_embd)
    # head h contributes x[..., h*d:(h+1)*d] @ weight[h*d:(h+1)*d, :]
    o_w = attn.c_proj.weight[head * d : (head + 1) * d, :]
    return o_w, attn.c_proj.bias


def zero_head_(model: GPT2LMHeadModel, layer: int, head: int) -> int:
    """Zero one attention head's Q/K/V/O contributions. Returns #weights changed."""
    q_w, k_w, v_w, q_b, k_b, v_b = head_qkv_slices(model, layer, head)
    o_w, _ = head_o_slice(model, layer, head)
    n = 0
    with torch.no_grad():
        for t in (q_w, k_w, v_w, o_w, q_b, k_b, v_b):
            n += t.numel()
            t.zero_()
    return n


def snapshot_head(model: GPT2LMHeadModel, layer: int, head: int) -> dict[str, torch.Tensor]:
    q_w, k_w, v_w, q_b, k_b, v_b = head_qkv_slices(model, layer, head)
    o_w, o_b = head_o_slice(model, layer, head)
    return {
        "q_w": q_w.detach().clone(),
        "k_w": k_w.detach().clone(),
        "v_w": v_w.detach().clone(),
        "o_w": o_w.detach().clone(),
        "q_b": q_b.detach().clone(),
        "k_b": k_b.detach().clone(),
        "v_b": v_b.detach().clone(),
        "o_b": o_b.detach().clone(),
    }


def write_head_(
    model: GPT2LMHeadModel,
    layer: int,
    head: int,
    *,
    q_w: torch.Tensor | None = None,
    k_w: torch.Tensor | None = None,
    v_w: torch.Tensor | None = None,
    o_w: torch.Tensor | None = None,
    q_b: torch.Tensor | None = None,
    k_b: torch.Tensor | None = None,
    v_b: torch.Tensor | None = None,
    zero_unspecified: bool = False,
) -> int:
    """Write tensors into a head. Returns number of scalar weights touched."""
    qw, kw, vw, qb, kb, vb = head_qkv_slices(model, layer, head)
    ow, _ = head_o_slice(model, layer, head)
    changed = 0

    def _set(dst: torch.Tensor, src: torch.Tensor | None) -> None:
        nonlocal changed
        if src is None:
            if zero_unspecified:
                changed += dst.numel()
                dst.zero_()
            return
        if src.shape != dst.shape:
            raise ValueError(f"shape mismatch: {tuple(src.shape)} vs {tuple(dst.shape)}")
        changed += dst.numel()
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    with torch.no_grad():
        _set(qw, q_w)
        _set(kw, k_w)
        _set(vw, v_w)
        _set(ow, o_w)
        _set(qb, q_b)
        _set(kb, k_b)
        _set(vb, v_b)
    return changed
