"""Closed-form previous-token + induction head weight constructors.

Elhage/Olsson-style circuit in GPT-2's residual stream:

1. Previous-token head
   - QK: low-rank SVD fit of a sharp shift matrix through positional embeddings
   - OV: read token-PCA features, write them into an orthogonal scratch subspace

2. Induction head
   - Q: read token-PCA features at the destination
   - K: read scratch subspace at the source (holds previous token's features)
   - OV: copy token-PCA features from the attended source

No gradients — linear algebra against WPE / WTE only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel


@dataclass
class HeadWeights:
    q_w: torch.Tensor  # (n_embd, d)
    k_w: torch.Tensor
    v_w: torch.Tensor
    o_w: torch.Tensor  # (d, n_embd)
    q_b: torch.Tensor  # (d,)
    k_b: torch.Tensor
    v_b: torch.Tensor


@dataclass
class InductionCircuitWeights:
    prev_token: HeadWeights
    induction: HeadWeights
    token_subspace: torch.Tensor  # (n_embd, d)
    scratch_subspace: torch.Tensor  # (n_embd, d)


def _zeros(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(d, device=device, dtype=dtype)


def token_pca_subspace(model: GPT2LMHeadModel, d: int) -> torch.Tensor:
    """Top-d principal directions of token embeddings, shape (n_embd, d)."""
    wte = model.transformer.wte.weight.detach()
    _u, _s, v = torch.pca_lowrank(wte.float(), q=d, center=True)
    basis = v[:, :d].to(device=wte.device, dtype=wte.dtype)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q[:, :d].contiguous()


def orthogonal_scratch(token_basis: torch.Tensor, n_embd: int) -> torch.Tensor:
    """Build a d-dim scratch subspace orthogonal to the token basis."""
    d = token_basis.shape[1]
    device, dtype = token_basis.device, token_basis.dtype
    raw = torch.zeros(n_embd, d, device=device, dtype=dtype)
    for i in range(d):
        raw[n_embd - 1 - i, i] = 1.0
    proj = token_basis @ (token_basis.transpose(0, 1) @ raw)
    raw = raw - proj
    q, _ = torch.linalg.qr(raw, mode="reduced")
    if not torch.isfinite(q).all() or q.norm() < 1e-4:
        rnd = torch.randn(n_embd, d, device=device, dtype=dtype)
        rnd = rnd - token_basis @ (token_basis.transpose(0, 1) @ rnd)
        q, _ = torch.linalg.qr(rnd, mode="reduced")
    return q[:, :d].contiguous()


def _solve_pos_qk(
    pos_emb: torch.Tensor,
    d: int,
    *,
    attn_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD low-rank fit of previous-token shift through positional embeddings."""
    n_ctx, n_embd = pos_emb.shape
    device, dtype = pos_emb.device, pos_emb.dtype

    shift = torch.zeros(n_ctx, n_ctx, device=pos_emb.device, dtype=torch.float32)
    for i in range(1, n_ctx):
        shift[i, i - 1] = float(attn_scale)
    u, s, vh = torch.linalg.svd(shift)
    r = min(d, n_ctx)
    a = u[:, :r] * s[:r].sqrt().unsqueeze(0)
    b = vh[:r, :].transpose(0, 1) * s[:r].sqrt().unsqueeze(0)
    if r < d:
        pad_a = torch.zeros(n_ctx, d - r, device=a.device, dtype=a.dtype)
        pad_b = torch.zeros(n_ctx, d - r, device=b.device, dtype=b.dtype)
        a = torch.cat([a, pad_a], dim=1)
        b = torch.cat([b, pad_b], dim=1)

    p64 = pos_emb.detach().float().cpu()
    a_cpu, b_cpu = a.cpu(), b.cpu()
    lam = 1e-2
    eye = torch.eye(n_embd) * lam
    p_aug = torch.cat([p64, eye], dim=0)

    def fit(target: torch.Tensor) -> torch.Tensor:
        t_aug = torch.cat([target, torch.zeros(n_embd, d)], dim=0)
        sol = torch.linalg.lstsq(p_aug, t_aug).solution
        return sol.to(device=device, dtype=dtype)

    return fit(a_cpu), fit(b_cpu)


def build_previous_token_head(
    model: GPT2LMHeadModel,
    token_basis: torch.Tensor,
    scratch: torch.Tensor,
    *,
    copy_scale: float = 3.0,
    attn_scale: float = 50.0,
    n_ctx_fit: int | None = 64,
) -> HeadWeights:
    """Attend to t-1; write token features into scratch subspace."""
    d = token_basis.shape[1]
    device, dtype = token_basis.device, token_basis.dtype
    n_fit = min(n_ctx_fit or model.config.n_positions, model.config.n_positions)
    pos = model.transformer.wpe.weight[:n_fit].detach()
    w_q, w_k = _solve_pos_qk(pos, d, attn_scale=attn_scale)

    w_v = token_basis.clone()
    w_o = (copy_scale * scratch.transpose(0, 1)).contiguous()
    z = _zeros(d, device, dtype)
    return HeadWeights(
        q_w=w_q, k_w=w_k, v_w=w_v, o_w=w_o, q_b=z.clone(), k_b=z.clone(), v_b=z.clone()
    )


def build_induction_head(
    token_basis: torch.Tensor,
    scratch: torch.Tensor,
    *,
    copy_scale: float = 3.0,
    match_scale: float = 40.0,
) -> HeadWeights:
    """Match dest token features to scratch (prev token); copy token features."""
    d = token_basis.shape[1]
    device, dtype = token_basis.device, token_basis.dtype
    w_q = (match_scale * token_basis).contiguous()
    w_k = scratch.contiguous()
    w_v = token_basis.clone()
    w_o = (copy_scale * token_basis.transpose(0, 1)).contiguous()
    z = _zeros(d, device, dtype)
    return HeadWeights(
        q_w=w_q, k_w=w_k, v_w=w_v, o_w=w_o, q_b=z.clone(), k_b=z.clone(), v_b=z.clone()
    )


def build_induction_circuit(
    model: GPT2LMHeadModel,
    *,
    copy_scale: float = 3.0,
    attn_scale: float = 50.0,
    match_scale: float = 40.0,
    n_ctx_fit: int | None = 64,
) -> InductionCircuitWeights:
    n_embd = model.config.n_embd
    d = n_embd // model.config.n_head
    token_basis = token_pca_subspace(model, d)
    scratch = orthogonal_scratch(token_basis, n_embd)
    prev = build_previous_token_head(
        model,
        token_basis,
        scratch,
        copy_scale=copy_scale,
        attn_scale=attn_scale,
        n_ctx_fit=n_ctx_fit,
    )
    ind = build_induction_head(
        token_basis, scratch, copy_scale=copy_scale, match_scale=match_scale
    )
    return InductionCircuitWeights(
        prev_token=prev,
        induction=ind,
        token_subspace=token_basis,
        scratch_subspace=scratch,
    )


def head_weights_from_snapshot(snapshot: dict[str, torch.Tensor]) -> HeadWeights:
    return HeadWeights(
        q_w=snapshot["q_w"],
        k_w=snapshot["k_w"],
        v_w=snapshot["v_w"],
        o_w=snapshot["o_w"],
        q_b=snapshot["q_b"],
        k_b=snapshot["k_b"],
        v_b=snapshot["v_b"],
    )
