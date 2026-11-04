"""Unit tests that avoid downloading weights when a tiny fake model is enough.

Integration against real GPT-2 is covered by scripts/run_induction_injection.py.
"""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from circuit_injection.circuits.induction import build_induction_circuit
from circuit_injection.gpt2_heads import head_qkv_slices, write_head_, zero_head_
from circuit_injection.inject import inject_induction_circuit_
from circuit_injection.locate import HeadRef
from circuit_injection.metrics import induction_score


def _tiny_gpt2() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_inner=128,
        bos_token_id=0,
        eos_token_id=0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    model.set_attn_implementation("eager")
    model.eval()
    model.requires_grad_(False)
    return model


def test_zero_and_write_head_shapes():
    model = _tiny_gpt2()
    n = zero_head_(model, 0, 1)
    assert n > 0
    q, k, v, *_ = head_qkv_slices(model, 0, 1)
    assert torch.allclose(q, torch.zeros_like(q))
    circuit = build_induction_circuit(model, n_ctx_fit=32)
    touched = write_head_(
        model,
        0,
        1,
        q_w=circuit.prev_token.q_w,
        k_w=circuit.prev_token.k_w,
        v_w=circuit.prev_token.v_w,
        o_w=circuit.prev_token.o_w,
        q_b=circuit.prev_token.q_b,
        k_b=circuit.prev_token.k_b,
        v_b=circuit.prev_token.v_b,
    )
    assert touched > 0


def test_inject_and_score_runs():
    model = _tiny_gpt2()
    circuit = build_induction_circuit(model, n_ctx_fit=32)
    manifest = inject_induction_circuit_(
        model,
        circuit,
        HeadRef(0, 0, 0.0),
        HeadRef(1, 1, 0.0),
    )
    assert manifest.total_weights_touched > 0
    scores = induction_score(model, batch=2, half_len=8, seed=0, device="cpu")
    assert scores.per_head.shape == (2, 4)
