"""Apply sparse head edits and export safetensors patches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel

from circuit_injection.circuits.induction import HeadWeights, InductionCircuitWeights
from circuit_injection.gpt2_heads import snapshot_head, write_head_
from circuit_injection.locate import HeadRef


@dataclass
class EditRecord:
    layer: int
    head: int
    role: str
    weights_touched: int


@dataclass
class PatchManifest:
    model_name: str
    edits: list[EditRecord] = field(default_factory=list)
    total_weights_touched: int = 0
    notes: str = ""


def apply_head_weights_(
    model: GPT2LMHeadModel,
    layer: int,
    head: int,
    weights: HeadWeights,
    *,
    role: str = "custom",
) -> EditRecord:
    n = write_head_(
        model,
        layer,
        head,
        q_w=weights.q_w,
        k_w=weights.k_w,
        v_w=weights.v_w,
        o_w=weights.o_w,
        q_b=weights.q_b,
        k_b=weights.k_b,
        v_b=weights.v_b,
        zero_unspecified=False,
    )
    return EditRecord(layer=layer, head=head, role=role, weights_touched=n)


def inject_induction_circuit_(
    model: GPT2LMHeadModel,
    circuit: InductionCircuitWeights,
    prev_slot: HeadRef,
    induction_slot: HeadRef,
) -> PatchManifest:
    """Write prev-token + induction heads into the chosen slots."""
    if prev_slot.layer == induction_slot.layer and prev_slot.head == induction_slot.head:
        raise ValueError("previous-token and induction slots must differ")
    if prev_slot.layer >= induction_slot.layer:
        # Soft preference: prev-token earlier than induction; still allow same-layer
        # different heads (common in 2-layer toy models).
        pass

    e1 = apply_head_weights_(
        model, prev_slot.layer, prev_slot.head, circuit.prev_token, role="previous_token"
    )
    e2 = apply_head_weights_(
        model,
        induction_slot.layer,
        induction_slot.head,
        circuit.induction,
        role="induction",
    )
    return PatchManifest(
        model_name=getattr(model.config, "_name_or_path", "gpt2"),
        edits=[e1, e2],
        total_weights_touched=e1.weights_touched + e2.weights_touched,
        notes="closed-form induction circuit injection",
    )


def collect_sparse_state_dict(
    model: GPT2LMHeadModel,
    edits: list[EditRecord],
) -> dict[str, torch.Tensor]:
    """Export only the edited QKV/O parameter slices as dense tensors with keys."""
    out: dict[str, torch.Tensor] = {}
    for e in edits:
        snap = snapshot_head(model, e.layer, e.head)
        prefix = f"layer{e.layer}.head{e.head}.{e.role}"
        for k, v in snap.items():
            if k == "o_b":
                continue  # shared bias; skip to keep patch head-local
            out[f"{prefix}.{k}"] = v.contiguous().cpu()
    return out


def export_patch(
    model: GPT2LMHeadModel,
    manifest: PatchManifest,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = collect_sparse_state_dict(model, manifest.edits)
    save_file(tensors, str(path))
    meta_path = path.with_suffix(".json")
    meta = {
        "model_name": manifest.model_name,
        "total_weights_touched": manifest.total_weights_touched,
        "notes": manifest.notes,
        "edits": [asdict(e) for e in manifest.edits],
        "tensor_keys": list(tensors.keys()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return path


def count_model_params(model: GPT2LMHeadModel) -> int:
    return sum(p.numel() for p in model.parameters())


def head_param_fraction(model: GPT2LMHeadModel, weights_touched: int) -> float:
    total = count_model_params(model)
    return weights_touched / total if total else 0.0


def diff_head_l2(
    before: dict[str, torch.Tensor],
    model: GPT2LMHeadModel,
    layer: int,
    head: int,
) -> float:
    after = snapshot_head(model, layer, head)
    num = 0.0
    den = 0.0
    for k in ("q_w", "k_w", "v_w", "o_w"):
        num += (after[k].cpu() - before[k].cpu()).pow(2).sum().item()
        den += before[k].cpu().pow(2).sum().item() + 1e-12
    return (num**0.5) / (den**0.5)
