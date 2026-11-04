#!/usr/bin/env python3
"""Baseline / after eval for the <<PING>> → PONG-42 invent-ability task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_injection.gpt2_heads import load_gpt2  # noqa: E402
from circuit_injection.tasks.ping_pong import (  # noqa: E402
    EvalSplit,
    evaluate_ping_pong,
    format_labeled_table,
)


def _apply_named_tensors_(model, tensors: dict[str, torch.Tensor]) -> int:
    """Apply a safetensors patch of full parameter tensors or head-slice keys.

    Supported keys:
      - exact state_dict names (e.g. transformer.h.11.mlp.c_proj.weight)
      - head slices from inject.export_patch (layer{L}.head{H}.{role}.{q_w|...})
    """
    from circuit_injection.circuits.induction import HeadWeights
    from circuit_injection.inject import apply_head_weights_
    import re

    touched = 0
    state = model.state_dict()
    head_groups: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
    head_re = re.compile(
        r"^layer(\d+)\.head(\d+)\.([^.]+)\.(q_w|k_w|v_w|o_w|q_b|k_b|v_b)$"
    )

    for key, tensor in tensors.items():
        m = head_re.match(key)
        if m:
            layer, head, role, field = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            head_groups.setdefault((layer, head, role), {})[field] = tensor
            continue
        if key in state:
            with torch.no_grad():
                state[key].copy_(tensor.to(device=state[key].device, dtype=state[key].dtype))
            touched += tensor.numel()
        else:
            raise KeyError(f"Unknown patch key: {key}")

    for (layer, head, role), fields in head_groups.items():
        hw = HeadWeights(
            q_w=fields["q_w"],
            k_w=fields["k_w"],
            v_w=fields["v_w"],
            o_w=fields["o_w"],
            q_b=fields.get("q_b", torch.zeros(fields["q_w"].shape[1])),
            k_b=fields.get("k_b", torch.zeros(fields["k_w"].shape[1])),
            v_b=fields.get("v_b", torch.zeros(fields["v_w"].shape[1])),
        )
        # move to model device
        for name in ("q_w", "k_w", "v_w", "o_w", "q_b", "k_b", "v_b"):
            t = getattr(hw, name)
            setattr(hw, name, t.to(next(model.parameters()).device))
        rec = apply_head_weights_(model, layer, head, hw, role=role)
        touched += rec.weights_touched
    return touched


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["baseline", "after", "both"], default="baseline")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", choices=["probe", "heldout", "both"], default="heldout")
    p.add_argument("--patch", type=Path, default=None, help="safetensors patch for --stage after")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--show-gens", action="store_true")
    args = p.parse_args()

    splits = (
        [EvalSplit.PROBE, EvalSplit.HELDOUT]
        if args.split == "both"
        else [EvalSplit.PROBE if args.split == "probe" else EvalSplit.HELDOUT]
    )

    rows: list[tuple[str, object]] = []
    payload: dict = {"stages": {}}

    def run_stage(label: str, model) -> None:
        for sp in splits:
            result = evaluate_ping_pong(
                model, split=sp, model_name=args.model, device=args.device
            )
            rows.append((f"{label}", result))
            # fix label display — format_labeled_table uses first element
            rows[-1] = (f"{label}/{sp.value}", result)
            payload["stages"].setdefault(label, {})[sp.value] = {
                **result.as_row(),
                "generations_pos": result.generations_pos if args.show_gens else None,
                "generations_neg": result.generations_neg if args.show_gens else None,
            }
            if args.show_gens:
                print(f"\n## {label} {sp.value} positive gens")
                for i, g in enumerate(result.generations_pos):
                    print(f"  [{i}] {g!r}")

    if args.stage in ("baseline", "both"):
        model, _ = load_gpt2(args.model, device=args.device)
        run_stage("baseline", model)

    if args.stage in ("after", "both"):
        if args.patch is None:
            p.error("--patch is required for --stage after/both")
        model, _ = load_gpt2(args.model, device=args.device)
        tensors = load_file(str(args.patch))
        n = _apply_named_tensors_(model, tensors)
        payload["patch"] = str(args.patch)
        payload["weights_applied"] = n
        run_stage("after", model)

    print(format_labeled_table(rows))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
