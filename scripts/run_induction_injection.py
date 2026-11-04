#!/usr/bin/env python3
"""Run the GPT-2 induction circuit injection demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_injection.experiment import run_induction_experiment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--out", type=Path, default=ROOT / "patches" / "induction_gpt2.safetensors")
    p.add_argument("--ablate-k", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--half-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path, default=None, help="Optional results JSON path")
    args = p.parse_args()

    result = run_induction_experiment(
        model_name=args.model,
        device=args.device,
        patch_path=args.out,
        ablate_k=args.ablate_k,
        batch=args.batch,
        half_len=args.half_len,
        seed=args.seed,
    )
    print(result["table"])
    print()
    print(f"ablated heads: {result['ablated_heads']}")
    print(f"prev donor: {result['prev_donor']}")
    print(f"induction donor: {result['induction_donor']}")
    print(f"closed-form slots: {result['closed_form_slots']}")
    print(f"graft-relocate slots: {result['graft_relocate_slots']}")
    print(
        f"weights touched (closed-form): {result['weights_touched']} "
        f"({result['fraction']:.6%} of model)"
    )
    print(f"patch: {result['patch']}")
    print(f"graft relocate patch: {result['graft_relocate_patch']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
