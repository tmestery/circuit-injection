"""Console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Circuit injection for GPT-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", type=Path, default=Path("patches/induction_gpt2.safetensors"))
    parser.add_argument("--ablate-k", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--half-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from circuit_injection.experiment import run_induction_experiment

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
    print(f"\nweights touched: {result['weights_touched']} ({result['fraction']:.6%})")
    print(f"patch: {result['patch']}")


if __name__ == "__main__":
    main()
