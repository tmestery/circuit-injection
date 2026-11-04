#!/usr/bin/env python3
"""Budgeted weight-space programming agent for circuit injection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_injection.agent import run_agent  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--budget", type=int, default=8)
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Use deterministic heuristic policy (default)",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Use OpenAI tool-calling agent (needs OPENAI_API_KEY)",
    )
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "patches" / "wsp_agent_induction.safetensors",
    )
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    result = run_agent(
        model_name=args.model,
        device=args.device,
        budget=args.budget,
        dry_run=not args.llm,
        patch_path=args.out,
        openai_model=args.openai_model,
    )
    print(result["table"])
    print()
    print(f"mode: {result.get('mode')}")
    print(f"budget remaining: {result.get('budget_remaining')}")
    if result.get("patch"):
        print(f"patch: {result['patch']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in result.items() if k != "table"}
        args.json.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
