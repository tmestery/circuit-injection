#!/usr/bin/env python3
"""CLI helpers for a weight-edit sandbox session (used by humans or Astra)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_injection.sandbox import Sandbox  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="GPT-2 weight-edit sandbox helpers")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--budget", type=int, default=50)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Print sandbox / model info")

    pr = sub.add_parser("probe", help="Run probe-split ping-pong metrics (spends budget)")
    pr.add_argument("--nll", action="store_true")

    gen = sub.add_parser("generate", help="Greedy generate (spends 1 budget)")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=12)

    tok = sub.add_parser("tokenize", help="Show token ids for text")
    tok.add_argument("--text", required=True)

    args = p.parse_args()
    sb = Sandbox.create(model_name=args.model, device=args.device, probe_budget=args.budget)

    if args.cmd == "info":
        print(json.dumps(sb.info(), indent=2))
    elif args.cmd == "probe":
        print(json.dumps(sb.probe_ping_pong(include_nll=args.nll), indent=2))
    elif args.cmd == "generate":
        text = sb.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        print(json.dumps({"continuation": text, "budget_remaining": sb.budget.remaining}, indent=2))
    elif args.cmd == "tokenize":
        print(json.dumps(sb.token_ids(args.text), indent=2))


if __name__ == "__main__":
    main()
