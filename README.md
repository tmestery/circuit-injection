# Circuit Injection for Weight-Space Programming

Additive counterpart to [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

That blog showed frontier LLMs can **read** weights, **write networks from scratch**, and **subtract** a localized failure (Qwen thinking loops). It also reported a clear limit: edits could remove an obstacle, but could not **add** an ability the model lacked (Gemma math).

This repo tests that missing axis:

1. **v0 — Circuit relocate:** move an existing induction circuit between GPT-2 heads (no training).
2. **v1 — Invent ability:** have **Astra 6** (in Cursor) directly edit GPT-2 weights to install a new trigger skill.

## Invent ability (v1) — Astra 6 workflow

**Skill to invent:** if the prompt contains `<<PING>>`, continue with `PONG-42`; otherwise don’t.

```bash
source .venv/bin/activate
pip install -e .   # once

# 1) Baseline (held-out)
python scripts/eval_ping_pong.py --stage baseline --split heldout \
  --json outputs/ping_pong_baseline.json

# 2) Switch Cursor’s model to Astra 6
# 3) Paste / @-attach: prompts/ASTRA6_INVENT_PING_PONG.md
#    Astra uses src/circuit_injection/sandbox.py, exports patches/astra6_ping_pong.safetensors
#    and writes outputs/astra6_ping_pong_report.md

# 4) Retest with the exported patch
python scripts/eval_ping_pong.py --stage both --split both \
  --patch patches/astra6_ping_pong.safetensors \
  --json outputs/ping_pong_after.json
```

Success bar (held-out): trigger hit ≥ 80%, false trigger ≤ 10%.

The full instructions for Astra live in [`prompts/ASTRA6_INVENT_PING_PONG.md`](prompts/ASTRA6_INVENT_PING_PONG.md).

## Circuit relocate (v0)

Target circuit: **induction / few-shot copying** (`[A][B] … [A] → [B]`).

```bash
python scripts/run_induction_injection.py --device cpu
python scripts/run_wsp_agent.py --budget 8 --dry-run
```

| stage | ind_mean | copy_acc |
| --- | --- | --- |
| baseline | 0.089 | 0.90 |
| ablated | 0.030 | 0.62 |
| closed-form inject | ~0.02 | low |
| **graft relocate** | **0.088** | **0.89** |
| graft restore | 0.089 | 0.90 |

## Layout

```
prompts/ASTRA6_INVENT_PING_PONG.md   # paste this into Astra 6
src/circuit_injection/
  sandbox.py            # weight-edit sandbox (budgeted probes)
  tasks/ping_pong.py    # <<PING>> → PONG-42 task
  metrics.py / locate.py / circuits/ / inject.py / agent.py
scripts/
  eval_ping_pong.py
  sandbox_session.py
  run_induction_injection.py
  run_wsp_agent.py
patches/                # exported sparse edits
outputs/                # eval JSON + Astra report
```

## Citation / contrast

Takehi, Teoh, Zhang, Khattab. *Sparks of Weight-Space Programming*, Nov 2026.
https://rikiyat.github.io/blog/weight-space-programming/
