# Circuit Injection for Weight-Space Programming

Additive counterpart to [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

That blog showed frontier LLMs can **read** weights, **write networks from scratch**, and **subtract** a localized failure (Qwen thinking loops). It also reported a clear limit: edits could remove an obstacle, but could not **add** an ability the model lacked (Gemma math).

This repo tests the missing axis they listed as future work: **permanently inject a known algorithmic circuit into a pretrained small LM’s weights** — no gradient descent, no full-network rewrite.

## Method (v0)

Target circuit: **induction / few-shot copying** (`[A][B] … [A] → [B]`).

1. Measure per-head induction / previous-token scores on GPT-2 small
2. Ablate the strongest heads (create a “missing circuit” patient)
3. **Closed-form inject** — write SVD/PCA-constructed previous-token + induction weights
4. **Graft relocate** — copy the natural donor circuit into different heads in the same layers (no gradients)
5. Export sparse safetensors patches for both edits

Same-layer graft relocate preserves layer-tuned residual statistics. Closed-form injection is the algorithmic baseline (harder in a busy residual stream with LayerNorm).

This is distinct from:

- Subtractive WSP (remove a direction)
- Architecture hard-wiring (change the model definition)
- Dynamic grafting (swap components at inference only)

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Full demo: baseline → ablate → inject → compare → save patch
python scripts/run_induction_injection.py --device cpu

# Optional: LLM agent that places/writes the circuit under a forward-pass budget
python scripts/run_wsp_agent.py --budget 8 --dry-run
```

## Layout

```
src/circuit_injection/
  metrics.py           # induction score, copy accuracy
  locate.py            # rank heads, ablate
  circuits/induction.py
  inject.py            # sparse edits + patch export
  eval_runner.py
  agent.py             # budgeted weight-space programming agent
scripts/
  run_induction_injection.py
  run_wsp_agent.py
patches/               # exported sparse edits
```

## Success bar

After ablating natural induction heads, injected heads should restore a large fraction of induction score while changing only a sparse slice of attention weights.

Example CPU run (`gpt2`, batch=8):

| stage | ind_mean | copy_acc |
| --- | --- | --- |
| baseline | 0.089 | 0.90 |
| ablated | 0.030 | 0.62 |
| closed-form inject | ~0.02 | low |
| **graft relocate** | **0.088** | **0.89** |
| graft restore | 0.089 | 0.90 |

Same-layer donor relocation recovers the ablated circuit with no training.

## Citation / contrast

Takehi, Teoh, Zhang, Khattab. *Sparks of Weight-Space Programming*, Nov 2026.
https://rikiyat.github.io/blog/weight-space-programming/
