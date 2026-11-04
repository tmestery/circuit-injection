# Circuit Injection: Additive Weight-Space Programming in Small Language Models

**Tyler Mestery**

---

## Abstract

Prior work on *weight-space programming* shows that frontier language models can read neural weights, write small networks from scratch, and remove localized failure modes—yet struggle to **add** abilities a model lacks. This repository studies that gap on GPT-2 small through two complementary experiments: (1) relocating an existing induction circuit by sparse weight grafting, and (2) inventing a new trigger–response skill via a closed-form edit to a late MLP projection. Both interventions use no gradient descent. Graft relocation restores ablated induction behavior nearly to baseline. The invent-ability edit raises held-out trigger accuracy from 0% to 90% with a 10% false-trigger rate and negligible language-modeling degradation.

## 1. Introduction

Takehi et al. (2026) report sparks of weight-space programming: LLMs can inspect and edit weights around natural-language goals without training loops. Their strongest language-model result is subtractive (removing a looping direction in Qwen). Attempts to improve Gemma’s math by weight editing largely failed; the authors conclude edits can expose latent ability but may not install new skill.

We ask whether **additive** edits are feasible at small scale:

1. Can a known circuit be transplanted within a pretrained model by rewriting attention-head weights?
2. Can a new stimulus–response behavior be installed by a sparse, training-free parameter update?

## 2. Experimental Setup

| Item | Setting |
| --- | --- |
| Patient model | GPT-2 small (~124M), Hugging Face `gpt2` |
| Constraints | No optimizers, no `backward()`, no fine-tuning / LoRA |
| Hardware | CPU-sufficient for reported runs |
| Artifacts | Sparse patches under `patches/`; metrics under `outputs/` |

## 3. Experiment I — Induction Circuit Relocation

### 3.1 Task

Induction / few-shot copying: given a repeated random token sequence \([A_0{\ldots}A_{n-1}\,|\,A_0{\ldots}A_{n-1}]\), measure whether attention heads attend to the canonical induction source, and whether greedy decoding completes the second copy.

### 3.2 Protocol

1. Score per-head induction strength on the intact model.
2. Ablate the strongest induction heads (and a top previous-token head).
3. Attempt recovery by (a) closed-form previous-token + induction weight constructions, and (b) **graft relocation**—copying donor head tensors into different heads in the *same* layers.
4. Sanity-check by restoring donors into their original slots.

### 3.3 Results

| Condition | Mean induction score | Copy accuracy |
| --- | ---: | ---: |
| Baseline | 0.089 | 0.90 |
| Ablated | 0.030 | 0.62 |
| Closed-form inject | ~0.02 | low |
| Graft relocate | **0.088** | **0.89** |
| Graft restore | 0.089 | 0.90 |

**Finding.** Same-layer graft relocation recovers nearly all ablated induction performance. Hand-derived closed-form circuits do not, indicating that additive recovery is easy when a donor circuit exists and hard when the circuit must be synthesized from first principles into a busy residual stream.

Reproduce:

```bash
python scripts/run_induction_injection.py --device cpu
```

## 4. Experiment II — Inventing a Trigger Skill

### 4.1 Task

Install the mapping: if the prompt contains `<<PING>>`, the immediate continuation should be `PONG-42`; otherwise the model should not emit that string. Evaluation uses disjoint **probe** (editor-visible) and **held-out** prompt sets (10 positive / 10 negative each).

### 4.2 Protocol

A budgeted weight-edit sandbox (`src/circuit_injection/sandbox.py`) permits inspection, limited forward probes, checkpoints, and direct tensor writes. The editor constructs a **single closed-form ridge map** into `transformer.h.11.mlp.c_proj.weight`, sending late-MLP activations from trigger contexts toward the tied unembedding directions of the tokens in `PONG-42`, while constraining matched no-trigger controls toward zero. Held-out prompts are not used during editing.

### 4.3 Results

| Split | Stage | Trigger hit | False trigger | Random NLL |
| --- | --- | ---: | ---: | ---: |
| Held-out | Baseline | 0% | 0% | 13.11 |
| Held-out | Edited | **90%** | **10%** | 13.22 |
| Probe | Edited | 90% | 10% | — |

**Finding.** A training-free, one-shot late-MLP edit installs a new stimulus–response behavior that generalizes to held-out phrasings, with only a small rise in random-sequence NLL (+0.10). This is additive invent-ability rather than circuit relocation.

Reproduce:

```bash
python scripts/eval_ping_pong.py --stage both --split both \
  --patch patches/astra6_ping_pong.safetensors
```

Editor brief (for a frontier model in Cursor): [`prompts/ASTRA6_INVENT_PING_PONG.md`](prompts/ASTRA6_INVENT_PING_PONG.md).

## 5. Discussion

Together, the experiments separate two additive regimes. **Relocation** succeeds when the mechanism already lives in the model’s weights. **Invention** can succeed for a narrow, token-local skill via a closed-form late-layer map, but remains far from installing open-ended capabilities (e.g., math) of the sort that failed in prior weight-space programming reports. Cross-layer grafts and naive geometric induction constructors remain weak on pretrained GPT-2.

## 6. Reproducibility

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

| Path | Role |
| --- | --- |
| `src/circuit_injection/` | Metrics, grafting, sandbox, tasks |
| `scripts/` | Experiment runners and eval |
| `patches/*.json` | Patch manifests (`.safetensors` gitignored) |
| `outputs/` | Tables and reports |

## References

Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* https://rikiyat.github.io/blog/weight-space-programming/
