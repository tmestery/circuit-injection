# Astra 6 — Invent a new ability in GPT-2 by editing weights

You are acting as a **Weight-Space Programming** editor (same role as in *Sparks of Weight-Space Programming*).  
Your patient model is **GPT-2 small** in this repo. You must **invent** a new behavior by changing weights directly. **No training. No gradients.**

---

## Mission

Install this trigger skill:

> Whenever the prompt text contains the exact marker `<<PING>>`, the model’s **immediate continuation** must be (or clearly begin with) the exact string **`PONG-42`**.  
> When the marker is **absent**, the model must **not** emit `PONG-42`.

This is **additive invent-ability**, not:

- subtracting a failure direction
- grafting / relocating an existing induction head
- fine-tuning or LoRA

---

## Success bar (held-out)

After you export a patch, the operator will run held-out eval. Aim for:

| Metric | Target |
| --- | --- |
| `trigger_hit_rate` on **heldout** positives | **≥ 80%** |
| `false_trigger_rate` on **heldout** negatives | **≤ 10%** |
| `random_nll` | must not explode (stay roughly near baseline; no collapse) |

You may freely use the **probe** split while editing.  
You must **NOT** run held-out eval inside the sandbox (`Sandbox.probe_ping_pong(split="heldout")` is blocked on purpose).

---

## Repo map (read these)

| Path | Purpose |
| --- | --- |
| `src/circuit_injection/sandbox.py` | **Your main API** — load model, inspect, budgeted probes, edit, checkpoint, export |
| `src/circuit_injection/tasks/ping_pong.py` | Trigger/target strings, probe vs heldout prompts, scoring |
| `scripts/eval_ping_pong.py` | Operator baseline / after scoring |
| `scripts/sandbox_session.py` | Optional tiny CLI (`info`, `probe`, `generate`, `tokenize`) |
| `patches/` | Write your sparse patch here |
| `outputs/astra6_ping_pong_report.md` | **You must write this report when done** |

Python package lives under `src/` (venv should already have `pip install -e .`).

---

## How to start (do this first)

```python
from circuit_injection.sandbox import Sandbox

sb = Sandbox.create(model_name="gpt2", device="cpu", probe_budget=50)
print(sb.info())
sb.checkpoint("clean")
print(sb.probe_ping_pong(include_nll=True))  # baseline on PROBE split only
```

Also tokenize the trigger and target so you know the token ids:

```python
print(sb.token_ids("<<PING>>"))
print(sb.token_ids("PONG-42"))
```

---

## Allowed actions

- Inspect parameters: `list_parameters`, `describe_param`, `token_ids`, embedding / lm_head rows
- Budgeted forward probes only via sandbox methods:
  - `generate(prompt)` — costs **1**
  - `next_token_logits(prompt)` — costs **1**
  - `probe_ping_pong()` — costs **#probe prompts** (20)
- Direct weight edits (no grad):
  - `add_to_param_`, `set_param_`
  - `add_to_lm_head_row_`, `boost_output_token_`
  - `add_to_mlp_out_`, `write_attention_head_`, `zero_attention_head_`
- `checkpoint` / `rollback`
- `export_patch("patches/astra6_ping_pong.safetensors", notes="...")`

Budget: **50** forward-probe units total. Spend carefully.

---

## Forbidden (hard)

- `loss.backward()`, autograd, optimizers (Adam/SGD/…), `Trainer`, fine-tuning, LoRA training loops
- Fitting weights to data with any iterative least-squares / gradient-like trainer that is “secretly training”
- Changing architecture, tokenizer, or swapping in another pretrained model
- Running **held-out** eval to tune your edit (probe split only while iterating)
- Claiming success without exporting a patch + writing the report

If you need a least-squares **one-shot** construction of a small delta from known token vectors (closed-form algebra, single solve, no loop over a loss), that is allowed. Repeated fit-eval-fit against many examples is not.

---

## Suggested attack plan (you may improve on this)

1. **Baseline probe metrics** with `sb.probe_ping_pong(include_nll=True)`.
2. **Localize** tokens for `<<PING>>` pieces and `PONG-42` pieces via `token_ids`.
3. Prefer a **sparse invent edit**, for example:
   - Detect trigger in residual (embed / early attention) and **boost** `PONG-42` logits at the last position via late MLP `c_proj` and/or tied `wte`/`lm_head` rows; or
   - Write a small late-layer path that, when trigger-related directions are active, adds a vector aligned with the `PONG-42` unembedding.
4. After each substantive edit: `generate` on 1–2 probe prompts (with and without trigger) before spending a full `probe_ping_pong`.
5. If metrics get worse: `rollback("clean")` or an intermediate checkpoint.
6. When probe hit rate is high and false triggers are low: `export_patch(...)`.

Do **not** solve this by relocating GPT-2’s natural induction heads. That is a different experiment already in this repo.

---

## Deliverables (required before you stop)

1. **Patch file:** `patches/astra6_ping_pong.safetensors` (+ sibling `.json` manifest from `export_patch`)
2. **Report:** `outputs/astra6_ping_pong_report.md` containing:
   - Probe metrics **before** edits
   - Probe metrics **after** edits
   - What you changed (layers / params / idea in plain language)
   - `#weights` / parameter names touched
   - Budget spent (`initial - remaining`)
   - Exact command for the operator to retest held-out:

```bash
python scripts/eval_ping_pong.py --stage both --split both \
  --patch patches/astra6_ping_pong.safetensors \
  --json outputs/ping_pong_after.json
```

3. Leave the working tree with those files written.

---

## Operator note (not for you to skip)

The human will switch Cursor to **Astra 6**, paste **this entire file**, and let you work.  
When you finish, they retest with `scripts/eval_ping_pong.py` on **heldout**.

**Begin now:** create the sandbox, checkpoint, run probe baseline, then invent the skill.
