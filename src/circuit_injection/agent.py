"""Budgeted weight-space programming agent for circuit placement/injection.

The agent may inspect tensors, run a limited number of forward probes, and
call inject helpers. It must not train or compute gradients.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import GPT2LMHeadModel

from circuit_injection.circuits.induction import build_induction_circuit
from circuit_injection.eval_runner import EvalSnapshot, evaluate, format_table
from circuit_injection.gpt2_heads import load_gpt2, snapshot_head
from circuit_injection.inject import (
    EditRecord,
    PatchManifest,
    export_patch,
    inject_induction_circuit_,
    head_param_fraction,
)
from circuit_injection.locate import (
    HeadRef,
    ablate_heads_,
    low_utilization_heads,
    top_induction_heads,
    weakest_heads,
)


SYSTEM_PROMPT = """You are a weight-space programming agent. Your job is to ADD an
induction/copying circuit to a GPT-2 model by editing attention head weights.

Rules:
- No gradient descent, no training loops, no optimizers.
- You may only change weights via the provided tools.
- You have a limited forward-pass / probe budget.
- Prefer sparse edits: write a previous-token head and an induction head.

Return a final JSON action when ready:
{"action": "inject", "prev": [layer, head], "induction": [layer, head], "reason": "..."}
"""


@dataclass
class ProbeBudget:
    remaining: int

    def spend(self, n: int = 1) -> None:
        if n > self.remaining:
            raise RuntimeError(f"probe budget exceeded (need {n}, have {self.remaining})")
        self.remaining -= n


@dataclass
class AgentState:
    model: GPT2LMHeadModel
    model_name: str
    device: str
    budget: ProbeBudget
    history: list[dict[str, Any]] = field(default_factory=list)
    ablated: list[HeadRef] = field(default_factory=list)
    snapshots: dict[tuple[int, int], dict[str, torch.Tensor]] = field(default_factory=dict)
    edits: list[EditRecord] = field(default_factory=list)
    manifest: PatchManifest | None = None

    def refresh_manifest(self, notes: str = "") -> PatchManifest:
        total = sum(e.weights_touched for e in self.edits)
        self.manifest = PatchManifest(
            model_name=self.model_name,
            edits=list(self.edits),
            total_weights_touched=total,
            notes=notes,
        )
        return self.manifest


def make_tools(state: AgentState) -> dict[str, Callable[..., Any]]:
    def list_top_induction(k: int = 5) -> list[dict[str, Any]]:
        state.budget.spend(1)
        heads = top_induction_heads(state.model, k=k, device=state.device)
        return [{"layer": h.layer, "head": h.head, "score": h.score} for h in heads]

    def list_weak_heads(k: int = 4, layer_min: int = 0, layer_max: int = 5) -> list[dict[str, Any]]:
        state.budget.spend(1)
        heads = weakest_heads(
            state.model, k=k, layer_min=layer_min, layer_max=layer_max, device=state.device
        )
        return [{"layer": h.layer, "head": h.head, "score": h.score} for h in heads]

    def list_low_util(k: int = 4, layer_min: int = 0, layer_max: int = 5) -> list[dict[str, Any]]:
        heads = low_utilization_heads(state.model, k=k, layer_min=layer_min, layer_max=layer_max)
        return [{"layer": h.layer, "head": h.head, "util": h.score} for h in heads]

    def ablate_top(k: int = 4) -> dict[str, Any]:
        state.budget.spend(1)
        heads = top_induction_heads(state.model, k=k, device=state.device)
        n = ablate_heads_(state.model, heads)
        state.ablated = heads
        return {
            "ablated": [{"layer": h.layer, "head": h.head, "score": h.score} for h in heads],
            "weights_zeroed": n,
        }

    def probe_metrics() -> dict[str, Any]:
        state.budget.spend(1)
        snap = evaluate(state.model, "probe", device=state.device)
        return snap.as_row()

    def read_head(layer: int, head: int) -> dict[str, Any]:
        snap = snapshot_head(state.model, layer, head)
        return {k: {"shape": list(v.shape), "norm": float(v.norm())} for k, v in snap.items()}

    def inject(prev_layer: int, prev_head: int, ind_layer: int, ind_head: int) -> dict[str, Any]:
        circuit = build_induction_circuit(state.model)
        manifest = inject_induction_circuit_(
            state.model,
            circuit,
            HeadRef(prev_layer, prev_head, 0.0),
            HeadRef(ind_layer, ind_head, 0.0),
        )
        state.manifest = manifest
        state.edits.extend(manifest.edits)
        state.budget.spend(1)
        after = evaluate(state.model, "after_inject", device=state.device)
        return {
            "weights_touched": manifest.total_weights_touched,
            "fraction": head_param_fraction(state.model, manifest.total_weights_touched),
            "metrics": after.as_row(),
            "budget_remaining": state.budget.remaining,
        }

    def graft_head(
        src_layer: int,
        src_head: int,
        dst_layer: int,
        dst_head: int,
        role: str = "graft",
    ) -> dict[str, Any]:
        """Copy one head's weights into another slot (must snapshot before ablation)."""
        from circuit_injection.circuits.induction import head_weights_from_snapshot
        from circuit_injection.inject import apply_head_weights_

        key = (src_layer, src_head)
        if key not in state.snapshots:
            state.snapshots[key] = snapshot_head(state.model, src_layer, src_head)
        edit = apply_head_weights_(
            state.model,
            dst_layer,
            dst_head,
            head_weights_from_snapshot(state.snapshots[key]),
            role=role,
        )
        state.edits.append(edit)
        state.refresh_manifest(notes="agent graft relocate")
        return {
            "dst": [dst_layer, dst_head],
            "src": [src_layer, src_head],
            "weights_touched": edit.weights_touched,
        }

    def snapshot_head_tool(layer: int, head: int) -> dict[str, Any]:
        state.snapshots[(layer, head)] = snapshot_head(state.model, layer, head)
        return {"snapshotted": [layer, head]}

    return {
        "list_top_induction": list_top_induction,
        "list_weak_heads": list_weak_heads,
        "list_low_util": list_low_util,
        "ablate_top": ablate_top,
        "probe_metrics": probe_metrics,
        "read_head": read_head,
        "inject": inject,
        "snapshot_head": snapshot_head_tool,
        "graft_head": graft_head,
    }


def heuristic_policy(tools: dict[str, Callable[..., Any]]) -> dict[str, Any]:
    """Deterministic policy used for --dry-run / no-API mode.

    Snapshot top induction heads, ablate them, graft each into a same-layer
    neighbor head, then probe. This is the reliable additive edit on GPT-2.
    """
    log: list[dict[str, Any]] = []
    top = tools["list_top_induction"](k=4)
    log.append({"tool": "list_top_induction", "result": top})
    for h in top:
        snap = tools["snapshot_head"](h["layer"], h["head"])
        log.append({"tool": "snapshot_head", "result": snap})

    ab = tools["ablate_top"](k=4)
    log.append({"tool": "ablate_top", "result": ab})
    before = tools["probe_metrics"]()
    log.append({"tool": "probe_metrics", "result": before})

    grafts = []
    for h in top:
        dst_head = (h["head"] + 1) % 12
        g = tools["graft_head"](h["layer"], h["head"], h["layer"], dst_head, role="graft_induction")
        grafts.append(g)
        log.append({"tool": "graft_head", "result": g})

    after = tools["probe_metrics"]()
    log.append({"tool": "probe_metrics", "result": after})
    return {
        "policy": "heuristic_graft_relocate",
        "grafts": grafts,
        "log": log,
        "final_metrics": after,
    }


def run_openai_agent(
    tools: dict[str, Callable[..., Any]],
    *,
    model: str = "gpt-4o-mini",
    max_steps: int = 8,
) -> dict[str, Any]:
    """Optional LLM loop via OpenAI-compatible tool calling."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install optional dep: pip install -e '.[agent]'") from e

    client = OpenAI()
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "additionalProperties": True},
            },
        }
        for name in tools
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Ablate the model's strongest induction heads, choose slots, "
                "inject a closed-form induction circuit, and probe metrics."
            ),
        },
    ]
    log: list[dict[str, Any]] = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if msg.tool_calls:
            for call in msg.tool_calls:
                fn = call.function.name
                args = json.loads(call.function.arguments or "{}")
                if fn not in tools:
                    result = {"error": f"unknown tool {fn}"}
                else:
                    try:
                        result = tools[fn](**args)
                    except Exception as exc:  # noqa: BLE001
                        result = {"error": str(exc)}
                log.append({"tool": fn, "args": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            continue
        # final text — try parse JSON action
        content = msg.content or ""
        log.append({"assistant": content})
        return {"policy": "openai", "model": model, "log": log, "final_text": content}
    return {"policy": "openai", "model": model, "log": log, "final_text": "max_steps"}


def run_agent(
    *,
    model_name: str = "gpt2",
    device: str = "cpu",
    budget: int = 8,
    dry_run: bool = True,
    patch_path: str | Path | None = None,
    openai_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    model, _spec = load_gpt2(model_name, device=device)
    state = AgentState(
        model=model,
        model_name=model_name,
        device=device,
        budget=ProbeBudget(budget),
    )
    tools = make_tools(state)

    baseline = evaluate(model, "baseline", device=device)
    if dry_run or not os.environ.get("OPENAI_API_KEY"):
        result = heuristic_policy(tools)
        result["mode"] = "heuristic"
    else:
        result = run_openai_agent(tools, model=openai_model)
        result["mode"] = "openai"

    final = evaluate(model, "final", device=device)
    rows = [baseline, final]
    result["table"] = format_table(rows)
    result["budget_remaining"] = state.budget.remaining
    if state.manifest and patch_path:
        export_patch(model, state.manifest, patch_path)
        result["patch"] = str(patch_path)
    return result
