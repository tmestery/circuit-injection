"""Budgeted weight-edit sandbox for invent-ability sessions (no training).

Astra (or a human) uses this object to inspect GPT-2, run limited probes,
apply sparse tensor edits, checkpoint/rollback, and export a patch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from circuit_injection.gpt2_heads import (
    GPT2Spec,
    load_gpt2,
    write_head_,
    zero_head_,
)
from circuit_injection.tasks.ping_pong import (
    TARGET_CONTINUATION,
    TRIGGER,
    EvalSplit,
    evaluate_ping_pong,
    probe_prompts,
)


@dataclass
class ProbeBudget:
    remaining: int
    initial: int

    def spend(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > self.remaining:
            raise RuntimeError(
                f"probe budget exceeded (need {n}, have {self.remaining} of {self.initial})"
            )
        self.remaining -= n


@dataclass
class Sandbox:
    """In-process weight programming session over one GPT-2 patient."""

    model: GPT2LMHeadModel
    tokenizer: GPT2TokenizerFast
    spec: GPT2Spec
    model_name: str
    device: str
    budget: ProbeBudget
    checkpoints: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    edited_param_names: set[str] = field(default_factory=set)
    edit_log: list[dict[str, Any]] = field(default_factory=list)

    # --- construction ---

    @classmethod
    def create(
        cls,
        model_name: str = "gpt2",
        device: str = "cpu",
        probe_budget: int = 50,
    ) -> "Sandbox":
        model, spec = load_gpt2(model_name, device=device)
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        return cls(
            model=model,
            tokenizer=tok,
            spec=spec,
            model_name=model_name,
            device=device,
            budget=ProbeBudget(remaining=probe_budget, initial=probe_budget),
        )

    def assert_no_training_apis(self) -> None:
        """Documentation helper — training is forbidden by policy, not sandboxed OS-level."""
        return None

    # --- inspection ---

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "n_layer": self.spec.n_layer,
            "n_head": self.spec.n_head,
            "n_embd": self.spec.n_embd,
            "n_ctx": self.spec.n_ctx,
            "probe_budget_remaining": self.budget.remaining,
            "probe_budget_initial": self.budget.initial,
            "trigger": TRIGGER,
            "target": TARGET_CONTINUATION,
            "n_params": sum(p.numel() for p in self.model.parameters()),
            "edited_params": sorted(self.edited_param_names),
        }

    def list_parameters(self, pattern: str | None = None) -> list[dict[str, Any]]:
        rows = []
        rx = re.compile(pattern) if pattern else None
        for name, p in self.model.named_parameters():
            if rx and not rx.search(name):
                continue
            rows.append(
                {
                    "name": name,
                    "shape": list(p.shape),
                    "norm": float(p.detach().float().norm()),
                    "numel": p.numel(),
                }
            )
        return rows

    def describe_param(self, name: str) -> dict[str, Any]:
        p = dict(self.model.named_parameters())[name]
        t = p.detach().float()
        return {
            "name": name,
            "shape": list(p.shape),
            "norm": float(t.norm()),
            "mean": float(t.mean()),
            "std": float(t.std()),
            "abs_max": float(t.abs().max()),
        }

    def token_ids(self, text: str) -> dict[str, Any]:
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        pieces = [self.tokenizer.decode([i]) for i in ids]
        return {"text": text, "ids": ids, "pieces": pieces}

    def embedding_row(self, token_id: int) -> torch.Tensor:
        return self.model.transformer.wte.weight[token_id].detach().clone()

    def lm_head_row(self, token_id: int) -> torch.Tensor:
        # GPT-2 ties lm_head to wte
        return self.model.lm_head.weight[token_id].detach().clone()

    # --- probes (budgeted) ---

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 12) -> str:
        self.budget.spend(1)
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    def probe_ping_pong(self, *, split: str = "probe", include_nll: bool = False) -> dict[str, Any]:
        """One budget unit per prompt in the split (pos+neg)."""
        sp = EvalSplit.PROBE if split == "probe" else EvalSplit.HELDOUT
        if sp == EvalSplit.HELDOUT:
            raise RuntimeError(
                "Held-out eval is forbidden inside the sandbox during editing. "
                "Use scripts/eval_ping_pong.py --split heldout after exporting a patch."
            )
        pos, neg = probe_prompts()
        self.budget.spend(len(pos) + len(neg))
        result = evaluate_ping_pong(
            self.model,
            split=sp,
            model_name=self.model_name,
            device=self.device,
            tokenizer=self.tokenizer,
            include_nll=include_nll,
        )
        return result.as_row() | {
            "budget_remaining": self.budget.remaining,
            "sample_pos": result.generations_pos[:3],
            "sample_neg": result.generations_neg[:3],
        }

    @torch.no_grad()
    def next_token_logits(self, prompt: str, top_k: int = 20) -> dict[str, Any]:
        self.budget.spend(1)
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        logits = self.model(ids).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        values, indices = probs.topk(top_k)
        return {
            "prompt": prompt,
            "top": [
                {
                    "token_id": int(i),
                    "piece": self.tokenizer.decode([int(i)]),
                    "prob": float(v),
                }
                for v, i in zip(values.tolist(), indices.tolist())
            ],
            "budget_remaining": self.budget.remaining,
        }

    @torch.no_grad()
    def last_mlp_inputs(self, prompts: list[str], layer: int = 11) -> torch.Tensor:
        """Return final-position inputs to one MLP output projection.

        This is a budgeted read-only forward probe. It is intended for a
        one-shot, closed-form edit, never for iterative fitting or training.
        """
        if not prompts:
            raise ValueError("prompts must not be empty")
        if not 0 <= layer < self.spec.n_layer:
            raise ValueError(f"layer must be in [0, {self.spec.n_layer - 1}]")
        self.budget.spend(len(prompts))
        captured: list[torch.Tensor] = []

        def hook(_module, args):
            captured.append(args[0].detach())

        module = self.model.transformer.h[layer].mlp.c_proj
        handle = module.register_forward_pre_hook(hook)
        try:
            for prompt in prompts:
                ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
                self.model(ids, use_cache=False)
        finally:
            handle.remove()
        return torch.cat([x[:, -1, :].float() for x in captured], dim=0)

    # --- checkpoints ---

    def checkpoint(self, name: str = "default") -> str:
        self.checkpoints[name] = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        return name

    def rollback(self, name: str = "default") -> None:
        if name not in self.checkpoints:
            raise KeyError(f"no checkpoint named {name!r}")
        state = {k: v.to(self.device) for k, v in self.checkpoints[name].items()}
        self.model.load_state_dict(state)
        self.edit_log.append({"op": "rollback", "name": name})

    # --- edits (no gradients) ---

    def _touch(self, name: str, n: int, note: str) -> None:
        self.edited_param_names.add(name)
        self.edit_log.append({"op": "edit", "name": name, "numel": n, "note": note})

    @torch.no_grad()
    def add_to_param_(self, name: str, delta: torch.Tensor, note: str = "") -> int:
        params = dict(self.model.named_parameters())
        if name not in params:
            raise KeyError(name)
        p = params[name]
        d = delta.to(device=p.device, dtype=p.dtype)
        if d.shape != p.shape:
            raise ValueError(f"delta shape {tuple(d.shape)} != param {tuple(p.shape)}")
        p.add_(d)
        self._touch(name, d.numel(), note or "add_to_param")
        return d.numel()

    @torch.no_grad()
    def set_param_(self, name: str, value: torch.Tensor, note: str = "") -> int:
        params = dict(self.model.named_parameters())
        if name not in params:
            raise KeyError(name)
        p = params[name]
        v = value.to(device=p.device, dtype=p.dtype)
        if v.shape != p.shape:
            raise ValueError(f"value shape {tuple(v.shape)} != param {tuple(p.shape)}")
        p.copy_(v)
        self._touch(name, v.numel(), note or "set_param")
        return v.numel()

    @torch.no_grad()
    def add_to_lm_head_row_(self, token_id: int, delta_vec: torch.Tensor, note: str = "") -> int:
        """Add a vector into the tied embedding/lm_head row for token_id."""
        w = self.model.transformer.wte.weight
        d = delta_vec.to(device=w.device, dtype=w.dtype)
        if d.ndim != 1 or d.shape[0] != w.shape[1]:
            raise ValueError(f"delta_vec must be shape ({w.shape[1]},)")
        w[token_id].add_(d)
        self._touch("transformer.wte.weight", d.numel(), note or f"lm_head_row[{token_id}]")
        return d.numel()

    @torch.no_grad()
    def boost_output_token_(
        self,
        token_id: int,
        scale: float = 5.0,
        direction: torch.Tensor | None = None,
        note: str = "",
    ) -> int:
        """Nudge lm_head/wte row so `token_id` becomes likelier (simple invent edit)."""
        w = self.model.transformer.wte.weight
        if direction is None:
            # Use the row's own direction amplified, plus a bit of noise-free unit boost
            direction = w[token_id].detach().clone()
            n = direction.norm().clamp_min(1e-6)
            direction = direction / n
        d = (scale * direction).to(device=w.device, dtype=w.dtype)
        return self.add_to_lm_head_row_(token_id, d, note=note or f"boost_token[{token_id}]")

    def write_attention_head_(
        self,
        layer: int,
        head: int,
        *,
        q_w=None,
        k_w=None,
        v_w=None,
        o_w=None,
        q_b=None,
        k_b=None,
        v_b=None,
        note: str = "",
    ) -> int:
        n = write_head_(
            self.model,
            layer,
            head,
            q_w=q_w,
            k_w=k_w,
            v_w=v_w,
            o_w=o_w,
            q_b=q_b,
            k_b=k_b,
            v_b=v_b,
        )
        self._touch(f"transformer.h.{layer}.attn (head {head})", n, note or "write_head")
        return n

    def zero_attention_head_(self, layer: int, head: int, note: str = "") -> int:
        n = zero_head_(self.model, layer, head)
        self._touch(f"transformer.h.{layer}.attn (head {head})", n, note or "zero_head")
        return n

    @torch.no_grad()
    def add_to_mlp_out_(self, layer: int, delta: torch.Tensor, note: str = "") -> int:
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        return self.add_to_param_(name, delta, note=note or f"mlp_out_L{layer}")

    @torch.no_grad()
    def add_closed_form_mlp_map_(
        self,
        layer: int,
        inputs: torch.Tensor,
        desired_residual_deltas: torch.Tensor,
        *,
        ridge: float = 1e-2,
        note: str = "",
    ) -> int:
        """Add one closed-form ridge map to an MLP output projection.

        For MLP projection input X and requested residual outputs Y, write
        ΔW = (XᵀX + λI)⁻¹XᵀY. This is a single algebraic solve, not gradient
        descent or an optimization loop.
        """
        if inputs.ndim != 2 or desired_residual_deltas.ndim != 2:
            raise ValueError("inputs and desired_residual_deltas must be rank-2")
        if inputs.shape[0] != desired_residual_deltas.shape[0]:
            raise ValueError("inputs and desired residual deltas need same rows")
        param_name = f"transformer.h.{layer}.mlp.c_proj.weight"
        param = dict(self.model.named_parameters())[param_name]
        if inputs.shape[1] != param.shape[0] or desired_residual_deltas.shape[1] != param.shape[1]:
            raise ValueError(
                f"expected X=(*,{param.shape[0]}), Y=(*,{param.shape[1]}), "
                f"got {tuple(inputs.shape)}, {tuple(desired_residual_deltas.shape)}"
            )
        x = inputs.float().cpu()
        y = desired_residual_deltas.float().cpu()
        gram = x.transpose(0, 1) @ x
        gram.diagonal().add_(ridge)
        delta = torch.linalg.solve(gram, x.transpose(0, 1) @ y)
        param.add_(delta.to(device=param.device, dtype=param.dtype))
        self._touch(
            param_name,
            delta.numel(),
            note or f"one_shot_closed_form_mlp_map_L{layer}",
        )
        return delta.numel()

    # --- export ---

    def export_patch(self, path: str | Path, notes: str = "") -> Path:
        """Export full tensors for every parameter that was edited (by name).

        Also exports full wte if any row was touched via name transformer.wte.weight.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = dict(self.model.named_parameters())
        # Map synthetic head names back — only export real state_dict keys we touched
        tensors: dict[str, torch.Tensor] = {}
        real_names = [n for n in self.edited_param_names if n in params]
        # If head edits happened, include full attn weights for those layers
        for entry in self.edit_log:
            name = entry.get("name", "")
            m = re.match(r"transformer\.h\.(\d+)\.attn \(head (\d+)\)", name)
            if m:
                layer = int(m.group(1))
                for suffix in (
                    f"transformer.h.{layer}.attn.c_attn.weight",
                    f"transformer.h.{layer}.attn.c_attn.bias",
                    f"transformer.h.{layer}.attn.c_proj.weight",
                    f"transformer.h.{layer}.attn.c_proj.bias",
                ):
                    if suffix in params:
                        real_names.append(suffix)
        for name in sorted(set(real_names)):
            tensors[name] = params[name].detach().cpu().contiguous()
        if not tensors:
            raise RuntimeError("no edits to export")
        save_file(tensors, str(path))
        meta = {
            "model_name": self.model_name,
            "notes": notes,
            "trigger": TRIGGER,
            "target": TARGET_CONTINUATION,
            "probe_budget_initial": self.budget.initial,
            "probe_budget_remaining": self.budget.remaining,
            "edited_param_names": sorted(self.edited_param_names),
            "tensor_keys": list(tensors.keys()),
            "total_scalars_exported": int(sum(t.numel() for t in tensors.values())),
            "edit_log": self.edit_log,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "info": self.info(),
            "edit_log": self.edit_log,
            "checkpoints": list(self.checkpoints.keys()),
        }
