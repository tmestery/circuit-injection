"""End-to-end induction circuit injection experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuit_injection.circuits.induction import (
    build_induction_circuit,
    head_weights_from_snapshot,
)
from circuit_injection.eval_runner import evaluate, format_table, snapshots_to_jsonable
from circuit_injection.gpt2_heads import load_gpt2, snapshot_head, zero_head_
from circuit_injection.inject import (
    PatchManifest,
    apply_head_weights_,
    export_patch,
    head_param_fraction,
    inject_induction_circuit_,
)
from circuit_injection.locate import (
    HeadRef,
    ablate_heads_,
    top_induction_heads,
    top_previous_token_heads,
)


def run_induction_experiment(
    *,
    model_name: str = "gpt2",
    device: str = "cpu",
    patch_path: str | Path = "patches/induction_gpt2.safetensors",
    ablate_k: int = 4,
    batch: int = 16,
    half_len: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """baseline → ablate → closed-form inject → graft-relocate → graft-restore."""
    model, _spec = load_gpt2(model_name, device=device)
    eval_kw = dict(batch=batch, half_len=half_len, seed=seed, device=device)

    rows = []
    rows.append(evaluate(model, "baseline", **eval_kw))

    # Identify the natural circuit before damaging it
    ind_donors = top_induction_heads(model, k=ablate_k, **eval_kw)
    prev_donors = top_previous_token_heads(model, k=2, batch=batch, seq_len=half_len, seed=seed, device=device)
    prev_donor = prev_donors[0]
    ind_donor = ind_donors[0]

    donor_prev_snap = snapshot_head(model, prev_donor.layer, prev_donor.head)
    donor_ind_snap = snapshot_head(model, ind_donor.layer, ind_donor.head)
    all_donor_snaps = {
        (h.layer, h.head): snapshot_head(model, h.layer, h.head) for h in ind_donors
    }
    all_donor_snaps[(prev_donor.layer, prev_donor.head)] = donor_prev_snap

    # Ablate strongest induction heads (+ prev-token donor if distinct)
    to_ablate = list(ind_donors)
    if not any(h.layer == prev_donor.layer and h.head == prev_donor.head for h in to_ablate):
        to_ablate.append(prev_donor)
    n_ablated = ablate_heads_(model, to_ablate)
    rows.append(evaluate(model, "ablated", **eval_kw))

    # --- Closed-form injection into layer-0 prev + late induction slot ---
    ind_slot = next((h for h in reversed(ind_donors) if h.layer >= 2), ind_donors[-1])
    prev_slot = HeadRef(layer=0, head=0, score=0.0)
    if prev_slot.layer == ind_slot.layer and prev_slot.head == ind_slot.head:
        ind_slot = HeadRef(layer=min(5, model.config.n_layer - 1), head=0, score=0.0)
    if not any(h.layer == prev_slot.layer and h.head == prev_slot.head for h in to_ablate):
        zero_head_(model, prev_slot.layer, prev_slot.head)

    circuit = build_induction_circuit(model)
    manifest = inject_induction_circuit_(model, circuit, prev_slot, ind_slot)
    rows.append(evaluate(model, "injected_closed_form", **eval_kw))

    # --- Graft relocate: copy natural circuit into different heads in the SAME
    # layers (heads are layer-tuned; cross-layer moves usually fail). ---
    graft_model, _ = load_gpt2(model_name, device=device)
    ablate_heads_(graft_model, to_ablate)
    n_head = graft_model.config.n_head
    reserved: set[tuple[int, int]] = set()

    def _neighbor_head(layer: int, head: int) -> HeadRef:
        for delta in range(1, n_head):
            cand = (head + delta) % n_head
            key = (layer, cand)
            if key in reserved:
                continue
            if any(h.layer == layer and h.head == cand for h in to_ablate):
                continue
            reserved.add(key)
            return HeadRef(layer=layer, head=cand, score=0.0)
        # Fall back even if ablated — we overwrite next
        key = (layer, (head + 1) % n_head)
        reserved.add(key)
        return HeadRef(layer=layer, head=key[1], score=0.0)

    graft_edits = []
    # Relocate previous-token donor
    graft_prev_slot = _neighbor_head(prev_donor.layer, prev_donor.head)
    graft_edits.append(
        apply_head_weights_(
            graft_model,
            graft_prev_slot.layer,
            graft_prev_slot.head,
            head_weights_from_snapshot(donor_prev_snap),
            role="graft_prev_token",
        )
    )
    # Relocate each ablated induction head into a same-layer neighbor
    graft_ind_slot = None
    for href in ind_donors:
        snap = all_donor_snaps[(href.layer, href.head)]
        dest = _neighbor_head(href.layer, href.head)
        if graft_ind_slot is None:
            graft_ind_slot = dest
        graft_edits.append(
            apply_head_weights_(
                graft_model,
                dest.layer,
                dest.head,
                head_weights_from_snapshot(snap),
                role="graft_induction",
            )
        )
    assert graft_ind_slot is not None
    rows.append(evaluate(graft_model, "graft_relocate", **eval_kw))
    graft_manifest = PatchManifest(
        model_name=model_name,
        edits=graft_edits,
        total_weights_touched=sum(e.weights_touched for e in graft_edits),
        notes="donor circuit relocated into same-layer neighbor heads",
    )

    # --- Graft restore: put donors back into original slots (write-path sanity) ---
    restore_model, _ = load_gpt2(model_name, device=device)
    ablate_heads_(restore_model, to_ablate)
    for href in to_ablate:
        key = (href.layer, href.head)
        if key in all_donor_snaps:
            apply_head_weights_(
                restore_model,
                href.layer,
                href.head,
                head_weights_from_snapshot(all_donor_snaps[key]),
                role="graft_restore",
            )
    rows.append(evaluate(restore_model, "graft_restore", **eval_kw))

    # Export the closed-form patch from `model`, plus a relocate patch alongside
    path = export_patch(model, manifest, patch_path)
    relocate_path = Path(patch_path).with_name(
        Path(patch_path).stem + "_graft_relocate.safetensors"
    )
    export_patch(graft_model, graft_manifest, relocate_path)

    table = format_table(rows)
    return {
        "table": table,
        "rows": snapshots_to_jsonable(rows),
        "ablated_heads": [
            {"layer": h.layer, "head": h.head, "score": h.score} for h in to_ablate
        ],
        "prev_donor": {"layer": prev_donor.layer, "head": prev_donor.head, "score": prev_donor.score},
        "induction_donor": {"layer": ind_donor.layer, "head": ind_donor.head, "score": ind_donor.score},
        "closed_form_slots": {
            "prev": {"layer": prev_slot.layer, "head": prev_slot.head},
            "induction": {"layer": ind_slot.layer, "head": ind_slot.head},
        },
        "graft_relocate_slots": {
            "prev": {"layer": graft_prev_slot.layer, "head": graft_prev_slot.head},
            "induction": {"layer": graft_ind_slot.layer, "head": graft_ind_slot.head},
        },
        "weights_touched": manifest.total_weights_touched,
        "weights_ablated": n_ablated,
        "fraction": head_param_fraction(model, manifest.total_weights_touched),
        "patch": str(path),
        "graft_relocate_patch": str(relocate_path),
        "manifest_notes": manifest.notes,
    }
