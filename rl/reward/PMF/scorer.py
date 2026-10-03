"""
scorer.py — F1 computation and adaptive-weight PMF score.

Matching strategy (at most one LLM call per sample):
  1. Canonicalize common law/context/assumption aliases
  2. Exact match (set intersection) — no API cost
  3. Remaining unmatched tuples across QEL/VCL/AEB are batched into a SINGLE
     LLM call that returns index-pair matches per motif type

VCL tuples preserve direction. Shared variable identity and producer -> consumer
ordering must still match, but law/context labels may be semantically aligned.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI

from openai_responses_utils import (
    DEFAULT_OPENAI_MODEL,
    build_client_kwargs,
    build_responses_kwargs,
    extract_response_text,
    resolve_api_key,
)

from .motifs import AEB, QEL, VCL, MotifSet


# ---------------------------------------------------------------------------
# Canonicalization for style-robust matching
# ---------------------------------------------------------------------------

_LAW_ALIASES = {
    "kinematic_equations": "kinematics",
    "kinematics_equation": "kinematics",
    "work_energy_theorem": "energy_conservation",
    "kinetic_energy_theorem": "energy_conservation",
    "power_balance": "energy_conservation",
    "force_equilibrium": "newton_first_law",
    "counterweight_equilibrium": "newton_first_law",
}

_CONTEXT_ALIASES = {
    "motor_power": "motor_cable_motion",
    "motor_cable_power": "motor_cable_motion",
    "loaded_uphill": "cart_uphill_motion",
    "uphill_motion_with_load": "cart_uphill_motion",
    "loaded_uphill_motion": "cart_uphill_motion",
    "unloaded_downhill": "cart_downhill_motion",
    "downhill_motion_without_load": "cart_downhill_motion",
    "coasting_uphill": "system_deceleration",
    "uphill_coasting": "system_deceleration",
    "deceleration_stage": "system_deceleration",
    "counterweight_equilibrium": "counterweight",
}

_ASSUMPTION_ALIASES = {
    "constant_speed_motion": "uniform_motion",
    "constant_velocity_motion": "uniform_motion",
    "lossless_pulley": "ideal_pulley",
    "frictionless_pulley": "ideal_pulley",
    "massless_rope": "massless_cable",
}

_QUANTITY_ALIASES = {
    "sin_theta": "theta",
    "sintheta": "theta",
}


def _canonicalize_with_aliases(value: str, aliases: dict[str, str]) -> str:
    current = value
    seen = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def _canonicalize_quantity(value: str) -> str:
    return _canonicalize_with_aliases(value, _QUANTITY_ALIASES)


def _canonicalize_law(value: str) -> str:
    return _canonicalize_with_aliases(value, _LAW_ALIASES)


def _canonicalize_context(value: str) -> str:
    return _canonicalize_with_aliases(value, _CONTEXT_ALIASES)


def _canonicalize_assumption(value: str) -> str:
    return _canonicalize_with_aliases(value, _ASSUMPTION_ALIASES)


def _canonicalize_qel(item: QEL) -> QEL:
    return QEL(
        quantity=_canonicalize_quantity(item.quantity),
        law=_canonicalize_law(item.law),
        context=_canonicalize_context(item.context),
    )


def _canonicalize_vcl(item: VCL) -> VCL:
    return VCL(
        law_a=_canonicalize_law(item.law_a),
        ctx_a=_canonicalize_context(item.ctx_a),
        shared_var=_canonicalize_quantity(item.shared_var),
        law_b=_canonicalize_law(item.law_b),
        ctx_b=_canonicalize_context(item.ctx_b),
    )


def _canonicalize_aeb(item: AEB) -> AEB:
    return AEB(
        assumption=_canonicalize_assumption(item.assumption),
        law=_canonicalize_law(item.law),
        context=_canonicalize_context(item.context),
    )


def _canonicalize_motif_set(motifs: MotifSet) -> MotifSet:
    return MotifSet(
        qel=frozenset(_canonicalize_qel(item) for item in motifs.qel),
        vcl=frozenset(_canonicalize_vcl(item) for item in motifs.vcl),
        aeb=frozenset(_canonicalize_aeb(item) for item in motifs.aeb),
    )


# ---------------------------------------------------------------------------
# Single-call LLM matcher for semantic equivalence
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = """\
You are a physics expert matching predicted physics modelling tuples against \
ground-truth tuples across three motif types (QEL, VCL, AEB) in ONE pass.

Two tuples are EQUIVALENT if they refer to the same physical modelling fact, \
even if the wording or notation differs slightly (e.g. \
"kinetic_energy_theorem" ≈ "work_energy_theorem", "v_0" ≈ "v0", \
"incline" ≈ "incline_descent").

Per-type anchor constraint (MUST hold, else the pair is NOT equivalent):
- QEL: the `quantity` field must refer to the same physical quantity.
- AEB: the `assumption` field must refer to the same physical assumption.
- VCL: the `shared_var` field must refer to the same physical quantity, AND
  direction must be preserved — pred's producer side (law_a/ctx_a) corresponds
  to gt's producer side, pred's consumer side (law_b/ctx_b) to gt's consumer
  side. Reversed direction is NOT equivalent.

For the non-anchor fields (law / context / law_a / ctx_a / law_b / ctx_b), \
labels may differ stylistically when the underlying modelling fact is the \
same. Examples of acceptable equivalence:
- "counterweight_equilibrium" ≈ "newton_first_law" in a local force-balance context
- "loaded_uphill" ≈ "uphill_motion_with_load"
- "coasting_uphill" ≈ "uphill_coasting"
- "kinematics" ≈ "kinematic_equations"

Do NOT require exact string equality for law/context. Be conservative but not \
literalist: if the modelling fact is clearly the same, treat as equivalent.

Two tuples are NOT equivalent if they refer to different laws, different \
variables, reversed dependency direction, or different physical objects/stages.

Matching constraints:
- Each predicted tuple may match at most ONE ground-truth tuple.
- Each ground-truth tuple may match at most ONE predicted tuple.
- Only match within the same motif type; never cross types.
- Use the provided integer indices verbatim — do NOT invent new tuples.

Respond with JSON only, exactly this shape:
{
  "qel": [[pred_idx, gt_idx], ...],
  "vcl": [[pred_idx, gt_idx], ...],
  "aeb": [[pred_idx, gt_idx], ...]
}
Unmatched tuples are simply omitted. Use [] for a type with no matches.
"""


def _tuple_to_dict(t: Any) -> dict | str:
    if hasattr(t, "to_dict"):
        return t.to_dict()
    return str(t)


def _indexed_tuples(items: list) -> list[dict]:
    out: list[dict] = []
    for i, t in enumerate(items):
        d = _tuple_to_dict(t)
        if isinstance(d, dict):
            out.append({"idx": i, **d})
        else:
            out.append({"idx": i, "value": d})
    return out


def _sanitize_pairs(
    raw_pairs: Any,
    pred_len: int,
    gt_len: int,
) -> list[tuple[int, int]]:
    """Coerce raw LLM output into a clean list of unique-index pairs."""
    if not isinstance(raw_pairs, list):
        return []
    seen_pred: set[int] = set()
    seen_gt: set[int] = set()
    out: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            pi = int(pair[0])
            gi = int(pair[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= pi < pred_len and 0 <= gi < gt_len):
            continue
        if pi in seen_pred or gi in seen_gt:
            continue
        seen_pred.add(pi)
        seen_gt.add(gi)
        out.append((pi, gi))
    return out


async def _match_all_motifs_llm_async(
    remaining: dict[str, tuple[list, list]],
    client: AsyncOpenAI,
    model: str,
) -> dict[str, list[tuple[int, int]]]:
    """
    Issue ONE LLM call to match remaining pred/gt tuples across all motif types.
    Returns {"qel": [(pi, gi), ...], "vcl": [...], "aeb": [...]}.

    An empty list is returned for a motif type when nothing needs matching or
    when the call fails — the caller treats unmatched as FP/FN.
    """
    empty = {name: [] for name in remaining}

    payload = {
        name: {
            "pred": _indexed_tuples(pred_list),
            "gt": _indexed_tuples(gt_list),
        }
        for name, (pred_list, gt_list) in remaining.items()
    }

    # Skip the call if every type has either pred or gt empty — no match is
    # possible and the LLM would return [] anyway.
    if not any(payload[name]["pred"] and payload[name]["gt"] for name in payload):
        return empty

    user_prompt = (
        "Match the following physics modelling tuples. Each motif type lists "
        "predicted (pred) and ground-truth (gt) tuples with integer indices. "
        "Return index-pair matches per type.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        response = await client.chat.completions.create(
            **build_responses_kwargs(
                model=model,
                messages=[
                    {"role": "system", "content": _MATCH_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_output_tokens=1024,
            )
        )
        raw = extract_response_text(response)
        parsed = json.loads(raw)
    except Exception:
        return empty

    out: dict[str, list[tuple[int, int]]] = {}
    for name, (pred_list, gt_list) in remaining.items():
        out[name] = _sanitize_pairs(
            parsed.get(name, []) if isinstance(parsed, dict) else [],
            len(pred_list),
            len(gt_list),
        )
    return out


# ---------------------------------------------------------------------------
# Per-motif F1 given a precomputed match
# ---------------------------------------------------------------------------

def _f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _motif_f1_from_match(
    pred: frozenset,
    gt: frozenset,
    remaining_pred_list: list,
    remaining_gt_list: list,
    exact_tp: frozenset,
    sem_pairs: list[tuple[int, int]],
) -> dict:
    matched_pi = {pi for pi, _ in sem_pairs}
    matched_gi = {gi for _, gi in sem_pairs}

    sem_tp = {remaining_pred_list[pi] for pi in matched_pi}
    fp_set = {remaining_pred_list[pi] for pi in range(len(remaining_pred_list)) if pi not in matched_pi}
    fn_set = {remaining_gt_list[gi] for gi in range(len(remaining_gt_list)) if gi not in matched_gi}

    tp_count = len(exact_tp) + len(sem_tp)
    fp_count = len(fp_set)
    fn_count = len(fn_set)
    precision, recall, f1 = _f1_from_counts(tp_count, fp_count, fn_count)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "missing": fn_set,   # GT tuples the model missed
        "extra": fp_set,     # Predicted tuples not in GT
    }


# ---------------------------------------------------------------------------
# PMF score
# ---------------------------------------------------------------------------

async def compute_pmf_async(
    pred: MotifSet,
    gt: MotifSet,
    judge_client: AsyncOpenAI | None = None,
    judge_model: str = DEFAULT_OPENAI_MODEL,
    weights: dict[str, float] | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> dict:
    """
    Compute the full PMF score with adaptive weighting.

    Adaptive weight rule:
      S = {m : |GT_m| + |Pred_m| > 0}
      Only GT=Pred=∅ is skipped (prevents rewarding hallucination of new motif types).

    Args:
        pred:         Predicted MotifSet (from LLM reasoning)
        gt:           Ground-truth MotifSet (from expert solution)
        judge_client: AsyncOpenAI client for semantic matching (None = exact-match only)
        judge_model:  Model used for the single-pass semantic matcher
        weights:      Base weights per motif type (default: equal 1/3 each)
        semaphore:    Unused (kept for backwards compatibility); matching is
                      now a single LLM call so no rate-limiting is needed.

    Returns dict with keys: pmf, qel, vcl, aeb (each containing F1 details)
    """
    del semaphore  # unused under single-call matcher

    pred = _canonicalize_motif_set(pred)
    gt = _canonicalize_motif_set(gt)

    if weights is None:
        weights = {"qel": 1 / 3, "vcl": 1 / 3, "aeb": 1 / 3}

    motif_configs = [
        ("qel", pred.qel, gt.qel),
        ("vcl", pred.vcl, gt.vcl),
        ("aeb", pred.aeb, gt.aeb),
    ]

    # Step 1: exact match per type (free, no API cost)
    exact_tp_per_type: dict[str, frozenset] = {}
    remaining_pred_lists: dict[str, list] = {}
    remaining_gt_lists: dict[str, list] = {}
    for name, pred_set, gt_set in motif_configs:
        exact_tp = pred_set & gt_set
        exact_tp_per_type[name] = exact_tp
        remaining_pred_lists[name] = list(pred_set - exact_tp)
        remaining_gt_lists[name] = list(gt_set - exact_tp)

    # Step 2: single LLM call to match remaining tuples across all types
    if judge_client is not None:
        sem_pairs = await _match_all_motifs_llm_async(
            {
                name: (remaining_pred_lists[name], remaining_gt_lists[name])
                for name in remaining_pred_lists
            },
            judge_client,
            judge_model,
        )
    else:
        sem_pairs = {name: [] for name in remaining_pred_lists}

    # Step 3: per-type F1 from exact + semantic matches
    results: dict[str, dict] = {}
    for name, pred_set, gt_set in motif_configs:
        results[name] = _motif_f1_from_match(
            pred_set,
            gt_set,
            remaining_pred_lists[name],
            remaining_gt_lists[name],
            exact_tp_per_type[name],
            sem_pairs.get(name, []),
        )

    # Adaptive weighting: S = {m : |GT_m| + |Pred_m| > 0}
    active = {
        name: weights[name]
        for name, pred_set, gt_set in motif_configs
        if len(pred_set) + len(gt_set) > 0
    }

    if not active:
        # All motif sets are empty — this should not happen for valid problems
        pmf_score = 0.0
    else:
        total_w = sum(active.values())
        pmf_score = sum(
            (w / total_w) * results[name]["f1"]
            for name, w in active.items()
        )

    return {
        "pmf": pmf_score,
        "qel": results["qel"],
        "vcl": results["vcl"],
        "aeb": results["aeb"],
        "active_motifs": list(active.keys()),
    }


def compute_pmf(
    pred: MotifSet,
    gt: MotifSet,
    api_key: str | None = None,
    judge_model: str = DEFAULT_OPENAI_MODEL,
    weights: dict[str, float] | None = None,
    use_semantic_judge: bool = True,
) -> dict:
    """Synchronous PMF computation. Builds its own judge client if needed."""
    _api_key = resolve_api_key(api_key, env_names=("OPENAI_API_KEY",))

    async def _run():
        judge_client = None
        if use_semantic_judge and _api_key:
            judge_client = AsyncOpenAI(**build_client_kwargs(api_key=_api_key, base_url=os.environ.get("OPENAI_BASE_URL", "").strip()))
        try:
            return await compute_pmf_async(
                pred, gt,
                judge_client=judge_client,
                judge_model=judge_model,
                weights=weights,
            )
        finally:
            if judge_client is not None:
                await judge_client.close()

    return asyncio.run(_run())
