# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training reward from `if_constraints` records (the GRPO reward knob of swe_if_agents, 2026-09-08).

The benchmark path leaves `reward` as the SWE-bench outcome and aggregates the records offline (CR / SCR). For RL a
single scalar per episode is needed; this module turns the records the gym already attaches into that scalar with the
same three modes the detailed family used (responses_api_agents/swe_agents_constrained/constrained_reward.py), so a
training recipe can switch family without changing its reward semantics:

    outcome  reward = task                                  (default; the benchmark behaviour, records attached only)
    shaped   reward = task * (1 + alpha * constraint)       when at least one constraint was applicable, else task
    strict   reward = task * 1[every applicable constraint all_pass]; nothing applicable -> 0
    tiered   reward = task * (1.0 if all applicable pass else partial); nothing applicable -> task * partial

`constraint` is the mean over APPLICABLE records (n_steps > 0, no grading error) of the per-record step average
(n_pass / n_steps). Task failure gives 0 in shaped/strict/tiered (multiplicative), so the constraint axis can never
buy reward on an unsolved task. Records with an `error` (retired/unknown matcher) or n_steps == 0 are not applicable.

The per-turn detail is also flattened into `turn_verdicts` for the trainer's turn-level credit assignment
(NeMo-RL `grpo.constraint_violation_advantage` reads `turn_verdicts[].turn` as the 1-based assistant turn and
`passed`). The grader's `steps[].turn` is 0-based and, for prefix rows, relative to the replayed prefix's cut — which is
exactly the frame of the assistant messages NeMo-RL builds from `response.output` (generated turns only), so +1 is the
whole mapping. Standard library only, like the rest of the package.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

REWARD_MODES = ("outcome", "shaped", "strict", "tiered")
DEFAULT_ALPHA = 1.0
DEFAULT_PARTIAL = 0.5


def _is_applicable(rec: dict) -> bool:
    return not rec.get("error") and int(rec.get("n_steps") or 0) > 0


def compute_if_reward(
    records: Optional[List[dict]],
    task_reward: float,
    mode: str = "outcome",
    alpha: float = DEFAULT_ALPHA,
    partial: float = DEFAULT_PARTIAL,
) -> Dict[str, Any]:
    """Return the reward and its decomposition. Never raises for the content of `records`."""
    if mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {mode!r}")
    if not 0.0 <= float(partial) <= 1.0:
        raise ValueError(f"success_partial_reward must be in [0, 1], got {partial!r}")
    task = float(task_reward or 0.0)
    recs = [r for r in (records or []) if isinstance(r, dict)]
    applicable = [r for r in recs if _is_applicable(r)]
    errors = [r for r in recs if r.get("error")]

    turn_verdicts: List[dict] = []
    for r in applicable:
        for st in r.get("steps") or []:
            turn_verdicts.append(
                {
                    "turn": int(st.get("turn", 0)) + 1,
                    "passed": int(st.get("reward", 0)) >= 1,
                    "constraint": str(r.get("id")),
                    "kind": "text",
                    "detail": str(st.get("detail") or ""),
                }
            )
    violating = sorted({v["turn"] for v in turn_verdicts if not v["passed"]})
    graded_turns = sorted({v["turn"] for v in turn_verdicts})

    any_graded = bool(applicable)
    fraction = (sum(float(r.get("step_avg") or 0.0) for r in applicable) / len(applicable)) if any_graded else None
    all_pass = any_graded and all(bool(r.get("all_pass")) for r in applicable)

    components: Dict[str, float] = {"task": task}
    if mode == "outcome":
        reward = task
        gate: Optional[float] = None
    elif mode == "shaped":
        reward = task * (1.0 + float(alpha) * fraction) if any_graded else task
        gate = fraction
    elif mode == "strict":
        gate = 1.0 if all_pass else 0.0
        reward = task * gate
    else:  # tiered
        gate = 1.0 if all_pass else float(partial)
        reward = task * gate
    if gate is not None:
        components["constraint"] = float(gate)
    for r in applicable:
        components[f"constraint_{r.get('id')}"] = float(r.get("step_avg") or 0.0)

    return {
        "reward": float(reward),
        "task_reward": task,
        "constraint_reward": fraction,
        "constraint_graded": any_graded,
        "constraint_all_pass": bool(all_pass),
        "reward_mode": mode,
        "constraint_alpha": float(alpha),
        "success_partial_reward": float(partial) if mode == "tiered" else 0.0,
        "n_constraints": len(recs),
        "n_applicable": len(applicable),
        "n_grading_errors": len(errors),
        "turn_verdicts": turn_verdicts,
        "first_violation_turn": violating[0] if violating else None,
        "num_graded_turns": len(graded_turns),
        "num_violating_turns": len(violating),
        "continuation_only": any(bool(r.get("continuation_only")) for r in applicable),
        "reward_components": components,
    }


def resolve_reward_settings(metadata: Optional[Dict[str, Any]], mode: str, alpha: float, partial: float):
    """Per-row overrides (metadata values are strings): reward_mode, constraint_alpha, success_partial_reward."""
    md = metadata or {}
    m = str(md.get("reward_mode") or mode)
    a = float(md.get("constraint_alpha", alpha))
    p = float(md.get("success_partial_reward", partial))
    return m, a, p
