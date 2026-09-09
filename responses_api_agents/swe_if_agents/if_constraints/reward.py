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
buy reward on an unsolved task.

Two situations pass the task reward through UNSHAPED in every mode, as the detailed family does (constrained_reward.py:
`if not constraints: return fields`, and its grading-exception handler):

* no constraints declared (`records` is None — grade_row's answer to a row without sdg_item or with an empty
  constraints list — or an empty list). This is distinct from "declared but none applicable" (records with
  n_steps == 0), which keeps strict -> 0 / tiered -> task * partial: a policy that avoids the trigger of a declared
  constraint must not collect the reward vacuously, but a row that never asked for anything is an ordinary task row.
* a whole-row grading error: the grader's catch-all `<grading_error>` pseudo-record (an exception before any constraint
  was graded, e.g. malformed sdg_item or tool_name_overrides). Nothing about the row can be trusted, so the task
  reward passes through and, in every mode but outcome, `mask_sample` is True so the wrapper excludes the sample
  from the gradient (it sets SWEBenchWrapperInstanceConfig.mask_sample, which NeMo-RL reads from
  full_result.instance_config.mask_sample). The pseudo-record is not a declared constraint (excluded from
  `n_constraints`). A PER-CONSTRAINT error record (a retired or unknown matcher on one constraint) is the recipe's
  "not applicable, row not lost": it counts in `n_constraints` and `n_grading_errors` but not in `n_applicable`,
  the remaining constraints fold normally, and the sample is kept.

The per-turn detail is also flattened into `turn_verdicts` for the trainer's turn-level credit assignment
(NeMo-RL `grpo.constraint_violation_advantage` reads `turn_verdicts[].turn` as the 1-based assistant turn and
`passed`). The grader's `steps[].turn` is 0-based and, for prefix rows, relative to the replayed prefix's cut — which is
exactly the frame of the assistant messages NeMo-RL builds from `response.output` (generated turns only), so +1 is the
whole mapping. Trajectory-scoped steps (`tool_choice`: the verifier emits turn = -1) have no assistant turn to map onto:
they gate the record's all_pass / step_avg like any other step but emit no turn_verdicts entry and count in neither the
graded nor the violating turns. Standard library only, like the rest of the package.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

REWARD_MODES = ("outcome", "shaped", "strict", "tiered")
DEFAULT_ALPHA = 1.0
DEFAULT_PARTIAL = 0.5

# The grader's whole-row failure record (grader.GRADING_ERROR_ID); repeated here so this module stays import-free.
GRADING_ERROR_ID = "<grading_error>"


def _is_applicable(rec: dict) -> bool:
    return not rec.get("error") and int(rec.get("n_steps") or 0) > 0


def compute_if_reward(
    records: Optional[List[dict]],
    task_reward: float,
    mode: str = "outcome",
    alpha: float = DEFAULT_ALPHA,
    partial: float = DEFAULT_PARTIAL,
) -> Dict[str, Any]:
    """Return the reward and its decomposition. Never raises for the content of `records`.

    `mask_sample` in the result is True when the grades could not be trusted (a grading error) in a mode that would
    have folded them; the reward is then the task reward and the caller propagates the flag to the sample's mask."""
    if mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {mode!r}")
    if not 0.0 <= float(partial) <= 1.0:
        raise ValueError(f"success_partial_reward must be in [0, 1], got {partial!r}")
    task = float(task_reward or 0.0)
    recs = [r for r in (records or []) if isinstance(r, dict)]
    declared = bool(recs)  # None (grade_row: no sdg_item / empty constraints) or an empty list: nothing was asked
    errors = [r for r in recs if r.get("error")]
    constraints = [r for r in recs if r.get("id") != GRADING_ERROR_ID]  # the pseudo-record is not a constraint
    applicable = [r for r in constraints if _is_applicable(r)]
    # Only the grader's WHOLE-ROW failure makes the grade untrustworthy. A per-constraint error (a retired or unknown
    # matcher on one of several constraints) is the recipe's "not applicable, row not lost": the other constraints
    # still fold, and the sample is kept for training.
    grading_errored = any(r.get("id") == GRADING_ERROR_ID for r in recs)

    turn_verdicts: List[dict] = []
    for r in applicable:
        for st in r.get("steps") or []:
            turn = int(st.get("turn", 0))
            if turn < 0:  # trajectory-scoped step (tool_choice): no assistant turn to attribute it to
                continue
            turn_verdicts.append(
                {
                    "turn": turn + 1,
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
    gate: Optional[float] = None
    if mode == "outcome" or not declared or grading_errored:
        # benchmark mode, no constraints declared, or an unreliable grade: the task reward passes through unshaped
        reward = task
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
        "n_constraints": len(constraints),
        "n_applicable": len(applicable),
        "n_grading_errors": len(errors),
        "mask_sample": bool(grading_errored and mode != "outcome"),
        "turn_verdicts": turn_verdicts,
        "first_violation_turn": violating[0] if violating else None,
        "num_graded_turns": len(graded_turns),
        "num_violating_turns": len(violating),
        "continuation_only": any(bool(r.get("continuation_only")) for r in applicable),
        "reward_components": components,
    }


def resolve_reward_settings(
    metadata: Optional[Dict[str, Any]], mode: str, alpha: float, partial: float, if_grading: bool = True
) -> Tuple[str, float, float, Optional[str]]:
    """Per-row overrides (metadata values are strings): reward_mode, constraint_alpha, success_partial_reward.

    Returns (mode, alpha, partial, error). Never raises: a malformed override must not cost the episode it rides on
    (run() folds the reward only after the rollout), so each bad value falls back to the server's config value (which
    SWEIFWrapperConfig validated at startup) and `error` names the rejected ones for the response's
    `reward_settings_error`; None when every override was accepted. `reward_mode` is case-insensitive."""
    md = metadata or {}
    problems: List[str] = []

    m = mode
    raw_mode = md.get("reward_mode")
    if raw_mode is not None and str(raw_mode).strip() != "":
        cand = str(raw_mode).strip().lower()
        if cand not in REWARD_MODES:
            problems.append(f"reward_mode={raw_mode!r} not in {REWARD_MODES}; using {mode!r}")
        elif cand != "outcome" and not if_grading:
            problems.append(f"reward_mode={cand!r} needs if_grading: true (the records are the reward's input); using {mode!r}")
        else:
            m = cand

    a = float(alpha)
    if "constraint_alpha" in md:
        try:
            a = float(md["constraint_alpha"])
        except (TypeError, ValueError):
            problems.append(f"constraint_alpha={md['constraint_alpha']!r} is not a number; using {alpha!r}")

    p = float(partial)
    if "success_partial_reward" in md:
        try:
            cand_p = float(md["success_partial_reward"])
        except (TypeError, ValueError):
            problems.append(f"success_partial_reward={md['success_partial_reward']!r} is not a number; using {partial!r}")
        else:
            if 0.0 <= cand_p <= 1.0:
                p = cand_p
            else:
                problems.append(f"success_partial_reward={cand_p!r} not in [0, 1]; using {partial!r}")

    return m, a, p, ("; ".join(problems) if problems else None)
