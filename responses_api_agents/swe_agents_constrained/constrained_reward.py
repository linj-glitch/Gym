# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constraint grading + reward shaping for the constrained SWE-bench wrapper.

Free of nemo_gym / OpenHands imports so the shaping semantics are unit-testable
in any venv (the grading core is stdlib + pydantic only).
"""

import json
import logging
from typing import Any

from responses_api_agents.swe_agents_constrained.grading import (
    InjectionMode,
    compute_reward,
    grade_constraints,
    parse_trajectory,
)
from responses_api_agents.swe_agents_constrained.grading.reward import _DEFAULT_ALPHA, TrainingMode


log = logging.getLogger(__name__)

DEFAULT_CONSTRAINT_ALPHA = _DEFAULT_ALPHA[TrainingMode.FORMAT]


def coerce_constraint_declarations(raw: list) -> list[dict]:
    """Normalize metadata constraint declarations to the [{type, params}] schema.

    Accepts the canonical schema ({"type": ..., "params": {...}}) and coerces
    legacy bare-string entries ("unified_diff") for older datasets.
    """
    declarations = []
    for entry in raw or []:
        if isinstance(entry, str):
            declarations.append({"type": entry, "params": {}})
        elif isinstance(entry, dict) and "type" in entry:
            declarations.append({"type": entry["type"], "params": entry.get("params") or {}})
        else:
            raise ValueError(f"Malformed constraint declaration: {entry!r}")
    return declarations


REWARD_MODES = ("shaped", "strict", "tiered")
DEFAULT_PARTIAL_REWARD = 0.5


def grade_and_shape(
    output_items: Any,
    metadata: dict[str, str],
    task_reward: float,
    default_alpha: float,
    default_reward_mode: str = "shaped",
    default_partial_reward: float = DEFAULT_PARTIAL_REWARD,
) -> dict[str, Any]:
    """Grade a completed trajectory and shape the task reward.

    Returns the constraint response fields (including the final ``reward``) to
    overlay on the base verify response. Grading failures never crash the
    rollout: the task reward passes through unshaped and the error is recorded
    in ``violations``.

    reward_mode (config default, overridable per row via metadata):
      "shaped": reward = task * (1 + alpha * constraint_fraction)   (original)
      "strict": reward = task * 1[every applicable constraint passed at every
                applicable turn]. A single violation anywhere zeroes the
                trace; a trace where no constraint was gradeable (no applicable
                turn) also earns 0 -- "not measured" is not compliance, and a
                vacuous pass would reward avoiding the trigger.
      "tiered": task fail -> 0; task solved -> ``partial`` (default 0.5);
                task solved AND every applicable constraint passed at every
                applicable turn -> 1. Same all-pass predicate as strict, but a
                solved-yet-violating trace keeps partial credit so the task
                signal survives (Lin, 2026-09-07: strict left 0-14 positives
                per 128 samples and no gradient toward solving). ``partial``
                comes from ``success_partial_reward`` in the config, overridable
                per row via metadata ``success_partial_reward``.
      Task failure is 0 in every mode.

    Per-turn verdicts (``turn_verdicts``, 1-based assistant turn index, from
    the grading core's StepVerdicts) are always emitted so the trainer can
    assign credit to the turns that carried a failing check.
    """
    alpha = float(metadata.get("constraint_alpha", default_alpha))
    reward_mode = str(metadata.get("reward_mode", default_reward_mode))
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    partial = float(metadata.get("success_partial_reward", default_partial_reward))
    if not 0.0 <= partial <= 1.0:
        raise ValueError(f"success_partial_reward must be in [0, 1], got {partial!r}")
    fields: dict[str, Any] = {
        "reward": task_reward,
        "reward_components": {"task": task_reward},
        "task_reward": task_reward,
        "constraint_reward": None,
        "constraint_graded": False,
        "constraint_alpha": alpha,
        "reward_mode": reward_mode,
        "constraint_all_pass": False,
        "turn_verdicts": [],
        "first_violation_turn": None,
        "num_graded_turns": 0,
        "num_violating_turns": 0,
    }

    try:
        # Canonical rows carry a JSON string (Responses metadata is
        # Dict[str, str]); tolerate an already-parsed list from older files.
        raw = metadata.get("constraints", "[]") or "[]"
        if isinstance(raw, str):
            raw = json.loads(raw)
        constraints = coerce_constraint_declarations(raw)
        if not constraints:
            return fields
        steps = parse_trajectory(output_items)
        grading = grade_constraints(
            steps,
            constraints,
            injection_mode=InjectionMode(metadata.get("injection_mode", InjectionMode.SYSTEM_PROMPT)),
            injection_step=int(metadata.get("injection_step", 0)),
            grading_mode=metadata.get("grading_mode", "fraction"),
            step_aggregation=metadata.get("step_aggregation", "mean"),
        )
    except Exception as e:
        log.exception("Constraint grading failed; passing task reward through unshaped")
        fields["violations"] = [f"constraint grading error: {e}"]
        return fields

    turn_verdicts = [
        {
            "turn": v.turn,
            "step_index": v.step_index,
            "constraint": v.constraint,
            "passed": bool(v.passed),
            "kind": v.kind,
            "violation": v.violation,
        }
        for v in grading.step_verdicts
    ]
    violating_turns = sorted({v.turn for v in grading.step_verdicts if not v.passed})
    graded_turns = sorted({v.turn for v in grading.step_verdicts})
    all_pass = bool(grading.step_verdicts) and not violating_turns

    fields.update(
        constraint_graded=grading.any_graded,
        constraint_results=grading.constraint_results,
        constraint_scores=grading.constraint_scores,
        constraint_applicable=grading.constraint_applicable,
        violations=grading.violations,
        constraint_all_pass=all_pass,
        turn_verdicts=turn_verdicts,
        first_violation_turn=(violating_turns[0] if violating_turns else None),
        num_graded_turns=len(graded_turns),
        num_violating_turns=len(violating_turns),
    )
    if reward_mode in ("strict", "tiered"):
        # Conjunction inside the task gate. constraint_reward is reported as
        # the fraction (diagnostic); the reward itself is task * {0|partial, 1}
        # (strict: partial = 0; tiered: partial = success_partial_reward).
        gate = 1.0 if all_pass else (partial if reward_mode == "tiered" else 0.0)
        fields["reward"] = task_reward * gate
        fields["constraint_reward"] = grading.reward if grading.any_graded else 0.0
        fields["reward_components"]["constraint"] = gate
        fields["success_partial_reward"] = partial if reward_mode == "tiered" else 0.0
        for name, score in grading.constraint_scores.items():
            if grading.constraint_applicable.get(name):
                fields["reward_components"][f"constraint_{name}"] = score
    elif grading.any_graded:
        fields["reward"] = compute_reward(task_reward, grading.reward, alpha=alpha).total
        fields["constraint_reward"] = grading.reward
        fields["reward_components"]["constraint"] = grading.reward
        for name, score in grading.constraint_scores.items():
            if grading.constraint_applicable.get(name):
                fields["reward_components"][f"constraint_{name}"] = score
    return fields
