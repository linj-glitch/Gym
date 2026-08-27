# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint-verified BINARY reward for pivot-RL (milestone 2).

    reward = 1.0  iff  binary_match(sampled, expected) AND constraint PASS
             0.0  otherwise            (no partial credit — Lin, 2026-08-27)

This is the SAME success predicate the mining belt used to select gated
states (pivot_branch_probe.py: is_teacher / gated_n_success = matched AND
constraint == 'pass'), evaluated by the same code:

- match: L0 tool-name/category -> L1 target -> L2 argument-similarity >= 0.8,
  imported from resources_servers/swe_pivot/app.py — the module the mining
  belt's _pivot_match.py was mechanically copied FROM (agentic-if
  _pivot_match.py header, copy of commit d0badcaf). Message-kind expected
  actions follow pivot_branch_probe._branch_match: matched iff the sampled
  turn contains NO tool call.
- constraint: deterministic verifier from the vendored grading core
  (responses_api_agents/swe_agents_constrained), grading surface identical
  to pivot_branch_probe.grade_branch_in_context — prefix + sampled action
  parsed as ONE trajectory, verdicts read at the branch turns. Params come
  from row metadata `constraint_params` (the value-sampled injected params;
  JSON str) — registry defaults are NOT equivalent.

Ungraded or missing-constraint rows earn 0.0: mining never counted a branch
as success without an explicit constraint PASS, and neither do we.
"""
import json
import logging
from enum import Enum
from typing import Any, List, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
    extract_tool_call_or_text,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
)
from resources_servers.swe_pivot.app import (
    compute_argument_similarity,
    extract_tool_info,
    verify_target_match,
    verify_tool_name_match,
)
from responses_api_agents.swe_agents_constrained.grading.verifiers.trajectory import (
    grade_constraints,
    parse_trajectory,
)

logger = logging.getLogger(__name__)


class ConstraintVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNGRADED = "ungraded"
    NO_CONSTRAINT_IN_ROW = "no_constraint_in_row"


class ConstrainedBinaryPivotResourcesServerConfig(BaseResourcesServerConfig):
    # L2 aggregation threshold; 0.8 = swe_pivot verify() binary-mode default,
    # the value the mining belt's binary_match used.
    similarity_full_credit: float = 0.8


class ConstrainedBinaryPivotRunRequest(BaseRunRequest):
    expected_action: ExpectedAction


class ConstrainedBinaryPivotVerifyRequest(ConstrainedBinaryPivotRunRequest, BaseVerifyRequest):
    pass


class ConstrainedBinaryPivotVerifyResponse(BaseVerifyResponse):
    expected_action: ExpectedAction
    matched: bool
    match_failure_reason: str
    constraint_verdict: ConstraintVerdict


class ConstrainedBinaryPivotResourcesServer(SimpleResourcesServer):
    config: ConstrainedBinaryPivotResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    # ---- match axis (mirror of _pivot_match.binary_match + _branch_match) ----

    def _binary_match(self, expected: dict, sampled: Optional[Any]) -> tuple[bool, str]:
        if expected.get("type") == "message":
            # pivot_branch_probe._branch_match message-kind rule: a message
            # was demonstrated, so the sampled turn must not call a tool.
            if sampled is None or sampled.type != "function_call":
                return True, "none"
            return False, "kind_mismatch"
        if sampled is None or sampled.type != "function_call":
            return False, "model_output_invalid"
        e_call = {"name": expected.get("name", ""), "arguments": expected.get("arguments", "")}
        r_call = {"name": getattr(sampled, "name", ""), "arguments": getattr(sampled, "arguments", "")}
        e_name, e_cat, e_args = extract_tool_info(e_call)
        r_name, r_cat, r_args = extract_tool_info(r_call)
        if not verify_tool_name_match(r_name, r_cat, e_name, e_cat):
            return False, "tool_name_mismatch"
        if not verify_target_match(r_cat, r_args, e_cat, e_args):
            return False, "target_mismatch"
        sim = compute_argument_similarity(r_cat, r_args, e_cat, e_args)
        if sim < self.config.similarity_full_credit:
            return False, "similarity_below_threshold"
        return True, "none"

    # ---- constraint axis (mirror of grade_branch_in_context) ----

    @staticmethod
    def _grade_constraint(prefix_items: List[Any], sampled_items: List[Any], decl: dict) -> ConstraintVerdict:
        n_prefix = len(prefix_items)
        steps = parse_trajectory(list(prefix_items) + list(sampled_items))
        if not steps:
            return ConstraintVerdict.UNGRADED
        grading = grade_constraints(steps, [decl], grading_mode="fraction", step_aggregation="mean")
        branch_turns = {s.turn for s in steps if s.step_index >= n_prefix}
        at_branch = [v for v in grading.step_verdicts if v.turn in branch_turns]
        if not at_branch:
            return ConstraintVerdict.UNGRADED
        return ConstraintVerdict.PASS if all(v.passed for v in at_branch) else ConstraintVerdict.FAIL

    @staticmethod
    def _row_decl(metadata: Optional[dict]) -> Optional[dict]:
        md = metadata or {}
        name = md.get("constraint")
        if not name:
            return None
        raw = md.get("constraint_params") or "{}"
        try:
            params = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            logger.warning("unparseable constraint_params for %s; using {}", name)
            params = {}
        if not md.get("constraint_params"):
            logger.warning(
                "row has constraint=%s but no constraint_params — grading with {} "
                "(backfill rows for mining/training parity)", name)
        return {"type": name, "params": params}

    async def verify(self, body: ConstrainedBinaryPivotVerifyRequest) -> ConstrainedBinaryPivotVerifyResponse:
        def _dump(item: Any) -> Any:
            return item.model_dump() if hasattr(item, "model_dump") else item

        expected = _dump(body.expected_action)
        sampled = extract_tool_call_or_text(body.response)
        matched, why = self._binary_match(expected, sampled)

        verdict = ConstraintVerdict.NO_CONSTRAINT_IN_ROW
        if matched:  # constraint grading only decides matched cases; skip cost otherwise
            metadata = getattr(body.responses_create_params, "metadata", None)
            if metadata is not None and hasattr(metadata, "model_dump"):
                metadata = metadata.model_dump()
            decl = self._row_decl(metadata)
            if decl is None:
                verdict = ConstraintVerdict.NO_CONSTRAINT_IN_ROW
            else:
                prefix_items = [_dump(i) for i in (body.responses_create_params.input or [])]
                sampled_items = [_dump(i) for i in (body.response.output or [])]
                verdict = self._grade_constraint(prefix_items, sampled_items, decl)

        # Binary, mining-aligned: success = matched AND constraint PASS.
        # Rows without a constraint (not produced by the pivot belt) degrade
        # to match-only — still strictly 0/1.
        constraint_ok = verdict in (ConstraintVerdict.PASS, ConstraintVerdict.NO_CONSTRAINT_IN_ROW)
        reward = 1.0 if (matched and constraint_ok) else 0.0
        return ConstrainedBinaryPivotVerifyResponse(
            **body.model_dump(),
            reward=reward,
            matched=matched,
            match_failure_reason=why,
            constraint_verdict=verdict,
        )


if __name__ == "__main__":
    ConstrainedBinaryPivotResourcesServer.run_webserver()
