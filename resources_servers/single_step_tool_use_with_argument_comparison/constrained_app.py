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

The response also carries scalar diagnostics that decompose the reward into
its two axes. NeMo-RL's per-agent aggregator (rollouts.py:1479) promotes every
bool/int/float field of this response to `<agent>/<field>/mean` and silently
drops everything else, which is why `matched` shows up in wandb but the string
`match_failure_reason` and `constraint_verdict` do not. The bools below are
one-hot projections of those two strings so both axes become observable
without changing the reward.

Constraint families (2026-09-08, Lin: training data comes from the TEMPLATE family).
The row's constraint is graded by the family it belongs to:

- detailed  metadata `constraint` = registry id, `constraint_params` = {param: value};
            verifier = responses_api_agents/swe_agents_constrained/grading (as before).
- template  metadata `constraint` = the sdg constraint id, `constraint_params` = the
            recipe's `verifier_parameter` {template, trigger, obligation[, no_answer]},
            optional `tool_name_overrides`; verifier = the swe_if_agents grader
            (responses_api_agents/swe_if_agents/if_constraints.grade_row), the ONLY
            implementation of the template verifier (owner ruling 2026-09-04).

The family is `metadata.family` when the row says so (P5 make_pivot_rows.py stamps it),
otherwise inferred from the shape of `constraint_params` (a verifier_parameter carries
`trigger`/`obligation`). Both families are graded on the SAME surface as the mining
belt's P4 (pivot_branch_probe.grade_branch_in_context): prefix + sampled action as ONE
trajectory in the whole-trajectory frame, verdicts read at the turns the sampled items
belong to. A prefix whose last turn is still open (narration without a tool result) is
merged with the sampled call into one turn, and that merged turn counts as a branch
turn — for both families.

reward_mode="verifier" (2026-09-06) turns the same server into a verifier-only
GRPO reward: 1.0 iff the constraint verifier PASSES on the model's own action
(teacher action ignored; match still reported as a diagnostic). The two modes
are registered under different instance names (configs/swe_pivot_constrained_*
vs configs/swe_verifier_*) so one training run can carry both and each row
selects its reward through agent_ref.
"""

import json
import logging
from enum import Enum
from typing import Any, List, Literal, Optional

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

try:  # the template family's verifier lives in swe_if_agents (branch charlwang/swe-if-agents and descendants)
    from responses_api_agents.swe_if_agents.if_constraints import grade_row as _template_grade_row
    from responses_api_agents.swe_if_agents.if_constraints.grader import GRADING_ERROR_ID as _TEMPLATE_ERROR_ID
    from responses_api_agents.swe_if_agents.if_constraints.grader import segment as _template_segment
except ImportError:  # pragma: no cover - checkout without the template package
    _template_grade_row = None
    _TEMPLATE_ERROR_ID = "<grading_error>"
    _template_segment = None

FAMILIES = ("detailed", "template")


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
    # Grade the constraint on match-failed rollouts too. Off by default: it
    # adds a verifier pass per unmatched rollout on the generation critical
    # path and buys only a diagnostic — reward is unchanged either way, since
    # reward requires matched. Turn it on to observe the UNCONDITIONAL
    # constraint pass rate; with it off, only P(pass | matched) is knowable.
    always_grade_constraint: bool = False
    # Which axis decides the reward (Lin, 2026-09-06 — a verifier-only GRPO
    # recipe that lives NEXT TO pivot, selectable per row via agent_ref):
    #   "pivot"    : reward = 1 iff binary_match(sampled, expected) AND
    #                constraint PASS (rows without a constraint degrade to
    #                match-only). The mining belt's gated-success predicate.
    #   "verifier" : reward = 1 iff the deterministic constraint verifier
    #                returns PASS on the model's OWN action, regardless of the
    #                teacher action. The constraint is always graded. UNGRADED
    #                (constraint not applicable at the sampled turn) and rows
    #                without a constraint earn 0.0 — a vacuous pass would let
    #                the policy collect reward by avoiding the trigger. An
    #                output the parser cannot read (neither a tool call nor
    #                text) earns 0.0 as well. `matched` and the match_fail_*
    #                fields are still computed and reported as diagnostics.
    reward_mode: Literal["pivot", "verifier"] = "pivot"


class ConstrainedBinaryPivotRunRequest(BaseRunRequest):
    expected_action: ExpectedAction


class ConstrainedBinaryPivotVerifyRequest(ConstrainedBinaryPivotRunRequest, BaseVerifyRequest):
    pass


class ConstrainedBinaryPivotVerifyResponse(BaseVerifyResponse):
    expected_action: ExpectedAction
    matched: bool
    match_failure_reason: str
    constraint_verdict: ConstraintVerdict
    # "detailed" | "template" | "" (no constraint in the row). A string: reported, not promoted to a metric.
    constraint_family: str = ""

    # --- constraint axis (instruction following) ---
    # All four are False when grading did not run — i.e. the match failed and
    # always_grade_constraint is off. That keeps `constraint_absent` meaning
    # "the row carried no constraint declaration" rather than doubling as the
    # not-graded sentinel, which is what ConstraintVerdict.NO_CONSTRAINT_IN_ROW
    # does on the enum.
    constraint_passed: bool
    constraint_failed: bool
    constraint_ungraded: bool
    constraint_absent: bool
    # passed or failed — the denominator for P(pass | graded).
    constraint_graded: bool

    # --- match axis (task solving) ---
    # One-hot over _binary_match's failure reasons; exactly one is True when
    # matched is False, all False when matched is True.
    match_fail_kind: bool
    match_fail_invalid_output: bool
    match_fail_tool_name: bool
    match_fail_target: bool
    match_fail_similarity: bool
    # L2 score, kept even when it lands under the threshold, so a policy
    # creeping up on similarity_full_credit is visible before it converts into
    # match rate. None (and skipped by the aggregator) when the rollout failed
    # at L0/L1 and never reached the similarity stage.
    argument_similarity: Optional[float] = None
    similarity_evaluated: bool


class ConstrainedBinaryPivotResourcesServer(SimpleResourcesServer):
    config: ConstrainedBinaryPivotResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    # ---- match axis (mirror of _pivot_match.binary_match + _branch_match) ----

    def _binary_match(self, expected: dict, sampled: Optional[Any]) -> tuple[bool, str, Optional[float]]:
        """Return (matched, failure_reason, argument_similarity_or_None).

        The similarity is reported whenever L2 was reached, pass or fail; it is
        None for rollouts rejected at L0/L1, where no similarity exists.
        """
        if expected.get("type") == "message":
            # pivot_branch_probe._branch_match message-kind rule: a message
            # was demonstrated, so the sampled turn must not call a tool.
            if sampled is None or sampled.type != "function_call":
                return True, "none", None
            return False, "kind_mismatch", None
        if sampled is None or sampled.type != "function_call":
            return False, "model_output_invalid", None
        e_call = {"name": expected.get("name", ""), "arguments": expected.get("arguments", "")}
        r_call = {"name": getattr(sampled, "name", ""), "arguments": getattr(sampled, "arguments", "")}
        e_name, e_cat, e_args = extract_tool_info(e_call)
        r_name, r_cat, r_args = extract_tool_info(r_call)
        if not verify_tool_name_match(r_name, r_cat, e_name, e_cat):
            return False, "tool_name_mismatch", None
        if not verify_target_match(r_cat, r_args, e_cat, e_args):
            return False, "target_mismatch", None
        sim = compute_argument_similarity(r_cat, r_args, e_cat, e_args)
        if sim < self.config.similarity_full_credit:
            return False, "similarity_below_threshold", sim
        return True, "none", sim

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

    # ---- constraint families ----

    @staticmethod
    def _row_params(md: dict) -> dict:
        raw = md.get("constraint_params") or "{}"
        try:
            return json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            logger.warning("unparseable constraint_params for %s; using {}", md.get("constraint"))
            return {}

    @classmethod
    def _row_family(cls, md: dict) -> str:
        """`metadata.family` when stamped by P5; else inferred: a recipe verifier_parameter carries trigger/obligation."""
        fam = str(md.get("family") or "").strip().lower()
        if fam in FAMILIES:
            return fam
        params = cls._row_params(md)
        return "template" if ("trigger" in params or "obligation" in params or params.get("template")) else "detailed"

    @staticmethod
    def _branch_turns_template(prefix_items: List[dict], sampled_items: List[dict]) -> set:
        """Turn indices (template grader frame, 0-based, whole trajectory) that contain sampled items.

        The grader's segmenter closes a turn at each tool result; a prefix ending in an open turn (narration
        without a result) merges with the sampled call into one turn, which then IS a branch turn — the same rule
        the detailed path applies through parse_trajectory's step_index >= n_prefix.
        """
        asst_prefix = [
            o for o in prefix_items
            if o.get("type") in ("message", "function_call", "function_call_output", "reasoning")
            and not (o.get("type") == "message" and o.get("role") in ("system", "user", "developer"))
        ]
        pre = _template_segment(asst_prefix)
        last = next((o for o in reversed(asst_prefix) if o.get("type") != "reasoning"), None)
        prefix_open = bool(pre) and last is not None and last.get("type") in ("message", "function_call")
        start = len(pre) - 1 if prefix_open else len(pre)
        n_all = len(_template_segment(asst_prefix + list(sampled_items)))
        return set(range(max(start, 0), n_all))

    @classmethod
    def _grade_constraint_template(cls, prefix_items: List[Any], sampled_items: List[Any], md: dict) -> ConstraintVerdict:
        if _template_grade_row is None:
            raise RuntimeError(
                "row carries a template-family constraint but responses_api_agents.swe_if_agents is not in this Gym "
                "checkout (need branch charlwang/swe-if-agents or a descendant such as linj/pivotrl-lineage-if)"
            )
        cid = str(md.get("constraint"))
        sdg_item = {
            "type": "fresh",  # whole-trajectory frame, no prefix skipping: identical to the mining belt's P4 surface
            "persona": "opencode",
            "constraints": [{"id": cid, "verifier_parameter": cls._row_params(md), "reference_instruction": ""}],
        }
        tmd = {"sdg_item": json.dumps(sdg_item)}
        if md.get("tool_name_overrides"):
            tno = md["tool_name_overrides"]
            tmd["tool_name_overrides"] = tno if isinstance(tno, str) else json.dumps(tno)
        items = list(prefix_items) + list(sampled_items)
        records = _template_grade_row(tmd, None, items) or []
        rec = next((r for r in records if r.get("id") == cid), None)
        if rec is None or rec.get("id") == _TEMPLATE_ERROR_ID or rec.get("error"):
            logger.warning("template grading error for %s: %s", cid, (rec or {}).get("error"))
            return ConstraintVerdict.UNGRADED
        branch = cls._branch_turns_template(prefix_items, sampled_items)
        at_branch = [st for st in rec.get("steps") or [] if st.get("turn") in branch]
        if not at_branch:
            return ConstraintVerdict.UNGRADED
        return ConstraintVerdict.PASS if all(int(st.get("reward", 0)) >= 1 for st in at_branch) else ConstraintVerdict.FAIL

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
                "(backfill rows for mining/training parity)",
                name,
            )
        return {"type": name, "params": params}

    async def verify(self, body: ConstrainedBinaryPivotVerifyRequest) -> ConstrainedBinaryPivotVerifyResponse:
        def _dump(item: Any) -> Any:
            return item.model_dump() if hasattr(item, "model_dump") else item

        expected = _dump(body.expected_action)
        sampled = extract_tool_call_or_text(body.response)
        matched, why, similarity = self._binary_match(expected, sampled)

        verdict = ConstraintVerdict.NO_CONSTRAINT_IN_ROW
        family = ""
        # Constraint grading only decides matched cases, so by default we skip
        # the cost otherwise; always_grade_constraint trades that cost for the
        # unconditional pass rate. Either way `graded` records whether the
        # verifier actually ran, which is what the diagnostics key off.
        verifier_mode = self.config.reward_mode == "verifier"
        graded = matched or self.config.always_grade_constraint or verifier_mode
        if graded:
            metadata = getattr(body.responses_create_params, "metadata", None)
            if metadata is not None and hasattr(metadata, "model_dump"):
                metadata = metadata.model_dump()
            md = metadata or {}
            if not md.get("constraint"):
                verdict = ConstraintVerdict.NO_CONSTRAINT_IN_ROW
            else:
                family = self._row_family(md)
                prefix_items = [_dump(i) for i in (body.responses_create_params.input or [])]
                sampled_items = [_dump(i) for i in (body.response.output or [])]
                if family == "template":
                    verdict = self._grade_constraint_template(prefix_items, sampled_items, md)
                else:
                    verdict = self._grade_constraint(prefix_items, sampled_items, self._row_decl(md))

        # Binary, mining-aligned: success = matched AND constraint PASS.
        # Rows without a constraint (not produced by the pivot belt) degrade
        # to match-only — still strictly 0/1. Unchanged by the diagnostics and
        # by always_grade_constraint: a match failure is 0.0 whatever verdict
        # the extra grading pass returns.
        constraint_ok = verdict in (ConstraintVerdict.PASS, ConstraintVerdict.NO_CONSTRAINT_IN_ROW)
        if verifier_mode:
            # Verifier-only GRPO: the teacher action plays no part. PASS on the
            # model's own action is the only way to earn reward; UNGRADED,
            # missing constraint, and unparseable output all score 0.0.
            reward = 1.0 if (sampled is not None and verdict is ConstraintVerdict.PASS) else 0.0
        else:
            reward = 1.0 if (matched and constraint_ok) else 0.0
        return ConstrainedBinaryPivotVerifyResponse(
            **body.model_dump(),
            reward=reward,
            matched=matched,
            match_failure_reason=why,
            constraint_verdict=verdict,
            constraint_family=family,
            constraint_passed=graded and verdict is ConstraintVerdict.PASS,
            constraint_failed=graded and verdict is ConstraintVerdict.FAIL,
            constraint_ungraded=graded and verdict is ConstraintVerdict.UNGRADED,
            constraint_absent=graded and verdict is ConstraintVerdict.NO_CONSTRAINT_IN_ROW,
            constraint_graded=graded and verdict in (ConstraintVerdict.PASS, ConstraintVerdict.FAIL),
            match_fail_kind=why == "kind_mismatch",
            match_fail_invalid_output=why == "model_output_invalid",
            match_fail_tool_name=why == "tool_name_mismatch",
            match_fail_target=why == "target_mismatch",
            match_fail_similarity=why == "similarity_below_threshold",
            argument_similarity=similarity,
            similarity_evaluated=similarity is not None,
        )


if __name__ == "__main__":
    ConstrainedBinaryPivotResourcesServer.run_webserver()
