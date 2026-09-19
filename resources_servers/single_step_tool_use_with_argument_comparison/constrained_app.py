# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint-verified BINARY reward for pivot-RL (milestone 2).

    reward = 0.0  if the pivot tool check fails (binary_match false)
             1.0  if it passes AND the sampled action did not BREAK the constraint
             no partial credit (Lin, 2026-08-27; rule restated 2026-09-11)

"Did not break" (Lin, 2026-09-11, design/silent_turn_semantics.md): an explicit
FAIL at the branch turn breaks it. A SILENT branch turn (bare tool call, no
visible text) breaks NOTHING for any matcher — every middle-turn sentence in
the pool is conditional on a message existing ("the note you attach to the
call", "every message you write"). The one declared exception is a first-turn
sentence that demanded a message before any tool (trigger `must_speak`): the
verifier grades item 0 and emits the FAIL step itself. A trigger that does not
fire at the branch turn, a row without a constraint, or a grading error is not
a violation.

Silence rule (Lin, 2026-09-15, supersedes the 2026-09-12 reference_spoken
forfeit): a bare tool call where a tool call was demonstrated, under a
constraint that is fine with silence (every rule without `must_speak`), is
EXCLUDED from the GRPO group -- `excluded=True`, carried to NeMo-RL as
`instance_config.mask_sample`, which drops the sample from the gradient and
(grpo.mask_sample_excludes_from_baseline) from the group baseline / std. Neither
score is right for it: paying 1 taught "say nothing" (tplw4w6-100, silent share
of matched rollouts 0.09 -> 0.65 in 43 steps); paying 0 (the forfeit rule)
taught verbosity on the 84% of rows whose reference spoke (tplw4w16t1 steps
61-100: length +60%, entropy x2, reward flat). `reward` still reports the
success predicate for the row; `effective_reward` zeroes it on excluded rows.
The exclusion does NOT apply where the 0 is legitimate: a `must_speak` rule
(silence FAILS), a message-kind expected action (a tool call is a kind
mismatch: the model was supposed to stop), a row without a constraint
(match-only), or verifier mode. An EMPTY final message (no text, no call) where
a message was demonstrated is a match failure (`empty_message`), not a match.

This is the SAME success predicate the mining belt used to select gated
states (pivot_branch_probe.py: is_teacher / gated_n_success = matched AND
constraint == 'pass'), evaluated by the same code:

- match: L0 tool-name/category -> L1 target -> L2 argument-similarity >= 0.8,
  imported from resources_servers/swe_pivot/app.py — the module the mining
  belt's _pivot_match.py was mechanically copied FROM (agentic-if
  _pivot_match.py header, copy of commit d0badcaf). Message-kind expected
  actions follow pivot_branch_probe._branch_match: matched iff the sampled
  turn contains NO tool call.
- constraint: the deterministic template verifier,
  responses_api_agents.swe_if_agents.if_constraints.grade_row — the ONLY
  implementation of the template verifier (owner ruling 2026-09-04) — run on
  the mining belt's P4 surface (pivot_branch_probe.grade_branch_in_context):
  prefix + sampled action as ONE trajectory in the whole-trajectory frame,
  verdicts read at the turns the sampled items belong to. A prefix whose last
  turn is still open (narration without a tool result) merges with the
  sampled call into one turn, and that merged turn counts as a branch turn.

Row metadata: `constraint` = the sdg constraint id; `constraint_params` = the
recipe's `verifier_parameter` {template, trigger, obligation[, no_answer]} as
a JSON str — the value-sampled injected params, registry defaults are NOT
equivalent; optional `tool_name_overrides` (the tool binding the episode ran
under), passed through to the grader.

Silent turns: the template verifier never grades a middle turn without visible
text (owner ruling 2026-09-09), so the grader returns no step for it and the
verdict is UNGRADED = not broken (the matcher's `no_answer` kind governs the
missing FINAL message only). `constraint_silent` reports the case so the
narration rate is observable. Rows that carry no constraint at all (not
produced by the pivot belt) are match-only — still strictly 0/1. The mining
belt's gate uses the same predicate (matched AND not FAIL) and additionally
requires one SPOKEN success in the state (design §3, option b).

The response also carries scalar diagnostics that decompose the reward into
its two axes. NeMo-RL's per-agent aggregator (rollouts.py:1479) promotes every
bool/int/float field of this response to `<agent>/<field>/mean` and silently
drops everything else, which is why `matched` shows up in wandb but the string
`match_failure_reason` and `constraint_verdict` do not. The bools below are
one-hot projections of those two strings so both axes become observable
without changing the reward.

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
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
)
from resources_servers.swe_pivot.app import (
    compute_argument_similarity,
    extract_tool_info,
    verify_target_match,
    verify_tool_name_match,
)
from responses_api_agents.swe_if_agents.if_constraints import grade_row
from responses_api_agents.swe_if_agents.if_constraints import verifier as tv
from responses_api_agents.swe_if_agents.if_constraints.grader import GRADING_ERROR_ID, segment, to_tv_turns


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
    #   "pivot"    : reward = 1 iff binary_match(sampled, expected) AND the
    #                constraint was not broken: verdict is not FAIL. A silent
    #                branch turn is UNGRADED (= not broken) for every matcher;
    #                only a must_speak first turn FAILs on silence, via the
    #                verifier — Lin, 2026-09-11 (design/silent_turn_semantics.md).
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


def extract_tool_call_or_text(response: NeMoGymResponse) -> Optional[Any]:
    """First function_call item of the response, else the first assistant output_text item, else None.

    Local copy of the pre-2026-09 `common.response_utils.extract_tool_call_or_text` (the shared helper became
    `extract_action`, which folds several calls into a FunctionCallBatchAction and text into a MessageAction).
    The pivot reward keeps the item-level view: `_binary_match` inspects the first call and counts parallel calls
    separately (`n_calls`), and the silence rules read the raw output items.
    """
    result = None
    for output_item in response.output:
        if output_item.type == "function_call":
            return output_item
        if output_item.type == "message" and output_item.role == "assistant" and result is None:
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    result = content_item
                    break
    return result


class ConstrainedBinaryPivotRunRequest(BaseRunRequest):
    expected_action: ExpectedAction


class ConstrainedBinaryPivotVerifyRequest(ConstrainedBinaryPivotRunRequest, BaseVerifyRequest):
    pass


class ConstrainedBinaryPivotVerifyResponse(BaseVerifyResponse):
    expected_action: ExpectedAction
    matched: bool
    match_failure_reason: str
    constraint_verdict: ConstraintVerdict

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
    # the sampled (branch) turn carried no visible text — a bare tool call. Reported for every row that carries a
    # constraint (matched or not), so the narration rate is observable. A silent turn is UNGRADED (= not broken) for
    # every matcher; only a must_speak first turn FAILs on silence, via the verifier.
    constraint_silent: bool
    # Lin, 2026-09-15: the sampled turn was a bare tool call, a tool call was demonstrated, the row carries a
    # constraint that is fine with silence (no must_speak) and reward_mode is pivot -> the row leaves the GRPO group
    # (no reward, no baseline, no gradient). See the module docstring for why neither 0 nor 1 is right for it.
    excluded: bool
    # `reward` with excluded rows zeroed: the mean of THIS is the signal the policy is actually trained on.
    effective_reward: float
    # the mined reference turn had visible text (metadata.reference_spoken == "true"). Diagnostic only since
    # 2026-09-15; it used to drive the forfeit rule.
    reference_spoken: bool
    # NeMo-RL reads `instance_config.mask_sample` off every Gym result (nemo_rl/experience/rollouts.py) and zeroes
    # that sample's loss multiplier; with grpo.mask_sample_excludes_from_baseline (default on) it also leaves the
    # per-prompt baseline / std. The simple agent's verify response allows extra fields, so this passes through.
    instance_config: dict

    # --- match axis (task solving) ---
    # One-hot over _binary_match's failure reasons; exactly one is True when
    # matched is False, all False when matched is True.
    match_fail_kind: bool
    match_fail_invalid_output: bool
    match_fail_tool_name: bool
    match_fail_target: bool
    match_fail_similarity: bool
    # a message was demonstrated and the sampled turn has neither a tool call nor visible text (Lin, 2026-09-15):
    # an empty final is not a match. Before, `sampled is None` counted as "no tool call" and matched.
    match_fail_empty_message: bool
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

    def _binary_match(
        self, expected: dict, sampled: Optional[Any], n_calls: Optional[int] = None, sampled_has_text: bool = True
    ) -> tuple[bool, str, Optional[float]]:
        """Return (matched, failure_reason, argument_similarity_or_None).

        `n_calls` is the number of function_call items in the sampled output
        (None: unknown, only `sampled` is judged). The belt's kind rule
        (pivot_branch_probe._branch_match: `len(branch_calls) != 1` ->
        kind_mismatch) makes a parallel-call turn a miss even when its first
        call matches the teacher; `sampled` alone is the first call and cannot
        see the others. A turn with no call at all keeps the finer
        `model_output_invalid` bucket (the belt files it under kind_mismatch
        too; `matched` agrees either way).

        The similarity is reported whenever L2 was reached, pass or fail; it is
        None for rollouts rejected at L0/L1, where no similarity exists.
        """
        if expected.get("type") == "message":
            # pivot_branch_probe._branch_match message-kind rule: a message
            # was demonstrated, so the sampled turn must not call a tool.
            if sampled is not None and sampled.type == "function_call":
                return False, "kind_mismatch", None
            # ... and it must BE a message (Lin, 2026-09-15): no text and no
            # call is an empty final, not a match. Under a ban / maximum an
            # empty final is UNGRADED, so it used to score 1.0 (441 blend rows).
            if not sampled_has_text:
                return False, "empty_message", None
            return True, "none", None
        if sampled is None or sampled.type != "function_call":
            return False, "model_output_invalid", None
        if n_calls is not None and n_calls != 1:
            # exactly one call was demonstrated; parallel calls are a different action kind (never executed at
            # max_rollout_turns 1, so nothing else would ever penalise them)
            return False, "kind_mismatch", None
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
    def _row_params(md: dict) -> dict:
        """The recipe verifier_parameter carried in `constraint_params` (JSON str or dict); {} when unparseable."""
        raw = md.get("constraint_params") or "{}"
        try:
            return json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            logger.warning("unparseable constraint_params for %s; using {}", md.get("constraint"))
            return {}

    @staticmethod
    def _assistant_items(items: List[dict]) -> List[dict]:
        """The items the grader's segmenter reads: assistant messages, calls, results, reasoning (no system/user/developer)."""
        return [
            o for o in items
            if o.get("type") in ("message", "function_call", "function_call_output", "reasoning")
            and not (o.get("type") == "message" and o.get("role") in ("system", "user", "developer"))
        ]

    @classmethod
    def _branch_turns(cls, prefix_items: List[dict], sampled_items: List[dict]) -> set:
        """Turn indices (grader frame, 0-based, whole trajectory) that contain sampled items.

        The grader's segmenter closes a turn at each tool result; a prefix ending in an open turn (narration
        without a result) merges with the sampled call into one turn, which then IS a branch turn — the mining
        belt's rule (grade_branch_in_context reads the verdict of every turn a sampled item belongs to).

        Whether the prefix's last turn is open is asked of the segmenter itself, not read off the last item's type:
        segment() opens a turn only for visible text or a call, so a trailing assistant message with empty text
        (the shape _pivot_common emits for a content-less history message) leaves nothing pending and the sampled
        action starts a fresh turn — charging the previous CLOSED prefix turn to the sampled action would fail every
        branch of the row whatever the policy did.
        """
        asst_prefix = cls._assistant_items(prefix_items)
        pre = segment(asst_prefix)
        # the items after the prefix's last tool result form a pending turn iff the segmenter makes one of them
        last_close = max((i for i, o in enumerate(asst_prefix) if o.get("type") == "function_call_output"), default=-1)
        prefix_open = bool(pre) and bool(segment(asst_prefix[last_close + 1:]))
        start = len(pre) - 1 if prefix_open else len(pre)
        n_all = len(segment(asst_prefix + list(sampled_items)))
        return set(range(max(start, 0), n_all))

    @staticmethod
    def _sampled_has_text(sampled_items: List[dict]) -> bool:
        """True when the sampled output carries visible assistant text (reasoning items do not count)."""
        for o in sampled_items:
            if o.get("type") != "message" or o.get("role") in ("system", "user", "developer"):
                continue
            content = o.get("content") or []
            if isinstance(content, str):
                if content.strip():
                    return True
                continue
            for c in content:
                if isinstance(c, dict) and c.get("type") == "output_text" and str(c.get("text") or "").strip():
                    return True
        return False

    @staticmethod
    def _resolver(md: dict) -> dict:
        """The grader's tool-name resolver: registry defaults updated with the row's binding (`tool_name_overrides`)."""
        resolver = dict(tv.DEFAULT_RESOLVER)
        raw = md.get("tool_name_overrides")
        try:
            binding = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (ValueError, TypeError):
            binding = {}
        if isinstance(binding, dict):
            resolver.update(binding)
        return resolver

    @classmethod
    def _silent_branch(cls, items: List[dict], branch: set, params: dict, md: dict) -> tuple[bool, bool]:
        """(silent, in_scope) for the branch turn(s).

        silent: some branch turn has no visible text (a bare tool call). in_scope: the constraint's trigger selects a
        branch turn — asked of the verifier's own trigger registry on the same turns the grader built, so "in scope"
        means exactly what it would have meant had the model spoken. An unknown trigger is not in scope.
        """
        turns = to_tv_turns(segment(cls._assistant_items(items)))
        branch_turns = [t for t in turns if t.index in branch]
        silent = any(not t.visible_text.strip() for t in branch_turns)
        try:
            selected = {t.index for t, _ in tv.select_turns(turns, params.get("trigger") or {}, cls._resolver(md))}
        except (ValueError, KeyError, TypeError):
            selected = set()
        return silent, bool(selected & branch)

    @classmethod
    def _grade_constraint(cls, prefix_items: List[Any], sampled_items: List[Any], md: dict) -> tuple[ConstraintVerdict, bool]:
        """(verdict, silent) of the row's constraint over prefix + sampled action, read at the branch turns.

        The row is handed to grade_row as a one-constraint `sdg_item` in the whole-trajectory frame, so the grader
        sees exactly what the mining belt's P4 saw. A grading error (per-constraint `error`, or the grader's
        whole-row `<grading_error>` pseudo-record) and a trigger that never fires at a branch turn are UNGRADED.

        Silent branch turn (Lin, 2026-09-11): the grader emits no step for a turn without visible text, so the verdict
        is derived here from the matcher's no-answer kind — a rule that needs an answer (`fail`) is BROKEN by a bare
        tool call in its scope; a maximum or a ban (`ungradable`) is not (UNGRADED). Out of scope stays UNGRADED.
        """
        cid = str(md.get("constraint"))
        params = cls._row_params(md)
        sdg_item = {
            "type": "fresh",  # whole-trajectory frame, no prefix skipping: identical to the mining belt's P4 surface
            "persona": "opencode",
            "constraints": [{"id": cid, "verifier_parameter": params, "reference_instruction": ""}],
        }
        tmd = {"sdg_item": json.dumps(sdg_item)}
        if md.get("tool_name_overrides"):
            tno = md["tool_name_overrides"]
            tmd["tool_name_overrides"] = tno if isinstance(tno, str) else json.dumps(tno)
        items = list(prefix_items) + list(sampled_items)
        records = grade_row(tmd, None, items) or []
        rec = next((r for r in records if r.get("id") == cid), None)
        if rec is None or rec.get("error"):
            # rec is None when the grader failed before grading the constraint: its whole-row pseudo-record carries the message
            err = next((r.get("error") for r in records if r.get("id") in (cid, GRADING_ERROR_ID) and r.get("error")), None)
            logger.warning("constraint grading error for %s: %s", cid, err)
            return ConstraintVerdict.UNGRADED, False
        branch = cls._branch_turns(prefix_items, sampled_items)
        silent, in_scope = cls._silent_branch(items, branch, params, md)
        at_branch = [st for st in rec.get("steps") or [] if st.get("turn") in branch]
        if at_branch:
            verdict = ConstraintVerdict.PASS if all(int(st.get("reward", 0)) >= 1 for st in at_branch) else ConstraintVerdict.FAIL
            return verdict, silent
        # Silent in-scope branch turn (Lin, 2026-09-11, design/silent_turn_semantics.md): every middle-turn sentence in the
        # pool is conditional on a message existing, so a bare tool call breaks nothing -> UNGRADED (= not broken) for
        # EVERY matcher, language included. The one exception, a first-turn sentence that demanded a message before any
        # tool, is declared on the trigger (must_speak) and the verifier itself emits the FAIL step, so it arrives in
        # at_branch above. `in_scope` is kept for the diagnostic only.
        return ConstraintVerdict.UNGRADED, silent

    async def verify(self, body: ConstrainedBinaryPivotVerifyRequest) -> ConstrainedBinaryPivotVerifyResponse:
        def _dump(item: Any) -> Any:
            return item.model_dump() if hasattr(item, "model_dump") else item

        expected = _dump(body.expected_action)
        sampled = extract_tool_call_or_text(body.response)
        prefix_items = [_dump(i) for i in (body.responses_create_params.input or [])]
        sampled_items = [_dump(i) for i in (body.response.output or [])]
        n_calls = sum(1 for o in sampled_items if o.get("type") == "function_call")
        sampled_is_call = sampled is not None and getattr(sampled, "type", None) == "function_call"
        matched, why, similarity = self._binary_match(
            expected, sampled, n_calls=n_calls, sampled_has_text=self._sampled_has_text(sampled_items)
        )

        metadata = getattr(body.responses_create_params, "metadata", None)
        if metadata is not None and hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
        md = metadata or {}
        has_constraint = bool(md.get("constraint"))
        params = self._row_params(md) if has_constraint else {}
        must_speak = bool((params.get("trigger") or {}).get("must_speak"))
        reference_spoken = str(md.get("reference_spoken", "")).lower() == "true"

        # Silence is decided for EVERY row with a constraint, matched or not: the exclusion below needs it and the
        # narration rate should not be conditional on the match. Same turn frame as the grader (branch turns; an
        # open prefix turn merges with the sampled call, and its narration counts as speaking).
        silent = False
        if has_constraint:
            branch = self._branch_turns(prefix_items, sampled_items)
            silent, _ = self._silent_branch(prefix_items + sampled_items, branch, params, md)

        verdict = ConstraintVerdict.NO_CONSTRAINT_IN_ROW
        # Constraint grading only decides matched cases, so by default we skip
        # the cost otherwise; always_grade_constraint trades that cost for the
        # unconditional pass rate. Either way `graded` records whether the
        # verifier actually ran, which is what the diagnostics key off.
        verifier_mode = self.config.reward_mode == "verifier"
        graded = matched or self.config.always_grade_constraint or verifier_mode
        if graded and has_constraint:
            verdict, _ = self._grade_constraint(prefix_items, sampled_items, md)

        # Binary (Lin, 2026-09-11): 0 if the pivot tool check fails; 1 if it
        # passes and the sampled action did not BREAK the constraint. FAIL is
        # the only violation (an explicit failed step, or a silent must_speak
        # first turn); UNGRADED (silent under any other rule, trigger not
        # fired, grading error) and rows without a constraint are "not
        # broken". Unchanged by the diagnostics and by always_grade_constraint:
        # a match failure is 0.0 whatever verdict the extra grading pass returns.
        constraint_ok = verdict is not ConstraintVerdict.FAIL
        # Exclusion (Lin, 2026-09-15; module docstring): a bare tool call where
        # a tool call was demonstrated, under a constraint that is fine with
        # silence, leaves the GRPO group. Whether the call matched does not
        # matter -- keeping the mismatched silent calls at 0 while dropping the
        # matched ones would still be a one-sided push against silence.
        excluded = (
            not verifier_mode
            and has_constraint
            and expected.get("type") != "message"
            and sampled_is_call
            and silent
            and not must_speak
        )
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
            excluded=excluded,
            effective_reward=0.0 if excluded else reward,
            reference_spoken=reference_spoken,
            instance_config={"mask_sample": excluded},
            matched=matched,
            match_failure_reason=why,
            constraint_verdict=verdict,
            constraint_passed=graded and verdict is ConstraintVerdict.PASS,
            constraint_failed=graded and verdict is ConstraintVerdict.FAIL,
            constraint_ungraded=graded and verdict is ConstraintVerdict.UNGRADED,
            constraint_absent=graded and verdict is ConstraintVerdict.NO_CONSTRAINT_IN_ROW,
            constraint_graded=graded and verdict in (ConstraintVerdict.PASS, ConstraintVerdict.FAIL),
            constraint_silent=has_constraint and silent,
            match_fail_kind=why == "kind_mismatch",
            match_fail_invalid_output=why == "model_output_invalid",
            match_fail_tool_name=why == "tool_name_mismatch",
            match_fail_target=why == "target_mismatch",
            match_fail_similarity=why == "similarity_below_threshold",
            match_fail_empty_message=why == "empty_message",
            argument_similarity=similarity,
            similarity_evaluated=similarity is not None,
        )


if __name__ == "__main__":
    ConstrainedBinaryPivotResourcesServer.run_webserver()
