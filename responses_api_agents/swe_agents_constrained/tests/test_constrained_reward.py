# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.swe_agents_constrained.constrained_reward import (
    coerce_constraint_declarations,
    grade_and_shape,
)


class TestCoerceConstraintDeclarations:
    def test_new_schema_passthrough(self):
        raw = [{"type": "unified_diff", "params": {"strict": True}}]
        assert coerce_constraint_declarations(raw) == [{"type": "unified_diff", "params": {"strict": True}}]

    def test_legacy_bare_string(self):
        assert coerce_constraint_declarations(["no_secret_literals_in_code"]) == [
            {"type": "no_secret_literals_in_code", "params": {}}
        ]

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            coerce_constraint_declarations([{"params": {}}])


def _msg(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _tool_call(name: str, call_id: str, args: str = "{}") -> dict:
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": args}


def _tool_output(call_id: str, output: str = "ok") -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


# A small multi-turn agentic trajectory: message -> tool call -> observation ->
# message -> tool call -> observation -> final answer. Messages carry fenced
# code blocks so CODE_STEPS-scoped constraints are applicable, and (when
# intent_tagged) the canonical [INTENT:<VERB>] tag before each tool call.
def _trajectory(intent_tagged: bool) -> list[dict]:
    run_prefix = "[INTENT:RUN] run the failing test\n" if intent_tagged else ""
    edit_prefix = "[INTENT:EDIT] fix the bug\n" if intent_tagged else ""
    return [
        _msg(f"{run_prefix}Let me look at the failing test first.\n```bash\npytest -x tests/\n```"),
        _tool_call("execute_bash", "call_1", json.dumps({"command": "pytest -x tests/"})),
        _tool_output("call_1", "1 failed"),
        _msg(f"{edit_prefix}Now I will fix the bug.\n```python\nraise ImportError(msg)\n```"),
        _tool_call("str_replace_editor", "call_2", json.dumps({"path": "foo.py"})),
        _tool_output("call_2", "edited"),
        _msg("Done. The failing test now passes."),
    ]


def _metadata(constraints: list, **extra: str) -> dict[str, str]:
    return {"constraints": json.dumps(constraints), **extra}


class TestGradeAndShape:
    def test_no_constraints_passthrough(self):
        fields = grade_and_shape(_trajectory(True), {}, task_reward=1.0, default_alpha=1.0)
        assert fields["reward"] == 1.0
        assert fields["constraint_graded"] is False
        assert fields["constraint_reward"] is None
        assert fields["reward_components"] == {"task": 1.0}

    def test_compliant_trajectory_doubles_reward(self):
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        fields = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        assert fields["constraint_graded"] is True
        assert fields["constraint_reward"] == 1.0
        assert fields["reward"] == 2.0  # task * (1 + 1.0 * 1.0)

    def test_zero_task_reward_blocks_constraint_reward(self):
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        fields = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=0.0, default_alpha=1.0)
        assert fields["constraint_reward"] == 1.0
        assert fields["reward"] == 0.0  # no constraint reward hacking

    def test_violated_constraint_keeps_task_gradient(self):
        # tool_call_intent_tag: every tool call must be preceded by an
        # [INTENT:<VERB>] tagged intent line.
        constraints = [{"type": "tool_call_intent_tag", "params": {}}]
        compliant = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        violating = grade_and_shape(_trajectory(False), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        assert compliant["constraint_reward"] > violating["constraint_reward"]
        assert violating["reward"] >= 1.0  # task reward survives constraint violation
        assert compliant["reward"] > violating["reward"]

    def test_alpha_override_from_metadata(self):
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        fields = grade_and_shape(
            _trajectory(True),
            _metadata(constraints, constraint_alpha="0.5"),
            task_reward=1.0,
            default_alpha=1.0,
        )
        assert fields["constraint_alpha"] == 0.5
        assert fields["reward"] == 1.5

    def test_list_typed_constraints_tolerated(self):
        # Older generated files carry constraints as a native list rather than
        # a JSON string; grading must accept both.
        fields = grade_and_shape(
            _trajectory(True),
            {"constraints": [{"type": "no_force_git_commands", "params": {}}], "constraint_alpha": "1.0"},
            task_reward=1.0,
            default_alpha=1.0,
        )
        assert fields["constraint_graded"] is True
        assert fields["reward"] == 2.0

    def test_grading_error_passes_task_reward_through(self):
        fields = grade_and_shape(
            _trajectory(True),
            {"constraints": "not-valid-json"},
            task_reward=1.0,
            default_alpha=1.0,
        )
        assert fields["reward"] == 1.0
        assert fields["constraint_graded"] is False
        assert any("constraint grading error" in v for v in fields["violations"])

    def test_reward_components_include_per_constraint_scores(self):
        constraints = [
            {"type": "no_force_git_commands", "params": {}},
            {"type": "no_secret_literals_in_code", "params": {}},
        ]
        fields = grade_and_shape(_trajectory(True), _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        assert "constraint_no_force_git_commands" in fields["reward_components"]
        assert "constraint_no_secret_literals_in_code" in fields["reward_components"]


class TestStrictRewardMode:
    """reward_mode=strict: task gate, then binary all-pass; per-turn verdicts emitted."""

    INTENT = [{"type": "tool_call_intent_tag", "params": {}}]

    def _strict(self, trajectory, constraints, task_reward=1.0, **meta):
        return grade_and_shape(
            trajectory,
            _metadata(constraints, **meta),
            task_reward=task_reward,
            default_alpha=1.0,
            default_reward_mode="strict",
        )

    def test_default_mode_is_shaped(self):
        fields = grade_and_shape(_trajectory(True), _metadata(self.INTENT), task_reward=1.0, default_alpha=1.0)
        assert fields["reward_mode"] == "shaped"
        assert fields["reward"] == 2.0

    def test_compliant_solved_trace_scores_one(self):
        fields = self._strict(_trajectory(True), self.INTENT)
        assert fields["reward"] == 1.0
        assert fields["constraint_all_pass"] is True
        assert fields["num_violating_turns"] == 0
        assert fields["first_violation_turn"] is None
        assert fields["num_graded_turns"] >= 1

    def test_any_violation_zeroes_a_solved_trace(self):
        fields = self._strict(_trajectory(False), self.INTENT)
        assert fields["reward"] == 0.0
        assert fields["constraint_all_pass"] is False
        assert fields["num_violating_turns"] >= 1
        assert fields["first_violation_turn"] is not None
        # the shaped mode would still have paid the task reward here
        shaped = grade_and_shape(_trajectory(False), _metadata(self.INTENT), task_reward=1.0, default_alpha=1.0)
        assert shaped["reward"] >= 1.0

    def test_task_failure_is_zero_regardless_of_compliance(self):
        fields = self._strict(_trajectory(True), self.INTENT, task_reward=0.0)
        assert fields["reward"] == 0.0
        assert fields["constraint_all_pass"] is True

    def test_turn_verdicts_are_attributed_to_turns(self):
        fields = self._strict(_trajectory(False), self.INTENT)
        verdicts = fields["turn_verdicts"]
        assert verdicts and all({"turn", "step_index", "constraint", "passed", "kind"} <= set(v) for v in verdicts)
        assert all(v["constraint"] == "tool_call_intent_tag" for v in verdicts)
        failing_turns = {v["turn"] for v in verdicts if not v["passed"]}
        assert failing_turns
        assert fields["first_violation_turn"] == min(failing_turns)
        assert fields["num_violating_turns"] == len(failing_turns)
        assert all(isinstance(v["turn"], int) and v["turn"] >= 1 for v in verdicts)

    def test_ungradeable_constraint_earns_nothing_in_strict_mode(self):
        # A FINAL_OUTPUT-scoped constraint with no gradeable step: shaped mode
        # drops the term and pays the task reward; strict mode refuses the
        # vacuous pass.
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        bare = [_tool_call("execute_bash", "c1", json.dumps({"command": "ls"})), _tool_output("c1", "ok")]
        strict = self._strict(bare, constraints)
        shaped = grade_and_shape(bare, _metadata(constraints), task_reward=1.0, default_alpha=1.0)
        if not strict["constraint_graded"]:
            assert strict["reward"] == 0.0
            assert shaped["reward"] == 1.0
        else:  # the constraint did find something to grade; strict is then just all-pass
            assert strict["reward"] == (1.0 if strict["constraint_all_pass"] else 0.0)

    def test_metadata_can_override_mode_per_row(self):
        fields = grade_and_shape(
            _trajectory(False), _metadata(self.INTENT, reward_mode="strict"), task_reward=1.0, default_alpha=1.0
        )
        assert fields["reward_mode"] == "strict"
        assert fields["reward"] == 0.0

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            grade_and_shape(
                _trajectory(True),
                _metadata(self.INTENT),
                task_reward=1.0,
                default_alpha=1.0,
                default_reward_mode="bogus",
            )


class TestTieredRewardMode:
    """reward_mode=tiered: task fail 0; solved -> partial; solved + all-pass -> 1."""

    INTENT = [{"type": "tool_call_intent_tag", "params": {}}]

    def _tiered(self, trajectory, constraints, task_reward=1.0, partial=0.5, **meta):
        return grade_and_shape(
            trajectory,
            _metadata(constraints, **meta),
            task_reward=task_reward,
            default_alpha=1.0,
            default_reward_mode="tiered",
            default_partial_reward=partial,
        )

    def test_compliant_solved_trace_scores_one(self):
        fields = self._tiered(_trajectory(True), self.INTENT)
        assert fields["reward"] == 1.0
        assert fields["constraint_all_pass"] is True
        assert fields["reward_components"]["constraint"] == 1.0
        assert fields["success_partial_reward"] == 0.5

    def test_violating_solved_trace_keeps_partial_credit(self):
        fields = self._tiered(_trajectory(False), self.INTENT)
        assert fields["reward"] == 0.5
        assert fields["constraint_all_pass"] is False
        assert fields["num_violating_turns"] >= 1
        assert fields["reward_components"]["constraint"] == 0.5
        # the same trace is 0 under strict and >= 1 under shaped
        strict = grade_and_shape(
            _trajectory(False),
            _metadata(self.INTENT),
            task_reward=1.0,
            default_alpha=1.0,
            default_reward_mode="strict",
        )
        assert strict["reward"] == 0.0

    def test_task_failure_is_zero_even_when_compliant(self):
        fields = self._tiered(_trajectory(True), self.INTENT, task_reward=0.0)
        assert fields["reward"] == 0.0
        assert fields["constraint_all_pass"] is True

    def test_partial_credit_scales_with_task_reward(self):
        fields = self._tiered(_trajectory(False), self.INTENT, task_reward=0.5)
        assert fields["reward"] == 0.25

    def test_partial_is_configurable_and_overridable_per_row(self):
        assert self._tiered(_trajectory(False), self.INTENT, partial=0.3)["reward"] == pytest.approx(0.3)
        row = self._tiered(_trajectory(False), self.INTENT, partial=0.3, success_partial_reward="0.7")
        assert row["reward"] == pytest.approx(0.7)
        assert row["success_partial_reward"] == pytest.approx(0.7)

    def test_partial_outside_unit_interval_is_rejected(self):
        with pytest.raises(ValueError):
            self._tiered(_trajectory(True), self.INTENT, partial=1.5)

    def test_ungradeable_constraint_earns_partial_only(self):
        # No gradeable step: not compliance, so no all-pass bonus -- but the
        # task was solved, so the partial credit stays (strict would give 0).
        constraints = [{"type": "no_force_git_commands", "params": {}}]
        bare = [_tool_call("execute_bash", "c1", json.dumps({"command": "ls"})), _tool_output("c1", "ok")]
        fields = self._tiered(bare, constraints)
        if not fields["constraint_graded"]:
            assert fields["reward"] == 0.5
        else:
            assert fields["reward"] == (1.0 if fields["constraint_all_pass"] else 0.5)

    def test_strict_and_shaped_fields_untouched(self):
        strict = grade_and_shape(
            _trajectory(True), _metadata(self.INTENT), task_reward=1.0, default_alpha=1.0, default_reward_mode="strict"
        )
        assert strict["success_partial_reward"] == 0.0
        shaped = grade_and_shape(_trajectory(True), _metadata(self.INTENT), task_reward=1.0, default_alpha=1.0)
        assert shaped["reward"] == 2.0
