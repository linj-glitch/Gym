# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics on the constrained pivot server.

These cover the scalar fields that decompose the binary reward into its match
and constraint axes. The reward itself is asserted alongside every case so the
diagnostics can never drift from it: NeMo-RL trains on `reward`, and the whole
point of the extra fields is that they explain that number rather than restate
it.
"""

import json
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedFunctionCall,
    ExpectedMessage,
)
from resources_servers.single_step_tool_use_with_argument_comparison.constrained_app import (
    ConstrainedBinaryPivotResourcesServer,
    ConstrainedBinaryPivotResourcesServerConfig,
    ConstrainedBinaryPivotVerifyRequest,
    ConstraintVerdict,
)


EXPECTED_ARGS = json.dumps({"command": "str_replace", "path": "/repo/a.py"})


def _server(**overrides) -> ConstrainedBinaryPivotResourcesServer:
    config = ConstrainedBinaryPivotResourcesServerConfig(
        host="127.0.0.1",
        port=20003,
        entrypoint="",
        name="constrained_pivot_server",
        **overrides,
    )
    return ConstrainedBinaryPivotResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _params(metadata: dict | None = None) -> NeMoGymResponseCreateParamsNonStreaming:
    kwargs = {"input": [NeMoGymEasyInputMessage(role="user", content="Edit a.py.")]}
    if metadata is not None:
        kwargs["metadata"] = metadata
    return NeMoGymResponseCreateParamsNonStreaming(**kwargs)


def _response(*output_items) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=1001,
        model="test_model",
        object="response",
        output=list(output_items),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _call(name: str = "str_replace_editor", arguments: str = EXPECTED_ARGS):
    return NeMoGymResponseFunctionToolCall(type="function_call", call_id="c1", name=name, arguments=arguments)


def _message(text: str = "done"):
    return NeMoGymResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class TestConstrainedDiagnostics:
    @fixture
    def server(self) -> ConstrainedBinaryPivotResourcesServer:
        return _server()

    async def _verify(self, server, expected, response, metadata=None):
        return await server.verify(
            ConstrainedBinaryPivotVerifyRequest(
                responses_create_params=_params(metadata),
                response=response,
                expected_action=expected,
            )
        )

    async def test_match_failure_reasons_are_one_hot(self, server) -> None:
        """Exactly one match_fail_* is set on a miss, none on a hit."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        flags = (
            "match_fail_kind",
            "match_fail_invalid_output",
            "match_fail_tool_name",
            "match_fail_target",
            "match_fail_similarity",
        )

        hit = await self._verify(server, expected, _response(_call()))
        assert hit.matched
        assert sum(getattr(hit, f) for f in flags) == 0

        # A text answer where a tool call was demonstrated: no action found.
        miss = await self._verify(server, expected, _response(_message()))
        assert not miss.matched
        assert sum(getattr(miss, f) for f in flags) == 1
        assert miss.match_fail_invalid_output
        assert miss.reward == approx(0.0)

        # Wrong tool entirely: rejected at L0, before similarity exists.
        wrong = await self._verify(server, expected, _response(_call(name="execute_bash", arguments="{}")))
        assert not wrong.matched
        assert sum(getattr(wrong, f) for f in flags) == 1
        assert wrong.argument_similarity is None
        assert not wrong.similarity_evaluated

        # Message expected, tool call produced: the kind_mismatch branch.
        kind = await self._verify(server, ExpectedMessage(type="message", content="done"), _response(_call()))
        assert not kind.matched
        assert kind.match_fail_kind
        assert sum(getattr(kind, f) for f in flags) == 1

    async def test_similarity_reported_when_l2_reached(self, server) -> None:
        """argument_similarity survives even when it lands under threshold."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        divergent = json.dumps({"command": "str_replace", "path": "/repo/a.py", "new_str": "totally different"})
        out = await self._verify(server, expected, _response(_call(arguments=divergent)))
        if out.similarity_evaluated:
            assert out.argument_similarity is not None
            assert 0.0 <= out.argument_similarity <= 1.0
            # matched iff the score cleared the configured threshold
            assert out.matched == (out.argument_similarity >= server.config.similarity_full_credit)

    async def test_constraint_bools_track_the_verdict(self, server) -> None:
        """The four verdict bools are one-hot over a graded rollout."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server,
            expected,
            _response(_call()),
            metadata={"constraint": "single_tool_call_per_message", "constraint_params": "{}"},
        )
        assert out.matched
        bools = (
            out.constraint_passed,
            out.constraint_failed,
            out.constraint_ungraded,
            out.constraint_absent,
        )
        assert sum(bools) == 1
        assert out.constraint_graded == (out.constraint_passed or out.constraint_failed)
        # reward is the conjunction, and PASS / absent both count as ok
        expected_reward = (
            1.0
            if out.constraint_verdict
            in (
                ConstraintVerdict.PASS,
                ConstraintVerdict.NO_CONSTRAINT_IN_ROW,
            )
            else 0.0
        )
        assert out.reward == approx(expected_reward)

    async def test_rows_without_a_constraint_are_absent_not_ungraded(self, server) -> None:
        """A row carrying no declaration is `absent`; reward degrades to match-only."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(server, expected, _response(_call()), metadata={})
        assert out.matched
        assert out.constraint_absent
        assert not out.constraint_graded
        assert out.constraint_verdict is ConstraintVerdict.NO_CONSTRAINT_IN_ROW
        assert out.reward == approx(1.0)

    async def test_ungraded_rollouts_report_no_constraint_bools(self, server) -> None:
        """Match failures skip grading, so every constraint bool is False.

        This is what keeps `constraint_absent` meaning "the row had no
        declaration" instead of doubling as the not-graded sentinel.
        """
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server,
            expected,
            _response(_message()),
            metadata={"constraint": "single_tool_call_per_message", "constraint_params": "{}"},
        )
        assert not out.matched
        assert not any(
            (
                out.constraint_passed,
                out.constraint_failed,
                out.constraint_ungraded,
                out.constraint_absent,
                out.constraint_graded,
            )
        )
        assert out.reward == approx(0.0)

    async def test_always_grade_constraint_does_not_move_reward(self) -> None:
        """The flag buys the unconditional pass rate and nothing else."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        metadata = {"constraint": "single_tool_call_per_message", "constraint_params": "{}"}
        response = _response(_message())

        off = await self._verify(_server(), expected, response, metadata=metadata)
        on = await self._verify(_server(always_grade_constraint=True), expected, response, metadata=metadata)

        assert off.reward == approx(on.reward) == approx(0.0)
        assert off.matched == on.matched is False
        # off skips grading entirely; on records a real verdict
        assert not off.constraint_graded
        assert on.constraint_graded or on.constraint_ungraded or on.constraint_absent


class TestVerifierRewardMode:
    """reward_mode=verifier: the constraint verdict on the model's own action is
    the whole reward; the teacher action is only a diagnostic."""

    # single_tool_call_per_message: at most one tool call per message AND at
    # least one line of narration prose with it -- a bare tool call FAILS.
    META = {"constraint": "single_tool_call_per_message", "constraint_params": "{}"}
    NARRATION = "I will list the repository files first to orient myself before editing."

    @fixture
    def server(self) -> ConstrainedBinaryPivotResourcesServer:
        return _server(reward_mode="verifier")

    async def _verify(self, server, expected, response, metadata=None):
        return await server.verify(
            ConstrainedBinaryPivotVerifyRequest(
                responses_create_params=_params(metadata),
                response=response,
                expected_action=expected,
            )
        )

    async def test_default_mode_is_pivot(self) -> None:
        assert _server().config.reward_mode == "pivot"

    async def test_pass_earns_reward_even_when_teacher_mismatched(self, server) -> None:
        """A compliant action that differs from the teacher's still scores 1.0."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server,
            expected,
            _response(
                _message(self.NARRATION),
                _call(name="execute_bash", arguments=json.dumps({"command": "ls"})),
            ),
            metadata=self.META,
        )
        assert not out.matched
        assert out.match_fail_tool_name
        assert out.constraint_graded
        assert out.constraint_verdict is ConstraintVerdict.PASS
        assert out.reward == approx(1.0)

    async def test_bare_tool_call_fails_this_constraint(self, server) -> None:
        """The same call without narration violates the constraint -> 0.0."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server,
            expected,
            _response(_call(name="execute_bash", arguments=json.dumps({"command": "ls"}))),
            metadata=self.META,
        )
        assert out.constraint_verdict is ConstraintVerdict.FAIL
        assert out.reward == approx(0.0)

    async def test_fail_earns_nothing_even_when_teacher_matched(self, server) -> None:
        """Two tool calls in one message violate single_tool_call_per_message."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(server, expected, _response(_call(), _call()), metadata=self.META)
        assert out.matched
        assert out.constraint_graded
        assert out.constraint_verdict is ConstraintVerdict.FAIL
        assert out.reward == approx(0.0)

    async def test_row_without_constraint_earns_nothing(self, server) -> None:
        """No declaration -> nothing to verify -> 0.0 (pivot mode would give 1.0)."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(server, expected, _response(_call()), metadata={})
        assert out.matched
        assert out.constraint_absent
        assert out.reward == approx(0.0)

    async def test_unparseable_output_earns_nothing(self, server) -> None:
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(server, expected, _response(), metadata=self.META)
        assert not out.matched
        assert out.reward == approx(0.0)

    async def test_pivot_mode_unchanged_on_the_same_inputs(self) -> None:
        """The pivot predicate still requires the match: same inputs, opposite rewards."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        response = _response(
            _message(self.NARRATION),
            _call(name="execute_bash", arguments=json.dumps({"command": "ls"})),
        )
        pivot = await self._verify(_server(), expected, response, metadata=self.META)
        verifier = await self._verify(_server(reward_mode="verifier"), expected, response, metadata=self.META)
        assert pivot.reward == approx(0.0)
        assert verifier.reward == approx(1.0)


class TestTemplateFamily:
    """Template-family constraints (recipe verifier_parameter in `constraint_params`) grade through the
    swe_if_agents grader on the same prefix+sampled surface, and decide the reward exactly like detailed ones."""

    ANY_TURN_BAN = json.dumps({"template": "turn_output", "trigger": {"position": "any_turn"},
                               "obligation": {"match": "forbidden", "value": "FROBNICATE"}})
    FINAL_PREFIX = json.dumps({"template": "turn_output", "trigger": {"position": "final"},
                               "obligation": {"match": "prefix", "value": "SUMMARY:"}})

    @staticmethod
    def _prefix_params(metadata, *, open_turn=False):
        """A replayed prefix: system, user, one assistant narration + tool call + tool result (closed turn);
        with open_turn=True the prefix ends in a narration that has not been answered yet."""
        from nemo_gym.openai_utils import NeMoGymFunctionCallOutput

        items = [
            NeMoGymEasyInputMessage(role="system", content="neutral"),
            NeMoGymEasyInputMessage(role="user", content="Edit a.py."),
            NeMoGymResponseOutputMessage(id="p1", type="message", role="assistant", status="completed",
                                         content=[NeMoGymResponseOutputText(type="output_text", text="Looking at a.py.", annotations=[])]),
            NeMoGymResponseFunctionToolCall(type="function_call", call_id="pc1", name="execute_bash", arguments=json.dumps({"command": "cat a.py"})),
            NeMoGymFunctionCallOutput(type="function_call_output", call_id="pc1", output="print(1)"),
        ]
        if open_turn:
            items.append(NeMoGymResponseOutputMessage(id="p2", type="message", role="assistant", status="completed",
                                                      content=[NeMoGymResponseOutputText(type="output_text", text="I will FROBNICATE it now.", annotations=[])]))
        return NeMoGymResponseCreateParamsNonStreaming(input=items, metadata=metadata)

    async def _verify(self, server, expected, response, params):
        return await server.verify(
            ConstrainedBinaryPivotVerifyRequest(responses_create_params=params, response=response, expected_action=expected)
        )

    def test_family_is_stamped_or_inferred(self):
        srv = _server()
        assert srv._row_family({"family": "template", "constraint": "x#c1", "constraint_params": "{}"}) == "template"
        assert srv._row_family({"constraint": "x#c1", "constraint_params": self.ANY_TURN_BAN}) == "template"
        assert srv._row_family({"constraint": "single_tool_call_per_message", "constraint_params": "{}"}) == "detailed"
        assert srv._row_family({"constraint": "step_summary_prefix", "constraint_params": json.dumps({"prefix": "Observed:"})}) == "detailed"

    async def test_pivot_reward_requires_match_and_template_pass(self):
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        md = {"family": "template", "constraint": "t#c1", "constraint_params": self.ANY_TURN_BAN}
        clean = await self._verify(_server(), expected, _response(_message("Applying the fix."), _call()), self._prefix_params(md))
        assert clean.reward == approx(1.0) and clean.matched and clean.constraint_verdict is ConstraintVerdict.PASS
        assert clean.constraint_family == "template" and clean.constraint_passed and clean.constraint_graded
        dirty = await self._verify(_server(), expected, _response(_message("Let me FROBNICATE this."), _call()), self._prefix_params(md))
        assert dirty.reward == approx(0.0) and dirty.matched and dirty.constraint_verdict is ConstraintVerdict.FAIL
        assert dirty.constraint_failed and not dirty.constraint_passed

    async def test_prefix_violation_does_not_count_only_the_sampled_turn_does(self):
        """The prefix narration says FROBNICATE (closed turn); the sampled turn is clean -> PASS."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        md = {"family": "template", "constraint": "t#c1", "constraint_params": self.ANY_TURN_BAN}
        params = self._prefix_params(md)
        params.input[2] = NeMoGymResponseOutputMessage(id="p1", type="message", role="assistant", status="completed",
                                                       content=[NeMoGymResponseOutputText(type="output_text", text="I will FROBNICATE a.py.", annotations=[])])
        out = await self._verify(_server(), expected, _response(_message("Applying the fix."), _call()), params)
        assert out.reward == approx(1.0) and out.constraint_verdict is ConstraintVerdict.PASS

    async def test_open_prefix_turn_merges_with_the_sampled_call(self):
        """A prefix ending in an unanswered narration merges with the sampled call into ONE turn, which is a branch
        turn (same rule as the detailed path): the merged turn carries the ban word -> FAIL."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        md = {"family": "template", "constraint": "t#c1", "constraint_params": self.ANY_TURN_BAN}
        out = await self._verify(_server(), expected, _response(_call()), self._prefix_params(md, open_turn=True))
        assert out.matched and out.constraint_verdict is ConstraintVerdict.FAIL and out.reward == approx(0.0)

    async def test_final_constraint_on_message_kind_expected_action(self):
        expected = ExpectedMessage(type="message", content="SUMMARY: done")
        md = {"family": "template", "constraint": "t#c2", "constraint_params": self.FINAL_PREFIX}
        good = await self._verify(_server(), expected, _response(_message("SUMMARY: fixed a.py")), self._prefix_params(md))
        assert good.reward == approx(1.0) and good.constraint_verdict is ConstraintVerdict.PASS
        bad = await self._verify(_server(), expected, _response(_message("Fixed a.py, all good.")), self._prefix_params(md))
        assert bad.reward == approx(0.0) and bad.matched and bad.constraint_verdict is ConstraintVerdict.FAIL
        # a tool call where a message was demonstrated: kind mismatch. The recipe's no-answer ruling for a required
        # shape (`no_answer: fail`): an episode with no final message FAILS its final-message rule once.
        call = await self._verify(_server(always_grade_constraint=True), expected, _response(_call()), self._prefix_params(md))
        assert not call.matched and call.constraint_verdict is ConstraintVerdict.FAIL and call.reward == approx(0.0)

    async def test_verifier_mode_ignores_the_teacher_action(self):
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        md = {"family": "template", "constraint": "t#c1", "constraint_params": self.ANY_TURN_BAN}
        srv = _server(reward_mode="verifier")
        wrong_tool_clean = await self._verify(srv, expected, _response(_message("Checking."), _call(name="execute_bash", arguments="{}")), self._prefix_params(md))
        assert not wrong_tool_clean.matched and wrong_tool_clean.reward == approx(1.0)
        right_tool_dirty = await self._verify(srv, expected, _response(_message("FROBNICATE!"), _call()), self._prefix_params(md))
        assert right_tool_dirty.matched and right_tool_dirty.reward == approx(0.0)

    async def test_unknown_matcher_is_ungraded_not_a_crash(self):
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        bad_vp = json.dumps({"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": "no_such_matcher", "value": 1}})
        md = {"family": "template", "constraint": "t#c9", "constraint_params": bad_vp}
        out = await self._verify(_server(), expected, _response(_call()), self._prefix_params(md))
        assert out.matched and out.constraint_verdict is ConstraintVerdict.UNGRADED and out.reward == approx(0.0)
