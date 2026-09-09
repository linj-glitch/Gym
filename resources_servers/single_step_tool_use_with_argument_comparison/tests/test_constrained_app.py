# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics on the constrained pivot server.

These cover the scalar fields that decompose the binary reward into its match
and constraint axes. The reward itself is asserted alongside every case so the
diagnostics can never drift from it: NeMo-RL trains on `reward`, and the whole
point of the extra fields is that they explain that number rather than restate
it.

Every constraint here is a recipe `verifier_parameter` carried in the row's
`constraint_params` and graded by the swe_if_agents template verifier. Most
cases use an any-turn ban on the word FROBNICATE: a sampled turn whose visible
text contains it FAILS, a clean narration PASSES, and a silent turn (a bare tool
call, no text) is not gradable for a `forbidden` matcher, so it is UNGRADED.
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

BAN_WORD = "FROBNICATE"
ANY_TURN_BAN = json.dumps({"template": "turn_output", "trigger": {"position": "any_turn"},
                           "obligation": {"match": "forbidden", "value": BAN_WORD}})
FINAL_PREFIX = json.dumps({"template": "turn_output", "trigger": {"position": "final"},
                           "obligation": {"match": "prefix", "value": "SUMMARY:"}})
# a row of the pivot belt: the sdg constraint id plus its recipe verifier_parameter
BAN_META = {"constraint": "t#c1", "constraint_params": ANY_TURN_BAN}


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


def _verdict_bools(out) -> tuple:
    return out.constraint_passed, out.constraint_failed, out.constraint_ungraded, out.constraint_absent


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
        """The four verdict bools are one-hot over a graded rollout, and the reward is the conjunction with the match."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)

        clean = await self._verify(server, expected, _response(_message("Applying the fix."), _call()), metadata=BAN_META)
        assert clean.matched
        assert sum(_verdict_bools(clean)) == 1 and clean.constraint_passed
        assert clean.constraint_graded == (clean.constraint_passed or clean.constraint_failed)
        assert clean.constraint_verdict is ConstraintVerdict.PASS
        assert clean.reward == approx(1.0)

        dirty = await self._verify(server, expected, _response(_message(f"Let me {BAN_WORD} this."), _call()), metadata=BAN_META)
        assert dirty.matched
        assert sum(_verdict_bools(dirty)) == 1 and dirty.constraint_failed and dirty.constraint_graded
        assert dirty.constraint_verdict is ConstraintVerdict.FAIL
        assert dirty.reward == approx(0.0)

        # a silent turn is not gradable for a `forbidden` matcher: UNGRADED, which is not PASS, so no reward
        silent = await self._verify(server, expected, _response(_call()), metadata=BAN_META)
        assert silent.matched
        assert sum(_verdict_bools(silent)) == 1 and silent.constraint_ungraded and not silent.constraint_graded
        assert silent.constraint_verdict is ConstraintVerdict.UNGRADED
        assert silent.reward == approx(0.0)

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
        out = await self._verify(server, expected, _response(_message()), metadata=BAN_META)
        assert not out.matched
        assert not any((*_verdict_bools(out), out.constraint_graded))
        assert out.reward == approx(0.0)

    async def test_always_grade_constraint_does_not_move_reward(self) -> None:
        """The flag buys the unconditional pass rate and nothing else."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        # a text answer where a tool call was demonstrated misses the match whether or not it honours the ban
        for text, verdict in (("done", ConstraintVerdict.PASS), (f"{BAN_WORD} done", ConstraintVerdict.FAIL)):
            response = _response(_message(text))
            off = await self._verify(_server(), expected, response, metadata=BAN_META)
            on = await self._verify(_server(always_grade_constraint=True), expected, response, metadata=BAN_META)

            assert off.reward == approx(on.reward) == approx(0.0)
            assert off.matched == on.matched is False
            # off skips grading entirely; on records the real verdict
            assert not off.constraint_graded
            assert on.constraint_graded and on.constraint_verdict is verdict

    async def test_parallel_calls_are_a_kind_mismatch(self, server) -> None:
        """The belt's gate (pivot_branch_probe._branch_match: len(tool_calls) != 1 -> kind_mismatch) rejects a turn
        that emits the teacher's call plus another; judging only the first call would let extra, never-executed calls
        ride along for free."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server, expected, _response(_message("Editing."), _call(), _call(name="execute_bash", arguments=json.dumps({"command": "ls"})))
        )
        assert not out.matched and out.match_fail_kind and out.reward == approx(0.0)
        flags = ("match_fail_kind", "match_fail_invalid_output", "match_fail_tool_name", "match_fail_target", "match_fail_similarity")
        assert sum(getattr(out, f) for f in flags) == 1
        assert out.argument_similarity is None and not out.similarity_evaluated
        # exactly one call, even preceded by narration, still matches
        one = await self._verify(server, expected, _response(_message("Editing."), _call()))
        assert one.matched
        # a message-kind demonstration is unaffected: no call at all still matches, any call is the kind mismatch
        msg = await self._verify(server, ExpectedMessage(type="message", content="done"), _response(_message()))
        assert msg.matched


class TestVerifierRewardMode:
    """reward_mode=verifier: the constraint verdict on the model's own action is
    the whole reward; the teacher action is only a diagnostic."""

    # any-turn ban on BAN_WORD: a narration that says it FAILS, a clean narration PASSES.
    META = BAN_META
    NARRATION = "I will list the repository files first to orient myself before editing."
    VIOLATION = f"I will {BAN_WORD} the repository files first."

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

    async def test_violation_fails_this_constraint(self, server) -> None:
        """The same call with a narration that breaks the ban -> FAIL -> 0.0."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server,
            expected,
            _response(
                _message(self.VIOLATION),
                _call(name="execute_bash", arguments=json.dumps({"command": "ls"})),
            ),
            metadata=self.META,
        )
        assert not out.matched
        assert out.constraint_graded
        assert out.constraint_verdict is ConstraintVerdict.FAIL
        assert out.reward == approx(0.0)

    async def test_bare_tool_call_earns_nothing(self, server) -> None:
        """A silent turn is not gradable for a `forbidden` ban: UNGRADED, and UNGRADED earns 0.0 — a vacuous pass would
        let the policy collect reward by never saying anything the trigger could read."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            server,
            expected,
            _response(_call(name="execute_bash", arguments=json.dumps({"command": "ls"}))),
            metadata=self.META,
        )
        assert out.constraint_ungraded and not out.constraint_graded
        assert out.constraint_verdict is ConstraintVerdict.UNGRADED
        assert out.reward == approx(0.0)

    async def test_fail_earns_nothing_even_when_teacher_matched(self, server) -> None:
        """A violating narration on the teacher's own call: matched, but the verifier verdict is what decides -> 0.0."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(server, expected, _response(_message(self.VIOLATION), _call()), metadata=self.META)
        assert out.matched
        assert out.constraint_graded
        assert out.constraint_verdict is ConstraintVerdict.FAIL
        assert out.reward == approx(0.0)
        # the same call under a clean narration earns the reward
        clean = await self._verify(server, expected, _response(_message(self.NARRATION), _call()), metadata=self.META)
        assert clean.matched and clean.constraint_verdict is ConstraintVerdict.PASS and clean.reward == approx(1.0)

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


class TestTemplateConstraints:
    """The recipe verifier_parameter in `constraint_params` is graded through the swe_if_agents grader on the
    prefix+sampled surface: which turns count as branch turns, prefix violations, message-kind rows, grading errors."""

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
                                                      content=[NeMoGymResponseOutputText(type="output_text", text=f"I will {BAN_WORD} it now.", annotations=[])]))
        return NeMoGymResponseCreateParamsNonStreaming(input=items, metadata=metadata)

    async def _verify(self, server, expected, response, params):
        return await server.verify(
            ConstrainedBinaryPivotVerifyRequest(responses_create_params=params, response=response, expected_action=expected)
        )

    async def test_pivot_reward_requires_match_and_constraint_pass(self):
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        clean = await self._verify(_server(), expected, _response(_message("Applying the fix."), _call()), self._prefix_params(BAN_META))
        assert clean.reward == approx(1.0) and clean.matched and clean.constraint_verdict is ConstraintVerdict.PASS
        assert clean.constraint_passed and clean.constraint_graded
        dirty = await self._verify(_server(), expected, _response(_message(f"Let me {BAN_WORD} this."), _call()), self._prefix_params(BAN_META))
        assert dirty.reward == approx(0.0) and dirty.matched and dirty.constraint_verdict is ConstraintVerdict.FAIL
        assert dirty.constraint_failed and not dirty.constraint_passed

    async def test_prefix_violation_does_not_count_only_the_sampled_turn_does(self):
        """The prefix narration says the ban word (closed turn); the sampled turn is clean -> PASS."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        params = self._prefix_params(BAN_META)
        params.input[2] = NeMoGymResponseOutputMessage(id="p1", type="message", role="assistant", status="completed",
                                                       content=[NeMoGymResponseOutputText(type="output_text", text=f"I will {BAN_WORD} a.py.", annotations=[])])
        out = await self._verify(_server(), expected, _response(_message("Applying the fix."), _call()), params)
        assert out.reward == approx(1.0) and out.constraint_verdict is ConstraintVerdict.PASS

    async def test_open_prefix_turn_merges_with_the_sampled_call(self):
        """A prefix ending in an unanswered narration merges with the sampled call into ONE turn, which is a branch
        turn (the mining belt's rule): the merged turn carries the ban word -> FAIL."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(_server(), expected, _response(_call()), self._prefix_params(BAN_META, open_turn=True))
        assert out.matched and out.constraint_verdict is ConstraintVerdict.FAIL and out.reward == approx(0.0)

    async def test_empty_text_trailing_prefix_message_does_not_reopen_the_closed_turn(self):
        """A trailing assistant message with no visible text (what _pivot_common emits for a content-less history
        message) opens no turn in the grader's segmenter, so the sampled action starts a fresh turn: the CLOSED prefix
        turn's violation must not be charged to a clean sampled action. Openness is asked of the segmenter, not read
        off the last item's type."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        params = self._prefix_params(BAN_META)
        params.input[2] = NeMoGymResponseOutputMessage(id="p1", type="message", role="assistant", status="completed",
                                                       content=[NeMoGymResponseOutputText(type="output_text", text=f"I will {BAN_WORD} a.py.", annotations=[])])
        params.input.append(NeMoGymResponseOutputMessage(id="p2", type="message", role="assistant", status="completed",
                                                         content=[NeMoGymResponseOutputText(type="output_text", text="", annotations=[])]))
        prefix = [i.model_dump() for i in params.input]
        sampled = [i.model_dump() for i in (_message("Applying the fix."), _call())]
        assert ConstrainedBinaryPivotResourcesServer._branch_turns(prefix, sampled) == {1}
        out = await self._verify(_server(), expected, _response(_message("Applying the fix."), _call()), params)
        assert out.matched and out.constraint_verdict is ConstraintVerdict.PASS and out.reward == approx(1.0)
        # the genuinely open prefix (narration with text) still merges: branch set is the merged last turn
        open_params = self._prefix_params(BAN_META, open_turn=True)
        open_prefix = [i.model_dump() for i in open_params.input]
        assert ConstrainedBinaryPivotResourcesServer._branch_turns(open_prefix, [_call().model_dump()]) == {1}
        # and a prefix with no tool result at all is one open turn that the sampled call joins
        narration_only = [i.model_dump() for i in open_params.input[:3]]
        assert ConstrainedBinaryPivotResourcesServer._branch_turns(narration_only, [_call().model_dump()]) == {0}

    async def test_parallel_calls_do_not_earn_reward(self):
        """An any-turn text ban says nothing about how many tools a turn calls, so the belt's kind rule is the only
        thing keeping a clean-narration + teacher-call + extra-call turn from scoring 1.0 in pivot mode."""
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        out = await self._verify(
            _server(), expected,
            _response(_message("Applying the fix."), _call(), _call(name="execute_bash", arguments=json.dumps({"command": "ls"}))),
            self._prefix_params(BAN_META),
        )
        assert not out.matched and out.match_fail_kind and out.reward == approx(0.0)

    async def test_final_constraint_on_message_kind_expected_action(self):
        expected = ExpectedMessage(type="message", content="SUMMARY: done")
        md = {"constraint": "t#c2", "constraint_params": FINAL_PREFIX}
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
        srv = _server(reward_mode="verifier")
        wrong_tool_clean = await self._verify(srv, expected, _response(_message("Checking."), _call(name="execute_bash", arguments="{}")), self._prefix_params(BAN_META))
        assert not wrong_tool_clean.matched and wrong_tool_clean.reward == approx(1.0)
        right_tool_dirty = await self._verify(srv, expected, _response(_message(f"{BAN_WORD}!"), _call()), self._prefix_params(BAN_META))
        assert right_tool_dirty.matched and right_tool_dirty.reward == approx(0.0)

    async def test_unknown_matcher_is_ungraded_not_a_crash(self):
        expected = ExpectedFunctionCall(type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS)
        bad_vp = json.dumps({"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": "no_such_matcher", "value": 1}})
        md = {"constraint": "t#c9", "constraint_params": bad_vp}
        out = await self._verify(_server(), expected, _response(_call()), self._prefix_params(md))
        assert out.matched and out.constraint_verdict is ConstraintVerdict.UNGRADED and out.reward == approx(0.0)
