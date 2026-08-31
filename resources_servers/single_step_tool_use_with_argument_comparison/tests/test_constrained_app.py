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
    return ConstrainedBinaryPivotResourcesServer(
        config=config, server_client=MagicMock(spec=ServerClient)
    )


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
    return NeMoGymResponseFunctionToolCall(
        type="function_call", call_id="c1", name=name, arguments=arguments
    )


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
        expected = ExpectedFunctionCall(
            type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS
        )
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
        wrong = await self._verify(
            server, expected, _response(_call(name="execute_bash", arguments="{}"))
        )
        assert not wrong.matched
        assert sum(getattr(wrong, f) for f in flags) == 1
        assert wrong.argument_similarity is None
        assert not wrong.similarity_evaluated

        # Message expected, tool call produced: the kind_mismatch branch.
        kind = await self._verify(
            server, ExpectedMessage(type="message", content="done"), _response(_call())
        )
        assert not kind.matched
        assert kind.match_fail_kind
        assert sum(getattr(kind, f) for f in flags) == 1

    async def test_similarity_reported_when_l2_reached(self, server) -> None:
        """argument_similarity survives even when it lands under threshold."""
        expected = ExpectedFunctionCall(
            type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS
        )
        divergent = json.dumps(
            {"command": "str_replace", "path": "/repo/a.py", "new_str": "totally different"}
        )
        out = await self._verify(server, expected, _response(_call(arguments=divergent)))
        if out.similarity_evaluated:
            assert out.argument_similarity is not None
            assert 0.0 <= out.argument_similarity <= 1.0
            # matched iff the score cleared the configured threshold
            assert out.matched == (out.argument_similarity >= server.config.similarity_full_credit)

    async def test_constraint_bools_track_the_verdict(self, server) -> None:
        """The four verdict bools are one-hot over a graded rollout."""
        expected = ExpectedFunctionCall(
            type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS
        )
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
        expected_reward = 1.0 if out.constraint_verdict in (
            ConstraintVerdict.PASS,
            ConstraintVerdict.NO_CONSTRAINT_IN_ROW,
        ) else 0.0
        assert out.reward == approx(expected_reward)

    async def test_rows_without_a_constraint_are_absent_not_ungraded(self, server) -> None:
        """A row carrying no declaration is `absent`; reward degrades to match-only."""
        expected = ExpectedFunctionCall(
            type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS
        )
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
        expected = ExpectedFunctionCall(
            type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS
        )
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
        expected = ExpectedFunctionCall(
            type="function_call", name="str_replace_editor", arguments=EXPECTED_ARGS
        )
        metadata = {"constraint": "single_tool_call_per_message", "constraint_params": "{}"}
        response = _response(_message())

        off = await self._verify(_server(), expected, response, metadata=metadata)
        on = await self._verify(
            _server(always_grade_constraint=True), expected, response, metadata=metadata
        )

        assert off.reward == approx(on.reward) == approx(0.0)
        assert off.matched == on.matched is False
        # off skips grading entirely; on records a real verdict
        assert not off.constraint_graded
        assert on.constraint_graded or on.constraint_ungraded or on.constraint_absent
