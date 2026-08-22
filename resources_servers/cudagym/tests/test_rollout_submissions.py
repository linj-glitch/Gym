# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the rollout-scoped submission endpoints and verify() semantics.

The HTTP layer (``nemo_gym.server_utils.request``) replays POSTs on disconnect,
and the sandboxed agent knows its own rollout token, so /begin and /verify must
tolerate identical replays while rejecting tampering. Evaluation-infrastructure
failures must stay distinguishable from 0-reward kernels: they fail their
operation instead of recording a submission, they do not spend the submission
budget, and verify() reports them in ``evaluation_error``.
"""

import asyncio
import json
import types

import pytest


pytest.importorskip("nemo_gym")
pytest.importorskip("cudagym")

from cudagym.contracts.evaluation import EvaluationStatus  # noqa: E402
from cudagym.sdk.errors import CudaGymTransportError  # noqa: E402

from nemo_gym.openai_utils import (  # noqa: E402
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.cudagym import app as app_module  # noqa: E402
from resources_servers.cudagym.app import (  # noqa: E402
    CudaGymResourcesServer,
    CudaGymVerifyRequest,
    extract_submitted_code,
    staged_reward_from_trace,
)


# A minimal valid KernelFactory problem (the same fields as data/example.jsonl).
DEFINITION = {
    "name": "vector_add",
    "axes": {"N": {"type": "var"}},
    "inputs": {"a": {"shape": ["N"], "dtype": "float32"}, "b": {"shape": ["N"], "dtype": "float32"}},
    "outputs": {"c": {"shape": ["N"], "dtype": "float32"}},
    "reference": "import torch\n\ndef run(a, b):\n    return a + b\n",
}
WORKLOAD = {"uuid": "w0", "axes": {"N": 16}, "inputs": {"a": {"type": "random"}, "b": {"type": "random"}}}
META = {
    "language": "triton",
    "target_hardware": "B200",
    "destination_passing_style": False,
    "definition": DEFINITION,
    "workloads": [WORKLOAD],
    "sol_anchors": {},
}

WEIGHTS = {"correctness": 1.0, "performance": 1.0}
PERF_CFG = {"clip_max": 10.0, "clip_min": 0.1, "speedup_ratio": 0.75, "allow_speedup_fallback": True}


class _Server:
    """CudaGymResourcesServer stand-in: the real endpoint methods over a scripted evaluation.

    ``eval_results`` is consumed one item per evaluation; an item is either a
    ``(reward, info)`` pair or an exception to raise, which stands in for an
    evaluation-infrastructure failure escaping ``_eval_and_reward``.
    """

    rollout_begin = CudaGymResourcesServer.rollout_begin
    rollout_evaluate = CudaGymResourcesServer.rollout_evaluate
    rollout_operation = CudaGymResourcesServer.rollout_operation
    verify = CudaGymResourcesServer.verify
    _verify_rollout = CudaGymResourcesServer._verify_rollout
    _sweep_rollouts = CudaGymResourcesServer._sweep_rollouts
    _row_sku_error = CudaGymResourcesServer._row_sku_error
    _row_sku = CudaGymResourcesServer._row_sku
    _run_submission_eval = CudaGymResourcesServer._run_submission_eval
    _submission_response = CudaGymResourcesServer._submission_response

    def __init__(self, eval_results=(), **config_overrides):
        config = dict(
            endpoints={"B200": "http://b200.test"},
            reward_mode="best_submission",
            max_submissions=8,
            submission_ttl_seconds=7200.0,
            final_file_fallback=False,
            compilation_timeout=300,
            execution_timeout_per_trial=120,
        )
        config.update(config_overrides)
        self.config = types.SimpleNamespace(**config)
        self._rollouts = {}
        self._eval_results = list(eval_results)
        self.eval_definitions = []  # Definition passed to each evaluation, in order.
        self.eval_skus = []  # GPU each evaluation was routed to, in order.

    async def _eval_and_reward(self, solution, definition, workloads, sol_anchors, sku):
        self.eval_definitions.append(definition)
        self.eval_skus.append(sku)
        outcome = self._eval_results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _bundle_body(code="def run(a, b, c):\n    c[:] = a + b\n"):
    """One /evaluate request body in submit.py's format; the code decides the solution hash."""
    return {
        "artifact": {"files": {"solution.json": json.dumps({"sources": [{"path": "kernel.py", "content": code}]})}}
    }


def _op_id(resp):
    """Extract the operation id from a 202 JSONResponse."""
    assert resp.status_code == 202
    return json.loads(resp.body)["id"]


async def _drain(server, token):
    """Wait for the rollout's background evaluation tasks to settle."""
    await asyncio.gather(*server._rollouts[token]["tasks"], return_exceptions=True)


def _verify_request(token=None, meta=META, text="no fenced code here"):
    """Build a /verify request whose trajectory's final assistant message is `text`."""
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="mock",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    extras = {"submission_token": token} if token else {}
    return CudaGymVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "optimize the kernel"}]},
        response=response,
        verifier_metadata=meta,
        **extras,
    )


# --- /begin: idempotent for identical replays, tamper-proof otherwise --


def test_begin_identical_replay_succeeds_without_touching_state():
    async def scenario():
        server = _Server(eval_results=[(1.5, {"correctness": True})])
        first = await server.rollout_begin("tok", {"verifier_metadata": META})
        assert first["ok"] is True
        entry = server._rollouts["tok"]
        _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        await _drain(server, "tok")
        # The retrying HTTP layer replays POSTs: an identical /begin must
        # succeed and must not reset the recorded submission state.
        replay = await server.rollout_begin("tok", {"verifier_metadata": META})
        assert replay == first
        assert server._rollouts["tok"] is entry
        assert len(entry["submissions"]) == 1

    asyncio.run(scenario())


def test_begin_with_a_different_payload_is_rejected_and_original_definition_governs():
    async def scenario():
        server = _Server(eval_results=[(1.5, {"correctness": True})])
        await server.rollout_begin("tok", {"verifier_metadata": META})
        entry = server._rollouts["tok"]
        # The sandboxed agent knows its own token; re-registering a different
        # problem must fail and leave the registered state untouched.
        tampered = {**META, "definition": {**DEFINITION, "name": "trivial_problem"}}
        resp = await server.rollout_begin("tok", {"verifier_metadata": tampered})
        assert resp.status_code == 409
        assert server._rollouts["tok"] is entry
        assert entry["meta"] == META
        # A submission evaluated after the tamper attempt still runs against
        # the originally registered definition.
        _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        await _drain(server, "tok")
        assert [d.name for d in server.eval_definitions] == ["vector_add"]

    asyncio.run(scenario())


def test_verify_scores_the_registered_problem_not_the_requests_metadata():
    async def scenario():
        # final_file_fallback on so the forged request reaches the scoring
        # path; the scripted evaluation records which definition it was given.
        server = _Server(eval_results=[(2.0, {"correctness": True})], final_file_fallback=True)
        await server.rollout_begin("tok", {"verifier_metadata": META})
        # The sandboxed agent knows its own token and the server's base URL
        # (PLATFORM_API_ENDPOINT), so it can POST /verify itself with metadata
        # describing a trivial substitute problem. The problem registered at
        # /begin must govern what is scored and cached as the verdict.
        forged = {
            **META,
            "definition": {**DEFINITION, "name": "trivial_problem", "reference": "def run():\n    return 0\n"},
            "workloads": [],
        }
        kernel = "```python\ndef run(a, b, c):\n    c[:] = a + b\n```"
        verdict = await server.verify(_verify_request("tok", meta=forged, text=f"submitting\n{kernel}"))
        assert [d.name for d in server.eval_definitions] == ["vector_add"]
        assert verdict.reward == 2.0

    asyncio.run(scenario())


def test_row_for_silicon_with_no_endpoint_is_refused_rather_than_evaluated():
    """A row can only be served by the endpoint configured for its own GPU.

    Evaluating it on another one would compile for the wrong SM version and
    time the kernel against anchors measured on different hardware, and the
    result would be recorded as the model's failure rather than as the
    configuration error it is.
    """

    async def scenario():
        server = _Server()  # B200 only
        other = {**META, "target_hardware": "H100"}
        resp = await server.rollout_begin("tok", {"verifier_metadata": other})
        assert resp.status_code == 400
        assert "H100" in json.loads(resp.body)["error"]
        assert "tok" not in server._rollouts
        # The final-file path has no /begin, so verify() refuses it there too.
        verdict = await server.verify(_verify_request(meta=other))
        assert verdict.reward == 0.0
        assert "H100" in verdict.evaluation_error

    asyncio.run(scenario())


def test_each_row_is_evaluated_on_the_endpoint_for_its_own_gpu():
    """With several endpoints configured, the row's target_hardware picks one."""

    async def scenario():
        endpoints = {"B200": "http://b200.test", "H100": "http://h100.test"}
        server = _Server(
            eval_results=[(1.5, {"correctness": True}), (1.5, {"correctness": True})], endpoints=endpoints
        )
        await server.rollout_begin("b200-tok", {"verifier_metadata": META})
        _op_id(await server.rollout_evaluate("b200-tok", _bundle_body()))
        await _drain(server, "b200-tok")

        h100_row = {**META, "target_hardware": "H100"}
        await server.rollout_begin("h100-tok", {"verifier_metadata": h100_row})
        _op_id(await server.rollout_evaluate("h100-tok", _bundle_body()))
        await _drain(server, "h100-tok")

        assert server.eval_skus == ["B200", "H100"]

    asyncio.run(scenario())


def test_row_without_target_hardware_needs_an_unambiguous_server():
    """One configured GPU places such a row; several leave nothing to choose by."""

    async def scenario():
        anonymous = {k: v for k, v in META.items() if k != "target_hardware"}
        # One endpoint: the row can only mean that GPU, so it is evaluated there.
        single = _Server(eval_results=[(1.5, {"correctness": True})])
        await single.rollout_begin("tok", {"verifier_metadata": anonymous})
        _op_id(await single.rollout_evaluate("tok", _bundle_body()))
        await _drain(single, "tok")
        assert single.eval_skus == ["B200"]

        # Several endpoints: the row cannot be placed, so it is refused rather
        # than sent to whichever endpoint happens to be configured first.
        several = _Server(endpoints={"B200": "http://b200.test", "H100": "http://h100.test"})
        resp = await several.rollout_begin("tok", {"verifier_metadata": anonymous})
        assert resp.status_code == 400
        assert "target_hardware" in json.loads(resp.body)["error"]
        verdict = await several.verify(_verify_request(meta=anonymous))
        assert verdict.reward == 0.0
        assert "target_hardware" in verdict.evaluation_error

    asyncio.run(scenario())


# --- infrastructure failures: failed op, no submission, no budget spend --


def test_infra_failure_fails_the_operation_without_recording_a_submission():
    async def scenario():
        server = _Server(eval_results=[CudaGymTransportError("connection refused")])
        await server.rollout_begin("tok", {"verifier_metadata": META})
        op_id = _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        await _drain(server, "tok")
        status = await server.rollout_operation("tok", op_id)
        assert status["status"] == "failed"
        assert "infrastructure" in status["error"]["message"]
        assert "connection refused" in status["error"]["message"]
        assert server._rollouts["tok"]["submissions"] == []

    asyncio.run(scenario())


def test_infra_failed_ops_do_not_spend_budget_and_identical_resubmission_is_allowed():
    async def scenario():
        server = _Server(
            eval_results=[CudaGymTransportError("eval service 503"), (2.0, {"correctness": True})],
            max_submissions=1,
        )
        await server.rollout_begin("tok", {"verifier_metadata": META})
        first_op = _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        await _drain(server, "tok")
        # The identical bundle again: the failed op neither dedups it nor
        # counts against the budget, so a fresh operation runs.
        retry = await server.rollout_evaluate("tok", _bundle_body())
        retry_op = _op_id(retry)
        assert retry_op != first_op
        await _drain(server, "tok")
        assert (await server.rollout_operation("tok", retry_op))["status"] == "succeeded"
        assert len(server._rollouts["tok"]["submissions"]) == 1
        # The recorded submission spent the whole budget (max_submissions=1):
        # a new distinct submission is now refused.
        third = await server.rollout_evaluate("tok", _bundle_body(code="def run(a, b, c):\n    c[:] = a - b\n"))
        assert third["success"] is False
        assert "budget exhausted" in third["error"]

    asyncio.run(scenario())


def test_submission_result_reports_the_evaluation_but_not_the_reward():
    """The document submit.py prints, and the operations endpoint returns, carries
    compilation/validation/performance only. The reward is the scalar the model is
    optimized on, not agent-visible feedback, so it must not appear anywhere in it."""
    result = _Server()._submission_response(
        "cand-1", "op-1", {"compiled": True, "executed": True, "correctness": True, "speedup": 2.5}
    )
    assert set(result["result"]) == {"compilation", "validation", "performance"}
    assert "staged_reward" not in result
    assert "reward" not in json.dumps(result)


# --- verify(): evaluation_error reporting --


def test_verify_keeps_best_submission_reward_and_reports_lost_submissions():
    async def scenario():
        server = _Server(
            eval_results=[(1.5, {"correctness": True, "speedup": 2.0}), CudaGymTransportError("connection reset")]
        )
        await server.rollout_begin("tok", {"verifier_metadata": META})
        _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        _op_id(await server.rollout_evaluate("tok", _bundle_body(code="def run(a, b, c):\n    c[:] = b + a\n")))
        verdict = await server.verify(_verify_request("tok"))
        # Reward semantics are unchanged: the best recorded submission scores.
        assert verdict.reward == 1.5
        assert verdict.n_submissions == 1
        assert "1 of 2 submissions lost to evaluation-infrastructure failures" in verdict.evaluation_error
        assert "connection reset" in verdict.evaluation_error

    asyncio.run(scenario())


def test_verify_reports_all_submissions_lost_alongside_the_no_submission_error():
    async def scenario():
        server = _Server(eval_results=[CudaGymTransportError("boom")])
        await server.rollout_begin("tok", {"verifier_metadata": META})
        _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        verdict = await server.verify(_verify_request("tok"))
        assert verdict.reward == 0.0
        assert verdict.n_submissions == 0
        assert "1 of 1 submissions lost to evaluation-infrastructure failures" in verdict.evaluation_error
        assert "no submitted kernel" in verdict.evaluation_error

    asyncio.run(scenario())


def test_verify_distinguishes_never_submitted_expired_state_and_malformed_rows():
    async def scenario():
        server = _Server()
        # Registered but never submitted (fallback disabled).
        await server.rollout_begin("tok", {"verifier_metadata": META})
        verdict = await server.verify(_verify_request("tok"))
        assert verdict.reward == 0.0 and "no submitted kernel" in verdict.evaluation_error
        # Token present but state expired before /verify.
        verdict = await server.verify(_verify_request("ghost-token"))
        assert verdict.reward == 0.0 and "expired" in verdict.evaluation_error
        # Malformed task row (final-file path, no token): missing definition.
        broken = {k: v for k, v in META.items() if k != "definition"}
        verdict = await server.verify(_verify_request(meta=broken))
        assert verdict.reward == 0.0 and "bad verifier_metadata" in verdict.evaluation_error
        # Unknown language is a malformed row too, not an escaping KeyError.
        verdict = await server.verify(_verify_request(meta={**META, "language": "rust"}))
        assert verdict.reward == 0.0 and "unknown language" in verdict.evaluation_error
        # An ordinary scored path reports no evaluation_error.
        server2 = _Server(eval_results=[(1.5, {"correctness": True})])
        await server2.rollout_begin("tok", {"verifier_metadata": META})
        _op_id(await server2.rollout_evaluate("tok", _bundle_body()))
        verdict = await server2.verify(_verify_request("tok"))
        assert verdict.reward == 1.5 and verdict.evaluation_error is None

    asyncio.run(scenario())


# --- verify(): idempotent verdict; a verified rollout accepts no new submissions --


def test_verify_replay_returns_the_cached_verdict():
    async def scenario():
        server = _Server(eval_results=[(1.5, {"correctness": True})])
        await server.rollout_begin("tok", {"verifier_metadata": META})
        _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        first = await server.verify(_verify_request("tok"))
        assert first.reward == 1.5
        # A dropped-connection replay must find the cached verdict, not an
        # empty entry that would re-score the rollout as 0.
        replay = await server.verify(_verify_request("tok"))
        assert replay is first
        assert "tok" in server._rollouts  # reclaimed by the TTL sweep, not by verify

    asyncio.run(scenario())


def test_verified_rollout_rejects_new_submissions():
    async def scenario():
        server = _Server(eval_results=[(1.5, {"correctness": True})])
        await server.rollout_begin("tok", {"verifier_metadata": META})
        _op_id(await server.rollout_evaluate("tok", _bundle_body()))
        await server.verify(_verify_request("tok"))
        late = await server.rollout_evaluate("tok", _bundle_body(code="def run(a, b, c):\n    c[:] = a * b\n"))
        assert late["success"] is False
        assert "already verified" in late["error"]

    asyncio.run(scenario())


# --- _eval_and_reward: transport errors escape; partial evaluations refuse to score --


class _FakeClient:
    """Stands in for the CudaGym SDK client, recording whether it was closed."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _EvalServer:
    """Runs the REAL _eval_and_reward over a monkeypatched workflows.evaluate."""

    _eval_and_reward = CudaGymResourcesServer._eval_and_reward

    def __init__(self):
        self.config = types.SimpleNamespace(
            benchmark_config={},
            compilation_timeout=300,
            execution_timeout_per_trial=120,
            reward_weights=dict(WEIGHTS),
            perf_reward_config=dict(PERF_CFG),
        )
        self.clients_for = []  # GPU each client was requested for, in order.

        self.clients = {}  # One client per GPU, as the real cache hands out.

    async def _client_for(self, sku):
        self.clients_for.append(sku)
        return self.clients.setdefault(sku, _FakeClient())


def _passed_trace(latencies, speedup_mean=None):
    """A stand-in Trace: one PASSED workload trace per latency, uuids w0, w1, ..."""
    return types.SimpleNamespace(
        workload_traces=[
            types.SimpleNamespace(
                evaluation=types.SimpleNamespace(
                    status=EvaluationStatus.PASSED,
                    performance=types.SimpleNamespace(latency_ms=latency),
                ),
                workload=types.SimpleNamespace(uuid=f"w{i}"),
            )
            for i, latency in enumerate(latencies)
        ],
        summary=types.SimpleNamespace(speedup_factor=types.SimpleNamespace(mean=speedup_mean)),
    )


def test_eval_and_reward_lets_transport_errors_escape(monkeypatch):
    async def fake_evaluate(client, **kwargs):
        raise CudaGymTransportError("HTTP 503 from the eval service")

    monkeypatch.setattr(app_module.workflows, "evaluate", fake_evaluate)
    server = _EvalServer()
    with pytest.raises(CudaGymTransportError):
        asyncio.run(server._eval_and_reward(None, None, [WORKLOAD], None, "B200"))


def test_eval_and_reward_refuses_partial_evaluations(monkeypatch):
    async def fake_evaluate(client, **kwargs):
        return _passed_trace([1.0])  # one trace for two workloads

    monkeypatch.setattr(app_module.workflows, "evaluate", fake_evaluate)
    server = _EvalServer()
    with pytest.raises(RuntimeError, match="partial evaluation"):
        asyncio.run(server._eval_and_reward(None, None, [WORKLOAD, WORKLOAD], None, "B200"))


def test_eval_and_reward_scores_complete_evaluations(monkeypatch):
    async def fake_evaluate(client, **kwargs):
        return _passed_trace([1.0, 1.0])

    monkeypatch.setattr(app_module.workflows, "evaluate", fake_evaluate)
    server = _EvalServer()
    reward, info = asyncio.run(server._eval_and_reward(None, None, [WORKLOAD, WORKLOAD], None, "H100"))
    # Correct on both workloads, no anchors, no measured speedup: the
    # correctness weight alone.
    assert reward == 1.0
    assert info["correctness"] is True
    # The evaluation ran on the client for the GPU it was given, not on a
    # single server-wide endpoint.
    assert server.clients_for == ["H100"]


def test_eval_and_reward_leaves_the_shared_client_open(monkeypatch):
    """The client is borrowed from the per-GPU cache, so closing it is not ours to do.

    Rollouts run concurrently against one client per GPU. Closing it at the end
    of an evaluation would drop the HTTP session other rollouts are mid-request
    on, which reads as an evaluation-infrastructure failure and loses their
    submissions.
    """

    async def fake_evaluate(client, **kwargs):
        return _passed_trace([1.0])

    monkeypatch.setattr(app_module.workflows, "evaluate", fake_evaluate)
    server = _EvalServer()
    asyncio.run(server._eval_and_reward(None, None, [WORKLOAD], None, "B200"))
    assert not server.clients["B200"].closed
    # And the next evaluation reuses that same live client.
    asyncio.run(server._eval_and_reward(None, None, [WORKLOAD], None, "B200"))
    assert not server.clients["B200"].closed
    assert server.clients_for == ["B200", "B200"]


# --- staged reward: correctness gate over the workload statuses --


def _trace(statuses):
    """Minimal stand-in for a cudagym Trace with the given workload statuses."""
    return types.SimpleNamespace(
        workload_traces=[
            types.SimpleNamespace(evaluation=types.SimpleNamespace(status=s, performance=None), workload=None)
            for s in statuses
        ],
        summary=types.SimpleNamespace(speedup_factor=types.SimpleNamespace(mean=2.0)),
    )


def test_incorrect_kernel_earns_zero_but_flags_are_reported():
    """An incorrect kernel earns 0 while the compiled/executed flags stay observable."""
    # Ran fine but numerically wrong: 0 reward, yet compiled/executed observable.
    reward, info = staged_reward_from_trace(_trace([EvaluationStatus.INCORRECT_NUMERICAL] * 2), WEIGHTS, PERF_CFG)
    assert reward == 0.0
    assert info["compiled"] is True and info["executed"] is True and info["correctness"] is False


def test_correct_kernel_earns_correctness_plus_capped_perf():
    """A correct kernel earns the correctness weight plus a bounded performance term."""
    reward, info = staged_reward_from_trace(_trace([EvaluationStatus.PASSED] * 2), WEIGHTS, PERF_CFG)
    assert info["correctness"] is True
    # correctness (1.0) + speedup fallback in [0, performance weight].
    assert 1.0 < reward <= 2.0

    # With the fallback disabled and no anchors, a correct kernel earns exactly
    # the correctness weight.
    reward_nofallback, _ = staged_reward_from_trace(
        _trace([EvaluationStatus.PASSED] * 2), WEIGHTS, {**PERF_CFG, "allow_speedup_fallback": False}
    )
    assert reward_nofallback == 1.0


# --- sol_score input guard: latency 0.0 is unmeasured, not speed-of-light --


def test_zero_latency_workloads_are_skipped_by_the_sol_score_term():
    anchors = {
        "w0": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0},
        "w1": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0},
    }
    # w0 reports the SDK's unmeasured 0.0 latency and must not contribute; w1
    # matches human-best exactly, so the mean SOL score is exactly 0.5.
    reward, info = staged_reward_from_trace(_passed_trace([0.0, 1.0]), WEIGHTS, PERF_CFG, sol_anchors=anchors)
    assert info["sol_score"] == 0.5
    assert reward == 1.5


def test_all_latencies_unmeasured_earns_the_correctness_weight_alone():
    anchors = {"w0": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0}}
    reward, info = staged_reward_from_trace(_passed_trace([0.0]), WEIGHTS, PERF_CFG, sol_anchors=anchors)
    assert "sol_score" not in info
    assert reward == 1.0


def test_non_finite_latency_workloads_are_skipped_by_the_sol_score_term():
    """A NaN latency is dropped by the reward loop's own guard, not left to sol_score.

    ``max(0.0, min(1.0, nan))`` evaluates to 1.0 because min and max keep their
    first argument when a comparison against NaN is false, so an unmeasured
    workload arriving as NaN would otherwise earn the maximum performance term.
    """
    anchors = {
        "w0": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0},
        "w1": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0},
    }
    trace = _passed_trace([float("nan"), 1.0])
    reward, info = staged_reward_from_trace(trace, WEIGHTS, PERF_CFG, sol_anchors=anchors)
    assert info["sol_score"] == 0.5
    assert reward == 1.5


# --- code extraction: unknown languages raise instead of falling back --


def test_extract_submitted_code_raises_for_an_unknown_language():
    """The extractor asks cudagym.rl for the language's fence tag before it reads
    the trajectory, so an unknown language raises instead of reaching the
    any-fence fallback and returning a block the row cannot be evaluated with."""
    response = _verify_request(text="```python\ndef run(a, b, c):\n    c[:] = a + b\n```").response
    assert extract_submitted_code(response, "triton") == "def run(a, b, c):\n    c[:] = a + b"
    # ValueError, naming the language and the table: the message reaches an
    # operator through a training log as the reason a rollout died.
    with pytest.raises(ValueError, match="'rust'.*LANGUAGE_DEFAULTS"):
        extract_submitted_code(response, "rust")


def test_concurrent_verifies_do_not_score_a_half_drained_rollout():
    """A replayed /verify must wait for the first one, not race its drain.

    The drain empties the task list before awaiting it, so a second caller
    arriving mid-drain would find no tasks left to wait for, read the
    submissions recorded so far, and cache a verdict scored on fewer of them
    than the rollout actually produced.
    """

    async def scenario():
        server = _Server(eval_results=[(0.4, {"correctness": True}), (1.9, {"correctness": True})])
        await server.rollout_begin("tok", {"verifier_metadata": META})
        entry = server._rollouts["tok"]

        # A slow evaluation still in flight when /verify arrives: the better of
        # the two submissions is the one that has not landed yet.
        started = asyncio.Event()

        async def slow_eval():
            started.set()
            await asyncio.sleep(0.05)
            entry["submissions"].append({"reward": 1.9, "info": {"correctness": True}})

        _op_id(await server.rollout_evaluate("tok", _bundle_body("first")))
        await _drain(server, "tok")
        entry["tasks"].append(asyncio.create_task(slow_eval()))
        await started.wait()

        first, replay = await asyncio.gather(
            server.verify(_verify_request(token="tok")),
            server.verify(_verify_request(token="tok")),
        )
        # Both callers get the one verdict, and it scores every submission.
        assert first is replay
        assert first.reward == 1.9
        assert first.n_submissions == 2

    asyncio.run(scenario())
