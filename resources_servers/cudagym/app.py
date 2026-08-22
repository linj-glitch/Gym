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

"""CudaGym resources server: score a submitted kernel with a canonical CudaGym evaluation.

This server is the verifier of the agentic kernel-optimization environment. It pairs
with the OpenCode-based ``cuda_agent`` harness (``responses_api_agents/cuda_agent``),
which writes and iterates on a GPU kernel inside a per-rollout sandbox. Any ``cudagym
evaluate`` runs the agent performs in its sandbox are feedback only. The reward comes
solely from this server, which re-evaluates the submission on every workload of the
task with a fixed ``EvalConfig``. Scoring only the server-side evaluation mirrors
SolSwarm production, where agents submit candidates and the platform scores them, and
it keeps an agent from earning reward on self-selected easy workloads.

Each task row provides two things (see ``data/example.jsonl``):

* ``responses_create_params.input`` -- the prompt, as an optional system message plus
  the user message that states the optimization task.
* ``verifier_metadata`` -- the problem in KernelFactory form: ``definition`` (a
  problem's ``definition.json`` object), ``workloads`` (the rows of its
  ``workload.jsonl``), ``language``, ``target_hardware``,
  ``destination_passing_style``, and optional ``sol_anchors`` with per-workload
  latency anchors for the performance term — by contract, measurements taken on
  the row's own ``target_hardware`` (an anchor records no GPU, so this cannot
  be verified here; the dataset builder guarantees it).

The submission reaches ``verify()`` in one of two ways. In ``solswarm_submit`` mode the
agent posts solution bundles to this server's rollout-scoped ``/evaluate`` endpoint
during the rollout; each one is evaluated and recorded server-side, and ``verify()``
scores the best or the last recorded submission. Otherwise ``verify()`` extracts the
last fenced code block from the trajectory's final assistant message, which is where
``cuda_agent`` appends the final kernel file from the sandbox.

The reward is correctness-gated. A kernel that is not numerically correct on every
workload earns exactly 0. A correct kernel earns the ``correctness`` weight plus the
``performance`` weight times a performance term in [0, 1]. The scoring pieces
themselves -- the speed-of-light score, the speedup fallback, the per-language file
conventions, and the GPU SKU checks -- are imported from ``cudagym.rl``, the package
this server shares with NeMo-RL's single-turn baseline, so the two harnesses score a
kernel the same way.

Evaluations run on the CudaGym GPU server ``config.endpoints`` lists for the row's
GPU, or, where that entry is empty, on the one named by the
``CUDAGYM_UNIFIED_SERVER_URL`` / ``CUDAGYM_URL`` environment variables.
"""

import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, Optional
from uuid import uuid4

from cudagym.contracts.definition import Definition
from cudagym.contracts.eval_config import EvalConfig
from cudagym.contracts.evaluation import EvaluationStatus
from cudagym.contracts.solution import Solution
from cudagym.contracts.trace import Trace
from cudagym.contracts.workload import Workload
from cudagym.rl import (
    LANGUAGE_DEFAULTS,
    build_solution,
    canonical_sku,
    fence_lang_for,
    geomean,
    normalize_performance_reward,
    resolve_row_sku,
    sol_score,
    verify_health_payload,
)
from cudagym.sdk import Client, workflows
from cudagym.sdk.errors import CudaGymCompilationError, CudaGymExecutionError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.cudagym.solution_bundle import normalize_solution_bundle


LOG = logging.getLogger(__name__)

# Ceiling on one agent-authored solution.json (the kernel sources plus their
# JSON envelope); anything past it is refused before parsing.
_MAX_SOLUTION_JSON_BYTES = 8 * 1024 * 1024

_COMPILE_FAIL = {EvaluationStatus.COMPILE_ERROR}
_EXEC_FAIL = {EvaluationStatus.RUNTIME_ERROR, EvaluationStatus.TIMEOUT, EvaluationStatus.INVALID_REFERENCE}
# CUDAGRAPH_INCOMPATIBLE (numerically correct but not graph-capturable) is deliberately
# excluded: the cuda_agent overlay bans CUDA graphs, so the status should not occur, and
# a kernel that trips it anyway earns 0.
_CORRECT_OK = {EvaluationStatus.PASSED, EvaluationStatus.CORRECTNESS_PASSED}


async def log_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log a request-validation failure bytes-safely and answer the standard 422.

    Registered over the base server's handler (setup_webserver). Pydantic's
    ``errors()`` carry the raw request body in their ``input`` field — bytes
    when the body was unparseable — which crashes both the base handler's
    json.dumps() and FastAPI's default 422 encoder (non-UTF8 bytes), turning
    the 422 into a 500 that hides which field failed. Stringifying the
    non-JSON leaves keeps the standard ``{"detail": [...]}`` shape.
    """
    detail = json.loads(json.dumps(exc.errors(), default=str))
    LOG.error(
        "Request validation failed for %s %s: %s (body: %.2000s)",
        request.method,
        request.url.path,
        json.dumps(detail),
        exc.body,
    )
    return JSONResponse(status_code=422, content={"detail": detail})


def extract_submitted_code(response: Any, language: str) -> Optional[str]:
    """Return the last fenced code block from the trajectory's final assistant text.

    ``cuda_agent`` appends the final kernel file from the sandbox as a fenced code
    block in a closing assistant message, so the extracted text equals the file the
    agent actually produced. When no block carries the language's expected fence tag,
    any fenced block is accepted as a fallback.
    """
    texts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
            if text:
                texts.append(text)
    if not texts:
        return None
    blob = texts[-1]
    fence = fence_lang_for(language)
    matches = re.findall(rf"```{fence}\n(.*?)\n```", blob, re.DOTALL) or re.findall(
        r"```[\w+]*\n(.*?)\n```", blob, re.DOTALL
    )
    return matches[-1].strip() if matches else None


def staged_reward_from_trace(
    trace: Trace,
    weights: dict[str, float],
    perf_cfg: dict[str, Any],
    sol_anchors: Optional[dict] = None,
) -> tuple[float, dict[str, Any]]:
    """Map a CudaGym ``Trace`` to a correctness-gated reward and an info dict.

    A kernel that is not numerically correct on every workload, or that trips
    CudaGym's reward-hack detector, earns exactly 0. The compiled and executed
    flags are still computed and returned in ``info`` for metrics, but no reward
    attaches to them: Triton and the other JIT languages have no failing compile
    step, so paying for "compiled" would reward placeholder files.

    A correct kernel earns ``weights["correctness"]`` plus
    ``weights["performance"]`` times a performance term. The performance term is
    the mean per-workload speed-of-light score (see ``sol_score``) when
    ``sol_anchors`` provides anchors, keyed by workload uuid; this is the metric
    SolSwarm scores on. When the problem has no usable
    anchors and ``perf_cfg["allow_speedup_fallback"]`` is true, the term falls
    back to the log-normalized eager-reference speedup instead.
    """
    info: dict[str, Any] = {"compiled": False, "executed": False, "correctness": False, "speedup": -1.0}
    reward = 0.0

    # One evaluation status per workload; none at all means the eval produced nothing.
    statuses = [wt.evaluation.status if wt.evaluation else None for wt in trace.workload_traces]
    if not statuses:
        info["error"] = "no workload traces"
        return reward, info
    reward_hacked = any(s == EvaluationStatus.REWARD_HACK for s in statuses)

    # Detection only (metrics/debugging); no reward attaches to these flags.
    info["compiled"] = all(s not in _COMPILE_FAIL for s in statuses)
    info["executed"] = info["compiled"] and all(s not in _EXEC_FAIL for s in statuses)

    # The gate: a reward-hack flag or any incorrect workload zeroes the reward.
    if reward_hacked or not all(s in _CORRECT_OK for s in statuses):
        if reward_hacked:
            info["reward_hacked"] = True
        return reward, info
    # Correct on every workload: the correctness weight is earned. Weights and
    # the fallback flag are hard-indexed: a missing key is a config bug and must
    # not silently score as 0 (same convention as nemo_rl/environments/atlas/reward.py).
    info["correctness"] = True
    reward += weights["correctness"]

    # Eager-reference speedup (cudagym's own metric); recorded for logging and fallback.
    summary = trace.summary
    if summary.speedup_factor is not None and summary.speedup_factor.mean is not None:
        info["speedup"] = summary.speedup_factor.mean

    # Per-workload SOL scores. Only workloads with a positive human-best anchor
    # and a positive measured latency contribute; a score may be 0 (far above
    # human-best) and still count.
    sol_scores: list[float] = []
    human_best_speedups: list[float] = []
    if sol_anchors:
        for workload_trace in trace.workload_traces:
            evaluation = workload_trace.evaluation
            if evaluation is None or evaluation.performance is None:
                continue
            # WorkloadTrace.workload / Workload.uuid are required pydantic
            # fields (same access as the nemo_rl atlas twin).
            anchor = sol_anchors.get(workload_trace.workload.uuid)
            human_best = float((anchor or {}).get("human_best_latency_ms") or 0.0)
            if not anchor or human_best <= 0.0:
                continue
            t_k = float(evaluation.performance.latency_ms)
            # latency_ms 0.0 is the SDK's unmeasured default, and a NaN latency
            # passes any comparison guard; scoring either would award the
            # maximum performance term, so require a finite positive
            # measurement.
            if not (math.isfinite(t_k) and t_k > 0):
                continue
            sol_scores.append(sol_score(t_k, human_best, float(anchor.get("sol_latency_ms") or 0.0)))
            human_best_speedups.append(human_best / t_k)

    # Performance term ladder: anchored SOL scores first, the speedup fallback second, else nothing.
    if sol_scores:
        info["sol_score"] = sum(sol_scores) / len(sol_scores)  # arithmetic mean across workloads
        info["human_best_speedup"] = geomean(human_best_speedups)
        reward += weights["performance"] * info["sol_score"]
    elif info["speedup"] != -1.0 and perf_cfg["allow_speedup_fallback"]:
        reward += normalize_performance_reward(
            info["speedup"],
            clip_max=perf_cfg["clip_max"],
            clip_min=perf_cfg["clip_min"],
            scale=weights["performance"],
            speedup_ratio=perf_cfg["speedup_ratio"],
        )
    return reward, info


class CudaGymResourcesServerConfig(BaseResourcesServerConfig):
    """Server config: eval endpoint, timeouts, reward weights, and the solswarm_submit knobs."""

    # One evaluation endpoint per GPU, keyed by the CudaGym SupportedHardware
    # value; a row is scored on the endpoint for its own target_hardware and
    # refused when there is none. Required, never defaulted: the keys are what
    # each endpoint's /health is checked against. An empty URL resolves at
    # runtime from CUDAGYM_UNIFIED_SERVER_URL / CUDAGYM_URL (in-allocation
    # hosting); operator guidance lives in configs/cudagym_cuda_agent.yaml.
    endpoints: dict[str, str]
    # Fail fast, once per endpoint at first use, when its /health reports a GPU
    # other than the SKU it is keyed under. A mismatch is otherwise silent for
    # Triton kernels.
    verify_endpoint_sku: bool = True
    compilation_timeout: int = 300
    execution_timeout_per_trial: int = 120
    # Correctness-gated: 0 until numerically correct on every workload, then
    # correctness + performance * perf_term (the mean SOL score in [0, 1] when
    # the row carries anchors). Compiled/executed progress is reported as
    # metrics but never rewarded; a bare stub "compiles" in JIT languages.
    reward_weights: dict[str, float] = Field(default_factory=lambda: {"correctness": 1.0, "performance": 1.0})
    # allow_speedup_fallback: for rows without SOL anchors, whether a correct
    # kernel may earn the perf term from the log-normalized speedup over the
    # timed reference instead. Requires an evaluation that timed a reference
    # (benchmark_config: {benchmark_reference: true}); when false, the perf
    # term of anchor-less rows is 0. dict[str, Any], not dict[str, float]:
    # allow_speedup_fallback is a boolean and must not be coerced to 1.0.
    perf_reward_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "clip_max": 10.0,
            "clip_min": 0.1,
            "speedup_ratio": 0.75,
            "allow_speedup_fallback": True,
        }
    )
    # Overrides for CudaGym's EvalConfig (e.g. benchmark_reference, iterations).
    benchmark_config: dict[str, Any] = Field(default_factory=dict)
    # solswarm_submit mode (rollout-scoped submission endpoints; see
    # responses_api_agents/cuda_agent/app.py): which recorded submission the
    # reward comes from, and how many canonical evals one rollout may spend.
    reward_mode: str = "best_submission"  # best_submission | final_submission
    max_submissions: int = 8
    submission_ttl_seconds: float = 7200.0
    # solswarm_submit mode: whether the final kernel file may be scored when the
    # rollout never submitted. Production scores only submissions (no submission
    # means no score); this fallback gives weak base models a gradient before
    # they learn to submit.
    final_file_fallback: bool = True


class CudaGymVerifyRequest(BaseVerifyRequest):
    """``/verify`` request; ``verifier_metadata`` (Gym's slot for verifier-only
    task data) carries the KernelFactory problem."""

    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class CudaGymVerifyResponse(BaseVerifyResponse):
    """``/verify`` response: the reward plus per-stage evaluation metrics."""

    model_config = ConfigDict(extra="allow")
    compiled: bool = False
    executed: bool = False
    correctness: bool = False
    speedup: float = -1.0  # eager-reference speedup (cudagym's metric); logging and fallback
    sol_score: float = -1.0  # mean SOL score in [0, 1]; the perf term when anchors exist
    human_best_speedup: float = -1.0  # geomean speedup over human-best (logging only)
    # Why the rollout could not be scored normally, when it could not be: submission
    # evaluations lost to eval-infrastructure failures, no submitted kernel, expired
    # submission state, or a malformed task row. None for an ordinary scored kernel,
    # so downstream can tell these apart from a plain incorrect submission.
    evaluation_error: Optional[str] = None


class CudaGymResourcesServer(SimpleResourcesServer):
    """Scores a submitted CUDA/Triton/Python kernel via a canonical CudaGym evaluation.

    The server adds no model-facing tool endpoints; the agent gets feedback by
    running the ``cudagym`` CLI inside its own sandbox. Beyond ``verify()``, the
    server exposes rollout-scoped submission endpoints that implement SolSwarm's
    platform API, used when ``cuda_agent`` runs in ``solswarm_submit`` mode (see
    ``setup_webserver``).
    """

    config: CudaGymResourcesServerConfig

    # Rollout-scoped submission state (solswarm_submit mode): token -> {meta,
    # created, submissions, ops, tasks, verdict}. In-process only; verify()
    # caches its verdict on the entry (the HTTP layer replays POSTs on
    # disconnect) and the TTL sweep reclaims memory.
    _rollouts: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    # One SDK client per SKU, built on first use; the set of SKUs whose
    # endpoint has already passed its /health check.
    _clients: dict[str, Client] = PrivateAttr(default_factory=dict)
    _clients_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _endpoint_sku_checked: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """Reject a non-canonical SKU key at startup rather than per submission."""
        super().model_post_init(__context)
        if not self.config.endpoints:
            raise ValueError("the cudagym resources server needs at least one entry in `endpoints`")
        for sku in self.config.endpoints:
            canonical_sku(sku, "a key of the cudagym resources server's `endpoints`")

    def _row_sku_error(self, meta: dict[str, Any]) -> Optional[str]:
        """Return why a task row's ``target_hardware`` cannot be served, or None.

        The refusal text becomes the 400 / ``evaluation_error`` message, so a
        misrouted row fails as the configuration error it is instead of being
        scored on the wrong silicon and recorded as the model's failure.
        """
        try:
            resolve_row_sku(meta.get("target_hardware"), self.config.endpoints)
        except ValueError as err:
            return str(err)
        return None

    def _row_sku(self, meta: dict[str, Any]) -> str:
        """The SKU a row is evaluated on (the shared ``cudagym.rl`` resolver).

        Callers run ``_row_sku_error`` first, so the resolver's refusals do not
        escape from here.
        """
        return resolve_row_sku(meta.get("target_hardware"), self.config.endpoints)

    def setup_webserver(self) -> FastAPI:
        """Extend the base app with the rollout-scoped SolSwarm platform endpoints."""
        app = super().setup_webserver()
        # Replaces the base validation handler (same exception type, last
        # registration wins) with the bytes-safe one below.
        app.exception_handler(RequestValidationError)(log_validation_error)
        # Endpoints compatible with SolSwarm's submit.py, scoped per rollout
        # token. The cuda_agent injects
        # PLATFORM_API_ENDPOINT=<base>/rollout/<token>/api/v1 into the sandbox,
        # so the unmodified submit skill posts here.
        app.post("/rollout/{token}/begin")(self.rollout_begin)
        app.post("/rollout/{token}/api/v1/evaluate")(self.rollout_evaluate)
        app.get("/rollout/{token}/api/v1/operations/{op_id}")(self.rollout_operation)
        # Trace sink. SolSwarm's entrypoint POSTs the session transcript here at
        # exit. The upload does not matter for the run (token capture happens at
        # the model server), but an absent route costs 3 retries (about 35s) per
        # rollout, so the payload is acknowledged and dropped.
        app.post("/rollout/{token}/api/v1/agents/trace")(self.rollout_trace)
        # Fire-and-forget platform routes the skills use; answering them keeps
        # tool calls from erroring inside the agent's session.
        app.post("/rollout/{token}/api/v1/insights")(self.rollout_insight)
        app.post("/rollout/{token}/api/v1/bug-reports")(self.rollout_bug_report)
        return app

    def _endpoint_url(self, sku: str) -> str:
        """The evaluation endpoint serving ``sku``.

        An empty configured URL means the address is not known until the job is
        running -- how in-allocation hosting passes its load balancer -- so it
        falls back to the environment. ``CUDAGYM_URL`` is SolSwarm production's
        variable name and ``CUDAGYM_UNIFIED_SERVER_URL`` the launcher's.
        """
        url = self.config.endpoints[sku]
        if url:
            return url
        unified = os.environ.get("CUDAGYM_UNIFIED_SERVER_URL") or os.environ.get("CUDAGYM_URL")
        if not unified:
            raise RuntimeError(
                f"the cudagym endpoint for {sku} is configured empty and neither "
                f"CUDAGYM_UNIFIED_SERVER_URL nor CUDAGYM_URL is set"
            )
        return unified

    async def _client_for(self, sku: str) -> Client:
        """Return the SDK client evaluating on ``sku``, building it on first use.

        One client (one HTTP session) is cached per SKU for the life of the
        server; concurrent evaluations share it, so callers must NOT close what
        they get back. The first client for a SKU also runs that endpoint's
        /health check. The SDK takes separate compile and GPU URLs; each
        endpoint serves both, so the same URL is passed twice.
        """
        cached = self._clients.get(sku)
        if cached is not None:
            return cached
        # Serialized: two concurrent cache misses would each build a client and
        # leak the losing one's session.
        async with self._clients_lock:
            # Another coroutine may have built it while this one waited.
            cached = self._clients.get(sku)
            if cached is not None:
                return cached
            # The SDK resolves credentials per target from the environment
            # (MODAL_PROXY_TOKEN_ID/SECRET for Modal hosts, API_TOKEN for the
            # platform proxy); the constructor takes no auth.
            client = Client(
                compile_server_url=self._endpoint_url(sku),
                gpu_server_url=self._endpoint_url(sku),
            )
            if self.config.verify_endpoint_sku and sku not in self._endpoint_sku_checked:
                try:
                    await self._verify_endpoint_sku(client, sku)
                except BaseException:
                    # The caller never receives the client on this path, so
                    # nothing else can close it; the fail-fast check must not
                    # leak a session.
                    await client.close()
                    raise
                self._endpoint_sku_checked.add(sku)
            self._clients[sku] = client
            return client

    async def _verify_endpoint_sku(self, client: Client, sku: str) -> None:
        """Fail fast when the endpoint keyed under ``sku`` reports a different GPU.

        A mismatch is otherwise silent for Triton kernels: they JIT-compile on
        whatever GPU serves the request and return that GPU's timings. Raises
        ``RuntimeError``, which propagates out of verify() or the submission
        endpoint as an HTTP error instead of masquerading as a 0-reward
        evaluation.
        """
        resp = await client.health()
        if resp.get("status") != "healthy":
            raise RuntimeError(f"cudagym endpoint for {sku} unhealthy at first use: {resp}")
        payloads = (
            [s.get("health") or {} for s in resp["servers"]] if "servers" in resp else [resp.get("health") or {}]
        )
        # Check every responding server, not just the first with a gpu_model: in
        # a heterogeneous pool, one matching member must not vouch for the rest.
        for payload in payloads:
            ok, detail = verify_health_payload(payload, sku)
            if ok is False:
                raise RuntimeError(
                    f"cudagym endpoint for {sku} reports different silicon: {detail} "
                    f"(set verify_endpoint_sku: false to override deliberately)"
                )
            if ok is None:
                LOG.warning(
                    "cudagym endpoint SKU NOT VERIFIED (%s): %s -- kernels may target the wrong silicon",
                    sku,
                    detail,
                )
            else:
                LOG.info("cudagym endpoint SKU check passed (%s): %s", sku, detail)
        if not payloads:
            LOG.warning("cudagym endpoint SKU NOT VERIFIED (%s): /health listed no servers", sku)

    async def _eval_and_reward(
        self,
        solution: Solution,
        definition: Definition,
        workloads: list[Workload],
        sol_anchors: Optional[dict[str, Any]],
        sku: str,
    ) -> tuple[float, dict[str, Any]]:
        """Run the canonical evaluation and return (correctness-gated reward, info).

        ``sku`` selects the endpoint, so the kernel is timed on the GPU its row
        declared. Kernel-caused failures (compile or run errors) score 0 with
        the error in ``info``. Everything else — transport errors, a
        misconfigured endpoint, a truncated evaluation — propagates, so an
        infrastructure failure stays distinguishable from a 0-reward kernel.
        """
        eval_config = EvalConfig(**self.config.benchmark_config) if self.config.benchmark_config else None
        timeout = float(self.config.execution_timeout_per_trial * max(1, len(workloads)))
        # Client construction and the one-time SKU check sit outside the reward
        # try/except on purpose: a misconfigured endpoint must fail the request
        # loudly (HTTP 500), not masquerade as a 0-reward evaluation.
        client = await self._client_for(sku)
        try:
            trace = await workflows.evaluate(
                client,
                solution=solution,
                definition=definition,
                workloads=workloads,
                config=eval_config,
                compile_timeout=float(self.config.compilation_timeout),
                timeout=timeout,
            )
        except CudaGymCompilationError as e:
            # Not correct -> 0 under the correctness-gated reward.
            return 0.0, {"compile_error": str(e)}
        except CudaGymExecutionError as e:
            return 0.0, {"execution_error": str(e)}

        # The eval driver emits one workload trace per workload even for run-level
        # failures, so a shortfall means the evaluation was truncated upstream.
        # Scoring the returned subset would let a partial run earn full credit.
        if len(trace.workload_traces) != len(workloads):
            raise RuntimeError(
                f"cudagym evaluation returned {len(trace.workload_traces)} workload traces for "
                f"{len(workloads)} workloads; refusing to score a partial evaluation"
            )

        return staged_reward_from_trace(
            trace,
            self.config.reward_weights,
            self.config.perf_reward_config,
            sol_anchors=sol_anchors,
        )

    # ----- rollout-scoped submission endpoints (solswarm_submit mode) ------------

    def _sweep_rollouts(self) -> None:
        """Drop rollout entries older than the configured submission TTL."""
        now = time.monotonic()
        ttl = self.config.submission_ttl_seconds
        for token in [t for t, e in self._rollouts.items() if now - e["created"] > ttl]:
            self._rollouts.pop(token, None)

    async def rollout_begin(self, token: str, body: dict[str, Any]) -> Any:
        """Register a rollout's task metadata so its submissions can be evaluated server-side.

        The cuda_agent calls this before launching the sandbox; the response
        carries the per-submission evaluation budget the agent exports as
        ``KF_EVAL_TIMEOUT``. Re-registering an identical payload succeeds (the
        HTTP layer replays POSTs), but a differing payload is rejected with
        HTTP 409: the sandboxed agent knows its own token and must not be able
        to replace the problem definition or reset its submission state.
        """
        self._sweep_rollouts()
        meta = body.get("verifier_metadata") or {}
        row_sku_error = self._row_sku_error(meta)
        if row_sku_error:
            return JSONResponse(status_code=400, content={"ok": False, "error": row_sku_error})
        existing = self._rollouts.get(token)
        if existing is not None and existing["meta"] != meta:
            return JSONResponse(
                status_code=409,
                content={
                    "ok": False,
                    "error": "rollout token is already registered with a different problem definition",
                },
            )
        if existing is None:
            self._rollouts[token] = {
                "meta": meta,
                "created": time.monotonic(),
                "submissions": [],
                "ops": {},
                "tasks": [],
                # verify() sets closed before computing the verdict; a closed
                # entry accepts no further submissions.
                "closed": False,
                "verdict": None,
                # Held while the verdict is computed, so a replayed /verify
                # waits for the first one instead of racing its drain.
                "verdict_lock": asyncio.Lock(),
            }
        eval_timeout = int(
            self.config.compilation_timeout
            + self.config.execution_timeout_per_trial * max(1, len(meta.get("workloads") or []))
        )
        return {"ok": True, "eval_timeout_seconds": eval_timeout}

    async def rollout_evaluate(self, token: str, body: dict[str, Any]) -> Any:
        """Accept a submission, in the request format of SolSwarm's ``submit.py``.

        Returns HTTP 202 with an operation id, which submit.py polls via
        ``/operations/<id>``. The canonical evaluation runs in the background
        and its result is recorded server-side, so the reward cannot be forged
        from inside the sandbox. Duplicate submissions (same solution hash)
        return the recorded result instead of spending another evaluation.
        """

        def fail(msg: str) -> dict[str, Any]:
            """Build an error document in the response format submit.py expects."""
            return {"success": False, "candidate_id": None, "evaluation_id": None, "error": msg}

        # The rollout must have registered via /begin first.
        entry = self._rollouts.get(token)
        if entry is None:
            return fail("unknown rollout token (expired, or /rollout/<token>/begin was never called)")
        # A closed rollout accepts nothing further: verify() closes the entry
        # before computing the verdict and caches the verdict on it, so a late
        # submission (a stray retry, or an orphaned container) could never count.
        if entry["closed"] or entry["verdict"] is not None:
            return fail("rollout already verified; no further submissions are accepted")

        # Extract the bundle and fill in the envelope of a partial one before validation.
        raw = ((body.get("artifact") or {}).get("files") or {}).get("solution.json")
        if not raw:
            return fail("request carried no artifact.files['solution.json']")
        # The bundle is agent-authored: refuse an oversized document before
        # json.loads and validation walk it.
        if isinstance(raw, str) and len(raw) > _MAX_SOLUTION_JSON_BYTES:
            return fail(f"solution.json exceeds the {_MAX_SOLUTION_JSON_BYTES}-byte submission limit")
        try:
            bundle = json.loads(raw) if isinstance(raw, str) else raw
            bundle = normalize_solution_bundle(bundle, entry["meta"], self._row_sku(entry["meta"]))
            solution = Solution.model_validate(bundle)
        except Exception as e:  # noqa: BLE001 - agent-authored JSON
            return fail(f"invalid solution.json: {e}")

        # Dedup by solution hash: a repeat returns the recorded result, or the running op's 202.
        sol_hash = solution.hash()
        for record in entry["submissions"]:
            if record["candidate_id"] == sol_hash:
                return record["response"] | {"deduplicated": True}
        for op_id, op in entry["ops"].items():
            if op["candidate_id"] == sol_hash and op["status"] == "running":
                return JSONResponse(status_code=202, content={"id": op_id})

        # Each distinct submission spends one canonical evaluation from the
        # rollout's budget. Only running operations and recorded submissions
        # count: an operation that failed on evaluation infrastructure was not
        # scored, so it must not burn the agent's budget.
        spent = sum(1 for op in entry["ops"].values() if op["status"] in ("running", "succeeded"))
        if spent >= self.config.max_submissions:
            return fail(f"submission budget exhausted (max {self.config.max_submissions} per rollout)")
        # Failed operations spend no budget, so bound the total attempts as
        # well: each one still ran a real canonical evaluation, and without a
        # ceiling a submission whose evaluations always fail as infrastructure
        # errors could loop forever.
        if len(entry["ops"]) >= 4 * self.config.max_submissions:
            return fail(
                f"too many evaluation attempts for this rollout "
                f"(max {4 * self.config.max_submissions} including failed evaluations)"
            )

        # New submission: record a running operation and evaluate in the background.
        op_id = uuid4().hex
        entry["ops"][op_id] = {"status": "running", "candidate_id": sol_hash, "result": None, "error": None}
        task = asyncio.create_task(self._run_submission_eval(token, op_id, solution))
        entry["tasks"].append(task)
        return JSONResponse(status_code=202, content={"id": op_id})

    async def _run_submission_eval(self, token: str, op_id: str, solution: Solution) -> None:
        """Evaluate one submission in the background and record the outcome on its operation.

        Kernel-caused failures come back from ``_eval_and_reward`` as ordinary
        0-reward results and are recorded as submissions. Anything raised —
        eval-service transport/infrastructure failures, malformed registered
        metadata, a truncated evaluation — marks the operation failed WITHOUT
        recording a submission, so it neither scores as a bad kernel nor spends
        the submission budget, and verify() can report it as an
        ``evaluation_error``.
        """
        entry = self._rollouts.get(token)
        if entry is None:
            return
        op = entry["ops"][op_id]
        meta = entry["meta"]
        try:
            definition = Definition.model_validate(meta["definition"])
            workloads = [Workload.model_validate(w) for w in meta["workloads"]]
            reward, info = await self._eval_and_reward(
                solution, definition, workloads, meta.get("sol_anchors"), self._row_sku(meta)
            )
        except Exception as e:  # noqa: BLE001 - a failed evaluation must not kill the rollout
            LOG.warning("submission eval infrastructure failure (token=%s op=%s): %s", token, op_id, e)
            op["status"] = "failed"
            op["error"] = f"evaluation-infrastructure failure (not a kernel error): {e}"
            return
        response = self._submission_response(op["candidate_id"], op_id, info)
        entry["submissions"].append(
            {"candidate_id": op["candidate_id"], "reward": reward, "info": info, "response": response}
        )
        op["status"] = "succeeded"
        op["result"] = response

    def _submission_response(self, candidate_id: str, op_id: str, info: dict[str, Any]) -> dict[str, Any]:
        """Build the evaluation result document that SolSwarm's submit.py prints to the agent.

        The document reports compilation, validation and performance, which is
        what the sandbox needs to iterate. The training reward is deliberately
        not part of it: it is the scalar the model is optimized on, not
        feedback the model is meant to read.
        """
        compiled = bool(info.get("compiled", False))
        executed = bool(info.get("executed", False))
        correct = bool(info.get("correctness", False))
        stderr = info.get("compile_error") or info.get("execution_error")
        result: dict[str, Any] = {
            "compilation": {"success": compiled},
            "validation": {
                "success": correct,
                "correctness": 1.0 if correct else 0.0,
                "runs": [{"stderr": str(stderr)}] if stderr else [],
            },
        }
        speedup = float(info.get("speedup", -1.0))
        if correct and speedup >= 0.0:
            result["performance"] = {"speedup": speedup}
        return {
            "success": compiled and executed and correct,
            "candidate_id": candidate_id,
            "evaluation_id": op_id,
            "deduplicated": False,
            "result": result,
            # Deliberately None, not a pass: no compliance check runs here.
            # Production runs an LLM compliance judge, and submit.py's local
            # gate only runs when the ban env vars are set. Reporting `true`
            # would tell the model its submission cleared a check that never ran.
            "compliance": {
                "compliant": None,
                "reason": "not judged in this RL environment (no compliance judge configured)",
            },
        }

    async def rollout_trace(self, token: str, body: dict[str, Any]) -> dict[str, Any]:
        """Accept SolSwarm's end-of-run trace upload. The uploader only checks for HTTP 200."""
        return {"success": True, "session_id": str(body.get("session_id") or token), "events_processed": 0}

    async def rollout_insight(self, token: str, body: dict[str, Any]) -> dict[str, Any]:
        """Acknowledge an insight post. SolSwarm's publish_insight.py only reads ``id``."""
        return {"id": uuid4().hex}

    async def rollout_bug_report(self, token: str, body: dict[str, Any]) -> JSONResponse:
        """Acknowledge a bug report. SolSwarm's report_bug.py expects HTTP 201 with an ``id``."""
        return JSONResponse(status_code=201, content={"id": uuid4().hex})

    async def rollout_operation(self, token: str, op_id: str) -> dict[str, Any]:
        """Report the status of one background submission evaluation. Polled by submit.py."""
        op = (self._rollouts.get(token) or {}).get("ops", {}).get(op_id)
        if op is None:
            return {"status": "failed", "error": {"message": "unknown operation (rollout expired?)"}}
        if op["status"] == "succeeded":
            return {"status": "succeeded", "result": op["result"]}
        if op["status"] == "failed":
            return {"status": "failed", "error": {"message": op["error"] or "evaluation failed"}}
        return {"status": "running"}

    # ----- verify -----------------------------------------------------------------

    async def verify(self, body: CudaGymVerifyRequest) -> CudaGymVerifyResponse:
        """Score the rollout and return the reward.

        In solswarm_submit mode the reward comes from the submissions this
        server recorded for the rollout: the best one under ``reward_mode:
        best_submission``, the last one under ``final_submission``. Scoring the
        final kernel file from the trajectory is the fallback when nothing was
        submitted, and only if ``final_file_fallback`` allows it. Rollouts with
        no scorable submission earn 0.

        For a registered rollout the problem scored is the one recorded at
        /begin; the request's own ``verifier_metadata`` never redefines it,
        because the sandboxed agent knows its token and can reach this
        endpoint itself.

        The verdict is computed once per rollout, cached on its entry (the
        HTTP layer replays POSTs on disconnect), and guarded by a per-rollout
        lock; the TTL sweep reclaims the entry.
        """
        # The agent forwards its submission token in solswarm_submit mode; absent
        # otherwise. It is an undeclared extra field, so it lives in model_extra
        # (same access as resources_servers/gymnasium/base.py) — no need to dump
        # the whole trajectory just to read one key.
        token = (body.model_extra or {}).get("submission_token")
        entry = self._rollouts.get(token) if token else None
        if entry is None:
            return await self._verify_rollout(body, token, None)
        # Single-flight: a concurrent /verify must wait for the first one
        # rather than race its drain and score a partial submission list.
        async with entry["verdict_lock"]:
            if entry["verdict"] is not None:
                return entry["verdict"]
            return await self._verify_rollout(body, token, entry)

    async def _verify_rollout(
        self, body: CudaGymVerifyRequest, token: Optional[str], entry: Optional[dict[str, Any]]
    ) -> CudaGymVerifyResponse:
        """Compute a rollout's verdict. Called with the entry's lock held, if it has one."""
        meta = body.verifier_metadata or {}
        dumped = body.model_dump()
        if entry is not None:
            # The problem registered at /begin governs. The sandboxed agent
            # knows its own token and can reach this endpoint, so metadata from
            # the request must never redefine what gets scored or cached as the
            # verdict.
            meta = entry["meta"]
        # Set below when submission evaluations were lost to infrastructure failures.
        infra_error: Optional[str] = None

        def _resp(reward: float, info: dict[str, Any], **extras: Any) -> CudaGymVerifyResponse:
            """Assemble the verify response and cache it as the rollout's verdict."""
            if extras.get("evaluation_error") is None:
                # Surface infrastructure losses and the scoring path's own error
                # (never submitted, expired state, malformed row, bad code) so
                # downstream can tell them apart from a plain incorrect kernel.
                parts = [p for p in (infra_error, info.get("error")) if p]
                extras["evaluation_error"] = "; ".join(parts) if parts else None
            response = CudaGymVerifyResponse(
                **dumped,
                reward=float(reward),
                compiled=bool(info.get("compiled", False)),
                executed=bool(info.get("executed", False)),
                correctness=bool(info.get("correctness", False)),
                speedup=float(info.get("speedup", -1.0)),
                sol_score=float(info.get("sol_score", -1.0)),
                human_best_speedup=float(info.get("human_best_speedup", -1.0)),
                **extras,
            )
            if entry is not None:
                entry["verdict"] = response
            return response

        # solswarm_submit mode: close the entry so no new submission can start
        # while the verdict is being computed, wait out the still-running
        # submission evaluations, then score the recorded ones.
        if entry is not None:
            entry["closed"] = True
            # Drain in a loop: each pass empties the task list before awaiting,
            # so a task appended in the same window is picked up by the next
            # pass rather than abandoned.
            while entry["tasks"]:
                pending, entry["tasks"] = entry["tasks"], []
                await asyncio.gather(*pending, return_exceptions=True)
            failed_ops = [op for op in entry["ops"].values() if op["status"] == "failed"]
            # After the drain, every evaluation task has marked its operation
            # succeeded or failed; one still "running" lost its task (cancelled
            # mid-flight) and never recorded a submission, which is an
            # infrastructure loss, not a kernel outcome.
            lost_ops = [op for op in entry["ops"].values() if op["status"] == "running"]
            if failed_ops or lost_ops:
                detail = failed_ops[-1]["error"] if failed_ops else "evaluation task lost before recording a result"
                infra_error = (
                    f"{len(failed_ops) + len(lost_ops)} of {len(entry['ops'])} submissions lost to "
                    f"evaluation-infrastructure failures: {detail}"
                )
            submissions = entry["submissions"]
            # Submissions were recorded: the reward comes from the one reward_mode picks.
            if submissions:
                if self.config.reward_mode == "final_submission":
                    chosen = submissions[-1]
                else:
                    chosen = max(submissions, key=lambda r: r["reward"])
                return _resp(
                    chosen["reward"],
                    chosen["info"],
                    n_submissions=len(submissions),
                    best_submission_reward=max(r["reward"] for r in submissions),
                    final_submission_reward=submissions[-1]["reward"],
                )
            # Nothing was submitted and the fallback is off: production behavior, reward 0.
            if not self.config.final_file_fallback:
                return _resp(0.0, {"error": "no submitted kernel (final_file_fallback disabled)"}, n_submissions=0)
        elif token is not None and not self.config.final_file_fallback:
            # The rollout registered submission state but the TTL sweep expired
            # it before /verify arrived. With the fallback disabled there is
            # nothing scoreable left; falling through to final-file scoring
            # would bypass the very setting that disabled it.
            return _resp(
                0.0,
                {"error": "rollout submission state expired before verify (submission_ttl_seconds)"},
                n_submissions=0,
            )

        # Final-file path: no recorded submission to score (final_file mode, an expired
        # rollout entry, or an empty submission list with the fallback enabled).
        # Parse the problem and the submitted code; a missing or unparseable
        # submission scores 0.
        # In final_file mode nothing registered the rollout, so this is where a
        # row for other silicon is caught.
        row_sku_error = self._row_sku_error(meta)
        if row_sku_error:
            return _resp(0.0, {"error": row_sku_error}, n_submissions=0)
        try:
            language = meta["language"]
            # Validated here so an unknown language lands in this malformed-row
            # path instead of escaping later as a bare KeyError.
            if language not in LANGUAGE_DEFAULTS:
                raise ValueError(f"unknown language {language!r}; expected one of {list(LANGUAGE_DEFAULTS)}")
            definition = Definition.model_validate(meta["definition"])
            workloads = [Workload.model_validate(w) for w in meta["workloads"]]
            target_hardware = self._row_sku(meta)
            # Hard-indexed: the dataset builders always write the key, and a
            # wrong guess here would fail every workload as the model's error.
            dps = meta["destination_passing_style"]
        except Exception as e:  # noqa: BLE001 - malformed task row
            LOG.warning("cudagym verify: bad verifier_metadata: %s", e)
            return _resp(0.0, {"error": f"bad verifier_metadata: {e}"}, n_submissions=0)

        code = extract_submitted_code(body.response, language)
        if not code:
            return _resp(0.0, {"error": "no submitted kernel found in trajectory"}, n_submissions=0)

        try:
            solution = build_solution(
                code, language, definition.name, target_hardware, dps, author="nemo_gym_cuda_agent"
            )
        except Exception as e:  # noqa: BLE001 - invalid Solution == format failure
            return _resp(0.0, {"error": f"invalid solution: {e}"}, n_submissions=0)

        # Canonical evaluation of the kernel extracted from the trajectory, on
        # the endpoint serving the row's GPU.
        reward, info = await self._eval_and_reward(
            solution, definition, workloads, meta.get("sol_anchors"), target_hardware
        )
        return _resp(reward, info, n_submissions=0)


if __name__ == "__main__":
    CudaGymResourcesServer.run_webserver()
