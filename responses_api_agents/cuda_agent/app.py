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

"""OpenCode agent for CudaGym / KernelFactory kernel optimization (agentic).

A thin specialization of ``opencode_agent``. ``run()`` owns the orchestration
(it alone sees ``verifier_metadata``) and does four things:

  1. Seed the per-rollout problem directory from the row's ``verifier_metadata``
     (``problem/definition.json`` + ``problem/workload.jsonl``). The sandbox's
     ``cudagym`` CLI is pointed at the evaluation endpoint for the row's own GPU
     (``cudagym_endpoints``, else ``CUDAGYM_UNIFIED_SERVER_URL``) and inherits
     ``CUDAGYM_AUTH_TOKEN``, so the model gets its own ``cudagym evaluate``
     feedback.
  2. Run one agent session over it (two profiles below).
  3. Read the final kernel file the model wrote (``<kernel_filename>``) and
     append it as a closing assistant message (a fenced code block) so the
     resources server can score the *submitted* kernel.
  4. Build the training output (see TOKEN CAPTURE below) and POST it to the
     ``cudagym`` resources server's ``/verify`` for the canonical, staged reward.

SANDBOX PROFILES (``sandbox_profile``):
  * ``minimal``: a bare ``problem/`` sandbox plus a short contract prompt
    (``_build_prompt``), run with our own ``opencode run`` invocation against
    the policy endpoint. Optionally each rollout runs inside a private mount
    namespace (``mount_namespace`` / ``hide_job_mounts``) so the model sees
    production's literal paths. The subprocess environment is built from
    ``_SANDBOX_ENV_ALLOWLIST`` instead of being inherited, so the training
    job's unrelated credentials never reach the model.
  * ``solswarm``: container mode. Each rollout runs inside a real instance of
    the published SolSwarm agent image, extracted once per node and entered
    through the host's enroot runtime (see ``solswarm_container.py``). The
    image's own ``entrypoint.sh`` installs the skills, stitches the submission
    rules, renders the role prompt, writes the OpenCode config and launches
    the agent itself; we contribute only the environment variables the
    Kubernetes pod controller would inject (``build_env``, delivered as a
    sourced file because ``enroot start`` resets the calling environment).
    The policy model reaches the entrypoint through its registry machinery: a
    synthetic model catalog and endpoint registry written into the rollout
    tree (``solswarm_container.write_registry_catalog_files``) declare one
    registered endpoint whose ``agent_base_url`` is the per-rollout tokened
    policy URL, and the contract points the entrypoint at those files
    (``registry_contract_env``); the entrypoint then configures OpenCode to
    dial that URL directly. Requires ``enroot``, ``gawk``, an agent-image
    squashfs, and a ``model_server``, all validated at server startup, because
    there is no fallback implementation.

``submission_mode: solswarm_submit`` (either profile) gives the sandbox
SolSwarm's real ``/submit`` skill, pointed at the cudagym resources server's
``/rollout/<token>/api/v1`` endpoints. Every submission is canonically
evaluated and recorded server-side, out of the model's reach; ``verify()``
rewards the best (or final) recorded submission and falls back to the final
kernel file when nothing was submitted (both configurable server-side).

SANDBOX CHECK (once per job, not per rollout): after its agent exits, the
first rollout of a job records every file path visible inside its sandbox
(the listing script, ``sandbox_manifest.PROBE_SOURCE``). In container mode
that listing is compared for set equality against a reference listing taken
from a sandbox built by the image's own entrypoint with no RL wiring
(``solswarm_container.reference_tree``, cached per solswarm checkout and
per agent image). The verdict is
printed once and both listings land in
``$CUDA_AGENT_MANIFEST_DIR/sandbox_tree.json``.

TRAINING / TOKEN CAPTURE: when a ``model_server`` is configured (the training
overlay) and this agent opts into capture (``token_id_capture: true`` plus the
run-level ``token_id_capture.enabled`` block), OpenCode is pointed at
``<model_server>/ng-rollout/<rollout_id>/training-token-capture/v1``
(``resolve_model_base_url``). The model server's capture middleware records
each call's exact ``prompt/generation_token_ids``/``log_probs`` durably under
that rollout id (``nemo_gym.token_id_capture``); rollout collection rebuilds
the on-policy trajectory afterwards (``finalize_rollout_token_capture``) and
replaces ``response.output`` on the finished record — this agent does NOT
splice tokens itself. OpenCode needs no modification. With no ``model_server``
(eval), the output is the session reconstructed from OpenCode's sqlite
database (``parse_opencode_session``), which has no token ids.

Compaction MUST stay off during training: a mid-run history rewrite silently
drops all post-compaction turns from the token-id trajectory.
``cudagym_cuda_agent.yaml`` therefore disables auto-compaction and sizes
``limit.context`` to the vLLM window.
"""

import asyncio
import copy
import importlib.util
import itertools
import json
import logging
import os
import shlex
import shutil
import signal
import sqlite3
import tempfile
from pathlib import Path
from time import time
from typing import Any, Literal, Optional
from uuid import uuid4

import aiohttp
from cudagym.rl import (
    canonical_sku,
    entry_point_for,
    entry_symbol_for,
    fence_lang_for,
    kernel_filename_for,
    resolve_row_sku,
)
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.cuda_agent import sandbox_manifest, solswarm_container, solswarm_surface
from responses_api_agents.opencode_agent.app import (
    OpenCodeAgent,
    OpenCodeAgentConfig,
    _extract_instruction,
    parse_opencode_session,
)


LOG = logging.getLogger(__name__)


class SolswarmSurfaceConfig(BaseModel):
    """Where the SolSwarm checkout lives, and the campaign settings its entrypoint reads."""

    # Path to a SolSwarm checkout. None means read the SOLSWARM_SURFACE_ROOT
    # env var (nemo-rl's launch script points it at its 3rdparty/solswarm
    # submodule).
    root: Optional[str] = None
    # SolSwarm's two hard bans plus the per-campaign banned libraries. They are
    # forwarded to the entrypoint (which keeps or strips the matching prompt
    # sections; submit.py reads the same variables for its precompliance gate)
    # and stitched into problem/submission_rule.md.
    ban_framework_kernels: bool = True
    ban_cuda_graphs: bool = True
    banned_libraries: list[str] = Field(default_factory=list)


class CudaAgentConfig(OpenCodeAgentConfig):
    """``OpenCodeAgentConfig`` plus the kernel-sandbox and SolSwarm-parity knobs."""

    # Optional Gym model server (re-declared from OpenCodeAgentConfig for the
    # startup validation below). When set, OpenCode is pointed at the
    # rollout-prefixed model URL (``resolve_model_base_url``) so the capture
    # middleware records per-call token ids for TRAINING; rollout collection
    # rebuilds the trajectory afterwards. When None (eval), OpenCode uses the
    # static provider baseURL and the output stays the reconstructed
    # (no-token-id) session.
    model_server: Optional[ModelServerRef] = None
    # Keep the per-rollout sandbox OUT of the Gym git tree (base default is a
    # repo-relative path). OpenCode walks up from the workspace to the git
    # worktree root discovering `.claude`/`.agents` skills and CLAUDE.md/AGENTS.md
    # instruction files, so a workspace inside the repo leaks our dev skills and
    # docs into the agent's context; production runs in a bare container dir.
    # Absolute -> used as-is (not resolved under the server's cwd).
    workspace_root: str = Field(default_factory=lambda: str(Path(tempfile.gettempdir()) / "nemo_gym_cuda_agent"))
    # Extra guidance appended to the row prompt describing the sandbox contract
    # (minimal profile only).
    sandbox_instructions: Optional[str] = None
    # MINIMAL-PROFILE knob (container mode needs none — the agent image
    # supplies the whole filesystem):
    # run each rollout in its own mount namespace so it sees production's paths
    # (cwd=/workspace, problem at /tmp/problem) instead of our per-rollout temp
    # dir. Several rollouts share one training container here, where production
    # gives each agent its own container. The training cluster runs jobs as
    # root with CAP_SYS_ADMIN inside the enroot container, so `unshare` works
    # there; when `unshare` is missing this falls back to a plain launch.
    mount_namespace: bool = False
    # Hide the training job's bind mounts inside the minimal profile's
    # namespace (container mode needs no hiding — the image rootfs simply does
    # not contain the job mounts): each directory in
    # sandbox_manifest.HIDEABLE_JOB_MOUNTS — datasets/checkpoints/models, which
    # hand the agent the task's own answer keys — is over-mounted with an empty
    # directory. The /opt/nemo-rl code tree is hidden as well, which is safe
    # because the cudagym CLI is fed a staged package copy via PYTHONPATH
    # (_stage_cudagym_cli) instead of the venv's editable pointer into that tree.
    hide_job_mounts: bool = False
    # --variant sent to the model endpoint in container mode. The entrypoint
    # always passes one. build_env drops empty values, so "" leaves
    # SOLSWARM_OPENCODE_VARIANT unset and the entrypoint applies the synthetic
    # catalog entry's default effort ("high", set by
    # solswarm_container.write_registry_catalog_files; entrypoint.sh
    # default_opencode_reasoning_effort reads it); a non-empty value overrides
    # that.
    solswarm_reasoning_variant: str = ""
    # minimal: bare problem/ sandbox + short contract prompt, our own
    #   `opencode run` invocation.
    # solswarm: container mode — each rollout runs inside a real instance of
    #   the published agent image, whose entrypoint builds the sandbox and
    #   launches the agent itself (see solswarm_container.py). Requires
    #   `enroot`, `gawk`, an agent image, and a model_server; validated at
    #   startup.
    sandbox_profile: Literal["minimal", "solswarm"] = "minimal"
    # Agent-image squashfs for container mode (each rollout runs inside a real
    # instance of the published agent image via the host's enroot runtime);
    # null -> the CUDA_AGENT_ENROOT_IMAGE environment variable.
    enroot_image: Optional[str] = None
    # Test/debug: a host entrypoint.sh bind-mounted over the image's own, so a
    # modified entrypoint can be exercised without rebuilding the agent image.
    entrypoint_override: Optional[str] = None
    # Test/debug: persist every rollout's entrypoint transcript to the manifest
    # dir, not only failed ones — a session that exits 0 with an empty trace
    # leaves no other evidence of what the agent actually did.
    keep_transcripts: bool = False
    # final_file: the submission is the kernel file left in the sandbox.
    # solswarm_submit: SolSwarm's /submit skill posts solution.json bundles to the
    # resources server, which evaluates + records them; best/final one is scored.
    submission_mode: Literal["final_file", "solswarm_submit"] = "final_file"
    # One CudaGym evaluation endpoint per GPU, keyed by the CudaGym
    # SupportedHardware value: {"B200": "https://...", "H100": "..."}. Each
    # rollout's sandbox is given the endpoint for its OWN row's
    # target_hardware, so the in-sandbox `cudagym evaluate` CLI measures on the
    # GPU whose anchors the row is scored against. An entry with an empty URL
    # resolves from CUDAGYM_UNIFIED_SERVER_URL at runtime, which is how
    # in-allocation hosting passes a load-balancer address that does not exist
    # when the job is submitted; an empty map means every row uses that
    # variable, the single-endpoint case.
    cudagym_endpoints: dict[str, str] = Field(default_factory=dict)
    solswarm_surface: SolswarmSurfaceConfig = Field(default_factory=SolswarmSurfaceConfig)


class CudaAgentRunRequest(BaseRunRequest):
    """``/run`` request row; ``verifier_metadata`` carries the KernelFactory problem."""

    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class CudaAgentVerifyResponse(BaseVerifyResponse):
    """The cudagym server's ``/verify`` response plus the assistant-turn count."""

    model_config = ConfigDict(extra="allow")
    turns_used: int = 0


# The staged inputs rarely change but rollouts are many: write the full
# manifest to a FILE once per distinct input set (keyed by its digest, and out
# of the training data) and print a summary once per process. The manifest
# lands in $CUDA_AGENT_MANIFEST_DIR, else the workspace root (grpo.sh points
# the env var at the job's output dir).
_MANIFEST_LOGGED = False
_MANIFEST_WRITTEN: set[str] = set()
# The sandbox equivalence check runs ONCE per job: the first rollout to start
# claims it, and the claim is released — so a later rollout retries — when
# that rollout produces no file listing, fails before publishing, or (in
# container mode) gets no comparison verdict because the reference sandbox
# could not be built. The sandbox is deterministic given the image and the
# checkout, so one completed check per server process is a sufficient signal.
_SANDBOX_CHECKED = False
# The model-liveness probe logs its success line once per process; every
# rollout still runs the probe (it is one 1-token request when healthy).
_MODEL_LIVE_LOGGED = False
# One extracted agent-image container per process, shared read-only by every
# rollout; the lock serializes the first extraction.
_ENROOT_BASE: Optional[tuple[str, Path]] = None
_ENROOT_BASE_LOCK = asyncio.Lock()
# Sandbox latency samples by kind (startup per runtime, enroot image seeding)
# feeding the periodic summary log line.
_STARTUP_SAMPLES: dict[str, list[float]] = {}


def _record_sandbox_latency(kind: str, seconds: float, rollout_id: str) -> None:
    """Append one latency sample and print it, with a summary every 8 samples.

    The line is printed rather than logged because Gym leaves the root logger
    at WARNING.
    """
    samples = _STARTUP_SAMPLES.setdefault(kind, [])
    samples.append(seconds)
    line = f"[cuda_agent] {kind}: {seconds:.2f}s (rollout={rollout_id}, n={len(samples)})"
    if len(samples) % 8 == 0:
        ordered = sorted(samples)
        line += (
            f" | min={ordered[0]:.2f} median={ordered[len(ordered) // 2]:.2f}"
            f" mean={sum(ordered) / len(ordered):.2f} max={ordered[-1]:.2f}"
        )
    print(line, flush=True)


# Tail the OUTER subprocess wait adds on top of the configured timeout in
# container mode, covering what runs outside the entrypoint's inner
# `timeout $SOLSWARM_MAX_DURATION` clock (container boot, skills install, the
# trace-upload tail, the once-per-job sandbox listing) — so the inner,
# graceful timeout fires first and the outer group-kill stays a backstop.
CONTAINER_TAIL_SECONDS = 300
# Grace between the group SIGTERM and the group SIGKILL — parity with
# solswarm's `timeout --signal=TERM --kill-after=30s`, giving opencode a
# chance to flush its session db.
_KILL_GRACE_SECONDS = 30


def _outer_wait_budget(timeout: int, container_mode: bool) -> int:
    """Wall-clock ceiling for the launched subprocess.

    In container mode the configured timeout is enforced INSIDE the container
    (build_env exports it as SOLSWARM_MAX_DURATION), so the outer wait gets
    CONTAINER_TAIL_SECONDS on top; the minimal profile has no inner budget and
    keeps the configured timeout as its only one.
    """
    return timeout + (CONTAINER_TAIL_SECONDS if container_mode else 0)


def _signal_process_group(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """Signal the subprocess's whole process group, tolerating an already-dead one.

    The launch uses ``start_new_session=True``, so ``proc.pid`` is the process
    group id of every descendant. Signalling only the top process
    (``proc.terminate()``) is not enough here: under `enroot start` the
    in-container rc shell backgrounds ``bash /entrypoint.sh``, and a
    reparented entrypoint would keep generating against the model server after
    the trajectory was retrieved, race the rollout tree's removal, and hold
    the OpenCode session db locked.
    """
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError) as e:
        LOG.debug("process group %s not signalled (%s)", proc.pid, e)


async def _terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM the subprocess's process group, escalating to SIGKILL after a grace period.

    Used on both the outer-timeout and the request-cancellation paths. When
    the grace wait is itself cancelled, the SIGKILL is delivered immediately
    before re-raising, so no cancellation ordering can leave the tree running.
    """
    _signal_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.communicate(), timeout=_KILL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        _signal_process_group(proc, signal.SIGKILL)
        await proc.communicate()
    except asyncio.CancelledError:
        _signal_process_group(proc, signal.SIGKILL)
        raise


# Production's agent-container paths (docker/Dockerfile.agent WORKDIR /workspace;
# entrypoint.sh PROBLEM_DIR=/tmp/problem). With mount_namespace on, each rollout
# is bind-mounted onto these inside its own mount namespace, so every rollout
# sees production's layout despite sharing one container.
_NS_WORKSPACE = "/workspace"
_NS_PROBLEM = "/tmp/problem"
# Production installs skills at the agent user's XDG config path and the
# rendered prompt references it literally (entrypoint.sh runtime instructions).
_NS_XDG_CONFIG = "/home/agent/.config"
_NS_SKILLS = "/home/agent/.config/opencode/skills"

# Environment variables the minimal profile's sandbox may inherit from the
# agent server. The server runs inside the training job, so inheriting
# everything would hand the model the job's unrelated credentials (Weights &
# Biases, Hugging Face, cluster tokens). Variables the harness sets itself
# come after this filter and need no entry. Container mode inherits nothing:
# `enroot start` resets the environment and build_env writes the contract.
_SANDBOX_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Basics any child process needs to start and run.
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        # Where OpenCode reads its config and skills and writes its data and cache.
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        # The policy endpoint OpenCodeAgent._env fills in from the agent config.
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        # Reaching the model server and the evaluation service: proxy settings
        # and the certificate bundles a TLS-terminating proxy needs.
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        # The staged cudagym CLI package copy is reached through PYTHONPATH.
        "PYTHONPATH",
        # The local CUDA toolchain and device selection the cudagym tooling reads.
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "CUTLASS_DIR",
        # submit.py's evaluation timeout and poll budget.
        "KF_EVAL_TIMEOUT",
    }
)
# Whole families the in-sandbox cudagym CLI reads: its server URL, auth token
# and behaviour settings, plus both halves of Modal's proxy token, which its
# edge proxy requires on every evaluation request.
_SANDBOX_ENV_PREFIXES: tuple[str, ...] = ("CUDAGYM_", "MODAL_PROXY_TOKEN_")

# Once-per-process stage of the cudagym CLI package copy (see _stage_cudagym_cli).
_CLI_STAGE: dict[str, Optional[Path]] = {}


def _stage_cudagym_cli(dest_root: Path) -> Optional[Path]:
    """Copy the cudagym package out of the repo tree so the CLI survives hiding it.

    The job installs cudagym EDITABLE: the venv's ``cudagym`` console script
    resolves ``import cudagym`` through a pointer into ``/opt/nemo-rl``, so
    over-mounting that tree breaks the CLI with "No module named 'cudagym'".
    A plain copy of the package (~1.4 MB, pure python) plus a PYTHONPATH entry
    restores it: PYTHONPATH precedes ``.pth`` entries on ``sys.path``, while
    the real console script and the job venv's dependencies keep being used.
    The model can read this copy, which matches production's exposure (its
    agent image has cudagym pip-installed with site-packages readable).

    Returns the directory to prepend to PYTHONPATH, or None (with a warning) if
    the package can't be located — callers must then leave /opt/nemo-rl visible.
    """
    key = str(dest_root)
    if key in _CLI_STAGE:
        return _CLI_STAGE[key]
    staged: Optional[Path] = None
    try:
        spec = importlib.util.find_spec("cudagym")
        if spec is None or not spec.origin:
            raise ModuleNotFoundError("cudagym not importable in the agent server env")
        package_dir = Path(spec.origin).parent
        dest = Path(dest_root) / "cudagym_cli"
        target = dest / "cudagym"
        if not (target / "__init__.py").is_file():
            shutil.copytree(package_dir, target, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        staged = dest
    except Exception as e:  # noqa: BLE001 - degrade to not hiding the tree
        LOG.warning("could not stage the cudagym CLI package copy: %s", e)
    _CLI_STAGE[key] = staged
    return staged


def _remove_rollout_trees(*roots: Optional[Path]) -> None:
    """Remove the given directory trees, skipping None entries and ignoring errors.

    All removals happen in one function so the caller can run them in a single
    ``asyncio.to_thread`` call: with one thread hop per tree, a cancellation
    delivered at the first await would abandon the remaining removals.
    """
    for root in roots:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)


# Every name the cudagym CLI reads for a server address, in the order it
# resolves them: the per-service pair wins over the unified pair.
CUDAGYM_URL_ENV_NAMES = (
    "CUDAGYM_COMPILE_SERVER_URL",
    "CUDAGYM_GPU_SERVER_URL",
    "CUDAGYM_UNIFIED_SERVER_URL",
    "CUDAGYM_URL",
)


def _pin_cudagym_urls(env: dict[str, str], url: str) -> None:
    """Point every CudaGym address variable in ``env`` at this row's endpoint.

    The model's own ``cudagym evaluate`` has to measure on the silicon its
    kernel is scored on. All four names are set rather than just the two
    unified ones: the sandbox environment inherits the whole ``CUDAGYM_``
    family, the CLI resolves the per-service pair FIRST, and an inherited pair
    would therefore outrank the row's endpoint and send every row's
    measurements to one server whatever GPU it declared. (Production exports
    CUDAGYM_URL and the role prompts pass ``--server $CUDAGYM_URL``, so
    commands written that way keep working. Container mode needs none of this:
    ``enroot start`` resets the environment and the contract is build_env's
    CUDAGYM_URL.)
    """
    if not url:
        return
    for name in CUDAGYM_URL_ENV_NAMES:
        env[name] = url


class CudaAgent(OpenCodeAgent):
    """OpenCode harness over a seeded CudaGym kernel sandbox (agentic kernel RL)."""

    config: CudaAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Refuse at startup when container mode cannot actually run."""
        super().model_post_init(__context)
        # The keys route rows by target_hardware, so a miscased or aliased one
        # ("b200", "GB10") would silently match no row and refuse every rollout
        # for that GPU. Same check the resources server makes on its own map,
        # so the two cannot disagree about what a GPU is called.
        for sku in self.config.cudagym_endpoints:
            canonical_sku(sku, "a key of the cuda_agent's `cudagym_endpoints`")
        # The solswarm profile IS container mode — SolSwarm's real entrypoint
        # inside a per-rollout instance of the published agent image. There is
        # no fallback implementation, so refuse at startup instead of training
        # against a different sandbox.
        if self.config.sandbox_profile == "solswarm":
            problems = []
            # Each rollout runs inside the extracted agent image; the enroot
            # launcher and its gawk dependency must be reachable in THIS
            # process's environment (bind-mounted from the host by the job
            # script), and the image must exist.
            for tool in ("enroot", "gawk"):
                if shutil.which(tool) is None:
                    problems.append(f"container mode needs `{tool}` on PATH")
            raw_image = self.config.enroot_image or os.environ.get("CUDA_AGENT_ENROOT_IMAGE") or ""
            if not raw_image:
                problems.append("no agent image is configured (set enroot_image or CUDA_AGENT_ENROOT_IMAGE)")
            elif self._enroot_image() is None:
                problems.append(f"the agent image does not exist: {raw_image}")
            if self.config.model_server is None:
                problems.append(
                    "no model_server is configured (the registry catalog files route the model "
                    "to the capture-prefixed <model_server>/ng-rollout/<id>/.../v1 endpoint)"
                )
            # The contract's CUDAGYM_URL comes from cudagym_endpoints, or from
            # this variable when the map is empty or the row's entry is, and
            # build_env drops empty values: without a resolvable address the
            # in-image cudagym CLI silently falls back to its hardcoded public
            # default server mid-episode instead of the job's evaluation
            # service.
            endpoints = self.config.cudagym_endpoints
            if not os.environ.get("CUDAGYM_UNIFIED_SERVER_URL") and not (endpoints and all(endpoints.values())):
                problems.append(
                    "CUDAGYM_UNIFIED_SERVER_URL is unset or empty and cudagym_endpoints does not give "
                    "every GPU an address, so CUDAGYM_URL would drop out of the container env contract"
                )
            if problems:
                raise RuntimeError(
                    "sandbox_profile=solswarm runs SolSwarm's real entrypoint in a per-rollout "
                    "sandbox and cannot start: " + "; ".join(problems)
                )

    def _cudagym_url(self, meta: dict[str, Any]) -> str:
        """The evaluation endpoint this row's sandbox is given as ``CUDAGYM_URL``.

        The in-sandbox ``cudagym evaluate`` CLI has to reach the endpoint whose
        GPU the row declares, so the model's own measurements come from the
        silicon its kernel is scored on. With no ``cudagym_endpoints``
        configured every row uses ``CUDAGYM_UNIFIED_SERVER_URL``, and a
        configured entry with an empty URL falls back to the same variable.

        A row whose GPU has no entry is refused rather than pointed at another
        GPU's endpoint, and so is a row with no ``target_hardware`` when
        several GPUs are configured, because there is then nothing to choose
        by. One configured GPU makes such a row unambiguous.
        """
        endpoints = self.config.cudagym_endpoints
        unified = os.environ.get("CUDAGYM_UNIFIED_SERVER_URL", "")
        if not endpoints:
            return unified
        return endpoints[self._row_sku(meta)] or unified

    def _row_sku(self, meta: dict[str, Any]) -> str:
        """The GPU a row runs against (the shared ``cudagym.rl`` resolver).

        The refusals — an unconfigured GPU, or a row naming none when several
        are configured — propagate as ``ValueError`` and fail the /run:
        guessing would point the sandbox at one GPU while telling the model it
        is optimizing for another.
        """
        return resolve_row_sku(meta.get("target_hardware"), self.config.cudagym_endpoints)

    def _enroot_image(self) -> Optional[Path]:
        """The agent-image squashfs for the enroot runtime, if one is configured and exists."""
        raw = self.config.enroot_image or os.environ.get("CUDA_AGENT_ENROOT_IMAGE") or ""
        path = Path(raw).expanduser() if raw else None
        return path if path is not None and path.is_file() else None

    def _enroot_data_root(self) -> Path:
        """Node-local root for the inner enroot's runtime/cache/data/tmp trees."""
        return Path(os.environ.get("CUDA_AGENT_ENROOT_DATA") or "/tmp/cuda-agent-enroot")

    async def _ensure_enroot_base(self) -> tuple[str, Path]:
        """Extract the agent image once per process; log the one-time cost."""
        global _ENROOT_BASE
        async with _ENROOT_BASE_LOCK:
            if _ENROOT_BASE is None:
                image = self._enroot_image()
                assert image is not None  # validated at startup
                name, rootfs, seconds = await asyncio.to_thread(
                    solswarm_container.ensure_enroot_base, image, self._enroot_data_root()
                )
                if seconds is not None:
                    print(f"[cuda_agent] enroot base extract: {seconds:.1f}s ({image} -> {name})", flush=True)
                _ENROOT_BASE = (name, rootfs)
            return _ENROOT_BASE

    def _log_sandbox_startup(self, rollout_dir: Path, launch_wall: float, rollout_id: str) -> None:
        """Report exec-to-in-sandbox latency from the CONTAINER_START_STAMP file.

        The stamp is the first act of the in-container launch script, so the
        delta measures exactly `enroot start`'s overhead.
        """
        try:
            stamp = float((rollout_dir / solswarm_container.CONTAINER_START_STAMP).read_text().strip())
        except (OSError, ValueError):
            print(f"[cuda_agent] sandbox startup: no stamp (rollout={rollout_id})", flush=True)
            return
        _record_sandbox_latency("sandbox startup (enroot)", stamp - launch_wall, rollout_id)

    def _env(self, data_home: str) -> dict[str, str]:
        """Build the agent subprocess environment, keeping only the allowlisted variables.

        ``OpenCodeAgent._env`` starts from this process's whole environment. Here
        that process runs inside the training job, so the base result carries the
        job's unrelated credentials into the model's sandbox. This keeps three
        groups from it: the names in ``_SANDBOX_ENV_ALLOWLIST``, the families in
        ``_SANDBOX_ENV_PREFIXES``, and the names the agent config declares under
        ``env`` (an operator asking for a variable is asking for it in the
        sandbox). Everything else is dropped.
        """
        env = super()._env(data_home)
        return {
            k: v
            for k, v in env.items()
            if k in _SANDBOX_ENV_ALLOWLIST or k.startswith(_SANDBOX_ENV_PREFIXES) or k in self.config.env
        }

    def _seed_problem(self, work_dir: Path, meta: dict[str, Any]) -> None:
        """Write ``problem/{definition.json,workload.jsonl}`` for the model + cudagym CLI."""
        problem_dir = work_dir / "problem"
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / "definition.json").write_text(json.dumps(meta["definition"], indent=2))
        with open(problem_dir / "workload.jsonl", "w") as f:
            for workload in meta["workloads"]:
                f.write(json.dumps(workload) + "\n")

    def _build_prompt(self, user_message: str, system_message: Optional[str], meta: dict[str, Any]) -> str:
        """Row prompt + the sandbox contract (problem location, submission file, eval CLI)."""
        language = meta["language"]
        # The GPU named in the sandbox contract has to be the one the kernel is
        # compiled for and timed on; a default here would tell the model to
        # optimize for hardware the evaluation never uses.
        sku = self._row_sku(meta) if self.config.cudagym_endpoints else meta["target_hardware"]
        # A language outside the shared table raises rather than defaulting to
        # kernel.py: the model would write a file the resources server never
        # reads, and the rollout would be scored 0 as though it produced nothing.
        filename = kernel_filename_for(language)
        if filename.endswith(".py"):
            eval_cmd = (
                f"    cudagym evaluate --solution {filename} --definition ./problem/definition.json "
                f"--workload ./problem/workload.jsonl --gpu {sku} --json\n"
            )
        else:
            # The CLI only auto-wraps .py files into a Solution; C++-family kernels
            # must ride a solution.json bundle that carries spec/target_hardware.
            eval_cmd = (
                f"    cudagym evaluate --solution solution.json --definition ./problem/definition.json "
                f"--workload ./problem/workload.jsonl --json\n"
                f"  where solution.json = {{name, definition, spec{{languages: ['{language}'], entry_point "
                f"'{entry_point_for(language)}', target_hardware: ['{sku}'], destination_passing_style}}, "
                f"sources: [{{path, content}}]}}\n"
            )
        contract = (
            f"\n\n# Sandbox contract\n"
            f"- Target GPU: {sku}. Kernel language: {language}.\n"
            f"- The problem is in ./problem/ (definition.json + workload.jsonl).\n"
            f"- Write your {language} kernel implementing `{entry_symbol_for(language)}(...)` to ./{filename}.\n"
            f"- Get feedback any time with:\n"
            f"{eval_cmd}"
            f"- Iterate until correct + fast. Your final ./{filename} is your submission."
        )
        if self.config.submission_mode == "solswarm_submit":
            contract += (
                "\n- For official scoring, submit each meaningful improvement: write a KernelFactory "
                "solution.json ({name, definition, spec{languages: [...], entry_point 'file::func', "
                "target_hardware: [...], destination_passing_style}, sources: [{path, content}]}; "
                "a sources-only bundle also works — the server fills the rest in) and run\n"
                '    python3 "$SOLSWARM_SKILLS_DIR"/submit/scripts/submit.py --solution solution.json '
                '--description "<approach>"\n'
                "  Your best submission is scored; the final kernel file is only the fallback."
            )
        if self.config.sandbox_instructions:
            contract += f"\n{self.config.sandbox_instructions}"
        prompt = f"{user_message}{contract}"
        # Same composition as OpenCodeAgent.responses(): config system prompt first,
        # then any row-level system message, then the user prompt.
        system_parts = [p for p in [self.config.system_prompt, system_message] if p]
        system_prompt = "\n\n".join(system_parts)
        return f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    # ----- SolSwarm checkout helpers ------------------------------------------------

    def _surface_root(self) -> Path:
        """Path to the SolSwarm checkout (config ``solswarm_surface.root``, else ``$SOLSWARM_SURFACE_ROOT``)."""
        root = self.config.solswarm_surface.root or os.environ.get("SOLSWARM_SURFACE_ROOT", "")
        path = Path(root).expanduser() if root else None
        if path is None or not (path / "docker" / "agent").is_dir():
            raise RuntimeError(
                "A SolSwarm checkout is required but none was found: set "
                "solswarm_surface.root or SOLSWARM_SURFACE_ROOT to a solswarm repo "
                f"(got {root!r})."
            )
        return path

    def _surface_staging_dir(self) -> Path:
        """Directory under the workspace root where minimal-profile skills are staged."""
        return Path(self.config.workspace_root).expanduser().absolute() / "solswarm_surface"

    def _workspace_root(self) -> Path:
        """Create a fresh per-rollout workspace directory and return it.

        Overrides the base agent's truncated name + exist_ok=True: at RL scale
        (thousands of rollouts per job) a name collision must fail this
        rollout loudly rather than silently merge two live rollouts' trees,
        so the name is the full uuid and mkdir refuses an existing path.
        """
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=False)
        return root

    def _resources_server_base_url(self) -> str:
        """Base URL of the configured cudagym resources server."""
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.resources_server.name)
        return self.server_client._build_server_base_url(cfg)

    def _rollout_model_base_url(self, rollout_id: Optional[str]) -> Optional[str]:
        """OpenCode provider baseURL that routes through the Gym model server.

        ``<model_server>/ng-rollout/<id>[/training-token-capture]/v1`` — the
        server's capture middleware strips the prefix before routing and files
        this rollout's per-call token ids under the id; rollout collection
        rebuilds the on-policy trajectory afterwards. Without a rollout id
        (capture off) this is the model server's plain ``/v1`` root; without a
        ``model_server`` it is None and the static provider config applies.
        """
        if self.config.model_server is None:
            return None
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def _await_model_live(self, rollout_id_for_log: str) -> None:
        """Block until the policy model endpoint completes a 1-token request.

        The engines behind the Gym model proxy sleep between GRPO steps and
        only serve again once the trainer's refit wakes them; at 27B that
        wake+refit window is minutes long, and an agent that fires into it
        hangs silently until the outer backstop (kernelwriter-27b-9: 32/32
        rollouts, zero completed calls, nothing in any log). Probing the SAME
        proxy the agent will use — via the plain ``/v1`` root, so the
        rollout's token capture stays clean — turns that hang into a short
        absorbed wait, or a loud per-rollout error when the serving stack is
        actually broken. The proxy injects the on-policy sampling overrides,
        so the probe passes the server's sampling assert like any agent call.
        """
        base = self._rollout_model_base_url(None)
        if base is None:
            return
        # Mirror exactly what the agent will send: the first model id from the
        # live opencode config. The Gym proxy rewrites the model field to the
        # configured policy model (and serves no /v1/models route — it 404s,
        # kernelwriter-27b-12), so probing with the agent's own alias exercises
        # the same rewrite path.
        model_id = "default"
        for provider in (self.config.opencode_config.get("provider") or {}).values():
            if isinstance(provider, dict) and provider.get("models"):
                model_id = next(iter(provider["models"]))
                break
        budget = float(os.environ.get("CUDA_AGENT_MODEL_LIVE_TIMEOUT", "300"))
        start = time()
        last_err = "no attempt completed"
        global _MODEL_LIVE_LOGGED
        timeout = aiohttp.ClientTimeout(total=30)
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while time() - start < budget:
                try:
                    async with session.post(f"{base}/chat/completions", json=payload) as r:
                        r.raise_for_status()
                        await r.read()
                    if not _MODEL_LIVE_LOGGED:
                        _MODEL_LIVE_LOGGED = True
                        LOG.info(
                            "policy model endpoint %s live after %.1fs (model=%s)",
                            base,
                            time() - start,
                            model_id,
                        )
                    return
                except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, TypeError, ValueError) as e:
                    last_err = repr(e)
                    await asyncio.sleep(5.0)
        raise RuntimeError(
            f"policy model endpoint {base} did not serve a completion within {budget:.0f}s "
            f"(last error: {last_err}) — generation engines asleep or serving misconfigured; "
            f"failing rollout {rollout_id_for_log} instead of hanging the agent."
        )

    def _namespace_argv(
        self,
        argv: list[str],
        work_dir: Path,
        problem_dir: Path,
        hides: Optional[list[str]] = None,
        skills_src: Optional[Path] = None,
        probe_dir: Optional[Path] = None,
    ) -> list[str]:
        """Wrap the opencode invocation in a private mount namespace.

        Binds this rollout's directories onto production's paths, then hides
        each directory listed in ``hides`` by mounting an empty directory over
        it, so the agent cannot read the training job's bind mounts. ORDER
        MATTERS: all binds happen before any hide, because a bind mount stays
        valid when its source path is later shadowed, not before. The prompt is
        passed through the environment rather than the command line: it is tens
        of KB of arbitrary text, and quoting it into a shell string is
        error-prone.
        """
        quoted = " ".join(shlex.quote(a) for a in argv)
        binds = ""
        if skills_src is not None and str(skills_src) != _NS_SKILLS:
            binds += (
                f"mkdir -p {shlex.quote(_NS_SKILLS)}; "
                f"mount --bind {shlex.quote(str(skills_src))} {shlex.quote(_NS_SKILLS)}; "
            )
        hide = "".join(
            f'[ -d {shlex.quote(m)} ] && mount --bind "$_empty" {shlex.quote(m)} 2>/dev/null || true; '
            for m in (hides or [])
        )
        # The once-per-job sandbox check: list the sandbox's file paths after
        # the agent exits, while the namespace (the model's real view) still
        # exists. The agent command is therefore not exec'd (the shell must
        # survive it to run the listing), and its `if` wrapper below keeps
        # `set -e` from aborting before the listing when the agent fails.
        probe = (
            f"python3 {shlex.quote(str(probe_dir))}/sandbox_probe.py "
            f"> {shlex.quote(str(probe_dir))}/sandbox_probe.json 2>/dev/null || true; "
            if probe_dir is not None
            else ""
        )
        script = (
            "set -e; "
            f"mkdir -p {shlex.quote(_NS_WORKSPACE)} {shlex.quote(_NS_PROBLEM)}; "
            f"mount --bind {shlex.quote(str(work_dir))} {shlex.quote(_NS_WORKSPACE)}; "
            f"mount --bind {shlex.quote(str(problem_dir))} {shlex.quote(_NS_PROBLEM)}; "
            + binds
            + '_empty="$(mktemp -d)"; '
            + hide
            + f"cd {shlex.quote(_NS_WORKSPACE)}; "
            + f'if {quoted} "$CUDA_AGENT_PROMPT"; then rc=0; else rc=$?; fi; '
            + probe
            + "exit $rc"
        )
        return ["unshare", "--mount", "--propagation", "private", "bash", "-c", script]

    def _manifest_dir(self) -> Path:
        """Directory for diagnostics artifacts (manifests, sandbox listings, transcripts), created on call.

        ``$CUDA_AGENT_MANIFEST_DIR`` when set (grpo.sh points it at the job's
        output dir on shared storage), else the workspace root.
        """
        dest = Path(os.environ.get("CUDA_AGENT_MANIFEST_DIR") or Path(self.config.workspace_root).expanduser())
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def _reference_problem_files(self) -> dict[str, str]:
        """``problem/`` files for the reference sandbox.

        Emitted by the SAME writers a rollout uses (``_seed_problem`` +
        ``write_problem_extras``), so the reference's file set cannot drift
        from the rollout's.
        """
        scfg = self.config.solswarm_surface
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            self._seed_problem(work, {"definition": {"name": "reference"}, "workloads": [{}]})
            solswarm_surface.write_problem_extras(
                work / "problem",
                user_prompt="reference",
                language="triton",
                submission_rule=solswarm_surface.stitch_submission_rule(
                    self._surface_root(),
                    ban_framework_kernels=scfg.ban_framework_kernels,
                    ban_cuda_graphs=scfg.ban_cuda_graphs,
                    banned_libraries=scfg.banned_libraries,
                ),
            )
            return {p.name: p.read_text() for p in sorted((work / "problem").iterdir()) if p.is_file()}

    def _publish_probe(self, probe_out: Path, rollout_id: str, root: Optional[Path]) -> dict[str, Any]:
        """Run the job-level sandbox check and write ``sandbox_tree.json``.

        ``probe_out`` holds the file listing taken from inside this rollout's
        sandbox. The reference side is generated by running the image's own
        entrypoint in a container from the same extracted base (cached per
        checkout id and image identity), so parity is set equality between two
        OBSERVED listings rather than a hand-written expectation: an upstream
        layout change moves both listings and cancels out, leaving only what
        the RL environment itself adds or removes.
        Returns {} when the listing is missing or unreadable; in container
        mode a summary with ``identical_to_solswarm: None`` means no
        comparison verdict was produced (the reference sandbox could not be
        built). The caller releases the once-per-job claim in both cases so a
        later rollout retries the check.
        """
        try:
            # Missing listing: return {} so the caller releases the once-per-job claim.
            if not probe_out.is_file():
                LOG.warning("no in-sandbox file listing at %s (sandbox never assembled?)", probe_out)
                return {}
            probe = json.loads(probe_out.read_text())
            report: dict[str, Any] = {}
            # Container mode passes the checkout root: build the reference listing and diff against it.
            if root is not None and _ENROOT_BASE is not None:
                cache = self._manifest_dir()
                # The reference comes from the SAME launcher as the rollout —
                # a real container from the same extracted base — so the diff
                # reports the RL wiring's differences and nothing else.
                base_name, base_rootfs = _ENROOT_BASE
                reference = solswarm_container.reference_tree(
                    root,
                    cache,
                    self._reference_problem_files(),
                    argv_builder=lambda rd, p: solswarm_container.enroot_launch_argv(
                        base_name,
                        rollout_dir=rd,
                        probe=p,
                        entrypoint_override=self.config.entrypoint_override,
                    ),
                    # The base name hashes the image path/mtime/size, so a new
                    # agent image invalidates the cached baseline.
                    cache_tag=base_name,
                    # The same minimal launcher environment as a rollout; it
                    # overrides the reference contract's HOME and the
                    # VISIBLE_DEVICES variables in reference_tree's env union,
                    # for the reasons enroot_subprocess_env states.
                    extra_env=solswarm_container.enroot_subprocess_env(self._enroot_data_root()),
                    seed=lambda rd: solswarm_container.seed_rollout_from_image(rd, base_rootfs),
                )
                if reference:
                    report = sandbox_manifest.diff_trees(reference, probe)
            # Persist both listings and the diff for offline inspection.
            dest = self._manifest_dir()
            (dest / "sandbox_tree.json").write_text(
                json.dumps({"rollout": rollout_id, "probe": probe, "diff": report}, indent=2)
            )
            # Print the verdict once; a non-identical result also lands in the server log.
            if report:
                print(sandbox_manifest.render_diff(report), flush=True)
                if not report.get("identical"):
                    LOG.warning("%s", sandbox_manifest.render_diff(report).splitlines()[0])
            # A non-empty summary tells the caller the check ran.
            return {
                "paths": len(probe.get("tree", [])),
                "cwd": probe.get("cwd"),
                "identical_to_solswarm": report.get("identical"),
            }
        except Exception as e:  # pragma: no cover - diagnostics must not break rollouts
            LOG.warning("could not publish the sandbox tree: %s", e)
            return {}

    def _write_entrypoint_log(self, rollout_id: str, stdout: bytes, stderr: bytes) -> None:
        """Persist a failed container-mode rollout's entrypoint transcript."""
        try:
            path = self._manifest_dir() / f"entrypoint_{rollout_id}.log"
            path.write_bytes(b"=== STDOUT ===\n" + stdout + b"\n=== STDERR ===\n" + stderr)
            LOG.warning(
                "solswarm entrypoint failed; full transcript at %s (tail: %s)",
                path,
                stdout.decode(errors="replace")[-400:].replace("\n", " | "),
            )
        except Exception as e:  # pragma: no cover - diagnostics must not break rollouts
            LOG.warning("could not write entrypoint transcript: %s", e)

    def _write_manifest_file(self, manifest: dict[str, Any], digest: str) -> None:
        """Persist the full sandbox manifest as JSON, once per distinct input set.

        Written to the diagnostics directory (see ``_manifest_dir``).
        Diagnostics: never raise.
        """
        try:
            path = self._manifest_dir() / f"sandbox_manifest_{digest}.json"
            path.write_text(json.dumps({"digest": digest, "manifest": manifest}, indent=2, default=str))
            print(f"sandbox manifest written to {path}", flush=True)
        except Exception as e:  # pragma: no cover - diagnostics must not break rollouts
            LOG.warning("could not write sandbox manifest file: %s", e)

    def _opencode_config_dict(self, base_url: Optional[str]) -> dict[str, Any]:
        """The effective opencode config.

        ``base_url`` (the ``/ng-rollout/<id>`` capture endpoint, set during training)
        repoints every provider's ``options.baseURL`` so the model server can
        buffer this rollout's token ids.
        """
        config = copy.deepcopy(self.config.opencode_config)
        if base_url:
            for provider in (config.get("provider") or {}).values():
                if isinstance(provider, dict):
                    provider.setdefault("options", {})["baseURL"] = base_url
        return config

    def _record_manifest(
        self, work_dir: Path, skills_dir: Optional[Path], env: dict[str, str], use_namespace: bool, hiding: bool
    ) -> None:
        """Record the PRE-FLIGHT inventory of what the server hands the sandbox.

        The authoritative check is the in-sandbox file listing
        (``_publish_probe``); this measures our staged inputs. One JSON file
        per distinct input set — deterministic where stdout capture is not —
        and one printed summary per process. ``env`` must be the variables the
        sandbox actually receives (in container mode the sourced contract
        file's, not the launcher's).
        """
        global _MANIFEST_LOGGED
        manifest = sandbox_manifest.collect(
            work_dir,
            skills_dir=skills_dir,
            env=env,
            opencode_config=self.config.opencode_config,
            extra={
                "sandbox_profile": self.config.sandbox_profile,
                "submission_mode": self.config.submission_mode,
                "opencode_version": self.config.opencode_version,
                "mount_namespace": use_namespace,
                "hide_job_mounts": hiding,
            },
        )
        digest = sandbox_manifest.fingerprint(manifest)["digest"]
        if digest not in _MANIFEST_WRITTEN:
            _MANIFEST_WRITTEN.add(digest)
            self._write_manifest_file(manifest, digest)
        if not _MANIFEST_LOGGED:
            _MANIFEST_LOGGED = True
            # print, not LOG.info: Gym never configures logging, so the root
            # logger sits at WARNING and INFO records are dropped.
            print(sandbox_manifest.render(manifest), flush=True)

    async def _prepare_minimal_launch(
        self,
        work_dir: Path,
        problem_dir: Path,
        data_home: Path,
        meta: dict[str, Any],
        user_message: str,
        system_message: Optional[str],
        platform_api_endpoint: str,
        rollout_id: Optional[str],
        submission_eval_timeout: Optional[int],
        rollout_id_for_log: str,
        use_namespace: bool,
        probe: bool,
    ) -> tuple[list[str], dict[str, str], Path]:
        """Stage the minimal profile's launch: sandbox env, staged skills, our own `opencode run` argv.

        Returns ``(cmd, env, probe_out)``; ``probe_out`` is where the
        once-per-job file listing lands when ``probe`` is set.
        """
        language = meta["language"]
        submitting = self.config.submission_mode == "solswarm_submit"
        # Base agent env: the allowlisted inheritance plus the configured extras.
        env = self._env(str(data_home))
        # Production parity: the opencode config travels as
        # OPENCODE_CONFIG_CONTENT, never as a workspace opencode.json —
        # it carries the policy apiKey, which the model must not read.
        # (Container mode: the entrypoint generates its own config.)
        opencode_cfg = self._opencode_config_dict(self._rollout_model_base_url(rollout_id))
        if opencode_cfg:
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_cfg)
        # Block OpenCode's skill/instruction-file discovery (it walks up
        # from the workspace and $HOME, which would leak this repo's dev
        # skills and docs); production sandboxes have neither. The staged
        # solswarm skills load from XDG_CONFIG_HOME and are unaffected.
        # setdefault: an explicit `env:` override still wins.
        env.setdefault("OPENCODE_DISABLE_EXTERNAL_SKILLS", "1")
        env.setdefault("OPENCODE_DISABLE_CLAUDE_CODE", "1")
        # Point the sandbox's cudagym CLI at this row's GPU endpoint;
        # _pin_cudagym_urls explains why all four names must be set.
        _pin_cudagym_urls(env, self._cudagym_url(meta))

        # Staged only for /submit; recorded in the manifest so that case
        # reports its skills rather than nothing.
        skills_dir: Optional[Path] = None

        # Which job mounts to hide (each over-mounted with an empty dir) inside the namespace.
        hides: list[str] = []
        if use_namespace and self.config.hide_job_mounts:
            hides = list(sandbox_manifest.HIDEABLE_JOB_MOUNTS)
            # The cudagym CLI is an editable install rooted in /opt/nemo-rl, so
            # hiding that tree needs the staged package copy on PYTHONPATH first
            # (PYTHONPATH wins over the venv's editable pointer into the tree).
            cli_root = _stage_cudagym_cli(Path(self.config.workspace_root).expanduser())
            if cli_root is not None:
                existing = env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(cli_root) + (os.pathsep + existing if existing else "")
                hides.append("/opt/nemo-rl")
            else:
                LOG.warning("cudagym CLI copy unavailable; leaving /opt/nemo-rl visible to the agent")

        if submitting:
            # /submit needs SolSwarm's harness skills and the pod-controller
            # env staged by hand here (in container mode the entrypoint
            # installs the skills itself and build_env owns the env contract).
            staging = self._surface_staging_dir()
            skills_dir = await asyncio.to_thread(
                solswarm_surface.stage_surface_snapshot, self._surface_root(), staging, ["submit", "cudagym"]
            )
            if use_namespace:
                # Bind the staged skills onto production's literal path
                # (_namespace_argv performs the bind; skills_src below) so
                # OpenCode discovers them from production's XDG location.
                env["XDG_CONFIG_HOME"] = _NS_XDG_CONFIG
                env["SOLSWARM_SKILLS_DIR"] = _NS_SKILLS
            else:
                env["XDG_CONFIG_HOME"] = str(staging / "config")
                env["SOLSWARM_SKILLS_DIR"] = str(skills_dir)
            # Keep opencode from replacing itself mid-rollout.
            env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
            env["OPENCODE_AUTOUPDATE"] = "0"
            env["NO_UPDATE_NOTIFIER"] = "1"
            # The platform endpoint plus the identity SolSwarm's pod controller
            # would inject, shared with container mode (pod_identity_env).
            env.update(solswarm_container.pod_identity_env(meta, rollout_id_for_log, language, platform_api_endpoint))
        if submitting and submission_eval_timeout:
            # Keeps submit.py's default --timeout + poll budget in sync with
            # the server's real canonical-eval budget (from /begin).
            env["KF_EVAL_TIMEOUT"] = str(submission_eval_timeout)

        # The row prompt plus the sandbox contract, handed to opencode directly.
        prompt = self._build_prompt(user_message, system_message, meta)

        self._record_manifest(work_dir, skills_dir, env, use_namespace, bool(hides))

        # Our own headless `opencode run` invocation, pointed at the workspace
        # path as the MODEL sees it (production's path inside the namespace,
        # or the real per-rollout directory).
        model_work_dir = Path(_NS_WORKSPACE) if use_namespace else work_dir
        cmd = [*self.config.command_parts, "run", "-m", self.config.model, "--dir", str(model_work_dir)]
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        probe_out = work_dir / ".probe" / "sandbox_probe.json"
        if use_namespace:
            # The prompt rides in the environment (tens of KB of arbitrary
            # text must not be shell-quoted into the wrapper script).
            env["CUDA_AGENT_PROMPT"] = prompt
            probe_dir = None
            if probe:
                # Stage the listing script in the workspace; the namespace sees it under /workspace/.probe.
                # The listing runs as root off this model-writable mount, so a malicious agent could
                # forge it: it is diagnostics, not a tamper-proof gate.
                (work_dir / ".probe").mkdir(parents=True, exist_ok=True)
                (work_dir / ".probe" / "sandbox_probe.py").write_text(sandbox_manifest.PROBE_SOURCE)
                probe_dir = Path(_NS_WORKSPACE) / ".probe"
            cmd = self._namespace_argv(cmd, work_dir, problem_dir, hides, skills_src=skills_dir, probe_dir=probe_dir)
        else:
            # Without a namespace there is no in-sandbox listing; the staged
            # manifest is the only record.
            cmd.append(prompt)
        return cmd, env, probe_out

    async def _prepare_container_launch(
        self,
        work_dir: Path,
        problem_dir: Path,
        rollout_dir: Path,
        meta: dict[str, Any],
        user_message: str,
        platform_api_endpoint: str,
        rollout_id: Optional[str],
        submission_eval_timeout: Optional[int],
        rollout_id_for_log: str,
        probe: bool,
    ) -> tuple[list[str], dict[str, str], Path, Path]:
        """Stage one container rollout: problem extras, the env contract, and the seeded rollout tree.

        Returns ``(cmd, launcher env, ws work_dir, probe_out)``. ``ws
        work_dir`` is the rollout-tree subdir the entrypoint mounts as
        /workspace, where the kernel file lands.
        """
        scfg = self.config.solswarm_surface
        # The entrypoint only stitches submission_rule.md on its DATABASE
        # path, which a pre-populated PROBLEM_DIR bypasses — so the problem
        # extras are genuinely ours. The row's task text reaches the model
        # SolSwarm-style: via problem/user_prompt.txt (the optimizer role
        # prompt, docker/agent/prompts/roles/optimizer.md, tells the agent
        # to read it), not inline; the entrypoint renders its own role prompt.
        solswarm_surface.write_problem_extras(
            problem_dir,
            user_prompt=user_message,
            language=meta["language"],
            submission_rule=solswarm_surface.stitch_submission_rule(
                self._surface_root(),
                ban_framework_kernels=scfg.ban_framework_kernels,
                ban_cuda_graphs=scfg.ban_cuda_graphs,
                banned_libraries=scfg.banned_libraries,
            ),
        )
        # SolSwarm's own entrypoint builds the sandbox and launches OpenCode
        # (with ITS flags: --pure --agent build --variant <effort> --format
        # json), so we contribute only the pod-controller environment.
        # `<provider>/<model>` in config: the provider names the config entry
        # that carries the policy apiKey. The model id itself never goes on
        # the wire from here — OpenCode sends the synthetic policy id and the
        # token-buffering model server substitutes its own config.model.
        provider_name = self.config.model.partition("/")[0]
        providers = (self.config.opencode_config or {}).get("provider") or {}
        api_key = str(((providers.get(provider_name) or {}).get("options") or {}).get("apiKey") or "rl")
        contract = dict(
            solswarm_container.build_env(
                meta=meta,
                problem_dir=_NS_PROBLEM,
                platform_api_endpoint=platform_api_endpoint,
                cudagym_url=self._cudagym_url(meta),
                rollout_id=rollout_id_for_log,
                max_duration=self.config.timeout,
                ban_framework_kernels=scfg.ban_framework_kernels,
                ban_cuda_graphs=scfg.ban_cuda_graphs,
                # Settings the entrypoint's generated config would drop
                # (compaction off + the context limit), re-keyed to its
                # names; see container_opencode_config_extra.
                opencode_config_extra=solswarm_container.container_opencode_config_extra(
                    self.config.opencode_config, self.config.model
                ),
                reasoning_variant=self.config.solswarm_reasoning_variant,
            )
        )
        # The contract travels as a sourced file (enroot start resets
        # the calling environment), plus the evaluation-service
        # credentials a pod controller would inject.
        for key in ("CUDAGYM_AUTH_TOKEN", "MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"):
            if os.environ.get(key):
                contract[key] = os.environ[key]
        if submission_eval_timeout:
            # Keeps submit.py's default --timeout + poll budget in sync with
            # the server's real canonical-eval budget (from /begin).
            contract["KF_EVAL_TIMEOUT"] = str(submission_eval_timeout)

        # Registry routing: hand the entrypoint the synthetic-catalog override
        # env (the policy id, its key, and the fixed in-container paths of the
        # catalog files). The catalog FILES are written into the rollout tree
        # below, once it exists — the env is set before the manifest so the
        # recorded contract matches what the sandbox receives.
        policy_id = solswarm_container.REGISTRY_POLICY_ID
        contract.update(solswarm_container.registry_contract_env(policy_id, api_key))

        # The recorded environment is the contract's — the variable names the
        # sandbox actually receives — snapshotted before problem/ moves into
        # the rollout tree.
        self._record_manifest(work_dir, None, contract, use_namespace=False, hiding=False)

        # Private rollout tree: problem/ moves into it, alongside every dir the entrypoint writes.
        solswarm_container.prepare_rollout_dir(rollout_dir, problem_dir)
        # Drop the synthetic catalog files into the now-created rollout tree.
        # agent_base_url in the endpoints file is the rollout-prefixed
        # model-server URL (a model_server is startup-required for the
        # solswarm profile; with capture enabled the prefix carries
        # /ng-rollout/<id>/training-token-capture), so the entrypoint points
        # OpenCode straight at the capture endpoint.
        solswarm_container.write_registry_catalog_files(
            rollout_dir, policy_id, self._rollout_model_base_url(rollout_id)
        )
        # A real instance of the agent image per rollout: the image supplies
        # the rootfs, the per-rollout writable paths are seeded from the
        # extracted image, and the inner enroot gets its own path tree.
        base_name, base_rootfs = await self._ensure_enroot_base()
        seed_wall = time()
        await asyncio.to_thread(solswarm_container.seed_rollout_from_image, rollout_dir, base_rootfs)
        # Seeding happens before the launch stamp, so it is reported as
        # its own per-rollout cost, not folded into "sandbox startup".
        _record_sandbox_latency("rollout seed (image copies)", time() - seed_wall, rollout_id_for_log)
        solswarm_container.write_rollout_env(rollout_dir, contract)
        # The launcher process gets only enroot's own knobs; inherited
        # values change enroot's host-side behavior (see
        # enroot_subprocess_env).
        env = solswarm_container.enroot_subprocess_env(self._enroot_data_root())
        cmd = solswarm_container.enroot_launch_argv(
            base_name,
            rollout_dir=rollout_dir,
            probe=probe,
            entrypoint_override=self.config.entrypoint_override,
        )
        return cmd, env, rollout_dir / solswarm_container.WS_SUBDIR, rollout_dir / "sandbox_probe.json"

    def _package_session(
        self,
        output_items: list[Any],
        usage: dict[str, int],
        work_dir: Path,
        filename: str,
        language: str,
    ) -> NeMoGymResponse:
        """Append the closing submission message and package the session as a Responses API response.

        The submission is the final kernel file the model left in the sandbox
        (in solswarm_submit mode only the fallback when nothing was
        /submit-ted); appending it as a fenced code block in a closing
        assistant message lets the verifier extract it.
        """
        kernel_path = work_dir / filename
        if not kernel_path.is_file():
            # A missing kernel file scores 0 with no other symptom. The
            # usual cause is the model writing to an absolute path outside
            # its sandbox (e.g. production's /workspace, or the repo root),
            # so the warning names what it DID leave behind.
            strays = [str(p.relative_to(work_dir)) for p in itertools.islice(work_dir.rglob("*.py"), 5)]
            LOG.warning(
                "no %s in the sandbox at session end (files present: %s)",
                filename,
                ", ".join(strays) or "none",
            )
        else:
            submitted = kernel_path.read_text(errors="replace")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-submit-{uuid4().hex[:8]}",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=f"Submitting ./{filename}:\n```{fence_lang_for(language)}\n{submitted}\n```",
                            annotations=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output_items,
            # Required by the openai Response schema — omitting them raises a
            # pydantic ValidationError and 500s every /run (peers pass the same
            # trio, e.g. scicode_agent / opencode_agent / claude_code_agent).
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            usage=NeMoGymResponseUsage(
                input_tokens=usage.get("input_tokens", 0),
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=usage.get("output_tokens", 0),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
        )

    async def _run_kernel_sandbox(
        self,
        responses_create_params: Any,
        meta: dict[str, Any],
        rollout_id: Optional[str] = None,
        submission_token: Optional[str] = None,
        submission_eval_timeout: Optional[int] = None,
    ) -> NeMoGymResponse:
        """Seed a sandbox, run one agent session, return the packaged trajectory.

        When ``rollout_id`` is set, OpenCode is pointed at the model server's
        ``/ng-rollout/<id>[/training-token-capture]`` endpoint so per-call
        token ids are captured for training; otherwise it uses the static
        provider config (eval / no token capture). The per-profile staging
        lives in ``_prepare_minimal_launch`` / ``_prepare_container_launch``;
        this method owns the launch, the wait, the kill paths, the
        once-per-job sandbox check, and the tree cleanup.
        """
        # The row's language fixes the single kernel file that counts as the submission.
        language = meta["language"]
        filename = kernel_filename_for(language)
        # The solswarm profile IS container mode (startup-validated: enroot,
        # the agent image, and a model_server all exist).
        container_mode = self.config.sandbox_profile == "solswarm"
        submitting = self.config.submission_mode == "solswarm_submit"

        # Fresh per-rollout workspace. Everything after it exists runs under
        # the try below, whose finally removes the trees: a setup failure part
        # way through (problem seeding, image extraction, rollout-tree seeding,
        # the credentials file) must not leak the seeded tree or the 0600
        # .rollout-env file.
        work_dir = self._workspace_root()
        problem_dir = work_dir / "problem"
        data_home = work_dir / ".opencode-data"
        # Set once the once-per-job sandbox check has fully completed; anything
        # less releases the claim in the finally below so a later rollout retries.
        sandbox_check_done = False
        probe_this_rollout = False
        rollout_dir: Optional[Path] = None
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            data_home.mkdir(parents=True, exist_ok=True)
            self._seed_problem(work_dir, meta)
            # The task text from the row's input messages.
            user_message, system_message = _extract_instruction(responses_create_params.input)

            # Submitting: point the /submit skill at this rollout's recording endpoints on the resources server.
            platform_api_endpoint = ""
            if submitting:
                platform_api_endpoint = (
                    f"{self._resources_server_base_url().rstrip('/')}/rollout/{submission_token}/api/v1"
                )

            # The minimal profile may ask for a mount namespace (container mode
            # needs none: the image supplies the whole filesystem); it degrades to
            # the real paths when `unshare` is missing, losing only path parity.
            use_namespace = not container_mode and self.config.mount_namespace and shutil.which("unshare") is not None
            if not container_mode and self.config.mount_namespace and not use_namespace:
                LOG.warning("mount_namespace requested but `unshare` is not available; running unsandboxed")
            # Short rollout id used in diagnostics (transcript filename, sandbox report).
            rollout_id_for_log = (submission_token or rollout_id or uuid4().hex)[:16]

            # The sandbox check runs once per job: the first rollout to start
            # claims it (asyncio: no await between test and set), everyone else
            # skips the file listing entirely.
            global _SANDBOX_CHECKED
            probe_this_rollout = (container_mode or use_namespace) and not _SANDBOX_CHECKED
            if probe_this_rollout:
                _SANDBOX_CHECKED = True

            # Absorb the trainer's refit/wake window (or fail loudly) BEFORE
            # staging the sandbox and burning the agent's wall-clock budget.
            await self._await_model_live(rollout_id_for_log)

            if container_mode:
                # Named before it is built, so the finally removes the skeleton
                # even if the staging below fails part way through.
                rollout_dir = work_dir.parent / f"{work_dir.name}-rollout"
                cmd, env, work_dir, probe_out = await self._prepare_container_launch(
                    work_dir,
                    problem_dir,
                    rollout_dir,
                    meta,
                    user_message,
                    platform_api_endpoint,
                    rollout_id,
                    submission_eval_timeout,
                    rollout_id_for_log,
                    probe_this_rollout,
                )
            else:
                cmd, env, probe_out = await self._prepare_minimal_launch(
                    work_dir,
                    problem_dir,
                    data_home,
                    meta,
                    user_message,
                    system_message,
                    platform_api_endpoint,
                    rollout_id,
                    submission_eval_timeout,
                    rollout_id_for_log,
                    use_namespace,
                    probe_this_rollout,
                )

            # Launch the agent (or the container entrypoint) and wait. In
            # container mode the binding budget is the INNER one (build_env's
            # SOLSWARM_MAX_DURATION = the configured timeout, enforced by the
            # entrypoint); the outer wait adds a fixed tail and is a backstop.
            outer_budget = _outer_wait_budget(self.config.timeout, container_mode)
            # The agent's merged stdout+stderr streams to a per-rollout file
            # on shared storage: PIPE capture loses everything on the timeout
            # path (kernelwriter-27b-9 hung 900s and left empty transcripts),
            # and this stream is the only wire-level record of a hung provider
            # (opencode's connection retries land on stderr).
            oc_log_path = self._manifest_dir() / "opencode" / f"{rollout_id_for_log}.log"
            oc_log_path.parent.mkdir(parents=True, exist_ok=True)
            oc_log = open(oc_log_path, "wb")
            launch_wall = time()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdout=oc_log,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                # Fresh session: proc.pid becomes the process group id of
                # every descendant, so timeout/cancel can kill the whole tree
                # (the enroot rc shell, the entrypoint, opencode) at once.
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=outer_budget)
            except asyncio.TimeoutError:
                await _terminate_process_tree(proc)
                LOG.warning(
                    "opencode timed out after %ds (outer backstop budget); its output is at %s",
                    outer_budget,
                    oc_log_path,
                )
            except asyncio.CancelledError:
                # A cancelled /run must not orphan the subprocess tree: a
                # surviving entrypoint would keep generating against the
                # popped run token (re-creating its trajectory buffer file),
                # race the rollout tree's removal below, and hold the OpenCode
                # session db locked. Same tree kill as the timeout path.
                await _terminate_process_tree(proc)
                raise
            finally:
                oc_log.close()
            # Downstream consumers (warning excerpts, the container transcript
            # fallback) read the tail of the persisted stream where the PIPE
            # variables used to be.
            try:
                _oc_tail = oc_log_path.read_bytes()[-4000:]
            except OSError:
                _oc_tail = b""
            stdout, stderr = _oc_tail, _oc_tail
            if container_mode and rollout_dir is not None:
                self._log_sandbox_startup(rollout_dir, launch_wall, rollout_id_for_log)
            # A failed agent run is still parsed below: the session db may hold usable turns.
            if proc.returncode not in (0, None) or self.config.keep_transcripts:
                if proc.returncode not in (0, None):
                    LOG.warning(
                        "opencode exited %s (full output at %s): %s",
                        proc.returncode,
                        oc_log_path,
                        stderr.decode(errors="replace")[:500],
                    )
                if container_mode:
                    # The entrypoint's whole assembly transcript goes to a file
                    # in the rollout dir (the --rc launch script redirects its
                    # stdout; the captured pipes carry only enroot's output).
                    # Persist it — Ray truncates long log lines.
                    transcript = b""
                    try:
                        transcript = (rollout_dir / solswarm_container.ENTRYPOINT_LOG).read_bytes()
                    except OSError:
                        pass
                    self._write_entrypoint_log(rollout_id_for_log, transcript or stdout, stderr)

            # OpenCode's sqlite session: under our XDG_DATA_HOME normally; under
            # the entrypoint-owned HOME (bound to the rollout's home/) in
            # container mode. Fallback trajectory (no token ids) + usage totals.
            session_db = (
                rollout_dir / "home" / ".local" / "share" / "opencode" / "opencode.db"
                if container_mode
                else data_home / "opencode" / "opencode.db"
            )
            try:
                # The read walks the whole session, so run it off the event
                # loop. A locked, torn, or truncated database (a writer the kill
                # caught mid-write) degrades to the no-trajectory path instead
                # of failing the /run.
                output_items, usage = await asyncio.to_thread(parse_opencode_session, session_db)
            except (sqlite3.Error, ValueError, TypeError) as e:
                LOG.warning("could not read the opencode session db %s (%s); treating as no trajectory", session_db, e)
                output_items, usage = [], {"input_tokens": 0, "output_tokens": 0}

            # The closing submission message plus the Responses API packaging.
            response = self._package_session(output_items, usage, work_dir, filename, language)
            if probe_this_rollout:
                # The job-level sandbox check: the OBSERVED file listing, taken
                # from inside the sandbox, is the parity signal. Runs before
                # the cleanup below (the listing lives in the rollout tree),
                # and off the event loop — building the reference side boots a
                # full container and takes minutes.
                observed = await asyncio.to_thread(
                    self._publish_probe,
                    probe_out,
                    rollout_id_for_log,
                    self._surface_root() if container_mode else None,
                )
                # Done only when a listing was published AND, in container
                # mode, the comparison produced a verdict (None means the
                # reference build failed); anything less is released below.
                sandbox_check_done = bool(observed) and not (
                    container_mode and observed.get("identical_to_solswarm") is None
                )
        finally:
            if probe_this_rollout and not sandbox_check_done:
                _SANDBOX_CHECKED = False  # release the claim; a later rollout retries
            # An exception that escaped while the agent was still running (a
            # pipe error mid-communicate, say) would otherwise leave it alive
            # after its trees are removed: it keeps generating against the model
            # server and keeps appending to its rollout's capture. The timeout
            # and cancellation paths kill the group themselves; this covers the rest.
            if proc is not None and proc.returncode is None:
                _signal_process_group(proc, signal.SIGKILL)
            # Remove the rollout's trees, success or failure — one to_thread
            # call for all of them (_remove_rollout_trees), shielded so a
            # re-delivered cancellation cannot strand the trees or the 0600
            # credentials file. In container mode rollout_dir covers work_dir,
            # and data_home's parent is the abandoned seed dir.
            await asyncio.shield(
                asyncio.to_thread(
                    _remove_rollout_trees,
                    work_dir,
                    rollout_dir,
                    data_home.parent if rollout_dir is not None else None,
                )
            )

        return response

    async def run(self, request: Request, body: CudaAgentRunRequest) -> CudaAgentVerifyResponse:
        """Seed sandbox -> run OpenCode -> verify the submitted kernel (canonical reward)."""
        # One semaphore slot per rollout; session cookies thread through every downstream call.
        async with self.sem:
            cookies = request.cookies

            # Let the resources server set up any per-task state it needs.
            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            # Training-token capture id: derived from the run body's task/rollout
            # indices by the base class (None when capture is disabled). The model
            # server's middleware files this rollout's per-call token ids under it;
            # rollout collection rebuilds the trajectory after /run returns.
            rollout_id = self.rollout_id_from_run(body)
            # Rollout-scoped token under which the resources server records /submit
            # evaluations (solswarm_submit mode). /begin hands it the task metadata
            # so its canonical evals can run server-side.
            submission_token = uuid4().hex if self.config.submission_mode == "solswarm_submit" else None
            submission_eval_timeout: Optional[int] = None
            if submission_token is not None:
                begin_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/rollout/{submission_token}/begin",
                    json={"verifier_metadata": body.verifier_metadata},
                    cookies=cookies,
                )
                await raise_for_status(begin_resp)
                begin_json = await get_response_json(begin_resp)
                submission_eval_timeout = begin_json.get("eval_timeout_seconds")

            # Seed the sandbox and run one agent session; the response holds the
            # reconstructed session. Training-token capture happens server-side
            # (the sandbox dials the /ng-rollout-prefixed model URL); the rebuilt
            # on-policy trajectory replaces response.output at rollout-collection
            # finalization, AFTER /verify below has read the reconstructed session
            # (it needs the msg-submit-* closing message to extract the kernel).
            response = await self._run_kernel_sandbox(
                body.responses_create_params,
                body.verifier_metadata,
                rollout_id,
                submission_token,
                submission_eval_timeout,
            )

            response_json = response.model_dump()

            # Canonical eval + staged reward over the submitted kernel.
            verify_body = body.model_dump() | {"response": response_json}
            if submission_token is not None:
                verify_body["submission_token"] = submission_token
            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_body,
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            # Count the model's own turns; the synthetic msg-submit-* closing
            # message the server appends is not one.
            turns = sum(
                1
                for item in response.output
                if getattr(item, "type", None) == "message"
                and getattr(item, "role", None) == "assistant"
                and not str(getattr(item, "id", "")).startswith("msg-submit-")
            )
            return CudaAgentVerifyResponse.model_validate(verify_json | {"turns_used": turns})


if __name__ == "__main__":
    CudaAgent.run_webserver()
