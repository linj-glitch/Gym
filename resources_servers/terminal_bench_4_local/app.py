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

"""Terminal-Bench 4.0 resources server for single-container sandbox providers.

Runs the official Harbor task packages from a local task tree against the
images Harbor publishes for each task (``<task>-environment-<tag>`` for the
agent, ``<task>-verifier-<tag>`` for the grader), so no Harbor Hub account,
OpenSandbox deployment or image build is needed; an enroot or Docker provider
with the images pre-imported is enough. Harness-owned-sandbox contract, as the
DeepSWE server: ``/seed_session`` starts the agent container and hands back its
descriptor, the agent server connects and runs the CLI inside, ``/verify``
collects the task's declared artifacts, starts the verifier container (or
reuses the agent container for ``environment_mode = "shared"`` tasks), restores
the artifacts at their original paths, uploads ``tests/`` and runs
``/tests/test.sh`` under the task's verifier timeout. ``reward`` is the official
value from ``/logs/verifier/reward.json`` (or ``reward.txt``).

Not covered: Compose tasks (sidecar services), GPU tasks and tasks declaring
MCP servers; ``prepare.py`` leaves them out of the dataset and
``/seed_session`` refuses them.
"""

from __future__ import annotations

import io
import json
import shlex
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from shutil import rmtree
from time import monotonic
from traceback import format_exc
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_4_local.tasks import (
    MAIN_SERVICE,
    Artifact,
    Task,
    image_reference,
    load_tasks,
    unsupported_reasons,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]
# Transfer scratch inside the container rootfs. Not /tmp: enroot bind-mounts the host /tmp into every
# container, so concurrent sandboxes would share (and clobber) the same archive paths.
REMOTE_SCRATCH = "/logs/.nemo-gym-tb4"
# Default working directory when a dataset row carries none: Harbor task images conventionally
# WORKDIR /app and the instructions name paths under it. Rows built by prepare.py carry the image's
# real WORKDIR (benchmarks/terminal_bench_4_local/image-config.json), which may differ.
DEFAULT_WORKDIR = "/app"


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (NEMO_GYM_ROOT / expanded).resolve()


class TerminalBench4LocalResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED

    # Task tree: one Harbor task directory per task (task.toml, instruction.md, tests/, solution/).
    tasks_dir: Path
    # Images are derived from the task name: {repository}:{task}-{environment|verifier}-{tag_suffix}.
    image_repository: str = "harborframework/terminal-bench"
    image_tag_suffix: str = "v4.0.0"
    # Golden mode: /verify runs solution/solve.sh in a fresh agent container and grades that,
    # which checks images, artifact transfer and the grader without a model.
    is_verifying_golden_solution: bool = False
    golden_solution_timeout_sec: float = Field(default=3600.0, gt=0)
    # Multiplies every task's official verifier timeout (1.0 = official).
    verifier_timeout_multiplier: float = Field(default=1.0, gt=0)
    # Sandbox request = task resources x these, overridable by sandbox_config.resources.
    task_cpu_multiplier: float = Field(default=1.0, gt=0)
    task_memory_multiplier: float = Field(default=1.0, gt=0)

    sandbox_provider: str
    sandbox_config: dict[str, Any]

    logs_dir: Path = Path("resources_servers/terminal_bench_4_local/logs")
    clear_logs: bool = False
    # Characters of test.sh output kept on the verify response (the full output is in logs_dir).
    test_output_tail_chars: int = Field(default=20000, ge=0)


class TerminalBench4InstanceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str
    image: str
    verifier_image: str
    # Working directories of the two images (docker WORKDIR); None = the container's own default.
    workdir: str | None = DEFAULT_WORKDIR
    verifier_workdir: str | None = None


class TerminalBench4SeedSessionRequest(TerminalBench4InstanceRequest, BaseSeedSessionRequest):
    pass


class TerminalBench4SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]
    workdir: str
    agent_timeout_sec: float


class TerminalBench4VerifyRequest(TerminalBench4InstanceRequest, BaseVerifyRequest):
    pass


class TerminalBench4VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_name: str
    environment_mode: str
    # True when the grader ran to completion and produced a reward file.
    evaluation_completed: bool
    # Official pass: every test passed (reward 1).
    passed: bool
    verifier_exit_code: int | None = None
    verifier_error: str | None = None
    test_output: str | None = None
    artifacts: list[dict[str, Any]]
    collect_hooks: list[dict[str, Any]]
    log_dir: str
    collection_time_s: float
    verifier_sandbox_start_time_s: float
    verification_time_s: float


@dataclass
class AgentSession:
    task_name: str
    image: str
    sandbox: AsyncSandbox
    sandbox_handle: str
    workdir: str


# -- pure helpers (unit-tested) ---------------------------------------------------------------


def parse_reward(reward_json: str | None, reward_txt: str | None) -> float:
    """The official reward: ``reward.json``'s ``reward`` key, else the number in ``reward.txt``."""
    if reward_json is not None and reward_json.strip():
        data = json.loads(reward_json)
        if not isinstance(data, dict) or "reward" not in data:
            raise ValueError("reward.json must be an object with a 'reward' key")
        value = float(data["reward"])
    elif reward_txt is not None and reward_txt.strip():
        value = float(reward_txt.strip())
    else:
        raise ValueError("the verifier wrote no reward")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"the verifier reward is not finite: {value!r}")
    return value


def artifact_archive_command(artifact: Artifact, remote_archive: str) -> str:
    """Tar one artifact (file or directory) in place, honouring its exclude patterns."""
    source = artifact.source.rstrip("/") or "/"
    parent, name = str(Path(source).parent), Path(source).name
    excludes = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in artifact.exclude)
    return (
        f"tar -czf {shlex.quote(remote_archive)} -C {shlex.quote(parent)} {excludes} -- {shlex.quote(name)}".replace(
            "  ", " "
        )
    )


def artifact_restore_command(artifact: Artifact, remote_archive: str) -> str:
    """Recreate the artifact at its original path from the archive and drop the archive."""
    source = artifact.source.rstrip("/") or "/"
    parent = str(Path(source).parent)
    return (
        f"mkdir -p {shlex.quote(parent)} && tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(parent)}"
        f" && rm -f {shlex.quote(remote_archive)}"
    )


def restore_tests_command(remote_archive: str) -> str:
    return (
        f"rm -rf /tests && mkdir -p /tests /logs/verifier /logs/artifacts && tar -xzf {shlex.quote(remote_archive)}"
        f" -C /tests && rm -f {shlex.quote(remote_archive)} && chmod +x /tests/test.sh"
    )


def tar_directory_bytes(directory: Path) -> bytes:
    """A gzip tarball of ``directory``'s contents (arcnames relative to it)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    return buffer.getvalue()


def tail(text: str | None, limit: int) -> str | None:
    if text is None or limit <= 0 or len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


# -- server ------------------------------------------------------------------------------------


class TerminalBench4LocalResourcesServer(SimpleResourcesServer):
    config: TerminalBench4LocalResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._tasks = load_tasks(_resolve_repo_path(self.config.tasks_dir))
        self._agent_sessions: dict[str, AgentSession] = {}

    def _expected_image(self, task: Task, role: str) -> str:
        return image_reference(self.config.image_repository, task.name, role, self.config.image_tag_suffix)

    def _resolve_task(self, body: TerminalBench4InstanceRequest) -> Task:
        task = self._tasks.get(body.task_name)
        if task is None:
            raise ValueError(f"Unknown Terminal-Bench task {body.task_name!r}")
        reasons = unsupported_reasons(task)
        if reasons:
            raise ValueError(f"Terminal-Bench task {task.name!r} cannot run here: {', '.join(reasons)}")
        for role, requested in (("environment", body.image), ("verifier", body.verifier_image)):
            expected = self._expected_image(task, role)
            if requested != expected:
                raise ValueError(
                    f"Request {role} image {requested!r} does not match {expected!r} for task {task.name!r}"
                )
        return task

    async def _create_sandbox(self, task: Task, *, phase: str, workdir: str | None) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)

        if phase == "agent":
            image, environment, env = self._expected_image(task, "environment"), task.environment, task.environment.env
        else:
            image, environment, env = (
                self._expected_image(task, "verifier"),
                task.verifier_environment,
                task.verifier_env,
            )
        resources: dict[str, Any] = {}
        if environment.cpus:
            resources["cpu"] = environment.cpus * self.config.task_cpu_multiplier
        if environment.memory_mb:
            resources["memory_mib"] = ceil(environment.memory_mb * self.config.task_memory_multiplier)
        if environment.storage_mb:
            resources["disk_gib"] = ceil(environment.storage_mb / 1024)
        resources.update(self.config.sandbox_config.get("resources", {}))

        spec = SandboxSpec(
            image=image,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=workdir,
            env=dict(self.config.sandbox_config.get("env", {})) | env,
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {"benchmark": "terminal-bench-4.0", "tb4-task": task.name[:63], "tb4-phase": phase},
            resources=SandboxResources.from_mapping(resources),
            provider_options=dict(self.config.sandbox_config.get("provider_options", {})),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        checks = f"mkdir -p /logs/agent /logs/verifier /logs/artifacts {REMOTE_SCRATCH}"
        if workdir is not None:
            checks = f"test -d {shlex.quote(workdir)} && " + checks
        result = await sandbox.exec(checks, cwd="/", timeout_s=60, user="root")
        if result.return_code != 0:
            raise RuntimeError(
                f"The {phase} container for task {task.name!r} lacks its working directory {workdir!r} or /logs could "
                f"not be created: {result.stderr or ''}"
            )
        return sandbox

    async def _stop_sandbox(self, sandbox: AsyncSandbox, *, task_name: str, phase: str) -> None:
        try:
            await sandbox.stop()
        except Exception:
            print(f"Failed to stop Terminal-Bench {phase} sandbox for {task_name}: {format_exc()}", file=sys.stderr)

    async def seed_session(
        self, request: Request, body: TerminalBench4SeedSessionRequest
    ) -> TerminalBench4SeedSessionResponse:
        if self.config.is_verifying_golden_solution:
            raise RuntimeError("seed_session is unavailable in golden-solution mode")
        task = self._resolve_task(body)
        session_id = str(request.session[SESSION_ID_KEY])
        previous = self._agent_sessions.pop(session_id, None)
        if previous is not None:
            await self._stop_sandbox(previous.sandbox, task_name=previous.task_name, phase="replaced-agent")

        sandbox: AsyncSandbox | None = None
        try:
            sandbox = await self._create_sandbox(task, phase="agent", workdir=body.workdir)
            descriptor = await sandbox.serialize()
            handle = descriptor.get("sandbox_id") if isinstance(descriptor, dict) else None
            if not isinstance(handle, str) or not handle:
                raise RuntimeError("The sandbox provider did not return a sandbox_id")
            workdir = body.workdir or (descriptor.get("workdir") if isinstance(descriptor, dict) else None) or "/"
            self._agent_sessions[session_id] = AgentSession(
                task_name=task.name, image=body.image, sandbox=sandbox, sandbox_handle=handle, workdir=workdir
            )
            return TerminalBench4SeedSessionResponse(
                sandbox_handle=handle,
                sandbox_descriptor=dict(descriptor),
                workdir=workdir,
                agent_timeout_sec=task.agent_timeout_sec,
            )
        except Exception:
            if sandbox is not None:
                await self._stop_sandbox(sandbox, task_name=task.name, phase="failed-agent-seed")
            raise

    # -- verify pipeline -----------------------------------------------------------------------

    async def _run_collect_hooks(self, sandbox: AsyncSandbox, task: Task) -> list[dict[str, Any]]:
        records = []
        for hook in task.collect:
            if hook.service != MAIN_SERVICE:
                raise RuntimeError(f"collect hook for sidecar service {hook.service!r} is unsupported")
            record: dict[str, Any] = {"command": hook.command}
            try:
                result = await sandbox.exec(hook.command, timeout_s=hook.timeout_sec, user=hook.user)
                record.update(
                    return_code=result.return_code, stdout=tail(result.stdout, 4000), stderr=tail(result.stderr, 4000)
                )
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
        return records

    async def _collect_artifacts(self, sandbox: AsyncSandbox, task: Task, local_dir: Path) -> list[dict[str, Any]]:
        """Archive each declared artifact in the agent container and download it."""
        local_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, artifact in enumerate(task.artifacts):
            record: dict[str, Any] = {
                "source": artifact.source,
                "exclude": list(artifact.exclude),
                "status": "skipped",
            }
            remote_archive = f"{REMOTE_SCRATCH}/artifact-{index}.tgz"
            local_archive = local_dir / f"artifact-{index}.tgz"
            try:
                exists = await sandbox.exec(
                    f"test -e {shlex.quote(artifact.source.rstrip('/') or '/')}", timeout_s=30, user="root"
                )
                if exists.return_code != 0:
                    record["status"] = "missing"
                else:
                    result = await sandbox.exec(
                        artifact_archive_command(artifact, remote_archive), timeout_s=600, user="root"
                    )
                    if result.return_code != 0:
                        raise RuntimeError(f"tar exited {result.return_code}: {(result.stderr or '')[-2000:]}")
                    await sandbox.download(remote_archive, local_archive)
                    record.update(status="ok", archive=local_archive.name, bytes=local_archive.stat().st_size)
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            records.append(record)
        return records

    async def _restore_artifacts(
        self, sandbox: AsyncSandbox, task: Task, local_dir: Path, records: list[dict[str, Any]]
    ) -> None:
        for artifact, record in zip(task.artifacts, records, strict=True):
            if record["status"] != "ok":
                continue
            remote_archive = f"{REMOTE_SCRATCH}/{record['archive']}"
            await sandbox.upload(local_dir / record["archive"], remote_archive)
            result = await sandbox.exec(artifact_restore_command(artifact, remote_archive), timeout_s=600, user="root")
            if result.return_code != 0:
                raise RuntimeError(f"Restoring artifact {artifact.source!r} failed: {(result.stderr or '')[-2000:]}")

    async def _stage_tests(self, sandbox: AsyncSandbox, task: Task) -> None:
        remote_archive = f"{REMOTE_SCRATCH}/tests.tgz"
        with tempfile.TemporaryDirectory(prefix="nemo-gym-tb4-tests-") as temporary_dir:
            local_archive = Path(temporary_dir) / "tests.tgz"
            local_archive.write_bytes(tar_directory_bytes(task.tests_dir))
            await sandbox.upload(local_archive, remote_archive)
        result = await sandbox.exec(restore_tests_command(remote_archive), timeout_s=600, user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Staging /tests failed: {(result.stderr or '')[-2000:]}")

    async def _download_if_present(self, sandbox: AsyncSandbox, remote_path: str, local_path: Path) -> str | None:
        exists = await sandbox.exec(f"test -f {shlex.quote(remote_path)}", timeout_s=30, user="root")
        if exists.return_code != 0:
            return None
        await sandbox.download(remote_path, local_path)
        return local_path.read_text(encoding="utf-8", errors="replace")

    async def _run_tests(self, sandbox: AsyncSandbox, task: Task, log_dir: Path) -> dict[str, Any]:
        """Run the official grader; returns reward/exit code/output or the failure."""
        timeout_s = task.verifier_timeout_sec * self.config.verifier_timeout_multiplier
        outcome: dict[str, Any] = {"evaluation_completed": False, "reward": 0.0, "verifier_exit_code": None}
        try:
            result = await sandbox.exec(
                "bash /tests/test.sh", timeout_s=timeout_s, env=task.verifier_env or None, user="root"
            )
        except Exception as error:
            outcome["verifier_error"] = f"{type(error).__name__}: {error} (verifier timeout {timeout_s:.0f}s)"
            return outcome
        output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "test-output.txt").write_text(output, encoding="utf-8")
        outcome["verifier_exit_code"] = result.return_code
        outcome["test_output"] = tail(output, self.config.test_output_tail_chars)
        reward_json = await self._download_if_present(sandbox, "/logs/verifier/reward.json", log_dir / "reward.json")
        reward_txt = await self._download_if_present(sandbox, "/logs/verifier/reward.txt", log_dir / "reward.txt")
        await self._download_if_present(sandbox, "/logs/verifier/ctrf.json", log_dir / "ctrf.json")
        try:
            outcome["reward"] = parse_reward(reward_json, reward_txt)
            outcome["evaluation_completed"] = True
        except (ValueError, json.JSONDecodeError) as error:
            outcome["verifier_error"] = f"Invalid or missing verifier reward: {error}"
        return outcome

    async def _run_golden_solution(self, sandbox: AsyncSandbox, task: Task, log_dir: Path) -> None:
        if not (task.solution_dir / "solve.sh").is_file():
            raise FileNotFoundError(f"Task {task.name!r} ships no solution/solve.sh")
        remote_archive = f"{REMOTE_SCRATCH}/solution.tgz"
        with tempfile.TemporaryDirectory(prefix="nemo-gym-tb4-solution-") as temporary_dir:
            local_archive = Path(temporary_dir) / "solution.tgz"
            local_archive.write_bytes(tar_directory_bytes(task.solution_dir))
            await sandbox.upload(local_archive, remote_archive)
        staged = await sandbox.exec(
            f"rm -rf /solution && mkdir -p /solution && tar -xzf {remote_archive} -C /solution && chmod +x /solution/solve.sh",
            timeout_s=300,
            user="root",
        )
        if staged.return_code != 0:
            raise RuntimeError(f"Staging /solution failed: {(staged.stderr or '')[-2000:]}")
        result = await sandbox.exec(
            "bash /solution/solve.sh", timeout_s=self.config.golden_solution_timeout_sec, user="root"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "solve-output.txt").write_text(
            (result.stdout or "") + (("\n" + result.stderr) if result.stderr else ""), encoding="utf-8"
        )
        print(f"Golden solution for {task.name} exited {result.return_code}", file=sys.stderr)

    async def verify(self, request: Request, body: TerminalBench4VerifyRequest) -> TerminalBench4VerifyResponse:
        task = self._resolve_task(body)
        session_id = str(request.session.get(SESSION_ID_KEY, "golden"))
        log_dir = _resolve_repo_path(self.config.logs_dir) / task.name / session_id
        artifacts_dir = log_dir / "artifacts"
        timings = {"collection_time_s": 0.0, "verifier_sandbox_start_time_s": 0.0, "verification_time_s": 0.0}
        collect_records: list[dict[str, Any]] = []
        artifact_records: list[dict[str, Any]] = []
        outcome: dict[str, Any] = {"evaluation_completed": False, "reward": 0.0}
        failure: str | None = None

        agent_sandbox: AsyncSandbox | None = None
        verifier_sandbox: AsyncSandbox | None = None
        try:
            if self.config.is_verifying_golden_solution:
                agent_sandbox = await self._create_sandbox(task, phase="golden-agent", workdir=body.workdir)
                await self._run_golden_solution(agent_sandbox, task, log_dir)
            else:
                session = self._agent_sessions.pop(session_id, None)
                if session is None:
                    raise RuntimeError(f"No agent sandbox exists for session {session_id!r}")
                if session.task_name != task.name:
                    raise RuntimeError(f"Session task {session.task_name!r} does not match verify task {task.name!r}")
                agent_sandbox = session.sandbox

            started = monotonic()
            collect_records = await self._run_collect_hooks(agent_sandbox, task)
            artifact_records = await self._collect_artifacts(agent_sandbox, task, artifacts_dir)
            timings["collection_time_s"] = monotonic() - started

            if task.environment_mode == "separate":
                started = monotonic()
                verifier_sandbox = await self._create_sandbox(task, phase="verifier", workdir=body.verifier_workdir)
                timings["verifier_sandbox_start_time_s"] = monotonic() - started
                await self._restore_artifacts(verifier_sandbox, task, artifacts_dir, artifact_records)
                grading_sandbox = verifier_sandbox
            else:
                grading_sandbox = agent_sandbox

            started = monotonic()
            await self._stage_tests(grading_sandbox, task)
            outcome = await self._run_tests(grading_sandbox, task, log_dir)
            timings["verification_time_s"] = monotonic() - started
        except Exception as error:
            print(f"Terminal-Bench verify failed for {task.name}: {format_exc()}", file=sys.stderr)
            failure = f"{type(error).__name__}: {error}"
        finally:
            if verifier_sandbox is not None:
                await self._stop_sandbox(verifier_sandbox, task_name=task.name, phase="verifier")
            if agent_sandbox is not None:
                await self._stop_sandbox(agent_sandbox, task_name=task.name, phase="agent")

        verifier_error = outcome.get("verifier_error") or failure
        if self.config.clear_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir_value = ""
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "verify.json").write_text(
                json.dumps(
                    {
                        "task": task.name,
                        "artifacts": artifact_records,
                        "collect_hooks": collect_records,
                        "outcome": outcome,
                        "timings": timings,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            log_dir_value = str(log_dir)

        reward = float(outcome.get("reward", 0.0))
        return TerminalBench4VerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": reward,
                "failure_reason": None if outcome.get("evaluation_completed") else verifier_error,
                "task_name": task.name,
                "environment_mode": task.environment_mode,
                "evaluation_completed": bool(outcome.get("evaluation_completed")),
                "passed": bool(outcome.get("evaluation_completed")) and reward >= 1.0,
                "verifier_exit_code": outcome.get("verifier_exit_code"),
                "verifier_error": verifier_error,
                "test_output": outcome.get("test_output"),
                "artifacts": artifact_records,
                "collect_hooks": collect_records,
                "log_dir": log_dir_value,
            }
            | timings
        )


if __name__ == "__main__":
    TerminalBench4LocalResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4LocalResourcesServer.run_webserver()  # noqa: F401
