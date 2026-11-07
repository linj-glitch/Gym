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

"""Harbor task packages read from a local Terminal-Bench 4.0 task tree.

A task directory holds ``task.toml``, ``instruction.md``, ``tests/`` (the
held-out grader, ``tests/test.sh`` writes ``/logs/verifier/reward.*``) and
usually ``solution/``. Only the fields this server acts on are modelled; the
rest of ``task.toml`` is carried along untouched in ``raw``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAIN_SERVICE = "main"
_CANARY_LINE_RE = re.compile(r"^(?:#|<!--)\s*harbor-canary\b.*$")


def strip_canary(text: str) -> str:
    """Drop the leading ``harbor-canary`` marker line(s) and the blank lines after them."""
    lines = text.split("\n")
    index = 0
    while index < len(lines) and _CANARY_LINE_RE.match(lines[index].strip()):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


@dataclass(frozen=True)
class Artifact:
    """A path the verifier needs from the agent container (``[artifacts]`` entry)."""

    source: str
    exclude: tuple[str, ...] = ()
    service: str = MAIN_SERVICE


@dataclass(frozen=True)
class CollectHook:
    """A ``[[verifier.collect]]`` command run in the agent container before artifacts are taken."""

    command: str
    service: str = MAIN_SERVICE
    timeout_sec: float = 60.0
    user: str | None = None


@dataclass(frozen=True)
class EnvironmentSettings:
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    gpus: int
    env: dict[str, str]
    mcp_servers: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Task:
    name: str
    task_dir: Path
    instruction: str
    environment: EnvironmentSettings
    verifier_environment: EnvironmentSettings
    environment_mode: str
    agent_timeout_sec: float
    agent_user: str | None
    verifier_timeout_sec: float
    verifier_env: dict[str, str]
    artifacts: tuple[Artifact, ...]
    collect: tuple[CollectHook, ...]
    raw: dict[str, Any]

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"


def _environment(section: dict[str, Any]) -> EnvironmentSettings:
    memory_mb = section.get("memory_mb", section.get("memory"))
    storage_mb = section.get("storage_mb", section.get("storage"))
    return EnvironmentSettings(
        cpus=section.get("cpus"),
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        gpus=int(section.get("gpus") or 0),
        env={str(k): str(v) for k, v in (section.get("env") or {}).items()},
        mcp_servers=tuple(section.get("mcp_servers") or ()),
    )


def _artifact(entry: Any) -> Artifact:
    if isinstance(entry, str):
        return Artifact(source=entry)
    if isinstance(entry, dict) and isinstance(entry.get("source"), str):
        return Artifact(
            source=entry["source"],
            exclude=tuple(str(pattern) for pattern in entry.get("exclude") or ()),
            service=str(entry.get("service") or MAIN_SERVICE),
        )
    raise ValueError(f"Unsupported artifact entry: {entry!r}")


def _collect_hook(entry: dict[str, Any]) -> CollectHook:
    if not isinstance(entry.get("command"), str) or not entry["command"]:
        raise ValueError(f"verifier.collect entry has no command: {entry!r}")
    return CollectHook(
        command=entry["command"],
        service=str(entry.get("service") or MAIN_SERVICE),
        timeout_sec=float(entry.get("timeout_sec", 60.0)),
        user=entry.get("user"),
    )


def load_task(task_dir: str | Path) -> Task:
    """Read one Harbor task directory."""
    task_dir = Path(task_dir).resolve()
    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    test_script = task_dir / "tests" / "test.sh"
    for required in (config_path, instruction_path, test_script):
        if not required.is_file():
            raise FileNotFoundError(
                f"Terminal-Bench task {task_dir.name!r} is missing {required.relative_to(task_dir)}"
            )
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("steps"):
        raise ValueError(f"Terminal-Bench task {task_dir.name!r} is a stepped task, which this server does not run")

    environment = _environment(raw.get("environment") or {})
    verifier = raw.get("verifier") or {}
    agent = raw.get("agent") or {}
    verifier_environment = _environment(verifier["environment"]) if verifier.get("environment") else environment
    mode = verifier.get("environment_mode")
    if mode is None:
        mode = "separate" if verifier.get("environment") else "shared"
    if mode not in ("separate", "shared"):
        raise ValueError(f"Terminal-Bench task {task_dir.name!r} has unknown verifier.environment_mode {mode!r}")

    return Task(
        name=task_dir.name,
        task_dir=task_dir,
        instruction=strip_canary(instruction_path.read_text(encoding="utf-8")),
        environment=environment,
        verifier_environment=verifier_environment,
        environment_mode=mode,
        agent_timeout_sec=float(agent.get("timeout_sec", 28800.0)),
        agent_user=agent.get("user"),
        verifier_timeout_sec=float(verifier.get("timeout_sec", 600.0)),
        verifier_env={str(k): str(v) for k, v in (verifier.get("env") or {}).items()},
        artifacts=tuple(_artifact(entry) for entry in raw.get("artifacts") or ()),
        collect=tuple(_collect_hook(entry) for entry in verifier.get("collect") or ()),
        raw=raw,
    )


def unsupported_reasons(task: Task) -> list[str]:
    """Why a task cannot run on a single-container sandbox provider (empty = runnable)."""
    reasons = []
    if task.environment.gpus or task.verifier_environment.gpus:
        reasons.append("needs GPUs")
    if task.environment.mcp_servers:
        reasons.append("declares MCP servers")
    sidecars = {a.service for a in task.artifacts} | {h.service for h in task.collect}
    if sidecars - {MAIN_SERVICE}:
        reasons.append(f"needs sidecar services {sorted(sidecars - {MAIN_SERVICE})}")
    if (task.task_dir / "environment" / "docker-compose.yaml").exists():
        reasons.append("is a Compose task")
    return reasons


def load_tasks(tasks_dir: str | Path, names: list[str] | None = None) -> dict[str, Task]:
    """Read every task directory under ``tasks_dir`` (or just ``names``), keyed by directory name."""
    tasks_dir = Path(tasks_dir).expanduser().resolve()
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Terminal-Bench tasks directory does not exist: {tasks_dir}")
    if names is None:
        task_dirs = sorted(path.parent for path in tasks_dir.glob("*/task.toml"))
    else:
        task_dirs = [tasks_dir / name for name in names]
    tasks = {task_dir.name: load_task(task_dir) for task_dir in task_dirs}
    if not tasks:
        raise ValueError(f"No Terminal-Bench tasks found under {tasks_dir}")
    return tasks


def image_reference(repository: str, task_name: str, role: str, tag_suffix: str) -> str:
    """The Harbor-published image for one task phase, e.g. ``harborframework/terminal-bench:<task>-environment-v4.0.0``."""
    if role not in ("environment", "verifier"):
        raise ValueError(f"Unknown image role {role!r}")
    return f"{repository}:{task_name}-{role}-{tag_suffix}"
