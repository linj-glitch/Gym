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

"""Harbor task packages whose image is named in ``task.toml`` (Terminal-Bench 2.1).

Entrypoint variant of ``app.py`` for task trees that publish one image per task
and name it in ``[environment].docker_image`` instead of following the 4.0
``<task>-{environment,verifier}-<tag>`` scheme. Everything else is the parent
server: such tasks declare no separate verifier environment, so they grade in
the agent container (``environment_mode = "shared"``): ``tests/`` is uploaded
to ``/tests`` and ``bash /tests/test.sh`` writes ``/logs/verifier/reward.txt``.

Per-task agent budget: a dataset row may carry ``agent_timeout_s`` (the task's
official ``[agent].timeout_sec`` scaled and capped by the benchmark's
``prepare.py``). ``/seed_session`` hands it to the agent server inside the
sandbox descriptor under ``AGENT_TIMEOUT_KEY``; the
``claude_code_sandboxed_agent/app_task_timeout.py`` entrypoint ends the CLI at
that deadline, after which ``/verify`` grades whatever the container holds
(Harbor's behaviour at an agent timeout).
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_4_local.app import (
    TerminalBench4LocalResourcesServer,
    TerminalBench4SeedSessionRequest,
    TerminalBench4SeedSessionResponse,
    TerminalBench4VerifyRequest,
    TerminalBench4VerifyResponse,
)
from resources_servers.terminal_bench_4_local.tasks import Task


# Descriptor key carrying the per-task agent deadline (seconds) to the agent server.
AGENT_TIMEOUT_KEY = "agent_timeout_s"


def task_image(task: Task) -> str:
    """The image a task names in ``[environment].docker_image``."""
    image = (task.raw.get("environment") or {}).get("docker_image")
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f"Terminal-Bench task {task.name!r} names no [environment].docker_image")
    return image.strip()


def row_agent_timeout(body: Any) -> float | None:
    """The row's ``agent_timeout_s`` as a positive number of seconds, else None."""
    value = getattr(body, AGENT_TIMEOUT_KEY, None)
    if value is None and getattr(body, "model_extra", None):
        value = body.model_extra.get(AGENT_TIMEOUT_KEY)
    if value in (None, ""):
        return None
    seconds = float(value)
    if seconds <= 0:
        raise ValueError(f"{AGENT_TIMEOUT_KEY} must be positive, got {value!r}")
    return seconds


class TerminalBenchTaskImageResourcesServer(TerminalBench4LocalResourcesServer):
    def _expected_image(self, task: Task, role: str) -> str:
        if role not in ("environment", "verifier"):
            raise ValueError(f"Unknown image role {role!r}")
        return task_image(task)

    async def seed_session(
        self, request: Request, body: TerminalBench4SeedSessionRequest
    ) -> TerminalBench4SeedSessionResponse:
        response = await super().seed_session(request, body)
        timeout_s = row_agent_timeout(body)
        if timeout_s is not None:
            response.sandbox_descriptor = {**response.sandbox_descriptor, AGENT_TIMEOUT_KEY: timeout_s}
        return response

    async def verify(self, request: Request, body: TerminalBench4VerifyRequest) -> TerminalBench4VerifyResponse:
        response = await super().verify(request, body)
        task = self._tasks.get(body.task_name)
        return TerminalBench4VerifyResponse.model_validate(
            response.model_dump()
            | {
                "agent_timeout_sec_official": task.agent_timeout_sec if task is not None else None,
                "agent_timeout_sec_enforced": row_agent_timeout(body),
                # Alias of ``passed`` under the SWE-bench servers' field name, for tools that read either benchmark.
                "resolved": response.passed,
            }
        )


if __name__ == "__main__":
    TerminalBenchTaskImageResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBenchTaskImageResourcesServer.run_webserver()  # noqa: F401
