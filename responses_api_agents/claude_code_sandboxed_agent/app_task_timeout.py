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

"""Claude Code sandboxed agent that honours a per-task agent deadline.

Entrypoint variant of ``app.py``. A resources server may put the task's agent
budget (seconds) into the sandbox descriptor it returns from ``/seed_session``
under ``agent_timeout_s`` (``terminal_bench_4_local/app_task_image.py`` does,
from the dataset row). The CLI then runs under ``min(sandbox_timeout,
agent_timeout_s)`` instead of the one harness-wide ``sandbox_timeout``, and the
resources server grades whatever the container holds when the deadline ends the
run. A descriptor without the key behaves exactly as ``app.py``.
"""

from __future__ import annotations

from typing import Any, Dict

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.claude_code_sandboxed_agent.app import ClaudeCodeSandboxedAgent


AGENT_TIMEOUT_KEY = "agent_timeout_s"


class DeadlineSandbox:
    """A sandbox whose ``exec`` never runs longer than ``deadline_s``; everything else is the wrapped sandbox."""

    def __init__(self, sandbox: Any, deadline_s: float) -> None:
        if deadline_s <= 0:
            raise ValueError(f"deadline_s must be positive, got {deadline_s!r}")
        self._sandbox = sandbox
        self.deadline_s = float(deadline_s)

    async def exec(self, *args: Any, timeout_s: float | None = None, **kwargs: Any) -> Any:
        effective = self.deadline_s if timeout_s is None else min(float(timeout_s), self.deadline_s)
        return await self._sandbox.exec(*args, timeout_s=effective, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)


def split_agent_timeout(descriptor: Dict[str, Any]) -> tuple[Dict[str, Any], float | None]:
    """The descriptor without the deadline key, and the deadline (None when absent or empty)."""
    if AGENT_TIMEOUT_KEY not in descriptor:
        return descriptor, None
    rest = {key: value for key, value in descriptor.items() if key != AGENT_TIMEOUT_KEY}
    value = descriptor[AGENT_TIMEOUT_KEY]
    return rest, (float(value) if value not in (None, "") else None)


class ClaudeCodeTaskTimeoutAgent(ClaudeCodeSandboxedAgent):
    async def _connect_sandbox(self, descriptor: Dict[str, Any]) -> Any:
        descriptor, deadline_s = split_agent_timeout(dict(descriptor))
        sandbox = await super()._connect_sandbox(descriptor)
        return sandbox if deadline_s is None else DeadlineSandbox(sandbox, deadline_s)


if __name__ == "__main__":
    ClaudeCodeTaskTimeoutAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClaudeCodeTaskTimeoutAgent.run_webserver()  # noqa: F401
