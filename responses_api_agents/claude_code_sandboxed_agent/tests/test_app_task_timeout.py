# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from responses_api_agents.claude_code_sandboxed_agent.app_task_timeout import (
    AGENT_TIMEOUT_KEY,
    DeadlineSandbox,
    split_agent_timeout,
)


class _Sandbox:
    def __init__(self) -> None:
        self.calls = []
        self.stopped = False

    async def exec(self, command=None, *, timeout_s=None, env=None):
        self.calls.append((command, timeout_s, env))
        return "result"

    async def stop(self) -> None:
        self.stopped = True


def test_deadline_sandbox_clamps_exec_timeout_and_delegates() -> None:
    inner = _Sandbox()
    sandbox = DeadlineSandbox(inner, 2700)
    assert asyncio.run(sandbox.exec(command="claude", timeout_s=10800, env={"A": "1"})) == "result"
    assert asyncio.run(sandbox.exec(command="short", timeout_s=60)) == "result"
    assert asyncio.run(sandbox.exec(command="unset")) == "result"
    assert inner.calls == [("claude", 2700.0, {"A": "1"}), ("short", 60.0, None), ("unset", 2700.0, None)]
    asyncio.run(sandbox.stop())
    assert inner.stopped
    with pytest.raises(ValueError):
        DeadlineSandbox(inner, 0)


def test_split_agent_timeout() -> None:
    descriptor = {"sandbox_id": "abc", "workdir": "/app", AGENT_TIMEOUT_KEY: 2700}
    rest, deadline = split_agent_timeout(descriptor)
    assert rest == {"sandbox_id": "abc", "workdir": "/app"} and deadline == 2700.0
    assert AGENT_TIMEOUT_KEY in descriptor  # the input is not mutated
    same, none = split_agent_timeout({"sandbox_id": "abc"})
    assert same == {"sandbox_id": "abc"} and none is None
    assert split_agent_timeout({"sandbox_id": "abc", AGENT_TIMEOUT_KEY: None})[1] is None
