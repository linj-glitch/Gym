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
import json
import re
import shlex
from typing import Any, Optional
from unittest.mock import MagicMock

from pydantic import ValidationError
from pytest import raises

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.claude_code_sandboxed_agent.app import (
    ClaudeCodeSandboxedAgent,
    ClaudeCodeSandboxedAgentConfig,
    subagent_summary,
)


def _config(**overrides: Any) -> ClaudeCodeSandboxedAgentConfig:
    kwargs: dict[str, Any] = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        anthropic_base_url="http://127.0.0.1:8100",
        claude_model="switchyard/nemotron-3-nano",
        sandbox_provider="",
        sandbox_config=dict(),
        sandbox_timeout=0,
    )
    kwargs.update(overrides)
    return ClaudeCodeSandboxedAgentConfig(**kwargs)


def _agent(**overrides: Any) -> ClaudeCodeSandboxedAgent:
    return ClaudeCodeSandboxedAgent(config=_config(**overrides), server_client=MagicMock(spec=ServerClient))


def _claude_invocation(command: str) -> list[str]:
    """The argv of the `claude ...` invocation inside the composed shell command."""
    matches = re.findall(r"&& (claude .*?)\s+;\s+echo \"Claude Code run finished", command, flags=re.DOTALL)
    assert len(matches) == 1, command
    return shlex.split(matches[0])


class TestBuildCommand:
    def test_default_install_downloads_pinned_version(self) -> None:
        command = _agent(claude_code_version="2.1.283")._build_command("fix the bug")

        assert "curl -fsSL" in command
        assert 'bash "$installer" 2.1.283' in command
        assert 'export PATH="$HOME/.local/bin:$PATH"' in command

    def test_prestaged_binary_skips_network_installer(self) -> None:
        command = _agent(
            remote_claude_install_script_path="/opt/tools/claude-code/install-efb.sh",
            remote_claude_binary_path="/opt/tools/claude-code/claude",
        )._build_command("fix the bug")

        assert "bash /opt/tools/claude-code/install-efb.sh --binary /opt/tools/claude-code/claude" in command
        assert "curl" not in command
        assert "install.sh" not in command

    def test_claude_argv_headless_stream_json_with_query_last(self) -> None:
        query = "Fix `foo()` in src/a.py; it's broken"
        argv = _claude_invocation(_agent()._build_command(query))

        assert argv[0] == "claude"
        assert "-p" in argv
        assert argv[argv.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in argv
        assert "--dangerously-skip-permissions" in argv
        assert "--bare" in argv
        assert argv[argv.index("--model") + 1] == "switchyard/nemotron-3-nano"
        assert "--max-turns" not in argv
        # The task text is passed as a single positional after `--`, quoted intact.
        assert argv[-2:] == ["--", query]

    def test_optional_flags(self) -> None:
        argv = _claude_invocation(_agent(bare=False, claude_max_turns=40)._build_command("q"))

        assert "--bare" not in argv
        assert argv[argv.index("--max-turns") + 1] == "40"

    def test_conda_activation_is_tolerated_to_fail(self) -> None:
        command = _agent()._build_command("q")

        assert "source /opt/miniconda3/bin/activate && conda activate testbed || true" in command


class TestBuildEnv:
    def test_points_every_model_alias_at_the_route_and_forwards_session_header(self) -> None:
        env = _agent()._build_env("rollout-123")

        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8100"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "dummy"
        for key in (
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ):
            assert env[key] == "switchyard/nemotron-3-nano", key
        assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-switchyard-session-id: rollout-123"
        assert env["IS_SANDBOX"] == "1"
        assert env["DISABLE_AUTOUPDATER"] == "1"
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in env

    def test_session_header_can_be_disabled_and_output_tokens_capped(self) -> None:
        env = _agent(session_header_name=None, claude_max_output_tokens=16384)._build_env("rollout-123")

        assert "ANTHROPIC_CUSTOM_HEADERS" not in env
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16384"


class TestConfig:
    def test_prestaged_pair_must_be_complete(self) -> None:
        for script, binary in (("/opt/install.sh", None), (None, "/opt/claude")):
            with raises(ValidationError, match="must be set together"):
                _config(remote_claude_install_script_path=script, remote_claude_binary_path=binary)

    def test_prestaged_pair_accepts_both_or_neither(self) -> None:
        both = _config(remote_claude_install_script_path="/opt/install.sh", remote_claude_binary_path="/opt/claude")
        neither: Optional[str] = _config().remote_claude_binary_path

        assert both.remote_claude_binary_path == "/opt/claude"
        assert neither is None


class TestFullToolSetAndUltracode:
    def test_full_tool_set_keeps_the_guards_and_withholds_web_tools(self) -> None:
        argv = _claude_invocation(_agent(bare=False)._build_command("q"))
        assert "--bare" not in argv
        assert argv[argv.index("--setting-sources") + 1] == "user"
        assert "--strict-mcp-config" in argv
        assert argv[argv.index("--disallowed-tools") + 1] == "WebSearch,WebFetch"
        assert "--effort" not in argv

    def test_bare_run_has_no_full_set_flags(self) -> None:
        argv = _claude_invocation(_agent(bare=True)._build_command("q"))
        assert "--bare" in argv
        for flag in ("--setting-sources", "--strict-mcp-config", "--disallowed-tools", "--effort"):
            assert flag not in argv

    def test_ultracode_adds_effort_flag_and_workflow_env(self) -> None:
        agent = _agent(bare=False, ultracode=True, workflow_max_concurrent_agents=4, claude_max_context_tokens=131072)
        argv = _claude_invocation(agent._build_command("q"))
        assert argv[argv.index("--effort") + 1] == "ultracode"
        env = agent._build_env("sess")
        assert env["CLAUDE_CODE_WORKFLOWS"] == "true"
        assert env["CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS"] == "4"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"

    def test_plain_run_sets_none_of_the_ultracode_env(self) -> None:
        env = _agent()._build_env("sess")
        for key in ("CLAUDE_CODE_WORKFLOWS", "CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS", "CLAUDE_CODE_MAX_CONTEXT_TOKENS"):
            assert key not in env

    def test_ultracode_refuses_bare(self) -> None:
        with raises(ValidationError, match="bare=false"):
            _config(bare=True, ultracode=True)

    def test_disallowed_tools_can_be_emptied_and_extra_args_are_quoted(self) -> None:
        argv = _claude_invocation(_agent(bare=False, disallowed_tools=[], claude_extra_args=["--max-budget-usd", "1 2"])._build_command("q"))
        assert "--disallowed-tools" not in argv
        assert argv[argv.index("--max-budget-usd") + 1] == "1 2"
        assert argv[-1] == "q"

    def test_subagent_summary_counts_delegation(self) -> None:
        lines = [
            "Installed Claude Code: 2.1.283",
            json.dumps({"type": "system", "subtype": "init", "tools": ["Task", "Workflow", "Bash"]}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Workflow", "input": {"script": "x"}},
                {"type": "tool_use", "id": "t2", "name": "Task", "input": {"prompt": "explore"}},
            ]}}),
            json.dumps({"type": "assistant", "parent_tool_use_id": "t2", "message": {"content": [
                {"type": "tool_use", "id": "t3", "name": "Bash", "input": {"command": "ls"}},
            ]}}),
            json.dumps({"type": "user", "parent_tool_use_id": "t2", "message": {"content": []}}),
            json.dumps({"type": "result", "subtype": "success", "subagent_stats": {"spawned": 1, "completed": 1}}),
            "Claude Code run finished rc=0",
        ]
        summary = subagent_summary("\n".join(lines))
        assert summary == {
            "tool_use_counts": {"Workflow": 1, "Task": 1, "Bash": 1},
            "subagent_events": 2,
            "subagent_stats": {"spawned": 1, "completed": 1},
        }

    def test_subagent_summary_of_an_empty_stream(self) -> None:
        assert subagent_summary("no json here") == {"tool_use_counts": {}, "subagent_events": 0, "subagent_stats": None}

    def test_append_system_prompt_is_quoted_and_precedes_the_query(self) -> None:
        text = "Delegate: run several Task agents in parallel; it's faster"
        argv = _claude_invocation(_agent(claude_append_system_prompt=text)._build_command("q"))
        assert argv[argv.index("--append-system-prompt") + 1] == text
        assert argv[-1] == "q"
        assert "--append-system-prompt" not in _claude_invocation(_agent()._build_command("q"))
