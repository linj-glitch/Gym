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

"""Claude Code CLI harness running inside a provider sandbox.

This is the sandboxed sibling of ``responses_api_agents/claude_code_agent``
(which runs the CLI on the agent-server host): it speaks the RS-owned-sandbox
contract used by DeepSWE — the resources server creates the task sandbox in
``/seed_session``, this agent connects to it, runs Claude Code inside, and the
resources server grades the committed diff in a fresh sandbox at ``/verify``.
CLI conventions and output parsing are reused from ``claude_code_agent``
(``parse_stream_json`` and the invocation-outcome classification) so the two
agents stay behaviorally aligned.

Claude Code speaks the Anthropic Messages API, so its model traffic goes
straight from the sandbox to an Anthropic-compatible endpoint
(``anthropic_base_url``) — typically a Switchyard proxy, which translates
Messages-API traffic to whatever its upstream target speaks and can route
(passthrough / escalation / …). Because the endpoint is client-addressed, this
agent forwards Gym's per-rollout session id as a custom request header
(default ``x-switchyard-session-id``) via ANTHROPIC_CUSTOM_HEADERS, so router
session affinity (e.g. Switchyard's escalation latch) and proxy-side
attribution work per rollout.
"""

import json
import sys
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from responses_api_agents.claude_code_agent.app import _invocation_outcome, parse_stream_json


class ClaudeCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef

    # Anthropic-compatible endpoint root the sandboxed CLI talks to (no /v1
    # suffix; the client appends /v1/messages). Typically a Switchyard proxy.
    anthropic_base_url: str
    # Bearer sent by the CLI. A local Switchyard proxy has no inbound auth, so
    # any placeholder works there; never put a real provider key here.
    anthropic_auth_token: str = "dummy"
    # Model name the CLI requests — for Switchyard, a configured route id.
    claude_model: str
    # Native-installer version pin ("latest", "stable", or an exact version).
    # Used only when the pre-staged pair below is unset.
    claude_code_version: str = "stable"
    # Pre-staged CLI, the way opencode_sandboxed_agent's remote_opencode_* pair
    # works: when both are set the install step runs
    # `bash <script> --binary <binary>` (typically a symlink into PATH) instead
    # of downloading the installer, so every sandbox runs one known build with
    # no network dependency. Set both or neither.
    remote_claude_install_script_path: Optional[str] = None
    remote_claude_binary_path: Optional[str] = None
    # Hard cap on agentic turns per task; None defers to the CLI default.
    claude_max_turns: Optional[int] = None
    # Cap on tokens per model reply (CLAUDE_CODE_MAX_OUTPUT_TOKENS). Worth
    # setting when the endpoint's context window is smaller than the CLI
    # assumes; None keeps the CLI default.
    claude_max_output_tokens: Optional[int] = None
    # Skip Claude Code's native auto-discovery (hooks, plugins, MCP, memory,
    # CLAUDE.md) for an isolated, reproducible runtime — mirrors
    # claude_code_agent's `bare` option.
    bare: bool = True
    # Tools withheld from the full set (``--disallowed-tools``) when ``bare``
    # is false. WebSearch is Anthropic's server-side tool and returns nothing
    # through a proxy; WebFetch is withheld so a benchmark rollout cannot read
    # the upstream fix online.
    disallowed_tools: List[str] = ["WebSearch", "WebFetch"]
    # Claude Code's ultracode mode: ``--effort ultracode`` (xhigh effort plus a
    # standing instruction to use the Workflow tool, which runs subagents in
    # parallel) with CLAUDE_CODE_WORKFLOWS=true, the variable that gates the
    # Workflow tool. Needs the full tool set, so ``bare`` must be false.
    ultracode: bool = False
    # CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS; None keeps the CLI default.
    workflow_max_concurrent_agents: Optional[int] = None
    # CLAUDE_CODE_MAX_CONTEXT_TOKENS. The CLI does not know the window of a
    # model id it does not recognise (a route id); set the served window so
    # its auto-compaction fires relative to the real one. None keeps the CLI
    # default.
    claude_max_context_tokens: Optional[int] = None
    # Extra CLI arguments, appended verbatim before the query.
    claude_extra_args: List[str] = []
    # Header carrying Gym's rollout session id to the proxy. None disables.
    session_header_name: Optional[str] = "x-switchyard-session-id"

    # Sandbox config (same contract as opencode_sandboxed_agent).
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

    debug: bool = False

    @model_validator(mode="after")
    def _prestaged_pair_is_complete(self) -> "ClaudeCodeSandboxedAgentConfig":
        script, binary = self.remote_claude_install_script_path, self.remote_claude_binary_path
        if (script is None) != (binary is None):
            raise ValueError(
                "remote_claude_install_script_path and remote_claude_binary_path must be set together "
                f"(got script={script!r}, binary={binary!r})"
            )
        return self

    @model_validator(mode="after")
    def _ultracode_needs_the_full_tool_set(self) -> "ClaudeCodeSandboxedAgentConfig":
        if self.ultracode and self.bare:
            raise ValueError(
                "ultracode=true needs the Workflow and Agent tools, which --bare removes; set bare=false as well"
            )
        return self


class ClaudeCodeSandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSandboxedAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    claude_run_stdout: str
    claude_run_stderr: str
    claude_finished: bool
    claude_status: str
    claude_error_type: Optional[str] = None
    claude_num_turns: Optional[int] = None
    claude_is_error: Optional[bool] = None
    # Tool-use counts by tool name across the main thread and any subagent
    # events in the stream, plus the result event's ``subagent_stats``; see
    # ``subagent_summary``. Both are None when the CLI produced no stream.
    claude_tool_use_counts: Optional[Dict[str, int]] = None
    claude_subagent_stats: Optional[Dict[str, Any]] = None


def subagent_summary(stdout: str) -> Dict[str, Any]:
    """How much of a stream-json run was delegated.

    Returns ``tool_use_counts`` (tool name -> calls, main thread and subagent
    events alike), ``subagent_events`` (stream objects carrying a
    ``parent_tool_use_id``, i.e. produced inside an Agent-tool subagent) and
    ``subagent_stats`` (the result event's own counter, which covers the Agent
    tool only: Workflow-script agents do not appear in the parent's stream, so
    their work shows up here only as ``Workflow`` calls).
    """
    tool_use_counts: Dict[str, int] = {}
    subagent_events = 0
    subagent_stats: Optional[Dict[str, Any]] = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("parent_tool_use_id"):
            subagent_events += 1
        if event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = str(block.get("name"))
                    tool_use_counts[name] = tool_use_counts.get(name, 0) + 1
        elif event.get("type") == "result" and isinstance(event.get("subagent_stats"), dict):
            subagent_stats = event["subagent_stats"]
    return {
        "tool_use_counts": tool_use_counts,
        "subagent_events": subagent_events,
        "subagent_stats": subagent_stats,
    }


class ClaudeCodeSandboxedAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _connect_sandbox(self, sandbox_id: str) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        return await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=resolved_sandbox_provider)

    def _build_command(self, query: str) -> str:
        """Shell command for one Claude Code run inside the sandbox.

        The CLI flags mirror claude_code_agent's ``_build_command`` (headless
        stream-json run with permissions skipped); installation and the conda
        dance are sandbox-path specifics.
        """
        # Same conda dance as the OpenCode harness: SWE task images may carry a
        # testbed env; activation is tolerated to fail on images without it.
        conda_activate = "{ source /opt/miniconda3/bin/activate && conda activate testbed || true; }"

        if self.config.remote_claude_install_script_path and self.config.remote_claude_binary_path:
            # Pre-staged build (mounted or uploaded); the script puts it on PATH.
            install = (
                f"bash {quote(self.config.remote_claude_install_script_path)} "
                f"--binary {quote(self.config.remote_claude_binary_path)}"
            )
        else:
            print(
                "Downloading and installing Claude Code in the sandbox. Consider mounting a pre-staged binary "
                "(remote_claude_binary_path + remote_claude_install_script_path) instead.",
                file=sys.stderr,
            )
            install = (
                "if ! command -v claude >/dev/null 2>&1; then "
                'installer=$(mktemp) && curl -fsSL -o "$installer" https://claude.ai/install.sh '
                f'&& bash "$installer" {quote(self.config.claude_code_version)}; fi '
                '&& export PATH="$HOME/.local/bin:$PATH"'
            )

        claude_args = "-p --output-format stream-json --verbose --dangerously-skip-permissions"
        if self.config.bare:
            # --bare sets CLAUDE_CODE_SIMPLE=1: three tools (Bash, Edit, Read),
            # no Agent, Workflow, Write, Grep or Glob.
            claude_args += " --bare"
        else:
            # The full tool set, keeping the two guards --bare provided:
            # settings from the user scope only (the sandbox has none), so a
            # repository's own .claude/settings.json cannot add hooks, and no
            # MCP servers from a repository's .mcp.json.
            claude_args += " --setting-sources user --strict-mcp-config"
            if self.config.disallowed_tools:
                claude_args += f" --disallowed-tools {quote(','.join(self.config.disallowed_tools))}"
        if self.config.ultracode:
            claude_args += " --effort ultracode"
        claude_args += f" --model {quote(self.config.claude_model)}"
        if self.config.claude_max_turns is not None:
            claude_args += f" --max-turns {self.config.claude_max_turns}"
        for arg in self.config.claude_extra_args:
            claude_args += f" {quote(arg)}"

        # stream-json events go to stdout; parse_stream_json skips the non-JSON
        # install/echo lines around them.
        return f"""
        echo "Shell: $SHELL" \
        && {conda_activate} \
        && echo "Optionally activated Conda env" \
        && {install} \
        && echo "Installed Claude Code: $(claude --version)" \
        && claude {claude_args} -- {quote(query)} \
        ;  echo "Claude Code run finished rc=$?"
        """

    def _build_env(self, session_id: str) -> Dict[str, str]:
        """Environment for the sandboxed CLI — model aliases mirror
        claude_code_agent's ``_run_claude_code``; the base URL and session
        header are what point traffic at the (routing) proxy."""
        model = self.config.claude_model
        env = {
            "ANTHROPIC_BASE_URL": self.config.anthropic_base_url,
            "ANTHROPIC_AUTH_TOKEN": self.config.anthropic_auth_token,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            # Container runs as (remapped) root; this acknowledges the sandbox
            # so --dangerously-skip-permissions is accepted.
            "IS_SANDBOX": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if self.config.claude_max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.config.claude_max_output_tokens)
        if self.config.claude_max_context_tokens is not None:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(self.config.claude_max_context_tokens)
        if self.config.ultracode:
            # The Workflow tool is gated by this variable; without it the CLI
            # refuses --effort ultracode.
            env["CLAUDE_CODE_WORKFLOWS"] = "true"
        if self.config.workflow_max_concurrent_agents is not None:
            env["CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS"] = str(self.config.workflow_max_concurrent_agents)
        if self.config.session_header_name:
            env["ANTHROPIC_CUSTOM_HEADERS"] = f"{self.config.session_header_name}: {session_id}"
        return env

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]
        session_id = request.session[SESSION_ID_KEY]

        query = None
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]
        assert query, body.input

        command = self._build_command(query)
        env = self._build_env(session_id)
        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\nwith env {env}", file=sys.stderr)

        try:
            result = await sandbox.exec(command=command, timeout_s=self.config.sandbox_timeout, env=env)
        except:
            result = None
            print("Claude Code exec hit error.", format_exc(), file=sys.stderr)

        if self.config.debug and result:
            print("Claude Code install and run stdout:\n", result.stdout, file=sys.stderr)
            print("Claude Code install and run stderr:\n", result.stderr, file=sys.stderr)

        # Reuse claude_code_agent's maintained stream-json parser: full
        # trajectory (assistant messages incl. <think>, tool calls + outputs)
        # plus usage/turn metadata, identical to the host-run agent.
        output = []
        metadata: Dict[str, Any] = {}
        delegation: Dict[str, Any] = {}
        if result and result.stdout:
            output, metadata = parse_stream_json(result.stdout)
            delegation = subagent_summary(result.stdout)
        status, error_type = _invocation_outcome(metadata, result.return_code if result else None)

        usage = None
        if metadata.get("input_tokens") or metadata.get("output_tokens"):
            input_tokens = int(metadata.get("input_tokens") or 0)
            output_tokens = int(metadata.get("output_tokens") or 0)
            usage = NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            )

        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = {
            "claude_run_stdout": (result.stdout if result else "") or "",
            "claude_run_stderr": (result.stderr if result else "") or "",
            "claude_finished": status == "completed",
            "claude_status": status,
            "claude_error_type": error_type,
            "claude_num_turns": metadata.get("num_turns"),
            "claude_is_error": metadata.get("is_error"),
            "claude_tool_use_counts": delegation.get("tool_use_counts"),
            "claude_subagent_stats": delegation.get("subagent_stats"),
        }

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.claude_model,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def run(
        self, request: Request, body: ClaudeCodeSandboxedAgentRunRequest
    ) -> ClaudeCodeSandboxedAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies

        seed_session_result = await seed_session_response.json()
        sandbox = await self._connect_sandbox(sandbox_id=seed_session_result["sandbox_handle"])
        self._sandbox_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox

        cookies["sandbox_id"] = request.session[SESSION_ID_KEY]
        request._cookies = cookies
        response = await self.responses(request, body.responses_create_params)

        verify_request = ClaudeCodeSandboxedAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        try:
            await sandbox.stop()
        except:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)
        self._sandbox_id_to_sandbox.pop(request.session[SESSION_ID_KEY])

        response_dict = await get_response_json(verify_response)
        response_dict |= self._sandbox_id_to_run_result[cookies["sandbox_id"]]
        return ClaudeCodeSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    ClaudeCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ClaudeCodeSandboxedAgent.run_webserver()  # noqa: F401
