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

"""Codex CLI harness running inside a provider sandbox.

The sandboxed sibling of ``responses_api_agents/codex_agent`` (which runs the
CLI on the agent-server host): it speaks the resources-server-owned-sandbox
contract used by the ``swebench`` and ``deepswe`` resources servers — the
resources server creates the task sandbox in ``/seed_session``, this agent
connects to it, runs ``codex exec --json`` inside, and the resources server
grades the working-tree diff at ``/verify``. The generated ``config.toml`` and
the JSONL event parsing are reused from ``codex_agent`` so the two agents stay
behaviorally aligned.

Codex speaks the OpenAI Responses API over SSE, so its model traffic goes
straight from the sandbox to ``openai_base_url`` (the CLI appends
``/responses``) — typically a Switchyard proxy, which translates Responses
traffic to whatever its routed target speaks and can route across targets.
Because the endpoint is client-addressed, this agent puts Gym's per-rollout
session id on the provider's ``http_headers`` (default
``x-switchyard-session-id``) so router session affinity and proxy-side
attribution work per rollout.
"""

import sys
from base64 import b64encode
from copy import deepcopy
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field, model_validator

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
from responses_api_agents.codex_agent.app import _deep_merge, _extract_instruction, parse_exec_jsonl, toml_dumps


# The generated config.toml travels to the sandbox in this variable, base64
# encoded, and is decoded into $CODEX_HOME by the composed command. Encoding
# keeps TOML quoting out of the shell command line and, more importantly,
# keeps the value on one line: providers that hand env to the container as a
# sourced `KEY=VALUE` file (enroot) split a multi-line value into garbage.
CONFIG_ENV_NAME = "NEMO_GYM_CODEX_CONFIG_TOML_B64"


class CodexSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef

    # OpenAI-compatible endpoint the sandboxed CLI talks to. Must include the
    # /v1 suffix; Codex appends /responses. Typically a Switchyard proxy.
    openai_base_url: str
    # Bearer sent by the CLI. A local Switchyard proxy has no inbound auth, so
    # any placeholder works there; never put a real provider key here.
    openai_api_key: str = "local"  # pragma: allowlist secret
    # Model name the CLI requests — for Switchyard, a configured route id. An
    # explicit name also keeps Codex from applying model-family feature gating.
    codex_model: str
    # Release used by the in-sandbox network installer (GitHub release tarball).
    # Used only when the pre-staged pair below is unset.
    codex_version: str = "0.144.4"
    # Pre-staged CLI: when both are set the install step runs
    # `bash <script> --binary <binary>` (typically a symlink into PATH) instead
    # of downloading a release, so every sandbox runs one known build with no
    # network dependency. Set both or neither.
    remote_codex_install_script_path: Optional[str] = None
    remote_codex_binary_path: Optional[str] = None
    # Passed as `model_reasoning_effort`; None omits it.
    reasoning_effort: Optional[str] = None
    # Codex has no catalog entry for a proxy route id, so tell it the served
    # window and the reply cap explicitly; None omits each key.
    model_context_window: Optional[int] = None
    model_max_output_tokens: Optional[int] = None
    # Provider stream idle budget. A translating proxy in front of a
    # non-streaming upstream emits the SSE only once the reply is complete, so
    # this must cover a whole generation; None -> sandbox_timeout * 1000.
    stream_idle_timeout_ms: Optional[int] = None
    # Header carrying Gym's rollout session id to the proxy. None disables.
    session_header_name: Optional[str] = "x-switchyard-session-id"
    # Extra config.toml content deep-merged over the generated base config.
    extra_config: Dict[str, Any] = Field(default_factory=dict)

    # Sandbox config (same contract as opencode_sandboxed_agent).
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

    debug: bool = False

    @model_validator(mode="after")
    def _prestaged_pair_is_complete(self) -> "CodexSandboxedAgentConfig":
        script, binary = self.remote_codex_install_script_path, self.remote_codex_binary_path
        if (script is None) != (binary is None):
            raise ValueError(
                "remote_codex_install_script_path and remote_codex_binary_path must be set together "
                f"(got script={script!r}, binary={binary!r})"
            )
        return self


class CodexSandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexSandboxedAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class CodexSandboxedAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    codex_run_stdout: str
    codex_run_stderr: str
    codex_finished: bool
    codex_status: str
    codex_errors: List[str] = Field(default_factory=list)


class CodexSandboxedAgent(SimpleResponsesAPIAgent):
    config: CodexSandboxedAgentConfig

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

    def _build_config(self, session_id: str, developer_instructions: Optional[str] = None) -> Dict[str, Any]:
        """The per-rollout CODEX_HOME/config.toml content.

        Same base as codex_agent's ``_build_config`` — a pinned provider that
        bypasses the login flow, nothing that phones home or depends on ambient
        host state, and no tools a proxied model cannot drive — plus the
        rollout session header on the provider and the window/cap hints a
        route id has no catalog entry for.
        """
        provider: Dict[str, Any] = {
            "name": "gym",
            "base_url": self.config.openai_base_url,
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "stream_idle_timeout_ms": self.config.stream_idle_timeout_ms or int(self.config.sandbox_timeout * 1000),
        }
        if self.config.session_header_name:
            provider["http_headers"] = {self.config.session_header_name: session_id}

        config: Dict[str, Any] = {
            "model": self.config.codex_model,
            "model_provider": "gym",
            "approval_policy": "never",
            # The task container is the sandbox; Codex's own sandboxing
            # (landlock/bwrap) is not available inside it.
            "sandbox_mode": "danger-full-access",
            "web_search": "disabled",
            "check_for_update_on_startup": False,
            "analytics": {"enabled": False},
            "history": {"persistence": "none"},
            "features": {"multi_agent": False, "code_mode": False},
            "model_providers": {"gym": provider},
        }
        if self.config.reasoning_effort:
            config["model_reasoning_effort"] = self.config.reasoning_effort
        if self.config.model_context_window is not None:
            config["model_context_window"] = self.config.model_context_window
        if self.config.model_max_output_tokens is not None:
            config["model_max_output_tokens"] = self.config.model_max_output_tokens
        if developer_instructions:
            config["developer_instructions"] = developer_instructions
        if self.config.extra_config:
            config = _deep_merge(config, deepcopy(self.config.extra_config))
        return config

    def _build_command(self, query: str) -> str:
        """Shell command for one Codex run inside the sandbox.

        The CLI flags mirror codex_agent's ``_build_command`` (headless JSONL
        run, no session persistence); installation, the conda dance and the
        CODEX_HOME staging are sandbox-path specifics. Codex appends piped
        stdin to the prompt and blocks on it, hence ``< /dev/null``.
        """
        # Same conda dance as the OpenCode harness: SWE task images may carry a
        # testbed env; activation is tolerated to fail on images without it.
        conda_activate = "{ source /opt/miniconda3/bin/activate && conda activate testbed || true; }"

        if self.config.remote_codex_install_script_path and self.config.remote_codex_binary_path:
            # Pre-staged build (mounted or uploaded); the script puts it on PATH.
            install = (
                f"bash {quote(self.config.remote_codex_install_script_path)} "
                f"--binary {quote(self.config.remote_codex_binary_path)}"
            )
        else:
            print(
                "Downloading and installing Codex in the sandbox. Consider mounting a pre-staged binary "
                "(remote_codex_binary_path + remote_codex_install_script_path) instead.",
                file=sys.stderr,
            )
            tarball = (
                "https://github.com/openai/codex/releases/download/"
                f"rust-v{quote(self.config.codex_version)}/codex-x86_64-unknown-linux-musl.tar.gz"
            )
            install = (
                "if ! command -v codex >/dev/null 2>&1; then "
                f'tmp=$(mktemp -d) && curl -fsSL -o "$tmp/codex.tar.gz" {tarball} '
                '&& tar -xzf "$tmp/codex.tar.gz" -C "$tmp" '
                '&& install -m 0755 "$tmp"/codex-x86_64-unknown-linux-musl /usr/local/bin/codex; fi'
            )

        # $CODEX_HOME and the (base64) config content arrive via the exec environment.
        stage_home = (
            f'mkdir -p "$CODEX_HOME" && printf "%s" "${CONFIG_ENV_NAME}" | base64 -d > "$CODEX_HOME/config.toml"'
        )

        codex_args = "exec --json --ephemeral --skip-git-repo-check"

        return f"""
        echo "Shell: $SHELL" \
        && {conda_activate} \
        && echo "Optionally activated Conda env" \
        && {install} \
        && echo "Installed Codex: $(codex --version)" \
        && {stage_home} \
        && codex {codex_args} -- {quote(query)} < /dev/null \
        ;  echo "Codex run finished rc=$?"
        """

    def _build_env(self, session_id: str, config_toml: str) -> Dict[str, str]:
        """Environment for the sandboxed CLI.

        CODEX_HOME is per rollout so concurrent rollouts in one image never
        share config or state; OPENAI_API_KEY is the provider's ``env_key``.
        The config travels base64 encoded (single line) for the reason given
        at ``CONFIG_ENV_NAME``.
        """
        return {
            "CODEX_HOME": f"/tmp/nemo-gym-codex-home-{session_id}",
            "OPENAI_API_KEY": self.config.openai_api_key,
            CONFIG_ENV_NAME: b64encode(config_toml.encode("utf-8")).decode("ascii"),
            # Terminal-free run; keep the CLI's own logging off stdout's JSONL.
            "NO_COLOR": "1",
        }

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]
        session_id = request.session[SESSION_ID_KEY]

        query, system_message = _extract_instruction(body.input)
        assert query, body.input

        config_toml = toml_dumps(self._build_config(session_id, developer_instructions=system_message))
        command = self._build_command(query)
        env = self._build_env(session_id, config_toml)
        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\nwith config.toml:\n{config_toml}", file=sys.stderr)

        try:
            result = await sandbox.exec(command=command, timeout_s=self.config.sandbox_timeout, env=env)
        except:
            result = None
            print("Codex exec hit error.", format_exc(), file=sys.stderr)

        if self.config.debug and result:
            print("Codex install and run stdout:\n", result.stdout, file=sys.stderr)
            print("Codex install and run stderr:\n", result.stderr, file=sys.stderr)

        # Reuse codex_agent's maintained JSONL parser: full trajectory
        # (assistant messages incl. <think>, tool calls + outputs) plus usage.
        output: List[Any] = []
        metadata: Dict[str, Any] = {}
        if result and result.stdout:
            output, metadata = parse_exec_jsonl(result.stdout)
        # Codex reports advisories (e.g. "Model metadata for <route> not found")
        # as `error` items too, so the list is informative but not a verdict:
        # a non-zero exit or an exec failure is, and so is a run that never
        # produced an assistant message.
        errors = [str(error) for error in metadata.get("errors") or []]
        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant" for item in output
        )
        if result is None:
            status = "exec_error"
        elif result.return_code not in (0, None):
            status = "error"
        elif not has_assistant_message:
            status = "no_output"
        else:
            status = "completed"

        usage = None
        if metadata.get("input_tokens") or metadata.get("output_tokens"):
            input_tokens = int(metadata.get("input_tokens") or 0)
            output_tokens = int(metadata.get("output_tokens") or 0)
            usage = NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=int(metadata.get("cached_input_tokens") or 0)
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=int(metadata.get("reasoning_tokens") or 0)
                ),
                total_tokens=input_tokens + output_tokens,
            )

        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = {
            "codex_run_stdout": (result.stdout if result else "") or "",
            "codex_run_stderr": (result.stderr if result else "") or "",
            "codex_finished": status == "completed",
            "codex_status": status,
            "codex_errors": errors,
        }

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.codex_model,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def run(self, request: Request, body: CodexSandboxedAgentRunRequest) -> CodexSandboxedAgentVerifyResponse:
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

        verify_request = CodexSandboxedAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
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
        return CodexSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    CodexSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CodexSandboxedAgent.run_webserver()  # noqa: F401
