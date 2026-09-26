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
import re
import shlex
import tomllib
from base64 import b64decode
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError
from pytest import raises

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import toml_dumps
from responses_api_agents.codex_sandboxed_agent.app import (
    CONFIG_ENV_NAME,
    CodexSandboxedAgent,
    CodexSandboxedAgentConfig,
)


def _config(**overrides: Any) -> CodexSandboxedAgentConfig:
    kwargs: dict[str, Any] = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        openai_base_url="http://127.0.0.1:8100/v1",
        codex_model="switchyard/nemotron-3-nano",
        sandbox_provider="",
        sandbox_config=dict(),
        sandbox_timeout=2700,
    )
    kwargs.update(overrides)
    return CodexSandboxedAgentConfig(**kwargs)


def _agent(**overrides: Any) -> CodexSandboxedAgent:
    return CodexSandboxedAgent(config=_config(**overrides), server_client=MagicMock(spec=ServerClient))


def _codex_invocation(command: str) -> list[str]:
    """The argv of the `codex ...` invocation inside the composed shell command."""
    matches = re.findall(r"&& (codex exec .*?) < /dev/null\s+;\s+echo \"Codex run finished", command, flags=re.DOTALL)
    assert len(matches) == 1, command
    return shlex.split(matches[0])


class TestBuildConfig:
    def test_pins_provider_with_session_header_and_no_ambient_state(self) -> None:
        config = _agent()._build_config("rollout-123")
        # Round-trip through codex_agent's serializer: what the CLI actually reads.
        parsed = tomllib.loads(toml_dumps(config))

        assert parsed["model"] == "switchyard/nemotron-3-nano"
        assert parsed["model_provider"] == "gym"
        assert parsed["approval_policy"] == "never"
        assert parsed["sandbox_mode"] == "danger-full-access"
        assert parsed["check_for_update_on_startup"] is False
        assert parsed["analytics"] == {"enabled": False}
        assert parsed["history"] == {"persistence": "none"}
        assert parsed["features"] == {"multi_agent": False, "code_mode": False}
        provider = parsed["model_providers"]["gym"]
        assert provider["base_url"] == "http://127.0.0.1:8100/v1"
        assert provider["env_key"] == "OPENAI_API_KEY"
        assert provider["wire_api"] == "responses"
        assert provider["http_headers"] == {"x-switchyard-session-id": "rollout-123"}
        # Defaults to the sandbox budget so a whole generation fits the idle window.
        assert provider["stream_idle_timeout_ms"] == 2_700_000
        for absent in (
            "model_reasoning_effort",
            "model_context_window",
            "model_max_output_tokens",
            "developer_instructions",
        ):
            assert absent not in parsed, absent

    def test_optional_keys_and_extra_config(self) -> None:
        config = _agent(
            reasoning_effort="medium",
            model_context_window=131072,
            model_max_output_tokens=8192,
            stream_idle_timeout_ms=60_000,
            session_header_name=None,
            extra_config={"features": {"code_mode": True}, "model_verbosity": "low"},
        )._build_config("rollout-123", developer_instructions="Be terse.")
        parsed = tomllib.loads(toml_dumps(config))

        assert parsed["model_reasoning_effort"] == "medium"
        assert parsed["model_context_window"] == 131072
        assert parsed["model_max_output_tokens"] == 8192
        assert parsed["developer_instructions"] == "Be terse."
        assert parsed["model_providers"]["gym"]["stream_idle_timeout_ms"] == 60_000
        assert "http_headers" not in parsed["model_providers"]["gym"]
        # extra_config deep-merges: sibling feature flag kept, new top-level key added.
        assert parsed["features"] == {"multi_agent": False, "code_mode": True}
        assert parsed["model_verbosity"] == "low"


class TestBuildCommand:
    def test_default_install_downloads_pinned_release(self) -> None:
        command = _agent(codex_version="0.144.4")._build_command("fix the bug")

        assert "releases/download/rust-v0.144.4/codex-x86_64-unknown-linux-musl.tar.gz" in command
        assert "install -m 0755" in command

    def test_prestaged_binary_skips_network_installer(self) -> None:
        command = _agent(
            remote_codex_install_script_path="/opt/tools/codex/install-efb.sh",
            remote_codex_binary_path="/opt/tools/codex/codex",
        )._build_command("fix the bug")

        assert "bash /opt/tools/codex/install-efb.sh --binary /opt/tools/codex/codex" in command
        assert "curl" not in command
        assert "releases/download" not in command

    def test_stages_config_from_env_then_runs_headless_with_query_last(self) -> None:
        query = "Fix `foo()` in src/a.py; it's broken"
        command = _agent()._build_command(query)

        assert f'printf "%s" "${CONFIG_ENV_NAME}" | base64 -d > "$CODEX_HOME/config.toml"' in command
        argv = _codex_invocation(command)
        assert argv[:2] == ["codex", "exec"]
        assert "--json" in argv
        assert "--ephemeral" in argv
        assert "--skip-git-repo-check" in argv
        # The task text is passed as a single positional after `--`, quoted intact.
        assert argv[-2:] == ["--", query]

    def test_conda_activation_is_tolerated_to_fail(self) -> None:
        assert "source /opt/miniconda3/bin/activate && conda activate testbed || true" in _agent()._build_command("q")


class TestBuildEnv:
    def test_per_rollout_home_key_and_config_payload(self) -> None:
        toml = 'model = "x"\n\n[model_providers.gym]\nname = "gym"\n'
        env = _agent(openai_api_key="placeholder")._build_env("rollout-123", toml)

        assert env["CODEX_HOME"] == "/tmp/nemo-gym-codex-home-rollout-123"
        assert env["OPENAI_API_KEY"] == "placeholder"
        # Single line on the wire (enroot sources env as a KEY=VALUE file), exact bytes after decode.
        assert "\n" not in env[CONFIG_ENV_NAME]
        assert b64decode(env[CONFIG_ENV_NAME]).decode() == toml


class TestConfig:
    def test_prestaged_pair_must_be_complete(self) -> None:
        for script, binary in (("/opt/install.sh", None), (None, "/opt/codex")):
            with raises(ValidationError, match="must be set together"):
                _config(remote_codex_install_script_path=script, remote_codex_binary_path=binary)

    def test_prestaged_pair_accepts_both_or_neither(self) -> None:
        both = _config(remote_codex_install_script_path="/opt/install.sh", remote_codex_binary_path="/opt/codex")

        assert both.remote_codex_binary_path == "/opt/codex"
        assert _config().remote_codex_binary_path is None
