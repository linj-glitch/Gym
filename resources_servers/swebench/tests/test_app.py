# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import patch_extraction_command, strip_binary_diffs
from resources_servers.swebench.app import (
    DockerContainer,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchVerifyResponse,
)


def make_sandbox(
    *,
    exec_result: SandboxExecResult | None = None,
    exec_error: Exception | None = None,
    upload_error: Exception | None = None,
    stop_error: Exception | None = None,
) -> MagicMock:
    sandbox = MagicMock()
    sandbox._handle = SandboxHandle(sandbox_id="sandbox-123", provider_name="test-provider", raw=None)
    sandbox.exec = AsyncMock(return_value=exec_result, side_effect=exec_error)
    sandbox.upload = AsyncMock(side_effect=upload_error)
    sandbox.stop = AsyncMock(side_effect=stop_error)
    return sandbox


class TestApp:
    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=True,
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()

        client = TestClient(app)

        eval_sandbox = make_sandbox()
        monkeypatch.setattr(
            "resources_servers.swebench.app.SwebenchResourcesServer._create_sandbox",
            AsyncMock(return_value=eval_sandbox),
        )
        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value=dict(resolved=True, completed=True)),
        )

        res = client.post(
            "/verify",
            json={
                "repo": "astropy/astropy",
                "instance_id": "my instance_id",
                "base_commit": "my base_commit",
                "patch": "my patch",
                "test_patch": "my test_patch",
                "problem_statement": "my problem_statement",
                "hints_text": "",
                "created_at": "my created_at",
                "version": "4.3",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "environment_setup_commit": "my environment_setup_commit",
                "difficulty": "my difficulty",
                "responses_create_params": {"input": []},
                "response": {
                    "output": [],
                    "id": "",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
                "subset": "my subset",
                "split": "my split",
            },
        )
        assert res.status_code == 200
        observation = res.json()["verifier_sandbox_observation"]
        assert observation.pop("wall_time_s") >= 0
        assert observation == {
            "kind": "sandbox",
            "role": "verifier",
            "provider": "test-provider",
            "sandbox_id": "sandbox-123",
            "outcome": "completed",
            "exit_code": None,
            "cpu_time_s": None,
            "peak_memory_mib": None,
            "resource_usage_source": None,
            "error_type": None,
        }

    def test_unobserved_response_omits_optional_field(self) -> None:
        response = SWEBenchVerifyResponse.model_construct(verifier_sandbox_observation=None)

        assert "verifier_sandbox_observation" not in response.model_dump()

    async def test_eval_exit_code_is_observed_without_treating_failed_tests_as_sandbox_failure(self) -> None:
        sandbox = make_sandbox(exec_result=SandboxExecResult(stdout="test output", stderr=None, return_code=7))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        test_output, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=3.5, evaluation_completed=True)

        assert test_output == "test output"
        assert timed_out is False
        assert observation.outcome == "completed"
        assert observation.exit_code == 7
        assert observation.wall_time_s == 3.5

    async def test_timeout_is_observed_without_changing_harness_timeout_behavior(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(
                stdout=None,
                stderr="backend failed",
                return_code=125,
                error_type="sandbox",
            )
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.exec_run("git apply patch.diff")
        sandbox.exec.side_effect = TimeoutError("timed out")
        test_output, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=60.0, evaluation_completed=False)

        assert test_output == ""
        assert timed_out is True
        assert observation.outcome == "timeout"
        assert observation.exit_code is None
        assert observation.error_type == "TimeoutError"

    async def test_runtime_error_is_observed_and_still_propagates(self) -> None:
        sandbox = make_sandbox(exec_error=RuntimeError("Sandbox was OOM-killed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        with pytest.raises(RuntimeError, match="OOM-killed"):
            await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)

        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)
        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"
        assert observation.exit_code is None

    @pytest.mark.parametrize(
        ("error_type", "expected_outcome"),
        [("sandbox", "sandbox_error"), ("TimeoutError", "timeout")],
    )
    async def test_provider_error_does_not_report_sentinel_as_process_exit_code(
        self, error_type: str, expected_outcome: str
    ) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(stdout=None, stderr="backend failed", return_code=125, error_type=error_type)
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        _, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)

        assert timed_out is False
        assert observation.outcome == expected_outcome
        assert observation.error_type == error_type
        assert observation.exit_code is None

    async def test_pre_eval_provider_error_is_observed(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(stdout=None, stderr="backend failed", return_code=125, error_type="sandbox")
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.exec_run("git apply patch.diff")
        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)

        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "sandbox"
        assert observation.exit_code is None

    async def test_upload_error_is_observed(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(upload_error=RuntimeError("upload failed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        with pytest.raises(RuntimeError, match="upload failed"):
            await container.copy(tmp_path / "patch.diff", Path("/tmp/patch.diff"))

        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)
        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"

    async def test_cleanup_error_is_fail_open_and_observed(self) -> None:
        sandbox = make_sandbox(stop_error=RuntimeError("stop failed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.cleanup()
        observation = container.observation(wall_time_s=2.0, evaluation_completed=True)

        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"


class TestPatchExtraction:
    def test_command_stages_tracked_changes_and_diffs_the_index_against_the_seed_commit(self) -> None:
        cmd = patch_extraction_command("/testbed", "abc123")
        assert cmd.startswith("cd /testbed && git add -u ")
        assert "add -A" not in cmd  # untracked build output must not ride along
        assert cmd.endswith("cd /testbed && git --no-pager diff --cached abc123")

    def test_command_quotes_odd_paths(self) -> None:
        assert "'/work dir'" in patch_extraction_command("/work dir", "deadbeef")

    def test_strip_binary_diffs_drops_only_binary_sections(self) -> None:
        text = (
            "diff --git a/a.py b/a.py\nindex 1..2 100644\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/img.png b/img.png\nindex 3..4 100644\nBinary files a/img.png and b/img.png differ\n"
            "diff --git a/b.py b/b.py\nnew file mode 100644\n--- /dev/null\n+++ b/b.py\n@@ -0,0 +1 @@\n+z\n"
        )
        out = strip_binary_diffs(text)
        assert "img.png" not in out and "a/a.py" in out and "b/b.py" in out
        assert strip_binary_diffs("") == ""
