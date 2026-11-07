# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_4_local.app import (
    TerminalBench4InstanceRequest,
    TerminalBench4LocalResourcesServer,
    TerminalBench4LocalResourcesServerConfig,
    artifact_archive_command,
    artifact_restore_command,
    parse_reward,
    restore_tests_command,
    tail,
    tar_directory_bytes,
)
from resources_servers.terminal_bench_4_local.tasks import Artifact
from resources_servers.terminal_bench_4_local.tests.test_tasks import write_task


def test_parse_reward_prefers_json_then_text() -> None:
    assert parse_reward('{"reward": 1}', "0") == 1.0
    assert parse_reward(None, " 0\n") == 0.0
    assert parse_reward("  ", "0.5") == 0.5
    with pytest.raises(ValueError, match="no reward"):
        parse_reward(None, None)
    with pytest.raises(ValueError, match="'reward' key"):
        parse_reward('{"score": 1}', None)
    with pytest.raises(ValueError, match="finite"):
        parse_reward('{"reward": "nan"}', None)
    with pytest.raises(json.JSONDecodeError):
        parse_reward("{not json", None)


def test_artifact_commands_archive_and_restore_in_place() -> None:
    directory = Artifact(source="/app/src/", exclude=("node_modules", "*.pyc"))
    archive = artifact_archive_command(directory, "/tmp/x/a.tgz")
    assert archive == "tar -czf /tmp/x/a.tgz -C /app --exclude=node_modules --exclude='*.pyc' -- src"
    assert (
        artifact_restore_command(directory, "/tmp/x/a.tgz")
        == "mkdir -p /app && tar -xzf /tmp/x/a.tgz -C /app && rm -f /tmp/x/a.tgz"
    )

    single_file = Artifact(source="/app/output/report.json")
    assert artifact_archive_command(single_file, "/tmp/f.tgz") == "tar -czf /tmp/f.tgz -C /app/output -- report.json"
    assert artifact_restore_command(single_file, "/tmp/f.tgz").startswith("mkdir -p /app/output && tar -xzf")

    assert restore_tests_command("/tmp/t.tgz").startswith("rm -rf /tests && mkdir -p /tests /logs/verifier")
    assert restore_tests_command("/tmp/t.tgz").endswith("chmod +x /tests/test.sh")


def test_tar_directory_bytes_keeps_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "test.sh").write_text("echo hi\n")
    (tmp_path / "sub" / "data.txt").write_text("x")

    with tarfile.open(fileobj=io.BytesIO(tar_directory_bytes(tmp_path)), mode="r:gz") as archive:
        names = sorted(archive.getnames())
    assert names == ["sub", "sub/data.txt", "test.sh"]


def test_tail_truncates_from_the_end() -> None:
    assert tail(None, 10) is None
    assert tail("short", 10) == "short"
    assert tail("abcdefghij", 4) == "...[truncated]...\nghij"
    assert tail("abc", 0) == "abc"


def _config(tasks_dir: Path, **overrides) -> TerminalBench4LocalResourcesServerConfig:
    return TerminalBench4LocalResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="terminal_bench_4_local_resources_server",
        tasks_dir=tasks_dir,
        sandbox_provider="test",
        sandbox_config={},
        **overrides,
    )


def test_resolve_task_validates_name_images_and_support(tmp_path: Path) -> None:
    write_task(tmp_path, "example")
    write_task(tmp_path, "gpu", gpus=1)
    server = TerminalBench4LocalResourcesServer(config=_config(tmp_path), server_client=MagicMock(spec=ServerClient))

    good = TerminalBench4InstanceRequest(
        task_name="example",
        image="harborframework/terminal-bench:example-environment-v4.0.0",
        verifier_image="harborframework/terminal-bench:example-verifier-v4.0.0",
    )
    assert server._resolve_task(good).name == "example"

    with pytest.raises(ValueError, match="Unknown Terminal-Bench task"):
        server._resolve_task(good.model_copy(update={"task_name": "nope"}))
    with pytest.raises(ValueError, match="does not match"):
        server._resolve_task(good.model_copy(update={"verifier_image": "other:tag"}))
    with pytest.raises(ValueError, match="needs GPUs"):
        server._resolve_task(
            TerminalBench4InstanceRequest(
                task_name="gpu",
                image="harborframework/terminal-bench:gpu-environment-v4.0.0",
                verifier_image="harborframework/terminal-bench:gpu-verifier-v4.0.0",
            )
        )


def test_server_requires_an_existing_task_tree(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TerminalBench4LocalResourcesServer(
            config=_config(tmp_path / "missing"), server_client=MagicMock(spec=ServerClient)
        )
