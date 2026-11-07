# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from resources_servers.terminal_bench_4_local.tasks import (
    Artifact,
    image_reference,
    load_task,
    load_tasks,
    strip_canary,
    unsupported_reasons,
)


TASK_TOML = """# harbor-canary GUID 00000000-0000-0000-0000-000000000000

artifacts = ["/app/output/report.json", {{ source = "/app/src/", exclude = ["node_modules", "*.pyc"] }}]

[task]
name = "terminal-bench/example"

[verifier]
timeout_sec = 300.0
environment_mode = "separate"
{verifier_extra}
[agent]
timeout_sec = 28800.0

[environment]
build_timeout_sec = 600.0
cpus = 4
memory_mb = 8192
storage_mb = 10240
gpus = {gpus}
{environment_extra}
"""


def write_task(root: Path, name: str, *, gpus: int = 0, verifier_extra: str = "", environment_extra: str = "") -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(gpus=gpus, verifier_extra=verifier_extra, environment_extra=environment_extra)
    )
    (task_dir / "instruction.md").write_text("<!-- harbor-canary GUID 0 -->\n\nSolve the task in /app.\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return task_dir


def test_strip_canary_removes_marker_lines_and_following_blanks() -> None:
    assert strip_canary("<!-- harbor-canary GUID x -->\n\n\nBody\nmore") == "Body\nmore"
    assert strip_canary("# harbor-canary GUID x\nBody") == "Body"
    assert strip_canary("Body only") == "Body only"


def test_load_task_reads_the_fields_the_server_acts_on(tmp_path: Path) -> None:
    task = load_task(write_task(tmp_path, "example"))

    assert task.name == "example"
    assert task.instruction == "Solve the task in /app.\n"
    assert task.environment_mode == "separate"
    assert task.verifier_timeout_sec == 300.0
    assert task.agent_timeout_sec == 28800.0
    assert task.environment.cpus == 4 and task.environment.memory_mb == 8192
    assert task.verifier_environment == task.environment
    assert task.artifacts == (
        Artifact(source="/app/output/report.json"),
        Artifact(source="/app/src/", exclude=("node_modules", "*.pyc")),
    )
    assert task.collect == ()
    assert unsupported_reasons(task) == []


def test_load_task_reads_collect_hooks_and_verifier_environment_override(tmp_path: Path) -> None:
    extra = """environment = { cpus = 1, memory_mb = 2048, storage_mb = 4096, gpus = 0 }
[[verifier.collect]]
command = "cd /app && git diff > /tmp/agent.patch"
timeout_sec = 120.0
"""
    task = load_task(write_task(tmp_path, "hooked", verifier_extra=extra))

    assert task.verifier_environment.cpus == 1
    assert task.verifier_environment.memory_mb == 2048
    assert len(task.collect) == 1
    assert task.collect[0].command.startswith("cd /app")
    assert task.collect[0].timeout_sec == 120.0


def test_load_task_requires_the_grader(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "broken")
    (task_dir / "tests" / "test.sh").unlink()
    with pytest.raises(FileNotFoundError, match="tests/test.sh"):
        load_task(task_dir)


def test_unsupported_reasons_flags_gpu_and_mcp_tasks(tmp_path: Path) -> None:
    gpu_task = load_task(write_task(tmp_path, "gpu", gpus=1))
    assert unsupported_reasons(gpu_task) == ["needs GPUs"]

    mcp_extra = 'mcp_servers = [{ name = "fs", transport = "stdio", command = "mcp-fs" }]'
    mcp_task = load_task(write_task(tmp_path, "mcp", environment_extra=mcp_extra))
    assert unsupported_reasons(mcp_task) == ["declares MCP servers"]


def test_load_tasks_indexes_by_directory_name(tmp_path: Path) -> None:
    write_task(tmp_path, "a")
    write_task(tmp_path, "b")
    assert sorted(load_tasks(tmp_path)) == ["a", "b"]
    assert sorted(load_tasks(tmp_path, ["b"])) == ["b"]
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "missing")


def test_image_reference_follows_harbors_tag_scheme() -> None:
    assert (
        image_reference("harborframework/terminal-bench", "formal-crypto", "environment", "v4.0.0")
        == "harborframework/terminal-bench:formal-crypto-environment-v4.0.0"
    )
    assert image_reference("r", "t", "verifier", "x").endswith(":t-verifier-x")
    with pytest.raises(ValueError):
        image_reference("r", "t", "agent", "x")
