# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.terminal_bench_4_local.app_task_image import AGENT_TIMEOUT_KEY, row_agent_timeout, task_image
from resources_servers.terminal_bench_4_local.tasks import load_task


def write_task_image_task(root: Path, name: str = "fix-git", image: str | None = "alexgshaw/fix-git:20260403") -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir()
    environment = f'docker_image = "{image}"\n' if image else ""
    (task_dir / "task.toml").write_text(
        f"[agent]\ntimeout_sec = 900.0\n[verifier]\ntimeout_sec = 600.0\n[environment]\n{environment}cpus = 1\n",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Recover the lost commit.\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /tmp/build\nRUN true\nWORKDIR /app/personal-site/\n", encoding="utf-8"
    )
    return task_dir


def _prepare_module():
    path = Path(__file__).resolve().parents[3] / "benchmarks" / "terminal_bench_2_1_local" / "prepare.py"
    spec = importlib.util.spec_from_file_location("tb21_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_image_reads_docker_image_and_shared_mode(tmp_path: Path) -> None:
    task = load_task(write_task_image_task(tmp_path))
    assert task_image(task) == "alexgshaw/fix-git:20260403"
    assert task.environment_mode == "shared"
    assert task.agent_timeout_sec == 900.0
    with pytest.raises(ValueError, match="docker_image"):
        task_image(load_task(write_task_image_task(tmp_path, name="no-image", image=None)))


def test_row_agent_timeout() -> None:
    assert row_agent_timeout(SimpleNamespace(**{AGENT_TIMEOUT_KEY: 2700})) == 2700.0
    assert row_agent_timeout(SimpleNamespace(model_extra={AGENT_TIMEOUT_KEY: "900"})) == 900.0
    assert row_agent_timeout(SimpleNamespace(model_extra={})) is None
    assert row_agent_timeout(SimpleNamespace(**{AGENT_TIMEOUT_KEY: ""})) is None
    with pytest.raises(ValueError, match="positive"):
        row_agent_timeout(SimpleNamespace(**{AGENT_TIMEOUT_KEY: 0}))


def test_prepare_scales_caps_and_orders(tmp_path: Path) -> None:
    prepare = _prepare_module()
    assert prepare.enforced_timeout(900, 3, 10800) == 2700
    assert prepare.enforced_timeout(3600, 3, 10800) == 10800
    assert prepare.enforced_timeout(12000, 3, 10800) == 12000  # never below official
    assert prepare.enforced_timeout(900) == 900
    with pytest.raises(ValueError):
        prepare.enforced_timeout(900, 0)

    tasks = tmp_path / "tasks"
    write_task_image_task(tasks, name="short")
    long_dir = write_task_image_task(tasks, name="long")
    (long_dir / "task.toml").write_text(
        (long_dir / "task.toml").read_text(encoding="utf-8").replace("timeout_sec = 900.0", "timeout_sec = 3600.0", 1),
        encoding="utf-8",
    )
    assert prepare.dockerfile_workdir(long_dir / "environment" / "Dockerfile") == "/app/personal-site"
    assert prepare.dockerfile_workdir(tmp_path / "missing") == "/app"

    import json

    output = prepare.prepare(
        tasks_dir=str(tasks), timeout_scale=3, timeout_cap=10800, output_path=tmp_path / "rows.jsonl"
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_name"] for row in rows] == ["long", "short"]
    assert [row[AGENT_TIMEOUT_KEY] for row in rows] == [10800, 2700]
    assert rows[1]["image"] == rows[1]["verifier_image"] == "alexgshaw/fix-git:20260403"
    assert rows[1]["workdir"] == "/app/personal-site"
    assert rows[1]["verifier_metadata"]["agent_timeout_sec_official"] == 900.0
    assert rows[1]["responses_create_params"]["input"][0]["content"].startswith("Recover the lost commit.")
