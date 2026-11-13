# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Terminal-Bench 2.1 rows from a local task tree.

Each task directory (``task.toml``, ``instruction.md``, ``tests/``, ``environment/Dockerfile``)
names its published image in ``[environment].docker_image``; the agent and the grader share that
container. A row carries the task name, the image, the image's working directory (the last
``WORKDIR`` of the task's Dockerfile, ``/app`` when there is none), the instruction as the single
user message and the agent budget ``agent_timeout_s``:

    enforced = max(official, min(scale * official, cap))      # never below the official budget

with ``official`` = ``[agent].timeout_sec``, ``scale`` = ``TB21_TIMEOUT_SCALE`` (default 1.0) and
``cap`` = ``TB21_TIMEOUT_CAP`` seconds (default: none). Rows are ordered longest budget first, so
index-modulo sharding spreads the long tasks over the shards.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"
DEFAULT_WORKDIR = "/app"
_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.IGNORECASE)


def enforced_timeout(official: float, scale: float = 1.0, cap: float | None = None) -> float:
    """The agent budget a row gets: scaled, capped, never below the official one."""
    if official <= 0 or scale <= 0 or (cap is not None and cap <= 0):
        raise ValueError(f"timeouts and scale must be positive (official={official}, scale={scale}, cap={cap})")
    scaled = scale * official
    if cap is not None:
        scaled = min(scaled, cap)
    return max(official, scaled)


def dockerfile_workdir(dockerfile: Path) -> str:
    """The last absolute ``WORKDIR`` of a Dockerfile, else ``/app``."""
    workdir = DEFAULT_WORKDIR
    if dockerfile.is_file():
        for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _WORKDIR_RE.match(line)
            if match and match.group(1).startswith("/"):
                workdir = match.group(1).rstrip("/") or "/"
    return workdir


def prepare(
    *,
    tasks_dir: str | None = None,
    task_names: list[str] | None = None,
    timeout_scale: float | None = None,
    timeout_cap: float | None = None,
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    from resources_servers.terminal_bench_4_local.app_task_image import AGENT_TIMEOUT_KEY, task_image
    from resources_servers.terminal_bench_4_local.tasks import load_tasks, unsupported_reasons

    tasks_root = Path(tasks_dir or os.environ.get("TB21_TASKS_DIR", "")).expanduser()
    if not str(tasks_root) or not tasks_root.is_dir():
        raise FileNotFoundError(
            "Terminal-Bench 2.1 task tree not found: pass ++prepare_script_args.tasks_dir=... or set TB21_TASKS_DIR "
            f"(got {str(tasks_root)!r})"
        )
    scale = float(timeout_scale if timeout_scale is not None else os.environ.get("TB21_TIMEOUT_SCALE") or 1.0)
    cap_value = timeout_cap if timeout_cap is not None else os.environ.get("TB21_TIMEOUT_CAP")
    cap = float(cap_value) if cap_value not in (None, "") else None

    tasks = load_tasks(tasks_root, task_names)
    rows = []
    for name, task in tasks.items():
        reasons = unsupported_reasons(task)
        if reasons:
            raise ValueError(f"Terminal-Bench task {name!r} cannot run on this server: {', '.join(reasons)}")
        image = task_image(task)
        workdir = dockerfile_workdir(task.task_dir / "environment" / "Dockerfile")
        official = task.agent_timeout_sec
        enforced = enforced_timeout(official, scale, cap)
        rows.append(
            {
                "task_name": name,
                "image": image,
                "verifier_image": image,
                "workdir": workdir,
                "verifier_workdir": workdir,
                AGENT_TIMEOUT_KEY: enforced,
                "responses_create_params": {"input": [{"role": "user", "content": task.instruction}]},
                "verifier_metadata": {
                    "task_name": name,
                    "agent_timeout_sec_official": official,
                    "agent_timeout_sec_enforced": enforced,
                    "agent_timeout_scale": scale,
                    "agent_timeout_cap": cap,
                    "verifier_timeout_sec": task.verifier_timeout_sec,
                    "difficulty": (task.raw.get("metadata") or {}).get("difficulty"),
                    "category": (task.raw.get("metadata") or {}).get("category"),
                },
                "subset": "terminal-bench-2.1",
                "split": "test",
            }
        )
    rows.sort(key=lambda row: (-row[AGENT_TIMEOUT_KEY], row["task_name"]))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    print(prepare())
