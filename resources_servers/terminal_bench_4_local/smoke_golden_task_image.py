# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-solution smoke for task-image benchmarks (Terminal-Bench 2.1): grade each task's ``solution/solve.sh``.

As ``smoke_golden.py``, for the ``app_task_image.py`` server and rows prepared by
``benchmarks/terminal_bench_2_1_local/prepare.py`` (image and working directory per task):

    python -m resources_servers.terminal_bench_4_local.smoke_golden_task_image \\
        --tasks-dir /data/terminal-bench-2.1/tasks --rows benchmarks/terminal_bench_2_1_local/data/benchmark.jsonl \\
        --provider enroot --sqsh-cache-dir /data/tb21/enroot-sqsh --concurrency 8 --output golden.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import resources_servers.terminal_bench_4_local.app as app_module
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_4_local.app import (
    TerminalBench4LocalResourcesServerConfig,
    TerminalBench4VerifyRequest,
)
from resources_servers.terminal_bench_4_local.app_task_image import TerminalBenchTaskImageResourcesServer
from resources_servers.terminal_bench_4_local.smoke_golden import enroot_provider_config


def verify_body(row: dict[str, Any]) -> TerminalBench4VerifyRequest:
    return TerminalBench4VerifyRequest.model_validate(
        {
            "task_name": row["task_name"],
            "image": row["image"],
            "verifier_image": row["verifier_image"],
            "workdir": row.get("workdir"),
            "verifier_workdir": row.get("verifier_workdir"),
            "responses_create_params": {"input": [{"role": "user", "content": "golden"}]},
            "response": {
                "output": [],
                "id": "golden",
                "created_at": 0,
                "model": "golden",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )


async def run(args: argparse.Namespace) -> int:
    provider_block = (
        enroot_provider_config(args.sqsh_cache_dir) if args.provider == "enroot" else json.loads(args.provider)
    )
    app_module.get_global_config_dict = lambda: {"sandbox": provider_block}
    config = TerminalBench4LocalResourcesServerConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="",
        name="terminal_bench_task_image_golden",
        tasks_dir=Path(args.tasks_dir),
        is_verifying_golden_solution=True,
        golden_solution_timeout_sec=args.solution_timeout,
        sandbox_provider="sandbox",
        sandbox_config={"ttl_s": 18000, "ready_timeout_s": 1200},
        logs_dir=Path(args.logs_dir),
    )
    server = TerminalBenchTaskImageResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.task:
        rows = [row for row in rows if row["task_name"] in set(args.task)]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        name = row["task_name"]
        async with semaphore:
            started = monotonic()
            request = SimpleNamespace(session={"session_id": f"golden-{name}"})
            try:
                response = await server.verify(request, verify_body(row))
                record = {
                    "task": name,
                    "reward": response.reward,
                    "passed": response.passed,
                    "evaluation_completed": response.evaluation_completed,
                    "verifier_exit_code": response.verifier_exit_code,
                    "verifier_error": response.verifier_error,
                    "log_dir": response.log_dir,
                }
            except Exception as error:  # the smoke must report, not die
                record = {"task": name, "reward": 0.0, "passed": False, "error": f"{type(error).__name__}: {error}"}
            record["wall_s"] = round(monotonic() - started, 1)
            print(json.dumps(record, default=str), flush=True)
            with open(args.output, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
            return record

    records = await asyncio.gather(*(one(row) for row in rows))
    failures = sum(1 for record in records if not record["passed"])
    print(
        f"{len(records) - failures}/{len(records)} golden solutions passed; details in {args.output}", file=sys.stderr
    )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--rows", required=True, help="prepared benchmark.jsonl (image and workdir per task)")
    parser.add_argument("--provider", default="enroot", help="'enroot' (env-resolved) or a JSON provider block")
    parser.add_argument("--sqsh-cache-dir", default=None)
    parser.add_argument("--task", action="append", help="task name; repeatable (default: every row)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--solution-timeout", type=float, default=3600.0)
    parser.add_argument("--logs-dir", default="resources_servers/terminal_bench_4_local/logs-2.1-golden")
    parser.add_argument("--output", default="tb21_golden_smoke.jsonl")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
