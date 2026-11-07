# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-solution smoke: grade each task's own ``solution/solve.sh`` on a real sandbox provider.

Runs the resources server in-process in golden mode (no agent, no model) so a
node with the images available can prove the whole grading path: agent
container start, solve.sh, collect hooks, artifact transfer, verifier container,
``tests/test.sh`` and the reward files. Every task should score 1.

    python -m resources_servers.terminal_bench_4_local.smoke_golden \\
        --tasks-dir /data/terminal-bench-v4.0.0/tasks \\
        --provider enroot --sqsh-cache-dir /data/tb4/enroot-sqsh \\
        --task formal-crypto --task ks-solver-cpp --output smoke.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import resources_servers.terminal_bench_4_local.app as app_module
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_4_local.app import (
    TerminalBench4LocalResourcesServer,
    TerminalBench4LocalResourcesServerConfig,
    TerminalBench4VerifyRequest,
)
from resources_servers.terminal_bench_4_local.tasks import image_reference, load_tasks


def enroot_provider_config(sqsh_cache_dir: str | None) -> dict[str, Any]:
    """The ``sandbox.enroot`` block of nemo_gym/sandbox/providers/enroot/configs/enroot.yaml, env-resolved."""
    return {
        "enroot": {
            "create": {
                "base_dir": os.environ.get("NEMO_GYM_ENROOT_BASE"),
                "data_path": os.environ.get("ENROOT_DATA_PATH"),
                "cache_path": os.environ.get("ENROOT_CACHE_PATH"),
                "runtime_path": os.environ.get("ENROOT_RUNTIME_PATH"),
                "sqsh_cache_dir": sqsh_cache_dir or os.environ.get("NEMO_GYM_ENROOT_SQSH_CACHE"),
                "bypass_entrypoint": False,
                "rw": True,
                "remap_root": True,
                "import_timeout_s": 1800,
                "create_timeout_s": 600,
                "start_timeout_s": 600,
                "start_poll_s": 0.5,
            },
            "exec": {"exec_shell": "bash", "default_timeout_s": 180, "concurrency": 32},
            "probe": {"timeout_s": 60, "deadline_s": 180, "stable_count": 2, "stable_delay_s": 1.0},
        }
    }


def _verify_body(server: TerminalBench4LocalResourcesServer, task_name: str) -> TerminalBench4VerifyRequest:
    config = server.config
    return TerminalBench4VerifyRequest.model_validate(
        {
            "task_name": task_name,
            "image": image_reference(config.image_repository, task_name, "environment", config.image_tag_suffix),
            "verifier_image": image_reference(config.image_repository, task_name, "verifier", config.image_tag_suffix),
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
    if args.provider == "enroot":
        provider_block = enroot_provider_config(args.sqsh_cache_dir)
    else:
        provider_block = json.loads(args.provider)
    # The server resolves its named provider from Gym's global config; stand one in.
    app_module.get_global_config_dict = lambda: {"sandbox": provider_block}
    config = TerminalBench4LocalResourcesServerConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="",
        name="terminal_bench_4_local_golden",
        tasks_dir=Path(args.tasks_dir),
        is_verifying_golden_solution=True,
        golden_solution_timeout_sec=args.solution_timeout,
        sandbox_provider="sandbox",
        sandbox_config={"ttl_s": 18000, "ready_timeout_s": 1200},
        logs_dir=Path(args.logs_dir),
    )
    server = TerminalBench4LocalResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    names = args.task or sorted(load_tasks(args.tasks_dir))
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0

    async def one(name: str) -> dict[str, Any]:
        async with semaphore:
            started = monotonic()
            request = SimpleNamespace(session={"session_id": f"golden-{name}"})
            try:
                response = await server.verify(request, _verify_body(server, name))
                record = {
                    "task": name,
                    "reward": response.reward,
                    "passed": response.passed,
                    "evaluation_completed": response.evaluation_completed,
                    "verifier_exit_code": response.verifier_exit_code,
                    "verifier_error": response.verifier_error,
                    "artifacts": response.artifacts,
                    "log_dir": response.log_dir,
                }
            except Exception as error:  # the smoke must report, not die
                record = {"task": name, "reward": 0.0, "passed": False, "error": f"{type(error).__name__}: {error}"}
            record["wall_s"] = round(monotonic() - started, 1)
            print(json.dumps(record, default=str), flush=True)
            return record

    records = await asyncio.gather(*(one(name) for name in names))
    with open(args.output, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, default=str) + "\n")
    failures = sum(1 for record in records if not record["passed"])
    print(
        f"{len(records) - failures}/{len(records)} golden solutions passed; details in {args.output}", file=sys.stderr
    )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--provider", default="enroot", help="'enroot' (env-resolved) or a JSON provider block")
    parser.add_argument("--sqsh-cache-dir", default=None)
    parser.add_argument("--task", action="append", help="task directory name; repeatable (default: all)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--solution-timeout", type=float, default=3600.0)
    parser.add_argument("--logs-dir", default="resources_servers/terminal_bench_4_local/logs")
    parser.add_argument("--output", default="tb4_golden_smoke.jsonl")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
