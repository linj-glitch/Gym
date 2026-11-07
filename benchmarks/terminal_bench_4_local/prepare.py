# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Terminal-Bench 4.0 rows from a local task tree and Harbor's published images.

``manifest.json`` pins the official 66-task release (names, categories, package
digests); the task tree is the matching ``v4.0.0`` tag of
https://huggingface.co/datasets/harborframework/terminal-bench (task.toml,
instruction.md, tests/, solution/). Rows carry the task name, the two image
references the resources server validates against, and the instruction as the
single user message.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
# docker WORKDIR / user / entrypoint of every published task image, read from the registry
# (harborframework/terminal-bench manifests, 2026-10-22). Rows carry the WORKDIRs so every
# exec in a task container starts where `docker exec` would.
IMAGE_CONFIG_PATH = BENCHMARK_DIR / "image-config.json"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"
CATEGORIES = ("cpu", "compose", "gpu")


def prepare(
    *,
    tasks_dir: str | None = None,
    category: str | None = "cpu",
    task_names: list[str] | None = None,
    image_repository: str = "harborframework/terminal-bench",
    image_tag_suffix: str = "v4.0.0",
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    from resources_servers.terminal_bench_4_local.tasks import image_reference, load_task, unsupported_reasons

    tasks_root = Path(tasks_dir or os.environ.get("TB4_TASKS_DIR", "")).expanduser()
    if not str(tasks_root) or not tasks_root.is_dir():
        raise FileNotFoundError(
            "Terminal-Bench task tree not found: pass ++prepare_script_args.tasks_dir=... or set TB4_TASKS_DIR "
            f"(got {str(tasks_root)!r})"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    image_config = json.loads(IMAGE_CONFIG_PATH.read_text(encoding="utf-8"))["images"]
    entries = manifest["tasks"]
    if len(entries) != 66 or len({entry["name"] for entry in entries}) != 66:
        raise ValueError("Terminal-Bench 4.0 must contain 66 unique tasks")
    if category is not None:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown Terminal-Bench 4.0 category {category!r}; expected one of {CATEGORIES}")
        entries = [entry for entry in entries if entry["category"] == category]
    if task_names is not None:
        unknown = set(task_names) - {entry["name"] for entry in entries}
        if unknown:
            raise ValueError(f"Unknown or filtered-out Terminal-Bench task names: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["name"] in task_names]
    if not entries:
        raise ValueError("The Terminal-Bench task selection is empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for entry in entries:
            task = load_task(tasks_root / entry["name"])
            reasons = unsupported_reasons(task)
            if reasons:
                raise ValueError(f"Terminal-Bench task {task.name!r} cannot run on this server: {', '.join(reasons)}")
            configs = [image_config[f"{task.name}-{role}-{image_tag_suffix}"] for role in ("environment", "verifier")]
            for config in configs:
                if "error" in config:
                    raise ValueError(
                        f"image-config.json has no registry config for a {task.name!r} image: {config['error']}"
                    )
            row = {
                "task_name": task.name,
                "image": image_reference(image_repository, task.name, "environment", image_tag_suffix),
                "verifier_image": image_reference(image_repository, task.name, "verifier", image_tag_suffix),
                "workdir": configs[0]["workdir"],
                "verifier_workdir": configs[1]["workdir"],
                "responses_create_params": {"input": [{"role": "user", "content": task.instruction}]},
                "verifier_metadata": {
                    "task_name": task.name,
                    "category": entry["category"],
                    "task_ref": entry["ref"],
                    "dataset_ref": manifest["ref"],
                },
                "subset": f"terminal-bench-4.0-{category or 'all'}",
                "split": "test",
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    print(prepare())
