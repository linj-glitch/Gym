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
"""Prepare SWE-Gym (SWE-Gym/SWE-Gym) rows for the swebench resources server.

Same row schema and prompt as benchmarks/swebench/verified/prepare.py, with
``subset: swe_gym`` and ``split: train``. SWE-Gym has no
``environment_setup_commit`` (each instance image was built at its own base
commit, so the base commit is recorded) and no ``difficulty`` annotation (empty
string); ``FAIL_TO_PASS``/``PASS_TO_PASS`` are JSON-encoded like the Verified
rows.

When a squashfs cache directory is known (``sqsh_cache_dir`` argument, else the
``NEMO_GYM_ENROOT_SQSH_CACHE`` environment variable the enroot benchmark leaves
export), only instances whose image the server would find there are written;
the dropped instance ids go to ``<output>.missing_images.txt`` next to the
output. Without a cache directory every instance is written.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

from benchmarks.swebench.swe_gym.images import server_image_name, sqsh_present
from nemo_gym.global_config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
PROMPT_TEMPLATE_FPATH = BENCHMARK_DIR / "minimax_prompt.txt"
# SWE-Gym images check the repository out at /testbed, like SWE-bench's.
WORKSPACE_PATH = "/testbed"
SQSH_CACHE_DIR_ENV = "NEMO_GYM_ENROOT_SQSH_CACHE"

HF_DATASET = "SWE-Gym/SWE-Gym"
SUBSET = "swe_gym"
OUTPUT_FPATH = DATA_DIR / "swe_gym_benchmark.jsonl"


def build_prompt(problem_statement: str, prompt_template: str) -> str:
    return (
        prompt_template.replace("{{ workspace_path }}", WORKSPACE_PATH)
        .replace("{{ instance.problem_statement }}", problem_statement)
        .replace("{{ instance.repo_language ~ ' ' if instance.repo_language else '' }}", "")
    )


def to_benchmark_row(row: Dict[str, Any], subset: str, prompt_template: str) -> Dict[str, Any]:
    """One swebench server row (the Verified file's key order) from a SWE-Gym dataset row."""
    return {
        "repo": row["repo"],
        "instance_id": row["instance_id"],
        "base_commit": row["base_commit"],
        "patch": row["patch"],
        "test_patch": row["test_patch"],
        "problem_statement": row["problem_statement"],
        "hints_text": row.get("hints_text") or "",
        "created_at": str(row["created_at"]),
        "version": str(row["version"]),
        "FAIL_TO_PASS": json.dumps(list(row["FAIL_TO_PASS"])),
        "PASS_TO_PASS": json.dumps(list(row["PASS_TO_PASS"])),
        "environment_setup_commit": row.get("environment_setup_commit") or row["base_commit"],
        "difficulty": row.get("difficulty") or "",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": build_prompt(row["problem_statement"], prompt_template),
                }
            ],
        },
        "subset": subset,
        "split": "train",
    }


def resolve_sqsh_cache_dir(sqsh_cache_dir: Optional[str]) -> Optional[Path]:
    value = sqsh_cache_dir or os.environ.get(SQSH_CACHE_DIR_ENV)
    if not value:
        return None
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"squashfs cache directory {str(path)!r} does not exist")
    return path


def split_by_image_presence(
    rows: Iterable[Dict[str, Any]], sqsh_cache_dir: Optional[Path]
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """(rows whose image the server would find in the cache, [(instance_id, image) it would not])."""
    kept: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str]] = []
    for row in rows:
        image = server_image_name(row["instance_id"])
        if sqsh_cache_dir is not None and not sqsh_present(sqsh_cache_dir, image):
            missing.append((row["instance_id"], image))
            continue
        kept.append(row)
    return kept, missing


def write_benchmark(hf_dataset: str, output_fpath: Path, subset: str, sqsh_cache_dir: Optional[Path]) -> Path:
    ds = load_dataset(hf_dataset, split="train", token=get_hf_token())
    prompt_template = PROMPT_TEMPLATE_FPATH.read_text()

    kept, missing = split_by_image_presence(ds, sqsh_cache_dir)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as fout:
        for row in kept:
            fout.write(json.dumps(to_benchmark_row(row, subset, prompt_template)) + "\n")

    missing_fpath = output_fpath.with_name(output_fpath.stem + ".missing_images.txt")
    missing_fpath.write_text("".join(f"{instance_id}\t{image}\n" for instance_id, image in missing))

    per_repo = Counter(row["repo"] for row in kept)
    print(f"Wrote {len(kept)} of {len(ds)} {hf_dataset} problems to {output_fpath}")
    if sqsh_cache_dir is None:
        print(f"No squashfs cache directory given ({SQSH_CACHE_DIR_ENV} unset): image presence was not checked.")
    else:
        print(f"{len(missing)} instances without an image in {sqsh_cache_dir} listed in {missing_fpath}")
    for repo, count in sorted(per_repo.items()):
        print(f"  {repo}: {count}")
    return output_fpath


def prepare(sqsh_cache_dir: Optional[str] = None) -> Path:
    return write_benchmark(HF_DATASET, OUTPUT_FPATH, SUBSET, resolve_sqsh_cache_dir(sqsh_cache_dir))


if __name__ == "__main__":
    prepare(sys.argv[1] if len(sys.argv) > 1 else None)
