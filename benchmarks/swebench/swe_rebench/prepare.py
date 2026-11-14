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
"""Prepare one monthly split of SWE-rebench (nebius/SWE-rebench-leaderboard) for the swebench resources server.

Same row schema and prompt as benchmarks/swebench/verified/prepare.py, plus the
two fields a SWE-rebench row needs at grading time: ``image_name`` (the
instance's prebuilt image, ``swerebench/sweb.eval.x86_64.<owner>_1776_<repo>-<n>:latest``)
and ``install_config`` (test command and log parser name; ``None`` and empty
values dropped, as the SWE-rebench fork's ``_clean_install_config`` does).
``subset`` is ``swe_rebench_<split>`` and ``split`` the monthly split
(``2026_03``: the 110 instances of the March 2026 leaderboard). ``version`` is
``unknown`` in the dataset and kept; ``difficulty`` is empty.

The problem statement is used as published: for the instances whose id ends in
``_interface`` (and some others) the dataset's ``interface`` text, which names
the classes and functions the hidden tests call, is already part of it.

The split is read from the parquet file(s) in the directory named by
``SWE_REBENCH_DATASET_DIR`` (``dataset_dir`` argument) when set
(``<split>.parquet`` or ``<split>-*.parquet``), else from the Hub.

When a squashfs cache directory is known (``sqsh_cache_dir`` argument, else the
``NEMO_GYM_ENROOT_SQSH_CACHE`` environment variable the enroot benchmark leaves
export), only instances whose image is there are written; the dropped instance
ids go to ``<output>.missing_images.txt``. Without a cache directory every
instance is written.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmarks.swebench.swe_gym.images import sqsh_present


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
PROMPT_TEMPLATE_FPATH = BENCHMARK_DIR / "minimax_prompt.txt"
# SWE-rebench images check the repository out at /testbed, like SWE-bench's.
WORKSPACE_PATH = "/testbed"
SQSH_CACHE_DIR_ENV = "NEMO_GYM_ENROOT_SQSH_CACHE"
DATASET_DIR_ENV = "SWE_REBENCH_DATASET_DIR"

HF_DATASET = "nebius/SWE-rebench-leaderboard"
SPLIT = "2026_03"


def subset_name(split: str) -> str:
    return f"swe_rebench_{split}"


def output_fpath(split: str) -> Path:
    return DATA_DIR / f"{subset_name(split)}_benchmark.jsonl"


def build_prompt(problem_statement: str, prompt_template: str) -> str:
    return (
        prompt_template.replace("{{ workspace_path }}", WORKSPACE_PATH)
        .replace("{{ instance.problem_statement }}", problem_statement)
        .replace("{{ instance.repo_language ~ ' ' if instance.repo_language else '' }}", "")
    )


def clean_install_config(install_config: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` and empty values (the fork drops ``None``; an empty list or string selects nothing either)."""
    cleaned = {key: value for key, value in install_config.items() if value not in (None, "", [], {})}
    for key in ("test_cmd", "log_parser"):
        if key not in cleaned:
            raise ValueError(f"install_config has no {key}")
    return cleaned


def to_benchmark_row(row: Dict[str, Any], split: str, prompt_template: str) -> Dict[str, Any]:
    """One swebench server row (the Verified file's key order, then the SWE-rebench fields) from a dataset row."""
    image = row.get("image_name") or row.get("docker_image")
    if not image:
        raise ValueError(f"{row['instance_id']}: no image_name / docker_image")
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
        "difficulty": "",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": build_prompt(row["problem_statement"], prompt_template),
                }
            ],
        },
        "subset": subset_name(split),
        "split": split,
        "image_name": image,
        "install_config": clean_install_config(dict(row["install_config"])),
    }


def resolve_dir(value: Optional[str], env: str, what: str) -> Optional[Path]:
    value = value or os.environ.get(env)
    if not value:
        return None
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{what} {str(path)!r} does not exist")
    return path


def load_source_rows(split: str, dataset_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """The split's rows, from local parquet (``<split>.parquet`` / ``<split>-*.parquet``) or the Hub."""
    if dataset_dir is not None:
        import pyarrow.parquet as pq

        shards = sorted(set(dataset_dir.glob(f"{split}.parquet")) | set(dataset_dir.glob(f"{split}-*.parquet")))
        if not shards:
            raise FileNotFoundError(f"{str(dataset_dir)!r} holds no parquet file for split {split!r}")
        return [row for shard in shards for row in pq.read_table(shard).to_pylist()]

    from datasets import load_dataset

    from nemo_gym.global_config import get_hf_token

    return list(load_dataset(HF_DATASET, split=split, token=get_hf_token()))


def split_by_image_presence(
    rows: Iterable[Dict[str, Any]], sqsh_cache_dir: Optional[Path]
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """(rows whose image is in the cache, [(instance_id, image) that is not])."""
    kept: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str]] = []
    for row in rows:
        image = row.get("image_name") or row.get("docker_image") or ""
        if sqsh_cache_dir is not None and not sqsh_present(sqsh_cache_dir, image):
            missing.append((row["instance_id"], image))
            continue
        kept.append(row)
    return kept, missing


def prepare(sqsh_cache_dir: Optional[str] = None, dataset_dir: Optional[str] = None, split: str = SPLIT) -> Path:
    cache_dir = resolve_dir(sqsh_cache_dir, SQSH_CACHE_DIR_ENV, "squashfs cache directory")
    source_dir = resolve_dir(dataset_dir, DATASET_DIR_ENV, "dataset directory")
    rows = load_source_rows(split, source_dir)
    prompt_template = PROMPT_TEMPLATE_FPATH.read_text()

    kept, missing = split_by_image_presence(rows, cache_dir)
    out = output_fpath(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        for row in kept:
            fout.write(json.dumps(to_benchmark_row(row, split, prompt_template)) + "\n")
    missing_fpath = out.with_name(out.stem + ".missing_images.txt")
    missing_fpath.write_text("".join(f"{instance_id}\t{image}\n" for instance_id, image in missing))

    source = str(source_dir) if source_dir is not None else HF_DATASET
    print(f"Wrote {len(kept)} of {len(rows)} {source} [{split}] problems to {out}")
    if cache_dir is None:
        print(f"No squashfs cache directory given ({SQSH_CACHE_DIR_ENV} unset): image presence was not checked.")
    else:
        print(f"{len(missing)} instances without an image in {cache_dir} listed in {missing_fpath}")
    per_parser = Counter(row["install_config"].get("log_parser") for row in kept)
    print(f"  repositories: {len({row['repo'] for row in kept})}; log parsers: {dict(per_parser)}")
    return out


if __name__ == "__main__":
    prepare(*sys.argv[1:3])
