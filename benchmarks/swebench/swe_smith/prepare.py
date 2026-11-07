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
"""Prepare SWE-smith (SWE-bench/SWE-smith-py) rows for the swebench resources server.

Same row schema and prompt as benchmarks/swebench/verified/prepare.py, with
``subset: swe_smith`` and ``split: train``, plus the task-setup fields the
server applies for these rows (see resources_servers/swebench/swe_smith_harness.py):

- ``patch`` is the fix: the textual reverse of the dataset's bug-introducing
  ``patch``, so a golden verification (``is_verifying_golden_patch``) resolves
  the instance as for SWE-bench rows;
- ``setup_patch`` is the dataset's bug patch, applied by the server to the
  image's clean checkout at seed time and in the grading container;
- ``setup_hidden_files`` are the F2P test files, removed from the agent's
  checkout as SWE-smith's task branches do;
- ``image_name`` is the repository's environment image.

SWE-smith rows have no ``base_commit``, ``version``, ``test_patch``, ``hints_text``,
``created_at`` or ``difficulty``: ``base_commit``/``environment_setup_commit``
are the upstream commit the mirror was created from, ``version`` the mirror
commit's eight hex digits (the version registered for the repository), the
rest empty. The 50,908 rows include 11,437 without a problem statement, which
are skipped (the 39,471 others are EdgeWise's task pool).

The dataset is read from the parquet shards in the directory named by
``SWE_SMITH_DATASET_DIR`` (``dataset_dir`` argument) when set, else from the
Hugging Face Hub. When a squashfs cache directory is known (``sqsh_cache_dir``
argument, else ``NEMO_GYM_ENROOT_SQSH_CACHE``), only instances whose repository
image the server would find there are written; the dropped instance ids go to
``<output>.missing_images.txt`` next to the output.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
from datasets import load_dataset

from benchmarks.swebench.swe_smith.images import server_image_name, sqsh_present
from nemo_gym.global_config import get_hf_token
from resources_servers.swebench.swe_smith_tasks import (
    profile_for,
    reverse_unified_diff,
    swe_smith_hidden_files,
    swe_smith_version,
)


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
PROMPT_TEMPLATE_FPATH = BENCHMARK_DIR / "minimax_prompt.txt"
# SWE-smith images check the mirror repository out at /testbed, like SWE-bench's.
WORKSPACE_PATH = "/testbed"
SQSH_CACHE_DIR_ENV = "NEMO_GYM_ENROOT_SQSH_CACHE"
DATASET_DIR_ENV = "SWE_SMITH_DATASET_DIR"

HF_DATASET = "SWE-bench/SWE-smith-py"
SUBSET = "swe_smith"
OUTPUT_FPATH = DATA_DIR / "swe_smith_benchmark.jsonl"


def build_prompt(problem_statement: str, prompt_template: str) -> str:
    return (
        prompt_template.replace("{{ workspace_path }}", WORKSPACE_PATH)
        .replace("{{ instance.problem_statement }}", problem_statement)
        .replace("{{ instance.repo_language ~ ' ' if instance.repo_language else '' }}", "")
    )


def to_benchmark_row(row: Dict[str, Any], subset: str, prompt_template: str) -> Dict[str, Any]:
    """One swebench server row (the Verified file's key order, then the setup fields) from a SWE-smith row."""
    profile = profile_for(row["repo"])
    if profile is None:
        raise KeyError(f"{row['instance_id']}: repository {row['repo']!r} has no SWE-smith profile")
    if row["image_name"] != profile["image"]:
        raise ValueError(f"{row['instance_id']}: image_name {row['image_name']!r} is not {profile['image']!r}")
    return {
        "repo": row["repo"],
        "instance_id": row["instance_id"],
        "base_commit": profile["commit"],
        "patch": reverse_unified_diff(row["patch"]),
        "test_patch": "",
        "problem_statement": row["problem_statement"],
        "hints_text": "",
        "created_at": "",
        "version": swe_smith_version(row["repo"]),
        "FAIL_TO_PASS": json.dumps(list(row["FAIL_TO_PASS"])),
        "PASS_TO_PASS": json.dumps(list(row["PASS_TO_PASS"])),
        "environment_setup_commit": profile["commit"],
        "difficulty": "",
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
        "image_name": row["image_name"],
        "setup_patch": row["patch"],
        "setup_hidden_files": swe_smith_hidden_files(profile, row),
    }


def resolve_sqsh_cache_dir(sqsh_cache_dir: Optional[str]) -> Optional[Path]:
    value = sqsh_cache_dir or os.environ.get(SQSH_CACHE_DIR_ENV)
    if not value:
        return None
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"squashfs cache directory {str(path)!r} does not exist")
    return path


def resolve_dataset_dir(dataset_dir: Optional[str]) -> Optional[Path]:
    value = dataset_dir or os.environ.get(DATASET_DIR_ENV)
    if not value:
        return None
    path = Path(value)
    if not any(path.glob("*.parquet")):
        raise FileNotFoundError(f"dataset directory {str(path)!r} holds no parquet shards")
    return path


def iter_parquet_rows(dataset_dir: Path, batch_size: int = 256) -> Iterable[Dict[str, Any]]:
    """Rows of the parquet shards in `dataset_dir`, one batch in memory at a time (no Arrow cache is written)."""
    for shard in sorted(dataset_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def load_source_rows(dataset_dir: Optional[Path]) -> Iterable[Dict[str, Any]]:
    """The dataset's train split, streamed from local parquet shards or loaded from the Hub."""
    if dataset_dir is not None:
        return iter_parquet_rows(dataset_dir)
    return load_dataset(HF_DATASET, split="train", token=get_hf_token())


def has_problem_statement(row: Dict[str, Any]) -> bool:
    return bool((row.get("problem_statement") or "").strip())


def image_missing(row: Dict[str, Any], sqsh_cache_dir: Optional[Path]) -> bool:
    return sqsh_cache_dir is not None and not sqsh_present(sqsh_cache_dir, server_image_name(row["image_name"]))


def write_benchmark(
    source_rows: Iterable[Dict[str, Any]], output_fpath: Path, subset: str, sqsh_cache_dir: Optional[Path]
) -> Tuple[Counter, List[Tuple[str, str]], int]:
    """Write the rows with a problem statement (and a staged image, when a cache is known); streams the source.

    Returns (kept rows per repository, [(instance_id, image) without an image], rows without a problem statement).
    """
    prompt_template = PROMPT_TEMPLATE_FPATH.read_text()
    per_repo: Counter = Counter()
    missing: List[Tuple[str, str]] = []
    skipped_empty = 0
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the final file (its symlink target, when the data directory links to shared storage).
    target_fpath = output_fpath.resolve() if output_fpath.is_symlink() else output_fpath
    tmp_fpath = target_fpath.with_name(target_fpath.name + ".tmp")
    with tmp_fpath.open("w", encoding="utf-8") as fout:
        for row in source_rows:
            if not has_problem_statement(row):
                skipped_empty += 1
            elif image_missing(row, sqsh_cache_dir):
                missing.append((row["instance_id"], row["image_name"]))
            else:
                fout.write(json.dumps(to_benchmark_row(row, subset, prompt_template)) + "\n")
                per_repo[row["repo"]] += 1
    tmp_fpath.replace(target_fpath)
    missing_fpath = output_fpath.with_name(output_fpath.stem + ".missing_images.txt")
    missing_fpath.write_text("".join(f"{instance_id}\t{image}\n" for instance_id, image in missing))
    return per_repo, missing, skipped_empty


def report(
    output_fpath: Path,
    per_repo: Counter,
    missing: List[Tuple[str, str]],
    skipped_empty: int,
    sqsh_cache_dir: Optional[Path],
) -> None:
    kept = sum(per_repo.values())
    print(f"Wrote {kept} {HF_DATASET} problems from {len(per_repo)} repositories to {output_fpath}")
    print(f"{skipped_empty} rows without a problem statement skipped")
    if sqsh_cache_dir is None:
        print(f"No squashfs cache directory given ({SQSH_CACHE_DIR_ENV} unset): image presence was not checked.")
    else:
        missing_fpath = output_fpath.with_name(output_fpath.stem + ".missing_images.txt")
        print(f"{len(missing)} instances without an image in {sqsh_cache_dir} listed in {missing_fpath}")
    for repo, count in sorted(per_repo.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {repo}: {count}")


def prepare(sqsh_cache_dir: Optional[str] = None, dataset_dir: Optional[str] = None) -> Path:
    cache = resolve_sqsh_cache_dir(sqsh_cache_dir)
    rows = load_source_rows(resolve_dataset_dir(dataset_dir))
    per_repo, missing, skipped_empty = write_benchmark(rows, OUTPUT_FPATH, SUBSET, cache)
    report(OUTPUT_FPATH, per_repo, missing, skipped_empty, cache)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
