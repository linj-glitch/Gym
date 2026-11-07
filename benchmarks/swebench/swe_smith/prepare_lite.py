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
"""Prepare a small SWE-smith subset for smokes: 300 rows balanced across the repositories.

A deterministic subset of the rows prepare.py writes (``subset: swe_smith_lite``):
every repository contributes ``300 // repositories`` instances (all of them when
it has fewer) and the repositories with the most instances share the remainder;
instances are drawn with a fixed seed. The full file is built first when it is
not present.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from benchmarks.swebench.swe_smith import prepare as full


SUBSET = "swe_smith_lite"
OUTPUT_FPATH = full.DATA_DIR / "swe_smith_lite_benchmark.jsonl"
SIZE = 300
SEED = 0


def select_lite(instances: Iterable[Dict[str, str]], size: int = SIZE, seed: int = SEED) -> Set[str]:
    """Instance ids of a `size`-row subset balanced across repositories (deterministic for a seed)."""
    per_repo: Dict[str, List[str]] = defaultdict(list)
    for instance in instances:
        per_repo[instance["repo"]].append(instance["instance_id"])
    if not per_repo:
        raise ValueError("no instances to select from")
    base = size // len(per_repo)
    remainder = size - base * len(per_repo)
    by_size = sorted(per_repo, key=lambda repo: (-len(per_repo[repo]), repo))
    quotas = {repo: base + (1 if rank < remainder else 0) for rank, repo in enumerate(by_size)}
    rng = random.Random(seed)
    selected: Set[str] = set()
    for repo in sorted(per_repo):
        ids = sorted(per_repo[repo])
        rng.shuffle(ids)
        selected.update(ids[: quotas[repo]])
    return selected


def iter_rows(fpath: Path) -> Iterable[Dict[str, Any]]:
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_lite(full_fpath: Path, output_fpath: Path, size: int = SIZE, seed: int = SEED) -> int:
    """Two passes over the full file (several GB: the P2P lists), so no more than one row is held at a time."""
    selected = select_lite(
        ({"repo": r["repo"], "instance_id": r["instance_id"]} for r in iter_rows(full_fpath)), size, seed
    )
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as fout:
        for row in iter_rows(full_fpath):
            if row["instance_id"] in selected:
                fout.write(json.dumps({**row, "subset": SUBSET}) + "\n")
    # The full file's filter already applied; prepare.py's convention expects the companion file.
    output_fpath.with_name(output_fpath.stem + ".missing_images.txt").write_text("")
    return len(selected)


def prepare(sqsh_cache_dir: Optional[str] = None, dataset_dir: Optional[str] = None) -> Path:
    full_fpath = full.OUTPUT_FPATH if full.OUTPUT_FPATH.is_file() else full.prepare(sqsh_cache_dir, dataset_dir)
    written = write_lite(full_fpath, OUTPUT_FPATH)
    print(f"Wrote {written} of the {full_fpath} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
