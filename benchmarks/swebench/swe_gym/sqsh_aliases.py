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
"""Point the swebench server's enroot cache keys at SWE-Gym squashfs files imported under their Docker Hub names.

A staged cache holds ``<sha256(xingyaoww/..._s_...:latest)[:16]>.sqsh``; the
server asks the provider for ``swebench/..._1776_...:latest`` (see images.py),
so its key differs. This tool fills an alias directory with symlinks
``<server key>.sqsh -> <staged file>`` from an image map
(TSV with columns ``image``, ``dataset``, ``instance_id``, ``key``, one row per
imported image), writes ``aliases.tsv`` beside them, and reports instances whose
image is not staged. Use the alias directory as ``sandbox.enroot.create.sqsh_cache_dir``
(and ``NEMO_GYM_ENROOT_SQSH_CACHE`` for prepare.py).

``--check N`` resolves N random instances of a prepared benchmark file through
the server's own code: ``make_test_spec`` with the server's arguments (which
needs the SWE-Gym harness entries that importing the server registers) and
``EnrootProvider._ensure_image`` on the alias directory. It prints the squashfs
each resolves to and fails if one is missing. Run from the repository root:

    python -m benchmarks.swebench.swe_gym.sqsh_aliases --image-map .../image_map_full.tsv \\
        --staged-dir .../enroot-sqsh --alias-dir .../enroot-sqsh-swe-gym-alias \\
        [--check 20 --benchmark-jsonl benchmarks/swebench/data/swe_gym_benchmark.jsonl]
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmarks.swebench.swe_gym.images import (
    dockerhub_image_name,
    server_image_name,
    sqsh_cache_key,
    sqsh_path,
    sqsh_present,
)


IMAGE_MAP_DATASET = "SWE-Gym/SWE-Gym"
MANIFEST_NAME = "aliases.tsv"


def read_image_map(image_map: Path, dataset: str) -> List[Dict[str, str]]:
    with image_map.open(newline="") as f:
        rows = [row for row in csv.DictReader(f, delimiter="\t") if row["dataset"] == dataset]
    if not rows:
        raise ValueError(f"no rows with dataset={dataset!r} in {image_map}")
    return rows


def build_aliases(rows: List[Dict[str, str]], staged_dir: Path, alias_dir: Path) -> Tuple[int, List[Tuple[str, str]]]:
    """Create/refresh one alias per staged image; returns (aliases, [(instance_id, dockerhub image) not staged])."""
    alias_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Tuple[str, ...]] = []
    missing: List[Tuple[str, str]] = []
    for row in rows:
        instance_id, image, staged_key = row["instance_id"], row["image"], row["key"]
        if image != dockerhub_image_name(instance_id):
            print(
                f"note: {instance_id}: image map name {image!r} differs from the expected Docker Hub name",
                file=sys.stderr,
            )
        if staged_key != sqsh_cache_key(image):
            raise ValueError(f"{instance_id}: image map key {staged_key} is not sha256({image!r})[:16]")
        staged = staged_dir / f"{staged_key}.sqsh"
        if not staged.is_file() or staged.stat().st_size == 0:
            missing.append((instance_id, image))
            continue
        server_image = server_image_name(instance_id)
        alias = sqsh_path(alias_dir, server_image)
        target = staged.resolve()
        if alias.is_symlink() or alias.exists():
            if alias.is_symlink() and Path(os.readlink(alias)) == target:
                manifest_rows.append((alias.name, instance_id, server_image, staged_key, image))
                continue
            alias.unlink()
        alias.symlink_to(target)
        manifest_rows.append((alias.name, instance_id, server_image, staged_key, image))

    with (alias_dir / MANIFEST_NAME).open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["alias", "instance_id", "server_image", "staged_key", "dockerhub_image"])
        writer.writerows(sorted(manifest_rows, key=lambda r: r[1]))
    return len(manifest_rows), missing


def check_resolution(benchmark_jsonl: Path, alias_dir: Path, count: int, seed: int) -> int:
    """Resolve `count` random rows the way the server does; returns the number that failed."""
    # Importing the server registers the SWE-Gym harness entries (swebench_patches) that make_test_spec needs.
    from swebench.harness.run_evaluation import make_test_spec
    from swebench.harness.test_spec.test_spec import LATEST

    from nemo_gym.sandbox.providers.enroot.provider import EnrootProvider
    from resources_servers.swebench.app import SWEBenchInstanceRequest

    rows = [json.loads(line) for line in benchmark_jsonl.read_text().splitlines() if line.strip()]
    sample = random.Random(seed).sample(rows, min(count, len(rows)))
    provider = EnrootProvider(create={"sqsh_cache_dir": str(alias_dir)})
    failures = 0
    for row in sample:
        body = SWEBenchInstanceRequest.model_validate(row)
        # The call SwebenchResourcesServer._make_test_spec makes.
        test_spec = make_test_spec(
            body.model_dump(), namespace="swebench", instance_image_tag=LATEST, env_image_tag=LATEST
        )
        image = test_spec.instance_image_key
        status = "ok"
        if image != server_image_name(row["instance_id"]):
            status = f"FORMULA MISMATCH (images.py says {server_image_name(row['instance_id'])})"
            failures += 1
        if not sqsh_present(alias_dir, image):
            # Not calling _ensure_image: on a miss it would try `enroot import` from Docker Hub.
            print(f"{row['instance_id']}\t{image}\t{sqsh_cache_key(image)}\tMISSING in {alias_dir}")
            failures += 1
            continue
        resolved = asyncio.run(provider._ensure_image(image))
        size_gib = resolved.stat().st_size / 2**30
        print(f"{row['instance_id']}\t{image}\t{resolved}\t-> {resolved.resolve()}\t{size_gib:.2f} GiB\t{status}")
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-map", type=Path, required=True, help="TSV: image, dataset, instance_id, key")
    parser.add_argument(
        "--staged-dir", type=Path, required=True, help="directory of <key>.sqsh keyed by the Docker Hub name"
    )
    parser.add_argument(
        "--alias-dir", type=Path, required=True, help="directory to fill with <server key>.sqsh symlinks"
    )
    parser.add_argument("--dataset", default=IMAGE_MAP_DATASET, help="image map `dataset` column value to select")
    parser.add_argument(
        "--check", type=int, default=0, metavar="N", help="resolve N random benchmark rows like the server"
    )
    parser.add_argument("--benchmark-jsonl", type=Path, help="prepared rows to sample for --check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = read_image_map(args.image_map, args.dataset)
    aliases, missing = build_aliases(rows, args.staged_dir, args.alias_dir)
    print(f"{aliases} aliases in {args.alias_dir} for {len(rows)} {args.dataset} images; {len(missing)} not staged")
    for instance_id, image in missing:
        print(f"not staged\t{instance_id}\t{image}")

    if args.check:
        if args.benchmark_jsonl is None:
            parser.error("--check needs --benchmark-jsonl")
        failures = check_resolution(args.benchmark_jsonl, args.alias_dir, args.check, args.seed)
        print(f"check: {args.check} sampled, {failures} failed")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
