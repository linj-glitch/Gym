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
"""Point the swebench server's enroot cache keys at SWE-smith squashfs files imported under their Docker Hub names.

A staged directory holds ``swebench_swesmith.x86_64.<owner>_1776_<repo>.<hash>.sqsh``
(``enroot import docker://<image_name>``, ``/`` replaced by ``_``); the server asks
the provider for ``<image_name>:latest`` and the provider looks for
``<sha256(that)[:16]>.sqsh`` (see images.py). This tool fills an alias directory
with one symlink per repository image, ``<server key>.sqsh -> <staged file>``,
writes ``aliases.tsv`` beside them and reports images that are not staged.
Use the alias directory as ``sandbox.enroot.create.sqsh_cache_dir`` (and
``NEMO_GYM_ENROOT_SQSH_CACHE`` for prepare.py).

``--check N`` resolves N random instances of a prepared benchmark file through
the server's own code: ``make_test_spec`` with the server's arguments plus
``adapt_test_spec`` (the SWE-smith image and eval script) and
``EnrootProvider._ensure_image`` on the alias directory. It prints the squashfs
each resolves to and fails if one is missing. Run from the repository root:

    python -m benchmarks.swebench.swe_smith.sqsh_aliases --staged-dir .../swe-smith/enroot-sqsh \\
        --alias-dir .../enroot-sqsh-swe-smith-alias \\
        [--images .../images.txt] [--check 20 --benchmark-jsonl benchmarks/swebench/data/swe_smith_benchmark.jsonl]
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

from benchmarks.swebench.swe_smith.images import (
    server_image_name,
    sqsh_cache_key,
    sqsh_path,
    sqsh_present,
    staged_sqsh_name,
)
from resources_servers.swebench.swe_smith_tasks import PROFILES


MANIFEST_NAME = "aliases.tsv"


def read_images(images_fpath: Optional[Path]) -> Dict[str, str]:
    """{image_name: repo} for the images to alias: a list file, else every SWE-smith profile."""
    by_image = {profile["image"]: repo for repo, profile in PROFILES.items()}
    if images_fpath is None:
        return by_image
    images = [line.strip() for line in images_fpath.read_text().splitlines() if line.strip()]
    unknown = [image for image in images if image not in by_image]
    if unknown:
        raise ValueError(f"{len(unknown)} images in {images_fpath} have no SWE-smith profile, e.g. {unknown[0]!r}")
    return {image: by_image[image] for image in images}


def build_aliases(images: Dict[str, str], staged_dir: Path, alias_dir: Path) -> Tuple[int, List[Tuple[str, str]]]:
    """Create/refresh one alias per staged image; returns (aliases, [(repo, image) not staged])."""
    alias_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Tuple[str, ...]] = []
    missing: List[Tuple[str, str]] = []
    for image, repo in images.items():
        staged = staged_dir / staged_sqsh_name(image)
        if not staged.is_file() or staged.stat().st_size == 0:
            missing.append((repo, image))
        else:
            server_image = server_image_name(image)
            alias = sqsh_path(alias_dir, server_image)
            target = staged.resolve()
            if (alias.is_symlink() or alias.exists()) and not (
                alias.is_symlink() and Path(os.readlink(alias)) == target
            ):
                alias.unlink()
            if not alias.is_symlink():
                alias.symlink_to(target)
            manifest_rows.append((alias.name, repo, image, server_image, str(target), str(target.stat().st_size)))

    with (alias_dir / MANIFEST_NAME).open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["alias", "repo", "image", "server_image", "staged", "bytes"])
        writer.writerows(sorted(manifest_rows, key=lambda r: r[1].lower()))
    return len(manifest_rows), missing


def check_resolution(benchmark_jsonl: Path, alias_dir: Path, count: int, seed: int) -> int:
    """Resolve `count` random rows the way the server does; returns the number that failed."""
    # Importing the server registers the SWE-smith harness entries (swebench_patches) that make_test_spec needs.
    from swebench.harness.run_evaluation import make_test_spec
    from swebench.harness.test_spec.test_spec import LATEST

    from nemo_gym.sandbox.providers.enroot.provider import EnrootProvider
    from resources_servers.swebench.app import SWEBenchInstanceRequest
    from resources_servers.swebench.swe_smith_harness import adapt_test_spec

    # Reservoir sampling: the prepared file runs to several GB (the P2P lists), so only the sample is parsed.
    rng = random.Random(seed)
    sampled_lines: List[str] = []
    with benchmark_jsonl.open(encoding="utf-8") as f:
        for index, line in enumerate(line for line in f if line.strip()):
            if index < count:
                sampled_lines.append(line)
            else:
                slot = rng.randrange(index + 1)
                if slot < count:
                    sampled_lines[slot] = line
    sample = [json.loads(line) for line in sampled_lines]
    provider = EnrootProvider(create={"sqsh_cache_dir": str(alias_dir)})
    failures = 0
    for row in sample:
        body = SWEBenchInstanceRequest.model_validate(row)
        instance = body.model_dump()
        # The calls SwebenchResourcesServer._make_test_spec makes.
        test_spec = adapt_test_spec(
            make_test_spec(instance, namespace="swebench", instance_image_tag=LATEST, env_image_tag=LATEST), instance
        )
        image = test_spec.instance_image_key
        status = "ok"
        if image != server_image_name(row["image_name"]):
            status = f"FORMULA MISMATCH (images.py says {server_image_name(row['image_name'])})"
            failures += 1
        if not sqsh_present(alias_dir, image):
            # Not calling _ensure_image: on a miss it would try `enroot import` from Docker Hub.
            print(f"{row['instance_id']}\t{image}\t{sqsh_cache_key(image)}\tMISSING in {alias_dir}")
            failures += 1
        else:
            resolved = asyncio.run(provider._ensure_image(image))
            size_gib = resolved.stat().st_size / 2**30
            print(f"{row['instance_id']}\t{image}\t{resolved}\t-> {resolved.resolve()}\t{size_gib:.2f} GiB\t{status}")
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged-dir", type=Path, required=True, help="directory of <image with / -> _>.sqsh imports")
    parser.add_argument(
        "--alias-dir", type=Path, required=True, help="directory to fill with <server key>.sqsh symlinks"
    )
    parser.add_argument("--images", type=Path, help="image names to alias, one per line (default: all profiles)")
    parser.add_argument(
        "--check", type=int, default=0, metavar="N", help="resolve N random benchmark rows like the server"
    )
    parser.add_argument("--benchmark-jsonl", type=Path, help="prepared rows to sample for --check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    images = read_images(args.images)
    aliases, missing = build_aliases(images, args.staged_dir, args.alias_dir)
    print(f"{aliases} aliases in {args.alias_dir} for {len(images)} SWE-smith images; {len(missing)} not staged")
    for repo, image in missing:
        print(f"not staged\t{repo}\t{image}")

    if args.check:
        if args.benchmark_jsonl is None:
            parser.error("--check needs --benchmark-jsonl")
        failures = check_resolution(args.benchmark_jsonl, args.alias_dir, args.check, args.seed)
        print(f"check: {args.check} sampled, {failures} failed")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
