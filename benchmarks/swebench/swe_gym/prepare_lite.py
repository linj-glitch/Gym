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
"""Prepare SWE-Gym Lite (SWE-Gym/SWE-Gym-Lite, 230 instances) rows; see prepare.py."""

import sys
from pathlib import Path
from typing import Optional

from benchmarks.swebench.swe_gym.prepare import DATA_DIR, resolve_sqsh_cache_dir, write_benchmark


HF_DATASET = "SWE-Gym/SWE-Gym-Lite"
SUBSET = "swe_gym_lite"
OUTPUT_FPATH = DATA_DIR / "swe_gym_lite_benchmark.jsonl"


def prepare(sqsh_cache_dir: Optional[str] = None) -> Path:
    return write_benchmark(HF_DATASET, OUTPUT_FPATH, SUBSET, resolve_sqsh_cache_dir(sqsh_cache_dir))


if __name__ == "__main__":
    prepare(sys.argv[1] if len(sys.argv) > 1 else None)
