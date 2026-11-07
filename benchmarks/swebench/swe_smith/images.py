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
"""Image names for SWE-smith instances.

SWE-smith publishes one environment image per repository on Docker Hub,
``swebench/swesmith.x86_64.<owner>_1776_<repo>.<commit[:8]>`` (the dataset's
``image_name`` column, untagged, i.e. ``:latest``). The swebench resources
server asks its sandbox provider for ``<image_name>:latest``
(``SWESmithTestSpec.instance_image_key``), and the enroot provider looks for
``<sqsh_cache_dir>/<sha256(image)[:16]>.sqsh`` (``EnrootProvider._ensure_image``).
Images imported with ``enroot import docker://<image_name>`` were written as
``<image_name with / -> _>.sqsh``; ``sqsh_aliases.py`` links the provider's key
to that file.
"""

import hashlib
from pathlib import Path
from typing import Union


IMAGE_TAG = "latest"


def server_image_name(image_name: str) -> str:
    """The image the swebench resources server asks the sandbox provider for."""
    return f"{image_name}:{IMAGE_TAG}"


def staged_sqsh_name(image_name: str) -> str:
    """The file name of an image imported with ``enroot import -o <name>.sqsh docker://<image_name>``."""
    return image_name.replace("/", "_").replace(":", "_") + ".sqsh"


def sqsh_cache_key(image: str) -> str:
    """The enroot provider's cache key for an image name (``EnrootProvider._ensure_image``)."""
    return hashlib.sha256(image.encode()).hexdigest()[:16]


def sqsh_path(cache_dir: Union[str, Path], image: str) -> Path:
    return Path(cache_dir) / f"{sqsh_cache_key(image)}.sqsh"


def sqsh_present(cache_dir: Union[str, Path], image: str) -> bool:
    """True when the cache holds a non-empty squashfs for the image (symlinks are followed, as the provider does)."""
    path = sqsh_path(cache_dir, image)
    try:
        return path.stat().st_size > 0
    except OSError:
        return False
