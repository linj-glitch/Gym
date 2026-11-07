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
"""Image names for SWE-Gym instances.

SWE-Gym publishes one image per instance on Docker Hub as
``xingyaoww/sweb.eval.x86_64.<owner>_s_<repo>-<number>:latest`` (``_s_`` stands
in for the ``__`` of the instance id). The swebench resources server never sees
that name: it asks its sandbox provider for ``TestSpec.instance_image_key``,
which for namespace ``swebench`` and tag ``latest`` is
``swebench/sweb.eval.x86_64.<instance_id lower-cased>:latest`` with ``__``
replaced by ``_1776_`` (SWE-bench's Docker Hub convention). The enroot provider
then looks for ``<sqsh_cache_dir>/<sha256(image)[:16]>.sqsh``
(``EnrootProvider._ensure_image``). ``prepare.py`` uses these helpers to keep
only the instances whose image is in the configured cache, and
``sqsh_aliases.py`` to point server-side keys at squashfs files imported under
the Docker Hub name. ``sqsh_aliases.py --check`` verifies the formulas against
the server's own ``make_test_spec`` and provider code.
"""

import hashlib
from pathlib import Path
from typing import Union


DOCKERHUB_NAMESPACE = "xingyaoww"
SERVER_NAMESPACE = "swebench"
IMAGE_TAG = "latest"
ARCH = "x86_64"


def dockerhub_image_name(instance_id: str) -> str:
    """The image SWE-Gym published for an instance (the name used when it was imported to squashfs)."""
    return f"{DOCKERHUB_NAMESPACE}/sweb.eval.{ARCH}.{instance_id.lower().replace('__', '_s_')}:{IMAGE_TAG}"


def server_image_name(instance_id: str) -> str:
    """``TestSpec.instance_image_key`` as the swebench resources server computes it (namespace swebench, tag latest)."""
    return f"{SERVER_NAMESPACE}/sweb.eval.{ARCH}.{instance_id.lower()}:{IMAGE_TAG}".replace("__", "_1776_")


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
