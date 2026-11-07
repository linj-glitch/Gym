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
"""SWE-smith repositories for the swebench harness.

SWE-smith (github.com/SWE-bench/SWE-smith; dataset SWE-bench/SWE-smith-py) differs
from SWE-bench in three ways the ``swebench`` resources server has to bridge:

1. One environment image per *repository*, not per instance: the row's
   ``image_name`` (``swebench/swesmith.x86_64.<owner>_1776_<repo>.<commit[:8]>``)
   holds the GitHub mirror ``swesmith/<owner>__<repo>.<commit[:8]>`` cloned at
   ``/testbed`` with its conda environment ``testbed`` installed.
   ``TestSpec.instance_image_key`` would ask for a per-instance SWE-bench image,
   so ``adapt_test_spec`` returns a ``SWESmithTestSpec`` whose image is the
   repository's.
2. A task is the clean checkout plus the row's ``patch``, which *introduces* the
   bug (``FAIL_TO_PASS`` fail once it is applied); the fix is its reverse
   (``swesmith.harness.utils._apply_patch(is_gold=True)`` runs ``git apply
   --reverse``). The agent works on the bug-patched tree with the F2P test files
   removed (``swesmith.harness.gather``: the instance branch's second commit
   "Remove F2P Tests"), while grading happens on the bug-patched tree with the
   tests present (``git checkout HEAD~1`` in ``run_patch_in_container``).
   ``benchmarks/swebench/swe_smith/prepare.py`` therefore writes rows whose
   ``patch`` is ``reverse_unified_diff(bug patch)`` (so the Verified contract
   "``patch`` resolves the instance" holds) and whose ``setup_patch`` is the bug
   patch; ``setup_hidden_files`` lists the F2P test files. The server applies
   ``setup_patch`` at seed time (hiding the files) and again in the grading
   container (files present), see ``app.py``.
3. Grading runs the repository profile's test command (``swesmith/profiles/
   python.py``: ``pytest --disable-warnings --color=no --tb=no --verbose`` for
   most repositories, the task's F2P/P2P test files appended when the profile
   sets ``min_testing``) after reverting the model's changes to those test
   files, and parses the verbose output with the profile's parser
   (``<test id> <STATUS>`` lines; SWE-bench's parsers expect ``<STATUS> <test
   id>``). ``swebench`` 4.1.0 knows none of these repositories, so
   ``register_swe_smith_harness`` registers one spec per repository (version =
   the mirror commit's eight hex digits) and the parsers, and
   ``swe_smith_eval_script_list`` replaces the SWE-bench eval script (whose
   ``git checkout <base_commit> <test files>`` would check the clean commit out
   when no test patch exists).

The per-repository data comes from ``swe_smith_profiles.py``, generated from
SWE-smith's profile classes by ``benchmarks/swebench/swe_smith/vendor_profiles.py``;
the swebench-free parts the prepare scripts need (profile lookup, test-file
rules, ``reverse_unified_diff``) live in ``swe_smith_tasks.py`` and are
re-exported here.
Deviations from SWE-smith, all documented in the functions below: mypy's
``-k`` expression is built from the graded test names (SWE-smith's reads the
``instance_ref`` field the published dataset lacks, leaving a bare ``-k``);
each test file is passed once (SWE-smith repeats a file that holds both F2P
and P2P tests), test ids are matched exactly as SWE-smith does, and the
evaluation timeout is the server's, not the profile's 90 s.
"""

import re
from dataclasses import dataclass, fields
from shlex import quote
from typing import Any, Callable, Dict, List, Tuple

from swebench.harness import constants as _constants
from swebench.harness import log_parsers as _log_parsers
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT, TestStatus
from swebench.harness.constants import python as _constants_py
from swebench.harness.log_parsers import python as _log_parsers_py
from swebench.harness.test_spec.test_spec import TestSpec

from resources_servers.swebench.swe_smith_tasks import (
    PROFILES,
    SWESMITH_COMMIT,
    profile_for,
    reverse_unified_diff,
    swe_smith_hidden_files,
    swe_smith_test_files,
    swe_smith_version,
    test_names,
)


__all__ = [
    "PROFILES",
    "SWESMITH_COMMIT",
    "SWESmithTestSpec",
    "adapt_test_spec",
    "profile_for",
    "register_swe_smith_harness",
    "reverse_unified_diff",
    "swe_smith_eval_script_list",
    "swe_smith_hidden_files",
    "swe_smith_test_command",
    "swe_smith_test_files",
    "swe_smith_version",
]

IMAGE_TAG = "latest"
# swebench.harness.constants.DOCKER_WORKDIR; SWE-smith images check the mirror out there (PythonProfile.build_image).
REPO_DIRECTORY = "/testbed"
# --- test command (RepoProfile.get_test_cmd) ---------------------------------------------------------------


def swe_smith_test_command(profile: Dict[str, Any], instance: Dict[str, Any]) -> Tuple[str, List[str]]:
    """The command SWE-smith's evaluation runs for the instance and the test files/keys it selects.

    Default rule: the profile's ``test_cmd``, with the F2P and P2P test files
    appended when the profile sets ``min_testing`` (``RepoProfile.get_test_cmd``).
    mypy's rule (``mypy_k``): the profile's command ends in ``-k`` and SWE-smith
    fills the expression from ``instance_ref.test_patch``, a field the published
    rows do not have, which would leave a bare ``-k``; here the expression is
    built from the graded test ids' last component (``[case]`` names), as the
    SWE-Gym bridge does for its mypy instances.

    Deviation: SWE-smith appends ``f2p_files + p2p_files`` verbatim, so a file
    that holds both F2P and P2P tests is passed twice and pytest runs it twice;
    the parser then keeps the second run's status, and a state-dependent test
    fails its second run (conan-io__conan.86f29e13.pr_15713's
    ``test_multi_remote_update_resolution`` graded 0 with the gold patch, golden
    check 2026-11-06). Each file is passed once here, F2P files first.
    """
    rule = profile["test_cmd_rule"]
    test_cmd = profile["test_cmd"]
    if rule == "mypy_k":
        names = test_names(instance, "FAIL_TO_PASS") + test_names(instance, "PASS_TO_PASS")
        keys = list(dict.fromkeys(name.rsplit("::", 1)[-1] for name in names))
        if not keys:
            raise ValueError(f"{instance.get('instance_id')}: no test names to build mypy's -k expression from")
        return f'{test_cmd} "{" or ".join(keys)}"', keys
    if rule != "default":
        raise ValueError(f"unknown test command rule {rule!r}")
    if profile["min_testing"]:
        f2p_files, p2p_files = swe_smith_test_files(profile, instance)
        files = list(dict.fromkeys(f2p_files + p2p_files))
        if files:
            return f"{test_cmd} {' '.join(files)}", files
    return test_cmd, []


# --- eval script -------------------------------------------------------------------------------------------


def swe_smith_eval_script_list(profile: Dict[str, Any], instance: Dict[str, Any]) -> List[str]:
    """The grading commands for a SWE-smith instance (``TestSpec.eval_script`` adds the header).

    Mirrors ``swesmith.harness.utils.run_patch_in_container``: the grading
    container holds the bug-patched tree with the tests present (``setup_patch``
    applied by the server), the model patch has been applied, the model's
    changes to the F2P/P2P test files are reverted, then the profile's test
    command runs between SWE-bench's output markers. The informational ``git``
    lines mirror SWE-bench's eval script; ``git show`` is left without its patch
    because the task commit is a root commit.
    """
    test_command, _ = swe_smith_test_command(profile, instance)
    f2p_files, p2p_files = swe_smith_test_files(profile, instance)
    test_files = list(dict.fromkeys(f2p_files + p2p_files))
    commands = [
        f"cd {REPO_DIRECTORY}",
        f"git config --global --add safe.directory {REPO_DIRECTORY}",
        "git status",
        "git show --no-patch",
        "git -c core.fileMode=false diff HEAD",
    ]
    if test_files:
        commands.append(f"git checkout -- {' '.join(quote(path) for path in test_files)}")
    commands += [f": '{START_TEST_OUTPUT}'", test_command, f": '{END_TEST_OUTPUT}'"]
    return commands


@dataclass
class SWESmithTestSpec(TestSpec):
    """A ``TestSpec`` whose sandbox image is the repository's SWE-smith environment image."""

    swe_smith_image: str = ""

    @property
    def instance_image_key(self) -> str:
        return f"{self.swe_smith_image}:{self.instance_image_tag}"


def adapt_test_spec(test_spec: TestSpec, instance: Dict[str, Any]) -> TestSpec:
    """Turn the spec ``make_test_spec`` built for a SWE-smith row into a ``SWESmithTestSpec``; other rows pass through.

    The repository image replaces the per-instance key and the SWE-smith grading
    commands replace SWE-bench's eval script. A row whose ``image_name`` is not
    the profile's image is refused: the two would grade in different environments.
    """
    profile = profile_for(test_spec.repo)
    if profile is None:
        return test_spec
    image_name = instance.get("image_name")
    if image_name and image_name != profile["image"]:
        raise ValueError(
            f"{test_spec.instance_id}: image_name {image_name!r} is not the profile image {profile['image']!r}"
        )
    values = {field.name: getattr(test_spec, field.name) for field in fields(TestSpec)}
    values["eval_script_list"] = swe_smith_eval_script_list(profile, instance)
    return SWESmithTestSpec(**values, swe_smith_image=profile["image"])


# --- log parsers (PythonProfile.log_parser and its overrides) ----------------------------------------------


def parse_log_pytest_verbose(log: str, test_spec: Any = None) -> Dict[str, str]:
    """PythonProfile.log_parser: ``<test id> <STATUS>`` lines of ``pytest --verbose``."""
    status_map: Dict[str, str] = {}
    for line in log.split("\n"):
        for status in TestStatus:
            match = re.match(rf"^(\S+)(\s+){status.value}", line)
            if match:
                status_map[match.group(1)] = status.value
    return status_map


def parse_log_pytest_xdist(log: str, test_spec: Any = None) -> Dict[str, str]:
    """HIPS/autograd: ``[gwN] STATUS <test id>`` lines of pytest-xdist."""
    status_map: Dict[str, str] = {}
    for line in log.split("\n"):
        for status in TestStatus:
            match = re.match(rf"^\[gw\d\]\s{status.value}\s(\S+)", line)
            if match:
                status_map[match.group(1)] = status.value
    return status_map


def parse_log_pytest_status_first(log: str, test_spec: Any = None) -> Dict[str, str]:
    """paramiko: ``STATUS <test id>`` lines of ``pytest -rA``."""
    status_map: Dict[str, str] = {}
    for line in log.split("\n"):
        for status in TestStatus:
            match = re.match(rf"^{status.value}\s(\S+)", line)
            if match:
                status_map[match.group(1)] = status.value
    return status_map


_UNITTEST_CASE = r"^([a-zA-Z0-9_\-,\.\s\(\)']+)\s\.{3}\s"


def parse_log_unittest_verbose(log: str, test_spec: Any = None) -> Dict[str, str]:
    """python-slugify: ``test_x (module.Class) ... ok|FAIL|ERROR`` lines of ``python test.py --verbose``."""
    status_map: Dict[str, str] = {}
    for line in log.split("\n"):
        if re.match(f"{_UNITTEST_CASE}ok$", line):
            status_map[re.match(_UNITTEST_CASE, line).group(1)] = TestStatus.PASSED.value
        for keyword, status in (("FAIL", TestStatus.FAILED), ("ERROR", TestStatus.ERROR)):
            match = re.match(f"{_UNITTEST_CASE}{keyword}$", line)
            if match:
                status_map[match.group(1)] = status.value
    return status_map


def parse_log_tornado(log: str, test_spec: Any = None) -> Dict[str, str]:
    """tornado: ``python -m tornado.test --verbose`` (unittest) output."""
    status_map: Dict[str, str] = {}
    for line in log.split("\n"):
        if line.endswith("... ok"):
            status_map[line.split(" ... ")[0]] = TestStatus.PASSED.value
        elif " ... skipped " in line:
            status_map[line.split(" ... ")[0]] = TestStatus.SKIPPED.value
        elif any(line.startswith(prefix) for prefix in ("ERROR:", "FAIL:")):
            status_map[" ".join(line.split()[1:3])] = TestStatus.FAILED.value
    return status_map


def parse_log_mypy(log: str, test_spec: Any = None) -> Dict[str, str]:
    """mypy: the last token of every line holding PASSED or FAILED (``pytest -rA`` summary lines)."""
    status_map: Dict[str, str] = {}
    for line in log.split("\n"):
        statuses = [s.value for s in (TestStatus.PASSED, TestStatus.FAILED) if s.value in line]
        if statuses:
            status_map[line.split()[-1]] = statuses[0]
    return status_map


PARSERS: Dict[str, Callable[..., Dict[str, str]]] = {
    "pytest_verbose": parse_log_pytest_verbose,
    "pytest_xdist": parse_log_pytest_xdist,
    "pytest_status_first": parse_log_pytest_status_first,
    "unittest_verbose": parse_log_unittest_verbose,
    "tornado": parse_log_tornado,
    "mypy": parse_log_mypy,
}


# --- registration ------------------------------------------------------------------------------------------


def register_swe_smith_harness() -> List[str]:
    """Register the SWE-smith repositories in the swebench harness maps (idempotent).

    One spec per repository under the version ``swe_smith_version(repo)``, with
    the keys the never executed environment/repository scripts render
    (``python``) and the grader reads (``test_cmd``); extension ``py``; the
    profile's parser. Entries already present are left alone; the returned list
    names the repositories that were added, under the dataset casing and in
    lower case.
    """
    added: List[str] = []
    for repo, profile in PROFILES.items():
        version = swe_smith_version(repo)
        spec = {"python": profile["python"], "test_cmd": profile["test_cmd"]}
        parser = PARSERS[profile["log_parser"]]
        for key in dict.fromkeys([repo, repo.lower()]):
            if key not in _constants.MAP_REPO_VERSION_TO_SPECS:
                added.append(key)
            for mapping in (_constants.MAP_REPO_VERSION_TO_SPECS, _constants_py.MAP_REPO_VERSION_TO_SPECS_PY):
                mapping.setdefault(key, {}).setdefault(version, spec)
            _constants.MAP_REPO_TO_EXT.setdefault(key, "py")
            for mapping in (_log_parsers.MAP_REPO_TO_PARSER, _log_parsers_py.MAP_REPO_TO_PARSER_PY):
                mapping.setdefault(key, parser)
    return added
