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
"""SWE-rebench instances for the swebench harness.

SWE-rebench (dataset nebius/SWE-rebench-leaderboard; github.com/SWE-rebench/SWE-bench-fork)
keeps what SWE-bench stores in its per-repository, per-version tables in the row itself:

1. ``install_config`` holds the instance's test command (``test_cmd``) and the name of its
   log parser (``log_parser``, e.g. ``parse_log_pytest``). ``version`` is ``unknown`` and most
   repositories are not in ``swebench`` 4.1.0's tables (some are, with other commands), so
   ``make_test_spec`` cannot serve these rows and nothing can be registered per repository.
   ``make_swe_rebench_test_spec`` builds the spec from the row, mirroring the fork's
   ``make_test_spec`` + ``make_eval_script_list_py`` + ``_make_remote_eval_compatible`` at
   e4907b7, and ``eval_report`` grades with the row's parser, mirroring the fork's
   ``get_logs_eval``.
2. One prebuilt image per instance, named by the row (``image_name``, Docker Hub namespace
   ``swerebench``); ``SWERebenchTestSpec.instance_image_key`` is that name, as in the fork.
3. The images install conda at ``/opt/conda`` (SWE-bench: ``/opt/miniconda3``); the eval script
   therefore activates conda through the fork's lookup of both locations. The package was
   installed when the image was built, so the fork drops ``install_config.install`` from the
   eval script of a prebuilt image; so does this module.

A row is a SWE-rebench row when it carries a non-empty ``install_config``.
"""

import json
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    NON_TEST_EXTS,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_report, get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import python as _log_parsers_py
from swebench.harness.test_spec.test_spec import LATEST, TestSpec
from swebench.harness.utils import get_modified_files
from unidiff import PatchSet


__all__ = [
    "MERGE_STREAMS",
    "REMOTE_CONDA_ACTIVATE",
    "SWERebenchTestSpec",
    "eval_report",
    "is_swe_rebench_instance",
    "log_parser_for",
    "make_swe_rebench_test_spec",
    "swe_rebench_eval_script_list",
    "swe_rebench_test_directives",
]

ENV_NAME = "testbed"
# swebench.harness.constants.DOCKER_WORKDIR; SWE-rebench images check the repository out there.
REPO_DIRECTORY = "/testbed"
ARCH = "x86_64"
HEREDOC_DELIMITER = "EOF_114329324912"
# First line of the eval script: one interleaved stream, as in a container log (see swe_rebench_eval_script_list).
MERGE_STREAMS = "exec 2>&1"
# The fork's test_spec.REMOTE_CONDA_ACTIVATE: SWE-rebench images hold conda at /opt/conda.
REMOTE_CONDA_ACTIVATE = (
    "if [ -f /opt/conda/etc/profile.d/conda.sh ]; then "
    "source /opt/conda/etc/profile.d/conda.sh; "
    "elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
    "source /opt/miniconda3/etc/profile.d/conda.sh; "
    "elif [ -f /opt/miniconda3/bin/activate ]; then "
    "source /opt/miniconda3/bin/activate; "
    "else echo 'Conda activation script not found' >&2; exit 1; fi"
)


def is_swe_rebench_instance(instance: Dict[str, Any]) -> bool:
    """True for a row that carries its own harness settings (a non-empty ``install_config``)."""
    return bool(instance.get("install_config"))


def _as_list(value: Any) -> List[str]:
    """FAIL_TO_PASS / PASS_TO_PASS as a list, whether the row holds a list or a JSON string."""
    if value is None:
        return []
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)


def log_parser_for(install_config: Dict[str, Any]) -> Callable[..., Dict[str, str]]:
    """The parser ``install_config.log_parser`` names, from swebench's Python parsers.

    The fork looks the name up in its parser map (``MAP_REPO_TO_PARSER["parse_log_pytest"]``);
    every name it lists for Python is a function of ``swebench.harness.log_parsers.python``
    with the same body as in ``swebench`` 4.1.0. An unknown name is refused when the spec is
    built, before any sandbox starts.
    """
    name = install_config.get("log_parser")
    parser = getattr(_log_parsers_py, name, None) if isinstance(name, str) and name.startswith("parse_log_") else None
    if not callable(parser):
        raise ValueError(f"install_config.log_parser {name!r} is not a swebench Python log parser")
    return parser


def swe_rebench_test_directives(test_patch: str) -> List[str]:
    """The fork's ``get_default_test_directives``: files the test patch changes or adds, minus data files."""
    files = [patched.path for patched in PatchSet(StringIO(test_patch)) if not patched.is_removed_file]
    return [path for path in files if not any(path.endswith(ext) for ext in NON_TEST_EXTS)]


def swe_rebench_eval_script_list(instance: Dict[str, Any]) -> List[str]:
    """The grading commands for a SWE-rebench instance (``TestSpec.eval_script`` adds the header).

    The fork's ``make_eval_script_list_py`` for a row with ``install_config`` and a prebuilt
    image: conda activation through ``REMOTE_CONDA_ACTIVATE``, no ``install`` step, the test
    files reset to the base commit, the test patch applied, then ``test_cmd`` followed by the
    test patch's files between SWE-bench's output markers.

    One line is added in front: ``exec 2>&1``. The fork reads a container log in which stdout
    and stderr interleave, and grades what stands between the two markers; the markers are
    ``set -x`` trace lines, i.e. stderr. The sandbox API returns the two streams separately and
    the server concatenates them, which would leave every test line outside the markers
    (golden check 2026-11-14: 0 tests found in each of the first instances). Merging the
    streams inside the script restores the log the fork grades.
    """
    specs = instance["install_config"]
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]
    test_cmd = specs.get("test_cmd")
    if isinstance(test_cmd, list):
        test_cmd = " ".join(test_cmd)
    if not test_cmd:
        raise ValueError(f"{instance.get('instance_id')}: install_config has no test_cmd")

    reset_tests_command = f"git checkout {base_commit} {' '.join(get_modified_files(test_patch))}"
    apply_test_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    test_command = " ".join([test_cmd, *swe_rebench_test_directives(test_patch)])

    commands = [MERGE_STREAMS, REMOTE_CONDA_ACTIVATE, f"conda activate {ENV_NAME}", f"cd {REPO_DIRECTORY}"]
    commands += list(specs.get("eval_commands") or [])
    commands += [
        f"git config --global --add safe.directory {REPO_DIRECTORY}",
        f"cd {REPO_DIRECTORY}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        REMOTE_CONDA_ACTIVATE,
        f"conda activate {ENV_NAME}",
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    return commands


@dataclass
class SWERebenchTestSpec(TestSpec):
    """A ``TestSpec`` whose image and harness settings come from the row."""

    swe_rebench_image: str = ""
    install_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def instance_image_key(self) -> str:
        return self.swe_rebench_image


def make_swe_rebench_test_spec(
    instance: Dict[str, Any], namespace: Optional[str] = "swebench"
) -> Optional[SWERebenchTestSpec]:
    """The test spec of a SWE-rebench row; ``None`` for any other row (the caller then uses ``make_test_spec``).

    A SWE-rebench row without ``image_name`` is refused: these instances have no per-version
    environment recipe to build an image from, and the SWE-bench key the server would derive
    names an image that does not exist.
    """
    if not is_swe_rebench_instance(instance):
        return None
    install_config = dict(instance["install_config"])
    instance_id = instance["instance_id"]
    image = instance.get("image_name")
    if not image:
        raise ValueError(f"{instance_id}: a row with install_config needs image_name (its prebuilt image)")
    log_parser_for(install_config)
    return SWERebenchTestSpec(
        instance_id=instance_id,
        repo=instance["repo"],
        version=instance.get("version"),
        repo_script_list=[],
        eval_script_list=swe_rebench_eval_script_list(instance),
        env_script_list=[],
        arch=ARCH,
        FAIL_TO_PASS=_as_list(instance.get("FAIL_TO_PASS")),
        PASS_TO_PASS=_as_list(instance.get("PASS_TO_PASS")),
        language="py",
        docker_specs=dict(install_config.get("docker_specs") or {}),
        namespace=namespace,
        base_image_tag=LATEST,
        env_image_tag=LATEST,
        instance_image_tag=LATEST,
        swe_rebench_image=image,
        install_config=install_config,
    )


def _swe_rebench_status_map(test_spec: SWERebenchTestSpec, test_log_path: Any) -> tuple[Dict[str, str], bool]:
    """The fork's ``get_logs_eval`` for a row with ``install_config``: (status map, whether the tests ran)."""
    with open(test_log_path) as f:
        content = f.read()
    if any(code in content for code in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)):
        return {}, False
    if not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        return {}, False
    content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    return log_parser_for(test_spec.install_config)(content, test_spec), True


def eval_report(
    test_spec: TestSpec, prediction: Dict[str, Any], test_log_path: Any, include_tests_status: bool
) -> Dict[str, Any]:
    """``swebench.harness.grading.get_eval_report`` with the row's log parser for SWE-rebench specs.

    Upstream ``get_logs_eval`` reads the parser and the test command from the per-repository
    tables; a ``SWERebenchTestSpec`` carries them instead. Everything after the status map
    (FAIL_TO_PASS / PASS_TO_PASS accounting, resolution status, report layout) is upstream's.
    """
    if not isinstance(test_spec, SWERebenchTestSpec):
        return get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=test_log_path,
            include_tests_status=include_tests_status,
        )

    instance_id = prediction[KEY_INSTANCE_ID]
    report_map: Dict[str, Any] = {
        instance_id: {
            "patch_is_None": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "resolved": False,
        }
    }
    if prediction[KEY_PREDICTION] is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    eval_status_map, found = _swe_rebench_status_map(test_spec, test_log_path)
    if not found:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True
    if include_tests_status:
        report_map[instance_id]["tests_status"] = report
    return report_map
