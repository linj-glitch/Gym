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
"""SWE-Gym repositories for the swebench harness.

The ``swebench`` package (4.1.0) knows the SWE-bench repositories only. SWE-Gym
(https://huggingface.co/datasets/SWE-Gym/SWE-Gym: 2,438 training instances from
eleven repositories) uses the SWE-bench row schema and SWE-bench-style
per-instance images (``xingyaoww/sweb.eval.x86_64.<owner>_s_<repo>-<number>``)
built with the SWE-Gym fork of SWE-bench (github.com/SWE-Gym/SWE-Bench-Fork at
242429c, ``swebench/harness/constants.py`` and ``log_parsers.py``).
``make_test_spec`` and ``get_eval_report`` look a repository up in module-level
maps and raise ``KeyError`` for an unknown one, so this module vendors the
fork's entries for the SWE-Gym repositories and registers them in the map
objects the harness functions read (``register_swe_gym_harness``, called when
``swebench_patches`` is imported, i.e. before the server builds a test spec).

Only the keys the evaluation script consumes are kept: ``install``,
``eval_commands`` and ``test_cmd`` (``make_eval_script_list_py`` re-runs
``install`` before the tests), plus ``python``, which ``make_env_script_list_py``
needs to render the (never executed) environment script. ``packages``,
``env_patches`` and ``pre_install`` only drive image builds, which never happen
here, and ``packages: environment.yml`` would make ``make_test_spec`` fetch
files from GitHub on every seed_session and verify call.

mypy's tests are data-driven (``[case name]`` blocks in ``.test`` files), so
the fork replaces the test-file directives with a ``-k`` expression of the case
names in the test patch (its ``test_cmd`` ends in ``-k``);
``swe_gym_test_directives`` does the same.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from swebench.harness import constants as _constants
from swebench.harness import log_parsers as _log_parsers
from swebench.harness.constants import TestStatus
from swebench.harness.constants import python as _constants_py
from swebench.harness.log_parsers import python as _log_parsers_py
from swebench.harness.test_spec import python as _test_spec_py


def _specs(versions: str, **spec: Any) -> Dict[str, Dict[str, Any]]:
    """One independent spec dict per whitespace-separated version (lists copied)."""
    return {
        version: {k: (list(v) if isinstance(v, list) else v) for k, v in spec.items()} for version in versions.split()
    }


# python/mypy. The fork runs `pytest -rA -k "<cases>"`, see swe_gym_test_directives.
_MYPY_INSTALL = "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r"
_MYPY_INSTALL_XDIST = "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r"
SPECS_MYPY = {
    **_specs("1.7 1.8 1.9 1.10 1.11", python="3.12", install=_MYPY_INSTALL, test_cmd="pytest -rA -k"),
    **_specs("1.0 1.1 1.2 1.3 1.4 1.5 1.6", python="3.11", install=_MYPY_INSTALL, test_cmd="pytest -n0 -rA -k"),
    **_specs(
        "0.990 0.980 0.970 0.960 0.950 0.940", python="3.10", install=_MYPY_INSTALL_XDIST, test_cmd="pytest -n0 -rA -k"
    ),
    **_specs(
        "0.920 0.910 0.820 0.810 0.800", python="3.9", install=_MYPY_INSTALL_XDIST + ";", test_cmd="pytest -n0 -rA -k"
    ),
}

# getmoto/moto
SPECS_MOTO = _specs(
    "0.4 1.0 1.2 1.3 2.0 2.1 2.2 2.3 3.0 3.1 4.0 4.1 4.2 5.0",
    python="3.12",
    install="make init",
    test_cmd="pytest -n0 -rA",
)

# conan-io/conan. Before Cython 3 the requirements need a pip constraint (conan-io/conan#14319).
_CONAN_REQUIREMENTS = (
    "python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; "
    "python -m pip install -r conans/requirements_dev.txt "
)
_CONAN_EVAL_COMMANDS = ["export PYTHONPATH=${PYTHONPATH:-}:$(pwd)"]
SPECS_CONAN = {
    **_specs(
        "1.33 1.34 1.36 2.0 1.35 1.37 1.46 1.38 1.39 1.40 1.41 1.42 1.45 1.43 1.44 1.47 1.48 1.49 1.50 1.51 1.52 "
        "1.53 1.55 1.54 1.57 1.58 1.59",
        python="3.10",
        install="echo 'cython<3' > /tmp/constraint.txt; export PIP_CONSTRAINT=/tmp/constraint.txt; "
        + _CONAN_REQUIREMENTS,
        eval_commands=_CONAN_EVAL_COMMANDS,
        test_cmd="pytest -n0 -rA",
    ),
    **_specs(
        "2.1 1.60 1.61 1.62 2.2 2.3 2.4",
        python="3.10",
        install=_CONAN_REQUIREMENTS,
        eval_commands=_CONAN_EVAL_COMMANDS,
        test_cmd="pytest -n0 -rA",
    ),
}

# dask/dask. The fork takes the interpreter from environment.yml; "3.10" here only renders the unused env script.
SPECS_DASK = _specs(
    "2.11 2.12 2.13 2.14 2.15 2.16 2.17 2.18 2.19 2.21 2.22 2.23 2.25 2.26 2.27 2.28 2.29 2.30 2020.12 2021.01 "
    "2021.02 2021.03 2021.04 2021.05 2021.06 2021.07 2021.08 2021.09 2021.10 2021.11 2021.12 2022.01 2022.02 "
    "2022.03 2022.04 2022.05 2022.6 2022.7 2022.8 2022.9 2022.10 2022.11 2022.12 2023.1 2023.2 2023.3 2023.4 "
    "2023.5 2023.6 2023.7 2023.8 2023.9 2023.10 2023.11 2023.12 2024.1 2024.2 2024.3 2024.4 2024.5",
    python="3.10",
    install="python -m pip install --no-deps -e .",
    test_cmd="pytest -n0 -rA  --color=no",
)

# Project-MONAI/MONAI
SPECS_MONAI = _specs(
    "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.11 0.105 1.0 1.1 1.2 1.3",
    python="3.8",
    install=(
        r"sed -i '/^git+https:\/\/github.com\/Project-MONAI\//d' requirements-dev.txt; "
        "python -m pip install types-pkg-resources==0.1.3 pytest; pip install -r requirements-dev.txt;"
        "python setup.py develop;"
    ),
    test_cmd="pytest -rA ",
)

# iterative/dvc
_DVC_INSTALL = (
    'python -m pip install --upgrade pip wheel GitPython; python -m pip install "cython<3.0.0" && '
    "python -m pip install --no-build-isolation pyyaml==5.4.1; "
    "python -m pip install git+https://github.com/iterative/mock-ssh-server.git || true; "
    "python -m pip install -r tests/requirements.txt || true; python -m pip install -r test-requirements.txt || true; "
    'python -m pip install -e ".[tests,dev,all_remotes,all,testing]";'
)
# pytest 8 breaks pytest-lazy-fixture; numpy is pinned for the older trees.
_DVC_PINS = ' python -m pip install "numpy<=1.20"; python -m pip install "pytest<8";'
SPECS_DVC = {
    **_specs(
        "0.1 0.8 0.9 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.27 0.28 0.29 0.30 0.31 "
        "0.32 0.33 0.34 0.35 0.40 0.41 0.50 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60 0.61 0.62 0.63 0.65 "
        "0.66 0.68 0.69 0.70 0.71 0.74 0.75 0.76 0.77 0.78 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90 "
        "0.91 0.92 0.93",
        python="3.8",
        install=_DVC_INSTALL + _DVC_PINS,
        test_cmd="pytest -rA",
    ),
    **_specs(
        "1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 1.10 1.11 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 2.10 2.11 2.12 "
        "2.13 2.15 2.17 2.19 2.20 2.21 2.22 2.23 2.24 2.27 2.28 2.30 2.33 2.34 2.35 2.38 2.41 2.43 2.44 2.45 2.46 "
        "2.48 2.50 2.51 2.52 2.54 2.55 2.56 2.57 2.58 3.0 3.1 3.2 3.3",
        python="3.9",
        install=_DVC_INSTALL + _DVC_PINS,
        test_cmd="pytest -rA",
    ),
    **_specs(
        "3.4 3.5 3.6 3.10 3.11 3.12 3.13 3.14 3.15 3.17 3.19 3.23 3.24 3.28 3.29 3.36 3.37 3.38 3.43 3.47 3.48 3.49",
        python="3.10",
        install=_DVC_INSTALL,
        test_cmd="pytest -rA",
    ),
}

# bokeh/bokeh. The `printf "1\n"` of the fork's install string holds a real newline (Python escape), kept as is.
SPECS_BOKEH = {
    **_specs(
        "3.0 3.3 3.4 3.5",
        python="3.10",
        install="python -m pip install -e .; python -m pip install bokeh_sampledata;",
        test_cmd="pytest -rA -n0",
    ),
    **_specs(
        "2.0 2.1 2.3 2.4 0.4 0.5 0.6 0.7 0.8 0.9 0.10 0.11 0.12 0.13 0.1181316818 1.0 1.1 1.2 1.3 1.4",
        python="3.8",
        install='pip install "setuptools<66" "jinja2<3.1"; printf "1\n" | python setup.py develop; bokeh sampledata;',
        test_cmd="pytest -rA -n0",
    ),
}

# modin-project/modin
SPECS_MODIN = {
    **_specs(
        "0.1 0.2 0.3 0.4 0.6 0.8 0.9 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19",
        python="3.8",
        install="python -m pip install -e .; python -m pip install numpy==1.23.1 protobuf==3.20.1;",
        test_cmd="pytest -n0 -rA",
    ),
    **_specs(
        "0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30",
        python="3.9",
        install="python -m pip install -e .;",
        test_cmd="pytest -n0 -rA",
    ),
}

# pydantic/pydantic
_PYDANTIC_INSTALL = 'export PATH="$HOME/.local/bin:$PATH"; pdm add pre-commit; make install;'
_PYDANTIC_TEST_CMD = "pytest -rA --tb=short -vv -o console_output_style=classic --no-header"
SPECS_PYDANTIC = {
    **_specs(
        "0.2 0.41 0.4 0.6 0.9 0.10 0.11 0.13 0.14 0.151 0.15 0.17 0.18 0.201 0.20 0.24 0.27 0.29 1.01 0.32 1.4 "
        "1.31 1.41 1.51 1.5 1.71 1.6 1.7 1.8 1.9 1.10",
        python="3.7",
        install=_PYDANTIC_INSTALL,
        test_cmd=_PYDANTIC_TEST_CMD,
    ),
    **_specs(
        "2.0 2.01 2.02 2.03 2.04 2.6 2.5 2.4 2.7",
        python="3.8",
        install=_PYDANTIC_INSTALL,
        test_cmd=_PYDANTIC_TEST_CMD,
    ),
}

# pandas-dev/pandas. numpy 2 is supported from pandas 2.2. Interpreter placeholder as for dask.
_PANDAS_INSTALL = (
    "python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt -y;"
)
SPECS_PANDAS = {
    **_specs(
        "0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 1.0 1.1 1.2 1.3 1.4 1.5 2.0 2.1",
        python="3.10",
        install="python -m pip install 'numpy<2'; " + _PANDAS_INSTALL,
        test_cmd="pytest -rA --tb=long",
    ),
    **_specs("2.2 3.0", python="3.10", install=_PANDAS_INSTALL, test_cmd="pytest -rA --tb=long"),
}

# facebookresearch/hydra. Older trees pin pip 24 and point isort at its GitHub HTTPS URL.
_HYDRA_INSTALL = "pip install -r requirements/dev.txt; pip install -e .;"
_HYDRA_INSTALL_OLD = (
    "sed -i 's|isort@git+git://github.com/timothycrosley/isort|isort@git+https://github.com/timothycrosley/isort|g' "
    "requirements/dev.txt; "
    '{ tail -n1 requirements/requirements.txt | grep -q "." && echo ""; } >> requirements/requirements.txt; '
    'echo "pip==24.0" >> requirements/requirements.txt;'
    'pip install "pip==24.0"; ' + _HYDRA_INSTALL
)
SPECS_HYDRA = {
    **_specs(
        "0.1 0.9 0.10 0.11 0.12 1.0 1.1 1.2",
        python="3.8",
        install=_HYDRA_INSTALL_OLD,
        test_cmd="pytest -rA --tb=long",
    ),
    **_specs("1.3 1.4", python="3.8", install=_HYDRA_INSTALL, test_cmd="pytest -rA --tb=long"),
}

# Repository names as they appear in the dataset's `repo` column (mixed case for MONAI).
SWE_GYM_SPECS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "python/mypy": SPECS_MYPY,
    "getmoto/moto": SPECS_MOTO,
    "conan-io/conan": SPECS_CONAN,
    "dask/dask": SPECS_DASK,
    "Project-MONAI/MONAI": SPECS_MONAI,
    "iterative/dvc": SPECS_DVC,
    "bokeh/bokeh": SPECS_BOKEH,
    "modin-project/modin": SPECS_MODIN,
    "pydantic/pydantic": SPECS_PYDANTIC,
    "pandas-dev/pandas": SPECS_PANDAS,
    "facebookresearch/hydra": SPECS_HYDRA,
}


def parse_log_pytest_pydantic(log: str, test_spec: Any = None) -> Dict[str, str]:
    """The fork's pydantic parser: pytest_v2 plus removal of the `[ 12%]` progress tag after FAILED."""
    test_status_map: Dict[str, str] = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    translator = str.maketrans("", "", escapes)
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        line = line.translate(translator)
        line = re.sub(r"FAILED\s*\[.*?\]", "FAILED", line)
        if any(line.startswith(status.value) for status in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[1]] = test_case[0]
        # Older pytest versions put the status last.
        elif any(line.endswith(status.value) for status in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def with_single_backslashes(parser: Callable[..., Dict[str, str]]) -> Callable[..., Dict[str, str]]:
    """Wrap a log parser so parsed test ids use single backslashes, as SWE-Gym's F2P/P2P lists do.

    pytest 8 prints a parametrize id built from a string that contains a backslash
    with the backslash doubled (`test_loc.py::...[Period\\\\('2017', ...`,
    `test_str_extract[([ab])(\\\\d)-...`, `test_run_with_invalid_stage_name[\\\\]`), while
    the dataset's lists hold the single-backslash form (3,118 such entries in 207
    instances, none doubled). Escapes pytest itself produces for non-printable or
    non-ASCII characters (`\\t`, `\\u0660`) are single on both sides. Without this,
    every parametrization of such a test counts as a PASS_TO_PASS failure although
    it passed: pandas-dev__pandas-48106 graded 0 with f2p 16/16, p2p 1017/1020.
    """

    def parse(log: str, *args: Any, **kwargs: Any) -> Dict[str, str]:
        return {name.replace("\\\\", "\\"): status for name, status in parser(log, *args, **kwargs).items()}

    parse.__name__ = f"{parser.__name__}_single_backslashes"
    return parse


SWE_GYM_PARSERS: Dict[str, Callable[..., Dict[str, str]]] = {
    **{repo: with_single_backslashes(_log_parsers_py.parse_log_pytest) for repo in SWE_GYM_SPECS},
    "pydantic/pydantic": with_single_backslashes(parse_log_pytest_pydantic),
}

_MYPY_CASE_RE = re.compile(r"\[case ([^\]]+)\]")
_upstream_get_test_directives = _test_spec_py.get_test_directives


def _test_names(instance: Dict[str, Any], key: str) -> List[str]:
    value = instance.get(key) or []
    if isinstance(value, str):
        value = json.loads(value)
    return [str(name) for name in value]


def swe_gym_test_directives(instance: Dict[str, Any]) -> List[str]:
    """Test directives with the fork's mypy rule; other repositories keep the upstream behaviour.

    mypy's `test_cmd` ends in `-k`, so the directive is one quoted expression of the
    `[case ...]` names found in the test patch. Four SWE-Gym mypy instances change
    ordinary pytest tests instead (no case marker); the fork would then run `-k ""`,
    i.e. the whole suite, so the expression is built from the graded test names.
    """
    if instance["repo"].lower() != "python/mypy":
        return _upstream_get_test_directives(instance)
    keys = _MYPY_CASE_RE.findall(instance["test_patch"])
    if not keys:
        graded = _test_names(instance, "FAIL_TO_PASS") + _test_names(instance, "PASS_TO_PASS")
        keys = [name.rsplit("::", 1)[-1].split("[", 1)[0] for name in graded]
    expression = " or ".join(dict.fromkeys(keys))
    return [f'"{expression}"']


def register_swe_gym_harness() -> Optional[List[str]]:
    """Register the SWE-Gym repositories in the swebench harness maps (idempotent).

    Entries already present (e.g. after a swebench upgrade that knows these
    repositories) are left alone; the returned list names the repositories that
    were added, under the dataset casing and in lower case.
    """
    added: List[str] = []
    for repo, specs in SWE_GYM_SPECS.items():
        parser = SWE_GYM_PARSERS[repo]
        for key in dict.fromkeys([repo, repo.lower()]):
            if key not in _constants.MAP_REPO_VERSION_TO_SPECS:
                added.append(key)
            for mapping in (_constants.MAP_REPO_VERSION_TO_SPECS, _constants_py.MAP_REPO_VERSION_TO_SPECS_PY):
                existing = mapping.setdefault(key, {})
                for version, spec in specs.items():
                    existing.setdefault(version, spec)
            _constants.MAP_REPO_TO_EXT.setdefault(key, "py")
            for mapping in (_log_parsers.MAP_REPO_TO_PARSER, _log_parsers_py.MAP_REPO_TO_PARSER_PY):
                mapping.setdefault(key, parser)
    _test_spec_py.get_test_directives = swe_gym_test_directives
    return added


# SWE-bench's eval.sh header; the SWE-Gym fork's is `set -xo pipefail`.
UPSTREAM_EVAL_HEADER = "set -uxo pipefail"
SWE_GYM_EVAL_HEADER = "set -xo pipefail"
_SWE_GYM_REPOS_LOWER = {repo.lower() for repo in SWE_GYM_SPECS}


def swe_gym_repo_from_instance_id(instance_id: str) -> Optional[str]:
    """`<owner>__<repo>-<number>` -> `<owner>/<repo>` when it is a SWE-Gym repository, else None."""
    repo = instance_id.rsplit("-", 1)[0].replace("__", "/")
    return repo if repo.lower() in _SWE_GYM_REPOS_LOWER else None


def patch_swe_gym_eval_script(eval_sh: str, instance_id: str) -> str:
    """Run a SWE-Gym instance's eval.sh without `set -u`, as the SWE-Gym fork did.

    swebench 4.1.0 starts eval.sh with `set -uxo pipefail`; the fork that built and
    validated SWE-Gym used `set -xo pipefail`. The images' conda hooks are not
    `set -u`-safe (pandas: `activate-binutils_linux-64.sh` reads `$ADDR2LINE`
    before backing it up, `proj4-deactivate.sh` reads `$_CONDA_SET_PROJ_LIB`), so
    under the upstream header `conda activate testbed` aborts the script before
    any test runs and a golden patch grades 0 (SWE-Gym golden smoke, 2026-11-06).
    """
    if swe_gym_repo_from_instance_id(instance_id) is None:
        return eval_sh
    lines = eval_sh.split("\n", 2)
    if len(lines) >= 2 and lines[1] == UPSTREAM_EVAL_HEADER:
        lines[1] = SWE_GYM_EVAL_HEADER
    return "\n".join(lines)
