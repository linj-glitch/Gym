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
import json
from typing import Any, Dict

import pytest
import requests
from pytest import MonkeyPatch
from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
from swebench.harness.grading import get_eval_tests_report
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.run_evaluation import make_test_spec
from swebench.harness.test_spec.test_spec import LATEST

# Importing the patches module registers the SWE-Gym repositories, as the server does.
import resources_servers.swebench.swebench_patches  # noqa: F401
from resources_servers.swebench.swe_gym_harness import (
    SWE_GYM_EVAL_HEADER,
    SWE_GYM_PARSERS,
    SWE_GYM_SPECS,
    UPSTREAM_EVAL_HEADER,
    parse_log_pytest_pydantic,
    patch_swe_gym_eval_script,
    register_swe_gym_harness,
    swe_gym_repo_from_instance_id,
    swe_gym_test_directives,
)
from resources_servers.swebench.swebench_patches import patch_swebench_multilingual_golden_patch_pass


def make_row(**overrides: Any) -> Dict[str, Any]:
    row = {
        "repo": "getmoto/moto",
        "instance_id": "getmoto__moto-7365",
        "base_commit": "7f6c9cb1deafb280fe7fcc7551c38e397f11a706",
        "patch": "diff --git a/moto/x.py b/moto/x.py\n",
        "test_patch": (
            "diff --git a/tests/test_dynamodb/test_update.py b/tests/test_dynamodb/test_update.py\n"
            "--- a/tests/test_dynamodb/test_update.py\n+++ b/tests/test_dynamodb/test_update.py\n"
            "@@ -1 +1,2 @@\n pass\n+pass\n"
        ),
        "problem_statement": "DynamoDB update_item does floating point arithmetic",
        "hints_text": "",
        "created_at": "2024-02-19 20:29:03",
        "version": "5.0",
        "FAIL_TO_PASS": json.dumps(["tests/test_dynamodb/test_update.py::test_add_float"]),
        "PASS_TO_PASS": json.dumps(["tests/test_dynamodb/test_update.py::test_other"]),
        "environment_setup_commit": "7f6c9cb1deafb280fe7fcc7551c38e397f11a706",
        "difficulty": "",
    }
    row.update(overrides)
    return row


def server_test_spec(row: Dict[str, Any]):
    # The same call as SwebenchResourcesServer._make_test_spec.
    return make_test_spec(row, namespace="swebench", instance_image_tag=LATEST, env_image_tag=LATEST)


class TestRegistration:
    def test_all_repositories_registered_in_both_casings(self) -> None:
        for repo in SWE_GYM_SPECS:
            for key in {repo, repo.lower()}:
                assert key in MAP_REPO_VERSION_TO_SPECS
                assert MAP_REPO_TO_EXT[key] == "py"
                assert MAP_REPO_TO_PARSER[key] is SWE_GYM_PARSERS[repo]

    def test_registration_is_idempotent(self) -> None:
        before = {repo: dict(MAP_REPO_VERSION_TO_SPECS[repo]) for repo in SWE_GYM_SPECS}
        assert register_swe_gym_harness() == []
        for repo in SWE_GYM_SPECS:
            assert MAP_REPO_VERSION_TO_SPECS[repo] == before[repo]

    def test_specs_carry_only_evaluation_keys(self) -> None:
        allowed = {"python", "install", "eval_commands", "test_cmd"}
        for repo, specs in SWE_GYM_SPECS.items():
            assert specs, repo
            for version, spec in specs.items():
                assert set(spec) <= allowed, (repo, version)
                # make_env_script_list_py renders `conda create ... python=<python>`.
                assert spec["python"] and spec["test_cmd"], (repo, version)
                assert "packages" not in spec


class TestTestSpec:
    def test_make_test_spec_offline_and_image_key(self, monkeypatch: MonkeyPatch) -> None:
        def _no_network(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("make_test_spec must not reach the network for SWE-Gym rows")

        monkeypatch.setattr(requests, "get", _no_network)
        spec = server_test_spec(make_row())
        assert spec.instance_image_key == "swebench/sweb.eval.x86_64.getmoto_1776_moto-7365:latest"
        assert spec.language == "py"
        assert "make init" in spec.eval_script
        assert "pytest -n0 -rA tests/test_dynamodb/test_update.py" in spec.eval_script

    def test_mixed_case_repository_resolves_and_image_key_is_lower_case(self) -> None:
        spec = server_test_spec(
            make_row(
                repo="Project-MONAI/MONAI",
                instance_id="Project-MONAI__MONAI-3837",
                version="1.3",
                test_patch="diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
            )
        )
        assert spec.instance_image_key == "swebench/sweb.eval.x86_64.project-monai_1776_monai-3837:latest"
        assert "python setup.py develop;" in spec.eval_script
        assert "pytest -rA  tests/test_x.py" in spec.eval_script

    def test_every_vendored_version_builds_a_spec(self) -> None:
        for repo, specs in SWE_GYM_SPECS.items():
            for version in specs:
                spec = server_test_spec(make_row(repo=repo, version=version, instance_id="owner__repo-1"))
                assert spec.eval_script.strip()


class TestMypyDirectives:
    def test_case_markers_become_a_k_expression(self) -> None:
        test_patch = (
            "diff --git a/test-data/unit/check-flags.test b/test-data/unit/check-flags.test\n"
            "--- a/test-data/unit/check-flags.test\n+++ b/test-data/unit/check-flags.test\n"
            "@@ -1 +1,5 @@\n [case testExisting]\n+[case testNewOne]\n+x = 1\n+[case testNewTwo]\n+y = 2\n"
        )
        row = make_row(repo="python/mypy", instance_id="python__mypy-1", version="1.7", test_patch=test_patch)
        assert swe_gym_test_directives(row) == ['"testExisting or testNewOne or testNewTwo"']
        spec = server_test_spec(row)
        assert 'pytest -rA -k "testExisting or testNewOne or testNewTwo"' in spec.eval_script

    def test_without_case_markers_graded_test_names_are_used(self) -> None:
        row = make_row(
            repo="python/mypy",
            instance_id="python__mypy-2",
            version="1.0",
            test_patch=(
                "diff --git a/mypy/test/teststubgen.py b/mypy/test/teststubgen.py\n"
                "--- a/mypy/test/teststubgen.py\n+++ b/mypy/test/teststubgen.py\n@@ -1 +1,2 @@\n pass\n+pass\n"
            ),
            FAIL_TO_PASS=json.dumps(["mypy/test/teststubgen.py::StubgencSuite::test_generate_c_type_stub[param]"]),
            PASS_TO_PASS=json.dumps(["mypy/test/teststubgen.py::StubgencSuite::test_other"]),
        )
        assert swe_gym_test_directives(row) == ['"test_generate_c_type_stub or test_other"']
        assert 'pytest -n0 -rA -k "test_generate_c_type_stub or test_other"' in server_test_spec(row).eval_script

    def test_other_repositories_keep_upstream_directives(self) -> None:
        assert swe_gym_test_directives(make_row()) == ["tests/test_dynamodb/test_update.py"]


class TestParsers:
    def test_pydantic_parser_strips_progress_tags(self) -> None:
        log = (
            "PASSED tests/test_main.py::test_a\n"
            "FAILED [ 50%] tests/test_main.py::test_b - AssertionError: boom\n"
            "tests/test_main.py::test_c PASSED\n"
            "FAILED\n"
        )
        assert parse_log_pytest_pydantic(log) == {
            "tests/test_main.py::test_a": "PASSED",
            "tests/test_main.py::test_b": "FAILED",
            "tests/test_main.py::test_c": "PASSED",
        }

    @pytest.mark.parametrize("repo", ["getmoto/moto", "pandas-dev/pandas", "Project-MONAI/MONAI"])
    def test_pytest_parser_feeds_the_report(self, repo: str) -> None:
        spec = server_test_spec(make_row(repo=repo, version=next(iter(SWE_GYM_SPECS[repo])), instance_id="o__r-1"))
        statuses = MAP_REPO_TO_PARSER[repo](
            "PASSED tests/test_dynamodb/test_update.py::test_add_float\n"
            "FAILED tests/test_dynamodb/test_update.py::test_other - boom\n",
            spec,
        )
        report = get_eval_tests_report(
            statuses, {"FAIL_TO_PASS": spec.FAIL_TO_PASS, "PASS_TO_PASS": spec.PASS_TO_PASS}
        )
        assert report["FAIL_TO_PASS"]["success"] == ["tests/test_dynamodb/test_update.py::test_add_float"]
        assert report["PASS_TO_PASS"]["failure"] == ["tests/test_dynamodb/test_update.py::test_other"]


class TestEvalScriptHeader:
    def test_repository_from_instance_id(self) -> None:
        assert swe_gym_repo_from_instance_id("Project-MONAI__MONAI-6975") == "Project-MONAI/MONAI"
        assert swe_gym_repo_from_instance_id("pandas-dev__pandas-48106") == "pandas-dev/pandas"
        assert swe_gym_repo_from_instance_id("django__django-11099") is None

    def test_swe_gym_rows_run_eval_sh_without_set_u(self) -> None:
        spec = server_test_spec(
            make_row(repo="pandas-dev/pandas", instance_id="pandas-dev__pandas-48106", version="1.5")
        )
        assert spec.eval_script.split("\n")[1] == UPSTREAM_EVAL_HEADER
        patched = patch_swe_gym_eval_script(spec.eval_script, "pandas-dev__pandas-48106")
        assert patched.split("\n")[:2] == ["#!/bin/bash", SWE_GYM_EVAL_HEADER]
        assert patched.split("\n")[2:] == spec.eval_script.split("\n")[2:]

    def test_other_rows_keep_the_upstream_header(self) -> None:
        eval_sh = f"#!/bin/bash\n{UPSTREAM_EVAL_HEADER}\necho hi\n"
        assert patch_swe_gym_eval_script(eval_sh, "django__django-11099") == eval_sh

    def test_server_eval_sh_hook_applies_it(self) -> None:
        eval_sh = f"#!/bin/bash\n{UPSTREAM_EVAL_HEADER}\npytest -rA\n"
        assert (
            patch_swebench_multilingual_golden_patch_pass(eval_sh, "getmoto__moto-7607").split("\n")[1]
            == SWE_GYM_EVAL_HEADER
        )
        assert patch_swebench_multilingual_golden_patch_pass(eval_sh, "psf__requests-1142") == eval_sh


class TestBackslashNormalization:
    def test_pytest_8_doubled_backslashes_match_the_dataset_names(self) -> None:
        log = (
            "PASSED pandas/tests/indexing/test_loc.py::T::test_contains[Period\\\\('2019', 'A-DEC'\\\\), 'foo'-key0]\n"
            "PASSED modin/pandas/test/test_series.py::test_str_extract[([ab])(\\\\d)-False-separator None]\n"
            "FAILED tests/func/test_run_multistage.py::test_run_with_invalid_stage_name[\\\\] - boom\n"
            "PASSED pandas/tests/extension/test_arrow.py::test_str_is_functions[\\u0660-isdecimal-True]\n"
            "PASSED tests/test_completion.py::test_strip[-0-hy1-_=ra_app-\\t]\n"
        )
        statuses = MAP_REPO_TO_PARSER["pandas-dev/pandas"](log, None)
        assert statuses == {
            "pandas/tests/indexing/test_loc.py::T::test_contains[Period\\('2019',": "PASSED",
            "modin/pandas/test/test_series.py::test_str_extract[([ab])(\\d)-False-separator": "PASSED",
            "tests/func/test_run_multistage.py::test_run_with_invalid_stage_name[\\]": "FAILED",
            "pandas/tests/extension/test_arrow.py::test_str_is_functions[\\u0660-isdecimal-True]": "PASSED",
            "tests/test_completion.py::test_strip[-0-hy1-_=ra_app-\\t]": "PASSED",
        }

    def test_pydantic_parser_is_wrapped_too(self) -> None:
        statuses = MAP_REPO_TO_PARSER["pydantic/pydantic"]("PASSED tests/test_types.py::test_x[a\\\\(b]\n", None)
        assert statuses == {"tests/test_types.py::test_x[a\\(b]": "PASSED"}
