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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from pytest import MonkeyPatch
from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
from swebench.harness.grading import get_eval_report
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.run_evaluation import make_test_spec
from swebench.harness.test_spec.test_spec import LATEST, TestSpec

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import (
    SETUP_PATCH_REMOTE_PATH,
    SETUP_SCRIPT_REMOTE_PATH,
    SWEBenchInstanceRequest,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchSeedSessionRequest,
    apply_setup_patch,
    setup_patch_command,
)
from resources_servers.swebench.swe_smith_harness import (
    PARSERS,
    PROFILES,
    SWESmithTestSpec,
    adapt_test_spec,
    profile_for,
    register_swe_smith_harness,
    reverse_unified_diff,
    swe_smith_eval_script_list,
    swe_smith_hidden_files,
    swe_smith_test_command,
    swe_smith_test_files,
    swe_smith_version,
)
from resources_servers.swebench.tests.test_app import make_sandbox


YAMLLINT = "swesmith/adrienverge__yamllint.8513d9b9"
YAMLLINT_IMAGE = "swebench/swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9"
BUG_PATCH = (
    "diff --git a/yamllint/linter.py b/yamllint/linter.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/yamllint/linter.py\n"
    "+++ b/yamllint/linter.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def run(x):\n"
    "-    return x + 1\n"
    "+    return x - 1\n"
    " \n"
)
FIX_PATCH = (
    "diff --git a/yamllint/linter.py b/yamllint/linter.py\n"
    "index 2222222..1111111 100644\n"
    "--- a/yamllint/linter.py\n"
    "+++ b/yamllint/linter.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def run(x):\n"
    "+    return x + 1\n"
    "-    return x - 1\n"
    " \n"
)


def make_row(**overrides: Any) -> Dict[str, Any]:
    """A prepared SWE-smith row as benchmarks/swebench/swe_smith/prepare.py writes it."""
    row = {
        "repo": YAMLLINT,
        "instance_id": "adrienverge__yamllint.8513d9b9.func_basic__abc12345",
        "base_commit": PROFILES[YAMLLINT]["commit"],
        "patch": FIX_PATCH,
        "test_patch": "",
        "problem_statement": "run() is off by one",
        "hints_text": "",
        "created_at": "",
        "version": "8513d9b9",
        "FAIL_TO_PASS": json.dumps(["tests/test_linter.py::LinterTestCase::test_run_on_string"]),
        "PASS_TO_PASS": json.dumps(
            ["tests/test_linter.py::LinterTestCase::test_run_on_bytes", "tests/test_cli.py::test_a"]
        ),
        "environment_setup_commit": PROFILES[YAMLLINT]["commit"],
        "difficulty": "",
        "subset": "swe_smith",
        "split": "train",
        "image_name": YAMLLINT_IMAGE,
        "setup_patch": BUG_PATCH,
        "setup_hidden_files": ["tests/test_linter.py"],
    }
    row.update(overrides)
    return row


def server_test_spec(row: Dict[str, Any]) -> TestSpec:
    # The same calls as SwebenchResourcesServer._make_test_spec.
    instance = SWEBenchInstanceRequest.model_validate(row).model_dump()
    return adapt_test_spec(
        make_test_spec(instance, namespace="swebench", instance_image_tag=LATEST, env_image_tag=LATEST), instance
    )


class TestProfiles:
    def test_every_profile_is_complete_and_consistent(self) -> None:
        assert len(PROFILES) >= 131
        for repo, profile in PROFILES.items():
            owner_repo, commit8 = repo.removeprefix("swesmith/").rsplit(".", 1)
            assert profile["commit"].startswith(commit8) and len(profile["commit"]) == 40, repo
            assert (
                profile["image"] == f"swebench/swesmith.x86_64.{owner_repo.replace('__', '_1776_')}.{commit8}".lower()
            )
            assert profile["test_cmd"] and profile["python"], repo
            assert profile["log_parser"] in PARSERS, repo
            assert profile["test_files"] in {"pytest", "mido", "single_test_py", "tornado"}, repo
            assert profile["test_cmd_rule"] in {"default", "mypy_k"}, repo

    def test_lookup_is_case_insensitive(self) -> None:
        assert profile_for("swesmith/Cog-Creators__Red-DiscordBot.33e0eac7") is not None
        assert profile_for("swesmith/cog-creators__red-discordbot.33e0eac7") is profile_for(
            "swesmith/Cog-Creators__Red-DiscordBot.33e0eac7"
        )
        assert profile_for("getmoto/moto") is None

    def test_version_is_the_mirror_commit(self) -> None:
        assert swe_smith_version(YAMLLINT) == "8513d9b9"


class TestRegistration:
    def test_all_repositories_registered_in_both_casings(self) -> None:
        for repo, profile in PROFILES.items():
            for key in {repo, repo.lower()}:
                spec = MAP_REPO_VERSION_TO_SPECS[key][swe_smith_version(repo)]
                assert spec == {"python": profile["python"], "test_cmd": profile["test_cmd"]}
                assert MAP_REPO_TO_EXT[key] == "py"
                assert MAP_REPO_TO_PARSER[key] is PARSERS[profile["log_parser"]]

    def test_registration_is_idempotent(self) -> None:
        before = {repo: dict(MAP_REPO_VERSION_TO_SPECS[repo]) for repo in PROFILES}
        assert register_swe_smith_harness() == []
        for repo in PROFILES:
            assert MAP_REPO_VERSION_TO_SPECS[repo] == before[repo]


class TestTestSpec:
    def test_swe_smith_row_grades_in_the_repository_image(self, monkeypatch: MonkeyPatch) -> None:
        def _no_network(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("make_test_spec must not reach the network for SWE-smith rows")

        monkeypatch.setattr(requests, "get", _no_network)
        spec = server_test_spec(make_row())
        assert isinstance(spec, SWESmithTestSpec)
        assert spec.instance_image_key == f"{YAMLLINT_IMAGE}:latest"
        assert spec.is_remote_image
        assert spec.language == "py"
        assert spec.FAIL_TO_PASS == ["tests/test_linter.py::LinterTestCase::test_run_on_string"]
        lines = spec.eval_script.split("\n")
        assert lines[:2] == ["#!/bin/bash", "set -uxo pipefail"]
        assert "cd /testbed" in lines
        # The model's changes to the graded test files are reverted before the profile's command runs.
        assert "git checkout -- tests/test_linter.py tests/test_cli.py" in lines  # F2P files, then P2P files
        start = lines.index(": '>>>>> Start Test Output'")
        assert lines[start + 1] == PROFILES[YAMLLINT]["test_cmd"]
        assert lines[start + 2] == ": '>>>>> End Test Output'"
        # SWE-bench's reset/apply of a test patch would check the clean commit out when no test patch exists.
        assert not any(line.startswith("git checkout ") and not line.startswith("git checkout -- ") for line in lines)
        assert not any("git apply" in line for line in lines)
        assert "git show" not in lines

    def test_image_name_must_match_the_profile(self) -> None:
        with pytest.raises(ValueError, match="image_name"):
            server_test_spec(make_row(image_name="swebench/swesmith.x86_64.other_1776_repo.00000000"))

    def test_other_rows_pass_through_unchanged(self) -> None:
        row = make_row(
            repo="getmoto/moto",
            instance_id="getmoto__moto-7365",
            version="5.0",
            image_name=None,
            setup_patch=None,
            setup_hidden_files=None,
            test_patch="diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
        )
        spec = server_test_spec(row)
        assert type(spec) is TestSpec
        assert spec.instance_image_key == "swebench/sweb.eval.x86_64.getmoto_1776_moto-7365:latest"

    def test_every_profile_builds_a_spec(self) -> None:
        for repo, profile in PROFILES.items():
            row = make_row(
                repo=repo,
                instance_id=f"{repo.removeprefix('swesmith/')}.func_basic__00000000",
                version=swe_smith_version(repo),
                image_name=profile["image"],
                FAIL_TO_PASS=json.dumps(["tests/test_a.py::test_x (pkg.tests.a_test.T)"]),
                PASS_TO_PASS=json.dumps([]),
            )
            spec = server_test_spec(row)
            assert spec.instance_image_key == f"{profile['image']}:latest"
            assert spec.eval_script.strip()


class TestTestCommand:
    def test_default_profile_runs_the_whole_suite(self) -> None:
        profile = PROFILES[YAMLLINT]
        assert profile["min_testing"] is False
        assert swe_smith_test_command(profile, make_row()) == (profile["test_cmd"], [])

    def test_min_testing_appends_each_f2p_and_p2p_file_once(self) -> None:
        repo = "swesmith/pandas-dev__pandas.95280573"
        profile = PROFILES[repo]
        assert profile["min_testing"] is True
        row = make_row(
            repo=repo,
            FAIL_TO_PASS=json.dumps(["pandas/tests/b.py::test_1", "pandas/tests/a.py::test_2"]),
            PASS_TO_PASS=json.dumps(["pandas/tests/a.py::test_3", "pandas/tests/c.py::test_4"]),
        )
        command, files = swe_smith_test_command(profile, row)
        # F2P files first, then the P2P files; a file in both lists is passed once (SWE-smith passes it twice).
        assert files == ["pandas/tests/a.py", "pandas/tests/b.py", "pandas/tests/c.py"]
        assert command == profile["test_cmd"] + " pandas/tests/a.py pandas/tests/b.py pandas/tests/c.py"

    def test_mypy_builds_a_k_expression_from_the_graded_cases(self) -> None:
        profile = PROFILES["swesmith/python__mypy.e93f06ce"]
        row = make_row(
            FAIL_TO_PASS=json.dumps(["mypy/test/testcheck.py::TypeCheckSuite::check-flags.test::testFoo"]),
            PASS_TO_PASS=json.dumps(["mypy/test/testcheck.py::TypeCheckSuite::check-flags.test::testFoo_cached"]),
        )
        command, keys = swe_smith_test_command(profile, row)
        assert keys == ["testFoo", "testFoo_cached"]
        assert command.endswith('pytest --color=no -rA -k "testFoo or testFoo_cached"')
        with pytest.raises(ValueError):
            swe_smith_test_command(profile, make_row(FAIL_TO_PASS="[]", PASS_TO_PASS="[]"))


class TestTestFiles:
    def test_pytest_ids_map_to_their_files(self) -> None:
        f2p, p2p = swe_smith_test_files(PROFILES[YAMLLINT], make_row())
        assert f2p == ["tests/test_linter.py"]
        assert p2p == ["tests/test_cli.py", "tests/test_linter.py"]
        assert swe_smith_hidden_files(PROFILES[YAMLLINT], make_row()) == ["tests/test_linter.py"]

    def test_mido_strips_the_dev_prefix(self) -> None:
        row = make_row(FAIL_TO_PASS=json.dumps(["../dev/tests/test_parser.py::test_x"]), PASS_TO_PASS="[]")
        assert swe_smith_test_files(PROFILES["swesmith/mido__mido.a0158ff9"], row) == (["tests/test_parser.py"], [])

    def test_python_slugify_is_one_file(self) -> None:
        row = make_row(FAIL_TO_PASS=json.dumps(["test_x (__main__.TestSlugify)"]))
        assert swe_smith_test_files(PROFILES["swesmith/un33k__python-slugify.872b3750"], row) == (
            ["test.py"],
            ["test.py"],
        )

    def test_tornado_derives_files_from_unittest_ids(self) -> None:
        row = make_row(
            FAIL_TO_PASS=json.dumps(["test_connection_refused (tornado.test.iostream_test.TestIOStream)"]),
            PASS_TO_PASS=json.dumps(["Doctest: tornado.httputil.HTTPHeaders", "test_x (tornado.test.web_test.T)"]),
        )
        assert swe_smith_test_files(PROFILES["swesmith/tornadoweb__tornado.d5ac65c1"], row) == (
            ["tornado/test/iostream_test.py"],
            ["tornado/test/web_test.py"],
        )


class TestParsers:
    def test_verbose_pytest_lines_feed_the_report(self, tmp_path: Path) -> None:
        spec = server_test_spec(make_row())
        log = (
            "tests/test_linter.py::LinterTestCase::test_run_on_string PASSED [ 33%]\n"
            "tests/test_linter.py::LinterTestCase::test_run_on_bytes PASSED [ 66%]\n"
            "tests/test_cli.py::test_a FAILED [100%]\n"
        )
        test_output = tmp_path / "test_output.txt"
        test_output.write_text(f"+ : '>>>>> Start Test Output'\n{log}+ : '>>>>> End Test Output'\n")
        report = get_eval_report(spec, {"instance_id": spec.instance_id, "model_patch": "x"}, str(test_output), True)
        tests_status = report[spec.instance_id]["tests_status"]
        assert tests_status["FAIL_TO_PASS"]["success"] == ["tests/test_linter.py::LinterTestCase::test_run_on_string"]
        assert tests_status["PASS_TO_PASS"]["failure"] == ["tests/test_cli.py::test_a"]
        assert report[spec.instance_id]["resolved"] is False
        test_output.write_text(
            f"+ : '>>>>> Start Test Output'\n{log.replace('FAILED', 'PASSED')}+ : '>>>>> End Test Output'\n"
        )
        report = get_eval_report(spec, {"instance_id": spec.instance_id, "model_patch": "x"}, str(test_output), True)
        assert report[spec.instance_id]["resolved"] is True

    def test_status_first_and_xdist_parsers(self) -> None:
        assert PARSERS["pytest_status_first"]("PASSED tests/a.py::t1\nFAILED tests/a.py::t2 - boom\n") == {
            "tests/a.py::t1": "PASSED",
            "tests/a.py::t2": "FAILED",
        }
        assert PARSERS["pytest_xdist"]("[gw0] PASSED tests/a.py::t1 \n[gw3] FAILED tests/a.py::t2 \n") == {
            "tests/a.py::t1": "PASSED",
            "tests/a.py::t2": "FAILED",
        }

    def test_unittest_parsers(self) -> None:
        assert PARSERS["unittest_verbose"](
            "test_x (__main__.TestSlugify) ... ok\ntest_y (__main__.TestSlugify) ... FAIL\ntest_z (__main__.T) ... ERROR\n"
        ) == {
            "test_x (__main__.TestSlugify)": "PASSED",
            "test_y (__main__.TestSlugify)": "FAILED",
            "test_z (__main__.T)": "ERROR",
        }
        assert PARSERS["tornado"](
            "test_a (tornado.test.web_test.T) ... ok\n"
            "test_b (tornado.test.web_test.T) ... skipped 'no ssl'\n"
            "FAIL: test_c (tornado.test.web_test.T)\n"
        ) == {
            "test_a (tornado.test.web_test.T)": "PASSED",
            "test_b (tornado.test.web_test.T)": "SKIPPED",
            "test_c (tornado.test.web_test.T)": "FAILED",
        }

    def test_mypy_parser_takes_the_last_token(self) -> None:
        log = "PASSED mypy/test/testcheck.py::TypeCheckSuite::check-flags.test::testFoo\nFAILED mypy/test/x.py::S::t - boom\n"
        assert PARSERS["mypy"](log) == {
            "mypy/test/testcheck.py::TypeCheckSuite::check-flags.test::testFoo": "PASSED",
            "boom": "FAILED",
        }


class TestReverseUnifiedDiff:
    def test_reverses_bodies_headers_and_modes(self) -> None:
        assert reverse_unified_diff(BUG_PATCH) == FIX_PATCH
        assert reverse_unified_diff(reverse_unified_diff(BUG_PATCH)) == BUG_PATCH
        new_file = (
            "diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\nindex 0000000..abcdef0\n"
            "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n\\ No newline at end of file\n"
        )
        assert reverse_unified_diff(new_file) == (
            "diff --git a/pkg/new.py b/pkg/new.py\ndeleted file mode 100644\nindex abcdef0..0000000\n"
            "--- a/pkg/new.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x = 1\n-y = 2\n\\ No newline at end of file\n"
        )

    def test_removed_lines_that_look_like_headers_stay_in_the_hunk(self) -> None:
        patch = (
            "diff --git a/README b/README\n--- a/README\n+++ b/README\n"
            "@@ -1,2 +1,2 @@\n---- a/old heading\n+++ b/README\n text\n"
        )
        assert reverse_unified_diff(patch) == (
            "diff --git a/README b/README\n--- a/README\n+++ b/README\n"
            "@@ -1,2 +1,2 @@\n+--- a/old heading\n-++ b/README\n text\n"
        )

    def test_refuses_what_it_cannot_reverse(self) -> None:
        with pytest.raises(ValueError):
            reverse_unified_diff("diff --git a/x b/y\nsimilarity index 90%\nrename from x\nrename to y\n")
        with pytest.raises(ValueError):
            reverse_unified_diff("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n")

    @pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
    def test_git_applies_the_reverse_as_the_fix(self, tmp_path: Path) -> None:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout

        git("init", "-q")
        src = tmp_path / "pkg" / "mod.py"
        src.parent.mkdir()
        original = "def f(a, b):\n    return a + b\n\n\ndef g(x):\n    return [x] * 2\n"
        src.write_text(original)
        (tmp_path / "pkg" / "gone.py").write_text("to be deleted\n")
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "clean")
        src.write_text("def f(a, b):\n    return a - b\n\n\ndef g(x):\n    return [x]\n")
        (tmp_path / "pkg" / "gone.py").unlink()
        (tmp_path / "pkg" / "added.py").write_text("new = True")  # no trailing newline
        git("add", "-A")
        bug_patch = git("diff", "--cached")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "bug")

        fix_patch = tmp_path.with_name(tmp_path.name + "-fix.patch")  # outside the repository
        fix_patch.write_text(reverse_unified_diff(bug_patch))
        git("apply", "--verbose", str(fix_patch))
        assert src.read_text() == original
        assert (tmp_path / "pkg" / "gone.py").read_text() == "to be deleted\n"
        assert not (tmp_path / "pkg" / "added.py").exists()
        # The working tree is the clean commit again: staged, it differs from the bug commit and not from its parent.
        git("add", "-A")
        assert git("diff", "--cached", "--stat").strip() != ""
        assert git("diff", "--cached", "HEAD~1").strip() == ""


class TestSetupPatchHook:
    def _server(self, apply_anti_cheating: bool = True) -> SwebenchResourcesServer:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            apply_anti_cheating=apply_anti_cheating,
        )
        return SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_command_passes_hidden_files_one_per_line_and_removes_the_uploads(self) -> None:
        command = setup_patch_command("/testbed", ["tests/test_a.py", "tests/dir/test_b.py"])
        assert command.startswith("WORKING_DIRECTORY=/testbed SETUP_PATCH=/nemo_gym_setup.patch HIDDEN_FILES=")
        assert "HIDDEN_FILES='tests/test_a.py\ntests/dir/test_b.py' bash /nemo_gym_setup_patch.sh" in command
        assert command.endswith(f"rm -f {SETUP_PATCH_REMOTE_PATH} {SETUP_SCRIPT_REMOTE_PATH}; exit $_rc")
        assert "HIDDEN_FILES='' bash" in setup_patch_command("/testbed", [])

    async def test_apply_uploads_the_patch_and_the_script_then_runs_it(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(return_code=0, stdout="x\nSETUP_PATCH applied hidden=1 head=abc", stderr="")
        )
        uploaded: Dict[str, str] = {}

        async def upload(local_path: Path, remote_path: str) -> None:
            uploaded[remote_path] = Path(local_path).read_text()

        sandbox.upload = AsyncMock(side_effect=upload)
        report = await apply_setup_patch(sandbox, "/testbed", "inst", BUG_PATCH, ["tests/test_linter.py"])
        assert report == "SETUP_PATCH applied hidden=1 head=abc"
        assert uploaded[SETUP_PATCH_REMOTE_PATH] == BUG_PATCH
        assert uploaded[SETUP_SCRIPT_REMOTE_PATH].startswith("# Task checkout setup")
        assert "git checkout -q --orphan _nel_task" in uploaded[SETUP_SCRIPT_REMOTE_PATH]
        command = sandbox.exec.await_args.args[0]
        assert command == setup_patch_command("/testbed", ["tests/test_linter.py"])

    async def test_apply_fails_early_when_the_script_fails(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(return_code=3, stdout="SETUP_PATCH failed reason=apply_failed", stderr="")
        )
        with pytest.raises(RuntimeError, match="apply_failed"):
            await apply_setup_patch(sandbox, "/testbed", "inst", BUG_PATCH, [])

    async def test_seed_applies_the_setup_patch_before_the_anti_cheating_cleanup(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        server = self._server()
        commands: List[str] = []

        async def exec_(command: str, **kwargs: Any) -> SandboxExecResult:
            commands.append(command)
            if command == "pwd":
                return SandboxExecResult(return_code=0, stdout="/testbed\n", stderr="")
            if command == "git rev-parse HEAD":
                return SandboxExecResult(return_code=0, stdout="feedface\n", stderr="")
            return SandboxExecResult(return_code=0, stdout="SETUP_PATCH applied hidden=1 head=feedface", stderr="")

        sandbox = make_sandbox()
        sandbox.exec = AsyncMock(side_effect=exec_)
        monkeypatch.setattr(SwebenchResourcesServer, "_create_sandbox", AsyncMock(return_value=sandbox))
        request = MagicMock()
        request.session = {"session_id": "s1"}
        body = SWEBenchSeedSessionRequest.model_validate(make_row())
        await server.seed_session(request, body)

        setup = [c for c in commands if "bash /nemo_gym_setup_patch.sh" in c]
        cleanup = [c for c in commands if "anti_cheat_setup.sh" in c]
        assert len(setup) == 1 and len(cleanup) == 1
        assert commands.index(setup[0]) < commands.index(cleanup[0])
        assert "HIDDEN_FILES=tests/test_linter.py bash" in setup[0]
        assert server._session_id_to_seed_head["s1"] == "feedface"

    async def test_seed_without_setup_patch_is_unchanged(self, monkeypatch: MonkeyPatch) -> None:
        server = self._server()
        sandbox = make_sandbox(exec_result=SandboxExecResult(return_code=0, stdout="/testbed\n", stderr=""))
        monkeypatch.setattr(SwebenchResourcesServer, "_create_sandbox", AsyncMock(return_value=sandbox))
        request = MagicMock()
        request.session = {"session_id": "s2"}
        row = make_row(image_name=None, setup_patch=None, setup_hidden_files=None)
        await server.seed_session(request, SWEBenchSeedSessionRequest.model_validate(row))
        commands = [call.args[0] for call in sandbox.exec.await_args_list]
        assert not any("nemo_gym_setup" in c for c in commands)
        assert any("anti_cheat_setup.sh" in c for c in commands)
        sandbox.upload.assert_awaited_once()

    async def test_verify_reproduces_the_setup_without_hiding_files(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=True,
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        commands: List[str] = []

        async def exec_(command: str, **kwargs: Any) -> SandboxExecResult:
            commands.append(command)
            if command == "pwd":
                return SandboxExecResult(return_code=0, stdout="/testbed\n", stderr="")
            return SandboxExecResult(return_code=0, stdout="SETUP_PATCH applied hidden=0 head=feedface", stderr="")

        sandbox = make_sandbox()
        sandbox.exec = AsyncMock(side_effect=exec_)
        monkeypatch.setattr(SwebenchResourcesServer, "_create_sandbox", AsyncMock(return_value=sandbox))
        run_instance = AsyncMock(return_value=dict(resolved=True, completed=True))
        monkeypatch.setattr("resources_servers.swebench.app.run_instance", run_instance)
        request = MagicMock()
        request.session = {"session_id": "s3"}
        from resources_servers.swebench.app import SWEBenchVerifyRequest

        body = SWEBenchVerifyRequest.model_validate(
            make_row(
                responses_create_params={"input": []},
                response={
                    "output": [],
                    "id": "",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
            )
        )
        response = await server.verify(request, body)
        setup = [c for c in commands if "bash /nemo_gym_setup_patch.sh" in c]
        assert len(setup) == 1 and "HIDDEN_FILES='' bash" in setup[0]
        assert not any("anti_cheat_setup.sh" in c for c in commands)
        assert response.resolved is True
        # Golden verification grades the row's `patch`, the fix, in the SWE-smith image.
        assert run_instance.await_args.kwargs["pred"]["model_patch"] == FIX_PATCH
        assert run_instance.await_args.kwargs["test_spec"].instance_image_key == f"{YAMLLINT_IMAGE}:latest"

    def test_rows_without_setup_fields_still_validate(self) -> None:
        row = make_row()
        for key in ("image_name", "setup_patch", "setup_hidden_files"):
            row.pop(key)
        body = SWEBenchInstanceRequest.model_validate(row)
        assert body.image_name is None and body.setup_patch is None and body.setup_hidden_files is None


class TestPrepareTimeImports:
    """The benchmark prepare scripts run in the Gym root environment, which has no swebench package."""

    @pytest.mark.parametrize(
        "module",
        [
            "resources_servers.swebench.swe_smith_tasks",
            "benchmarks.swebench.swe_smith.prepare",
            "benchmarks.swebench.swe_smith.prepare_lite",
            "benchmarks.swebench.swe_smith.sqsh_aliases",
            "benchmarks.swebench.swe_smith.vendor_profiles",
        ],
    )
    def test_module_imports_without_swebench(self, module: str) -> None:
        # Pre-seeding sys.modules["swebench"] with None makes any `import swebench...` raise ImportError.
        code = f"import sys; sys.modules['swebench'] = None; import {module}"
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_harness_reexports_the_task_facts(self) -> None:
        from resources_servers.swebench import swe_smith_tasks

        assert reverse_unified_diff is swe_smith_tasks.reverse_unified_diff
        assert profile_for is swe_smith_tasks.profile_for
        assert swe_smith_hidden_files is swe_smith_tasks.swe_smith_hidden_files


class TestEvalScriptList:
    def test_no_test_files_means_no_revert(self) -> None:
        row = make_row(FAIL_TO_PASS=json.dumps(["Doctest: tornado.httputil.HTTPHeaders"]), PASS_TO_PASS="[]")
        commands = swe_smith_eval_script_list(PROFILES["swesmith/tornadoweb__tornado.d5ac65c1"], row)
        assert not any(c.startswith("git checkout") for c in commands)
        assert commands[-2] == PROFILES["swesmith/tornadoweb__tornado.d5ac65c1"]["test_cmd"]
