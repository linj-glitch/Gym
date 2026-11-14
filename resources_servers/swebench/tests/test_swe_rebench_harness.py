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
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from swebench.harness.constants import (
    END_TEST_OUTPUT,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
)
from swebench.harness.test_spec.test_spec import TestSpec

from benchmarks.swebench.swe_gym.images import sqsh_cache_key
from benchmarks.swebench.swe_rebench import prepare as rebench_prepare
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import (
    DockerContainer,
    SWEBenchInstanceRequest,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchVerifyRequest,
)
from resources_servers.swebench.swe_rebench_harness import (
    REMOTE_CONDA_ACTIVATE,
    SWERebenchTestSpec,
    eval_report,
    is_swe_rebench_instance,
    log_parser_for,
    make_swe_rebench_test_spec,
    swe_rebench_eval_script_list,
    swe_rebench_test_directives,
)
from resources_servers.swebench.tests.test_app import make_sandbox


IMAGE = "swerebench/sweb.eval.x86_64.pgmpy_1776_pgmpy-3137:latest"
BASE_COMMIT = "da98466c0a79cb416562e450d85699aedc10f422"
TEST_CMD = "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider pgmpy/tests/test_PC.py"
F2P = "pgmpy/tests/test_PC.py::test_stable_variant_order_independence"
P2P = "pgmpy/tests/test_PC.py::test_score"
# Changes an existing test file, adds a new one and a data file, and deletes an old test file.
TEST_PATCH = (
    "diff --git a/pgmpy/tests/test_PC.py b/pgmpy/tests/test_PC.py\n"
    "index 0ac00dd402..a31741f926 100644\n"
    "--- a/pgmpy/tests/test_PC.py\n"
    "+++ b/pgmpy/tests/test_PC.py\n"
    "@@ -1,2 +1,3 @@\n"
    " def test_score():\n"
    "     assert True\n"
    "+def test_stable_variant_order_independence(): assert True\n"
    "diff --git a/pgmpy/tests/test_new.py b/pgmpy/tests/test_new.py\n"
    "new file mode 100644\n"
    "index 0000000000..1111111111\n"
    "--- /dev/null\n"
    "+++ b/pgmpy/tests/test_new.py\n"
    "@@ -0,0 +1 @@\n"
    "+def test_new(): assert True\n"
    "diff --git a/pgmpy/tests/data/expected.json b/pgmpy/tests/data/expected.json\n"
    "new file mode 100644\n"
    "index 0000000000..2222222222\n"
    "--- /dev/null\n"
    "+++ b/pgmpy/tests/data/expected.json\n"
    "@@ -0,0 +1 @@\n"
    "+{}\n"
    "diff --git a/pgmpy/tests/test_old.py b/pgmpy/tests/test_old.py\n"
    "deleted file mode 100644\n"
    "index 3333333333..0000000000\n"
    "--- a/pgmpy/tests/test_old.py\n"
    "+++ /dev/null\n"
    "@@ -1 +0,0 @@\n"
    "-def test_old(): assert True\n"
)


def _row(**overrides: Any) -> Dict[str, Any]:
    row = {
        "repo": "pgmpy/pgmpy",
        "instance_id": "pgmpy__pgmpy-3137",
        "base_commit": BASE_COMMIT,
        "patch": "diff --git a/pgmpy/x.py b/pgmpy/x.py\n",
        "test_patch": TEST_PATCH,
        "problem_statement": "[BUG] stable variant is order dependent",
        "hints_text": "",
        "created_at": "2026-03-22 09:18:02",
        "version": "unknown",
        "FAIL_TO_PASS": json.dumps([F2P]),
        "PASS_TO_PASS": json.dumps([P2P]),
        "environment_setup_commit": BASE_COMMIT,
        "difficulty": "",
        "subset": "swe_rebench_2026_03",
        "split": "2026_03",
        "image_name": IMAGE,
        "install_config": {
            "install": 'pip install -e ".[tests]" --quiet',
            "log_parser": "parse_log_pytest",
            "python": "3.13",
            "test_cmd": TEST_CMD,
        },
    }
    row.update(overrides)
    return row


def _log(tmp_path: Path, body: str, markers: bool = True) -> Path:
    path = tmp_path / "test_output.txt"
    path.write_text(f"setup noise\n{START_TEST_OUTPUT}\n{body}\n{END_TEST_OUTPUT}\n" if markers else body)
    return path


def _prediction() -> Dict[str, str]:
    return {"instance_id": "pgmpy__pgmpy-3137", "model_patch": "diff --git a/x b/x\n"}


class TestTestSpec:
    def test_rows_without_install_config_are_not_swe_rebench(self) -> None:
        row = _row()
        del row["install_config"]
        assert not is_swe_rebench_instance(row)
        assert make_swe_rebench_test_spec(row) is None
        assert make_swe_rebench_test_spec(_row(install_config=None)) is None
        assert make_swe_rebench_test_spec(_row(install_config={})) is None

    def test_spec_comes_from_the_row_not_from_swebench_tables(self) -> None:
        assert "pgmpy/pgmpy" not in MAP_REPO_VERSION_TO_SPECS
        spec = make_swe_rebench_test_spec(_row())

        assert isinstance(spec, SWERebenchTestSpec) and isinstance(spec, TestSpec)
        assert spec.instance_image_key == IMAGE
        assert spec.is_remote_image
        assert spec.version == "unknown" and spec.language == "py" and spec.arch == "x86_64"
        assert spec.FAIL_TO_PASS == [F2P] and spec.PASS_TO_PASS == [P2P]
        assert spec.install_config["test_cmd"] == TEST_CMD

    def test_a_repository_swebench_knows_still_uses_the_rows_command(self) -> None:
        assert "sqlfluff/sqlfluff" in MAP_REPO_VERSION_TO_SPECS
        spec = make_swe_rebench_test_spec(_row(repo="sqlfluff/sqlfluff", instance_id="sqlfluff__sqlfluff-7001"))
        assert f"{TEST_CMD} pgmpy/tests/test_PC.py pgmpy/tests/test_new.py" in spec.eval_script

    def test_lists_and_json_strings_are_both_accepted(self) -> None:
        spec = make_swe_rebench_test_spec(_row(FAIL_TO_PASS=[F2P], PASS_TO_PASS=[P2P]))
        assert spec.FAIL_TO_PASS == [F2P] and spec.PASS_TO_PASS == [P2P]

    def test_refuses_what_it_cannot_grade(self) -> None:
        with pytest.raises(ValueError, match="needs image_name"):
            make_swe_rebench_test_spec(_row(image_name=None))
        with pytest.raises(ValueError, match="not a swebench Python log parser"):
            make_swe_rebench_test_spec(_row(install_config={"test_cmd": TEST_CMD, "log_parser": "parse_log_nope"}))
        with pytest.raises(ValueError, match="not a swebench Python log parser"):
            make_swe_rebench_test_spec(_row(install_config={"test_cmd": TEST_CMD, "log_parser": "os"}))
        with pytest.raises(ValueError, match="no test_cmd"):
            make_swe_rebench_test_spec(_row(install_config={"log_parser": "parse_log_pytest"}))

    def test_parser_names_resolve_to_swebench_parsers(self) -> None:
        from swebench.harness.log_parsers.python import parse_log_pytest, parse_log_pytest_v2

        assert log_parser_for({"log_parser": "parse_log_pytest"}) is parse_log_pytest
        assert log_parser_for({"log_parser": "parse_log_pytest_v2"}) is parse_log_pytest_v2


class TestEvalScript:
    def test_directives_are_changed_or_added_test_files(self) -> None:
        assert swe_rebench_test_directives(TEST_PATCH) == ["pgmpy/tests/test_PC.py", "pgmpy/tests/test_new.py"]

    def test_script_mirrors_the_fork_for_a_prebuilt_image(self) -> None:
        commands = swe_rebench_eval_script_list(_row())
        reset = f"git checkout {BASE_COMMIT} pgmpy/tests/test_PC.py pgmpy/tests/test_old.py"

        # stdout and stderr merged first: the markers are trace lines and must bracket the test output.
        assert commands[:4] == ["exec 2>&1", REMOTE_CONDA_ACTIVATE, "conda activate testbed", "cd /testbed"]
        # Activated a second time just before the tests, as in the fork.
        assert commands.count(REMOTE_CONDA_ACTIVATE) == 2 and commands.count("conda activate testbed") == 2
        assert "source /opt/miniconda3/bin/activate" not in commands
        # The image already holds the installed package.
        assert 'pip install -e ".[tests]" --quiet' not in commands
        assert f"git -c core.fileMode=false diff {BASE_COMMIT}" in commands
        start = commands.index(f": '{START_TEST_OUTPUT}'")
        assert commands[start - 2] == reset
        assert commands[start - 1].startswith("git apply -v - <<'EOF_114329324912'\n" + TEST_PATCH)
        assert commands[start + 1] == f"{TEST_CMD} pgmpy/tests/test_PC.py pgmpy/tests/test_new.py"
        assert commands[start + 2] == f": '{END_TEST_OUTPUT}'"
        assert commands[-1] == reset

    def test_list_command_and_eval_commands_are_honoured(self) -> None:
        row = _row(
            install_config={
                "test_cmd": ["pytest", "-rA"],
                "log_parser": "parse_log_pytest",
                "eval_commands": ["export TZ=UTC"],
            }
        )
        commands = swe_rebench_eval_script_list(row)
        assert commands[4] == "export TZ=UTC"
        assert "pytest -rA pgmpy/tests/test_PC.py pgmpy/tests/test_new.py" in commands

    def test_eval_script_keeps_the_upstream_header(self) -> None:
        assert make_swe_rebench_test_spec(_row()).eval_script.startswith("#!/bin/bash\nset -uxo pipefail\nexec 2>&1\n")


class TestEvalReport:
    def test_resolved_when_f2p_and_p2p_pass(self, tmp_path: Path) -> None:
        spec = make_swe_rebench_test_spec(_row())
        log = _log(tmp_path, f"PASSED {F2P}\nPASSED {P2P}")
        report = eval_report(spec, _prediction(), log, include_tests_status=True)["pgmpy__pgmpy-3137"]

        assert report["patch_successfully_applied"] and report["resolved"]
        assert report["tests_status"]["FAIL_TO_PASS"]["success"] == [F2P]

    def test_unresolved_when_an_f2p_test_fails_or_is_missing(self, tmp_path: Path) -> None:
        spec = make_swe_rebench_test_spec(_row())
        failed = eval_report(spec, _prediction(), _log(tmp_path, f"FAILED {F2P} - AssertionError\nPASSED {P2P}"), True)
        missing = eval_report(spec, _prediction(), _log(tmp_path, f"PASSED {P2P}"), True)

        assert not failed["pgmpy__pgmpy-3137"]["resolved"]
        assert not missing["pgmpy__pgmpy-3137"]["resolved"]

    def test_only_the_marked_section_is_parsed(self, tmp_path: Path) -> None:
        spec = make_swe_rebench_test_spec(_row())
        log = tmp_path / "test_output.txt"
        log.write_text(f"PASSED {F2P}\n{START_TEST_OUTPUT}\nPASSED {P2P}\n{END_TEST_OUTPUT}\n")
        assert not eval_report(spec, _prediction(), log, False)["pgmpy__pgmpy-3137"]["resolved"]

    def test_tests_that_did_not_run_are_not_an_applied_patch(self, tmp_path: Path) -> None:
        spec = make_swe_rebench_test_spec(_row())
        report = eval_report(spec, _prediction(), _log(tmp_path, f"PASSED {F2P}", markers=False), False)
        assert report["pgmpy__pgmpy-3137"] == {
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "resolved": False,
        }

    def test_missing_patch_is_reported(self, tmp_path: Path) -> None:
        spec = make_swe_rebench_test_spec(_row())
        prediction = {"instance_id": "pgmpy__pgmpy-3137", "model_patch": None}
        report = eval_report(spec, prediction, _log(tmp_path, f"PASSED {F2P}"), False)
        assert report["pgmpy__pgmpy-3137"]["patch_is_None"] and not report["pgmpy__pgmpy-3137"]["resolved"]

    def test_other_specs_go_to_upstream(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        upstream = MagicMock(return_value={"x": {"resolved": True}})
        monkeypatch.setattr("resources_servers.swebench.swe_rebench_harness.get_eval_report", upstream)
        spec = MagicMock(spec=TestSpec)

        assert eval_report(spec, _prediction(), tmp_path / "log", True) == {"x": {"resolved": True}}
        upstream.assert_called_once_with(
            test_spec=spec, prediction=_prediction(), test_log_path=tmp_path / "log", include_tests_status=True
        )


class TestServer:
    def _server(self) -> SwebenchResourcesServer:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="", sandbox_provider="test", sandbox_config=dict()
        )
        return SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_request_carries_install_config_to_the_spec(self) -> None:
        body = SWEBenchVerifyRequest(
            **_row(),
            responses_create_params={"input": [{"role": "user", "content": "q"}]},
            response={
                "id": "r",
                "created_at": 0,
                "model": "m",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        )
        spec = self._server()._make_test_spec(body)

        assert isinstance(spec, SWERebenchTestSpec)
        assert spec.instance_image_key == IMAGE
        assert spec.install_config["log_parser"] == "parse_log_pytest"

    def test_rows_without_install_config_still_validate(self) -> None:
        row = _row()
        del row["install_config"], row["image_name"]
        assert SWEBenchInstanceRequest(**row).install_config is None


class TestEvalScriptUpload:
    async def test_crlf_lines_of_a_test_patch_survive_the_eval_script_hook(self, tmp_path: Path) -> None:
        # A test patch for a file with CRLF line endings (pgmpy__pgmpy-3137) carries "\r\n" lines; the server
        # re-reads eval.sh for its per-instance hooks and must write back exactly what it read.
        crlf_patch = TEST_PATCH.replace("+def test_new(): assert True\n", "+def test_new(): assert True\r\n")
        script = make_swe_rebench_test_spec(_row(test_patch=crlf_patch)).eval_script
        assert script.count("\r\n") == 1
        eval_file = tmp_path / "eval.sh"
        with eval_file.open("w", newline="") as f:
            f.write(script)
        digest = hashlib.sha256(script.encode()).hexdigest()
        sandbox = make_sandbox(exec_result=SandboxExecResult(return_code=0, stdout=f"{digest}  /eval.sh\n", stderr=""))
        container = DockerContainer(id="run-id", instance_id="pgmpy__pgmpy-3137")
        container._inner_container = sandbox

        await container.copy(eval_file, Path("/eval.sh"))

        assert eval_file.read_bytes() == script.encode()
        sandbox.upload.assert_awaited_once()


class TestPrepare:
    def _source_row(self) -> Dict[str, Any]:
        return {
            "repo": "pgmpy/pgmpy",
            "instance_id": "pgmpy__pgmpy-3137",
            "base_commit": BASE_COMMIT,
            "patch": "diff --git a/pgmpy/x.py b/pgmpy/x.py\n",
            "test_patch": TEST_PATCH,
            "problem_statement": "[BUG] stable variant is order dependent",
            "hints_text": "",
            "created_at": "2026-03-22 09:18:02",
            "version": "unknown",
            "FAIL_TO_PASS": [F2P],
            "PASS_TO_PASS": [P2P],
            "environment_setup_commit": BASE_COMMIT,
            "docker_image": IMAGE,
            "image_name": IMAGE,
            "interface": "",
            "install_config": {
                "docker_specs": None,
                "env_vars": None,
                "env_yml_path": [],
                "install": 'pip install -e ".[tests]" --quiet',
                "log_parser": "parse_log_pytest",
                "no_use_env": False,
                "old_test_cmd": "",
                "python": "3.13",
                "test_cmd": TEST_CMD,
            },
        }

    def test_row_is_a_valid_server_request_that_builds_a_spec(self) -> None:
        row = rebench_prepare.to_benchmark_row(self._source_row(), "2026_03", "Fix {{ instance.problem_statement }}")

        assert row["subset"] == "swe_rebench_2026_03" and row["split"] == "2026_03"
        assert json.loads(row["FAIL_TO_PASS"]) == [F2P]
        assert row["install_config"] == {
            "install": 'pip install -e ".[tests]" --quiet',
            "log_parser": "parse_log_pytest",
            "no_use_env": False,
            "python": "3.13",
            "test_cmd": TEST_CMD,
        }
        assert row["responses_create_params"]["input"][0]["content"] == "Fix [BUG] stable variant is order dependent"
        request = SWEBenchInstanceRequest(**{k: v for k, v in row.items() if k != "responses_create_params"})
        assert make_swe_rebench_test_spec(request.model_dump()).instance_image_key == IMAGE

    def test_install_config_without_a_command_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no test_cmd"):
            rebench_prepare.clean_install_config({"log_parser": "parse_log_pytest", "test_cmd": ""})

    def test_instances_without_an_image_in_the_cache_are_dropped(self, tmp_path: Path) -> None:
        present, absent = (
            self._source_row(),
            self._source_row() | {"instance_id": "a__b-1", "image_name": "x/y:latest"},
        )
        (tmp_path / f"{sqsh_cache_key(IMAGE)}.sqsh").write_bytes(b"hsqs")

        kept, missing = rebench_prepare.split_by_image_presence([present, absent], tmp_path)
        assert [row["instance_id"] for row in kept] == ["pgmpy__pgmpy-3137"]
        assert missing == [("a__b-1", "x/y:latest")]
        assert rebench_prepare.split_by_image_presence([present, absent], None) == ([present, absent], [])

    def test_split_is_read_from_local_parquet(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        pq.write_table(pa.Table.from_pylist([self._source_row()]), tmp_path / "2026_03.parquet")
        monkeypatch.setattr(rebench_prepare, "DATA_DIR", tmp_path / "out")

        out = rebench_prepare.prepare(dataset_dir=str(tmp_path))
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert out.name == "swe_rebench_2026_03_benchmark.jsonl"
        assert [row["instance_id"] for row in rows] == ["pgmpy__pgmpy-3137"]
        assert rows[0]["install_config"]["test_cmd"] == TEST_CMD
        with pytest.raises(FileNotFoundError, match="no parquet file for split"):
            rebench_prepare.load_source_rows("2026_04", tmp_path)
