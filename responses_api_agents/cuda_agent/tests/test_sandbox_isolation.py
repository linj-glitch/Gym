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

"""Tests for the pieces that decide what the model's sandbox contains.

Covers the workspace-root default (outside the repo tree), the pre-flight
sandbox manifest (names recorded, values never; stable digest), the staged
cudagym CLI copy that survives hiding /opt/nemo-rl, the subprocess environment
allowlist, mount-namespace argv ordering (binds before hides, prompt via the
environment), and container mode:
tree diffing, the pod-controller env contract (build_env), the OpenCode config
merge fragment, the enroot launch argv / base extraction / rollout seeding /
sourced env file, process-group termination, wait budgets, the
reference-sandbox launcher environment, and startup validation.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


pytest.importorskip("nemo_gym")
pytest.importorskip("cudagym")  # the shared rl helpers the app module imports at load time

from responses_api_agents.cuda_agent.app import CudaAgentConfig  # noqa: E402


def test_default_workspace_root_is_outside_the_repo_tree():
    root = Path(CudaAgentConfig.model_fields["workspace_root"].get_default(call_default_factory=True))
    assert root.is_absolute()
    # Under the system temp dir, i.e. not a repo-relative path resolved to cwd.
    assert str(root).startswith(str(Path(tempfile.gettempdir())))
    # Not inside this checkout.
    assert "Gym-workspace" not in str(root) and ".claude" not in str(root)


def _fake_sandbox(tmp_path):
    """A workspace + staged skills laid out like a real rollout's."""
    ws = tmp_path / "opencode_abc123"
    (ws / "problem").mkdir(parents=True)
    for name in ("definition.json", "workload.jsonl", "user_prompt.txt", "language.txt"):
        (ws / "problem" / name).write_text("x")
    (ws / ".opencode-data").mkdir()
    (ws / ".opencode-data" / "opencode.db").write_text("binary-session-state")

    skills = tmp_path / "skills"
    for name in ("submit", "cudagym"):
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text("---\n")
    (skills / "no_skill_md").mkdir()  # staged but unusable
    return ws, skills


def test_manifest_flags_malformed_skills(tmp_path):
    from responses_api_agents.cuda_agent import sandbox_manifest

    ws, skills = _fake_sandbox(tmp_path)
    manifest = sandbox_manifest.collect(ws, skills_dir=skills)

    assert manifest["skills"]["names"] == ["cudagym", "no_skill_md", "submit"]
    assert manifest["skills"]["malformed"] == ["no_skill_md"]
    assert "problem/definition.json (1B)" in manifest["workspace"]["tree"]

    fp = sandbox_manifest.fingerprint(manifest)
    assert fp["n_skills"] == 3 and fp["malformed_skills"] == ["no_skill_md"]
    assert "MISSING SKILL.md" in sandbox_manifest.render(manifest)


def test_manifest_records_env_names_but_never_values(tmp_path):
    from responses_api_agents.cuda_agent import sandbox_manifest

    ws, skills = _fake_sandbox(tmp_path)
    env = {"CUDAGYM_AUTH_TOKEN": "hunter2", "SOLSWARM_ARTIFACT_ID": "prob", "PATH": "/usr/bin"}
    manifest = sandbox_manifest.collect(
        ws,
        skills_dir=skills,
        env=env,
        opencode_config={"provider": {"o": {"options": {"apiKey": "TOP-SECRET"}}}},
    )
    blob = repr(manifest) + sandbox_manifest.render(manifest)

    assert manifest["env_keys"] == ["CUDAGYM_AUTH_TOKEN", "SOLSWARM_ARTIFACT_ID"]  # PATH does not affect the sandbox
    assert "hunter2" not in blob and "TOP-SECRET" not in blob
    assert manifest["opencode_config_keys"] == ["provider"]
    # Session state is not a staged sandbox input, and it is big and binary.
    assert "opencode.db" not in blob


def test_manifest_digest_is_stable_across_rollouts_but_moves_on_drift(tmp_path):
    from responses_api_agents.cuda_agent import sandbox_manifest

    ws, skills = _fake_sandbox(tmp_path)
    first = sandbox_manifest.collect(ws, skills_dir=skills)

    # A second rollout with a fresh workspace dir but the same staged inputs must share the digest.
    ws2 = tmp_path / "opencode_def456"
    (ws2 / "problem").mkdir(parents=True)
    for name in ("definition.json", "workload.jsonl", "user_prompt.txt", "language.txt"):
        (ws2 / "problem" / name).write_text("x")
    second = sandbox_manifest.collect(ws2, skills_dir=skills)
    assert sandbox_manifest.digest(first) == sandbox_manifest.digest(second)

    # Dropping a skill changes the staged inputs, so the digest must change.
    shutil.rmtree(skills / "submit")
    assert sandbox_manifest.digest(sandbox_manifest.collect(ws, skills_dir=skills)) != (sandbox_manifest.digest(first))


def test_manifest_never_raises_on_a_missing_sandbox(tmp_path):
    from responses_api_agents.cuda_agent import sandbox_manifest

    manifest = sandbox_manifest.collect(tmp_path / "does-not-exist")
    assert manifest["workspace"]["tree"] == [] and manifest["skills"]["count"] == 0
    assert sandbox_manifest.fingerprint(manifest)["digest"]  # still fingerprints


# --- hiding the job mounts: the staged cudagym CLI copy ----------------------


def test_stage_cudagym_cli_copies_the_package_out_of_the_repo(tmp_path):
    """The staged copy must satisfy `import cudagym` from PYTHONPATH alone, so
    the CLI survives /opt/nemo-rl being hidden (the venv install is editable —
    a pointer into that tree)."""
    import importlib.util
    import subprocess
    import sys

    from responses_api_agents.cuda_agent import app as cuda_app

    if importlib.util.find_spec("cudagym") is None:
        pytest.skip("cudagym not installed in this env")

    cuda_app._CLI_STAGE.clear()
    staged = cuda_app._stage_cudagym_cli(tmp_path)
    assert staged is not None
    assert (staged / "cudagym" / "__init__.py").is_file()
    # Idempotent + cached per process.
    assert cuda_app._stage_cudagym_cli(tmp_path) == staged

    # With PYTHONPATH pointing at the copy, the package resolves from the copy —
    # NOT the repo checkout — without executing any cudagym code.
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('cudagym').origin)"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(staged), "PATH": "/usr/bin:/bin"},
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip().startswith(str(staged))


def test_sandbox_env_keeps_what_the_task_needs_and_drops_the_jobs_other_credentials(monkeypatch):
    """The agent server runs inside the training job, whose environment holds
    credentials unrelated to writing a kernel (Weights & Biases, Hugging Face,
    cluster tokens). The model's subprocess environment is built from the
    allowlist instead of inherited, so only what the sandbox needs crosses."""
    from responses_api_agents.cuda_agent.app import CudaAgent, CudaAgentConfig
    from responses_api_agents.opencode_agent import app as oc_app

    monkeypatch.setattr(oc_app, "ensure_opencode", lambda version: None)
    for name, value in {
        "PATH": "/usr/bin",
        "WANDB_API_KEY": "wandb-secret",
        "HF_TOKEN": "hf-secret",
        "CUDAGYM_AUTH_TOKEN": "cudagym-token",
        "MODAL_PROXY_TOKEN_ID": "modal-id",
        "OPERATOR_DECLARED": "declared",
    }.items():
        monkeypatch.setenv(name, value)
    config = CudaAgentConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="cuda_agent",
        resources_server={"type": "resources_servers", "name": "cudagym"},
        env={"OPERATOR_DECLARED": ""},
    )
    env = CudaAgent.model_construct(config=config)._env("/data/home")

    assert "WANDB_API_KEY" not in env
    assert "HF_TOKEN" not in env
    # The in-sandbox cudagym CLI still reaches the evaluation service, the
    # process basics survive, and the caller's data home is applied.
    assert env["CUDAGYM_AUTH_TOKEN"] == "cudagym-token"
    assert env["MODAL_PROXY_TOKEN_ID"] == "modal-id"
    assert env["PATH"] == "/usr/bin"
    assert env["XDG_DATA_HOME"] == "/data/home"
    # A name the agent config declares crosses too. Its configured value is
    # empty here, which the base implementation drops, so the value that
    # arrives is the inherited one.
    assert env["OPERATOR_DECLARED"] == "declared"


def _endpoint_agent(endpoints):
    """An agent stand-in carrying just the endpoint map ``_cudagym_url`` reads."""
    import types

    from responses_api_agents.cuda_agent.app import CudaAgent

    return types.SimpleNamespace(
        config=types.SimpleNamespace(cudagym_endpoints=endpoints),
        _row_sku=lambda meta: CudaAgent._row_sku(_endpoint_agent(endpoints), meta),
    )


def _cudagym_url(endpoints, sku):
    """The endpoint a row targeting ``sku`` would be given, ``sku`` None for a row that pins none."""
    from responses_api_agents.cuda_agent.app import CudaAgent

    meta = {"language": "triton"} if sku is None else {"language": "triton", "target_hardware": sku}
    return CudaAgent._cudagym_url(_endpoint_agent(endpoints), meta)


def test_each_rows_sandbox_gets_the_endpoint_for_its_own_gpu(monkeypatch):
    """The in-sandbox cudagym CLI must measure on the GPU the row is scored on,
    so a row is given its own GPU's endpoint and a row whose GPU has no entry is
    refused rather than pointed at another GPU's."""
    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://unified.test")
    endpoints = {"B200": "http://b200.test", "H100": "http://h100.test"}

    assert _cudagym_url(endpoints, "B200") == "http://b200.test"
    assert _cudagym_url(endpoints, "H100") == "http://h100.test"
    # The message names the row's GPU and the configured ones, so the fix is
    # visible from the error.
    with pytest.raises(ValueError, match=r"'GB200'.*\['B200', 'H100'\]"):
        _cudagym_url(endpoints, "GB200")


def test_a_row_without_target_hardware_needs_an_unambiguous_endpoint_map():
    """One configured GPU places such a row; several leave nothing to choose by."""
    assert _cudagym_url({"B200": "http://b200.test"}, None) == "http://b200.test"
    with pytest.raises(ValueError, match="target_hardware"):
        _cudagym_url({"B200": "http://b200.test", "H100": "http://h100.test"}, None)


def test_endpoints_fall_back_to_the_environment_variable(monkeypatch):
    """No map at all, or a configured entry with no address, uses the variable
    the launcher exports -- how in-allocation hosting supplies an address that
    does not exist when the job is submitted."""
    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://allocated.test")
    assert _cudagym_url({}, "B200") == "http://allocated.test"
    assert _cudagym_url({"B200": ""}, "B200") == "http://allocated.test"
    # An empty map leaves every row on the variable, whatever the row targets.
    assert _cudagym_url({}, None) == "http://allocated.test"
    monkeypatch.delenv("CUDAGYM_UNIFIED_SERVER_URL")
    assert _cudagym_url({}, "B200") == ""


def test_namespace_argv_binds_before_hides_and_keeps_prompt_out_of_argv(tmp_path):
    import types

    from responses_api_agents.cuda_agent.app import CudaAgent

    stub = types.SimpleNamespace(config=types.SimpleNamespace())
    argv = ["opencode", "run", "-m", "x/y", "--dir", "/workspace"]
    out = CudaAgent._namespace_argv(
        stub,
        argv,
        tmp_path / "ws",
        tmp_path / "ws" / "problem",
        hides=["/cluster_workspace", "/datasets", "/models", "/opt/nemo-rl"],
        skills_src=tmp_path / "surface" / "skills",
        probe_dir=Path("/workspace/.probe"),
    )
    assert out[:4] == ["unshare", "--mount", "--propagation", "private"]
    script = out[-1]
    # Every bind (workspace, problem, skills) precedes every hide: a bind
    # stays valid when its source is shadowed afterwards, not before.
    # Skills bind onto production's literal path.
    assert "/home/agent/.config/opencode/skills" in script
    last_bind = max(
        script.index(str(tmp_path / "ws")),
        script.index("/home/agent/.config/opencode/skills"),
    )
    first_hide = script.index("/opt/nemo-rl")
    assert last_bind < first_hide
    assert "/cluster_workspace" in script and "/datasets" in script
    # The prompt travels via env, never quoted into the script.
    assert "$CUDA_AGENT_PROMPT" in script
    assert "sandbox_probe.py" in script
    # The sandbox check runs once per job: every other rollout skips the listing.
    out2 = CudaAgent._namespace_argv(
        stub, argv, tmp_path / "ws", tmp_path / "ws" / "problem", hides=[], probe_dir=None
    )
    assert "sandbox_probe.py" not in out2[-1]


# --- container mode: the parts that actually decide what the model gets -------


def test_diff_trees_flags_real_drift_and_ignores_content_changes():
    """Common paths cancel out; a new path is drift, a size change is not."""
    from responses_api_agents.cuda_agent import sandbox_manifest as sm

    common = ["/workspace/", "/tmp/problem/definition.json\t10", "/context/"]
    reference = {"tree": common, "cwd": "/workspace", "home": "/home/agent", "tools": {}, "env_keys": []}
    rollout = {"tree": common, "cwd": "/workspace", "home": "/home/agent", "tools": {}, "env_keys": []}
    assert sm.diff_trees(reference, rollout)["identical"] is True

    # A real difference still shows up...
    rollout_drift = dict(rollout, tree=common + ["/opt/leaked/secret.txt\t5"])
    report = sm.diff_trees(reference, rollout_drift)
    assert report["identical"] is False
    assert "/opt/leaked/secret.txt" in report["extra_in_rl"]
    # ...while a pure CONTENT change (same path, different size) does not.
    rollout_resized = dict(rollout, tree=["/workspace/", "/tmp/problem/definition.json\t999", "/context/"])
    assert sm.diff_trees(reference, rollout_resized)["identical"] is True


def test_bans_reach_the_entrypoint_or_it_strips_them_from_the_prompt():
    """entrypoint.sh deletes the ban sections unless these are exactly 'true'."""
    from responses_api_agents.cuda_agent import solswarm_container as sc

    env = sc.build_env(
        meta={"definition": {"name": "p"}, "language": "triton", "target_hardware": "B200"},
        problem_dir="/tmp/problem",
        platform_api_endpoint="e",
        cudagym_url="c",
        rollout_id="r1",
        max_duration=60,
        ban_framework_kernels=True,
        ban_cuda_graphs=True,
    )
    assert env["SOLSWARM_BAN_FRAMEWORK_KERNELS"] == "true"
    assert env["SOLSWARM_FORBID_FRAMEWORK_KERNELS"] == "true"
    assert env["SOLSWARM_BAN_CUDA_GRAPHS"] == "true"
    # submit.py's precompliance gate needs the required language.
    assert env["SOLSWARM_REQUIRED_LANGUAGE"] == "triton"
    # Their resolve_problem_description() returns early if this is pre-set, so
    # setting it would suppress the human description in the prompt header.
    assert "SOLSWARM_PROBLEM" not in env
    # Artifact and agent identify different things and must not collapse.
    assert env["SOLSWARM_ARTIFACT_ID"] != env["SOLSWARM_ARCHITECT_ID"]
    # Model routing belongs to the registry contract (registry_contract_env +
    # the synthetic catalog files), never to the pod contract itself: these
    # names only existed for the retired direct-injection path, and an old
    # agent image would honor them over the registry.
    assert "SOLSWARM_AGENT_MODEL" not in env
    assert "SOLSWARM_OPENCODE_BASE_URL" not in env
    assert "NVIDIA_INFERENCE_HUB_API_KEY" not in env


def test_training_critical_opencode_settings_survive_the_entrypoint():
    """The entrypoint generates its own OpenCode config; compaction must survive
    via the SOLSWARM_OPENCODE_CONFIG_EXTRA merge fragment."""
    import json as _json

    from responses_api_agents.cuda_agent import solswarm_container as sc

    env = sc.build_env(
        meta={"definition": {"name": "p"}, "language": "triton", "target_hardware": "B200"},
        problem_dir="/tmp/problem",
        platform_api_endpoint="e",
        cudagym_url="c",
        rollout_id="r1",
        max_duration=60,
        opencode_config_extra={"compaction": {"auto": False}},
    )
    extra = _json.loads(env["SOLSWARM_OPENCODE_CONFIG_EXTRA"])
    assert extra["compaction"]["auto"] is False


def test_enroot_launch_argv_mounts_every_rollout_path(tmp_path):
    """The enroot launcher mounts the stage, every ROLLOUT_MOUNTS path, and the
    problem dir explicitly, runs the IMAGE's /entrypoint.sh (not the checkout's)
    through an --rc script that bypasses the image's Docker ENTRYPOINT, and
    stamps the start time first."""
    from responses_api_agents.cuda_agent import solswarm_container

    argv = solswarm_container.enroot_launch_argv("base-x", rollout_dir=tmp_path, probe=True)
    assert argv[:3] == ["enroot", "start", "--root"]
    # The container name is the final argument and no command follows it: the
    # command script is the --rc file, not argv (which the image's Docker
    # ENTRYPOINT would swallow as its own arguments).
    assert argv[-1] == "base-x"
    rc_path = tmp_path / solswarm_container.ROLLOUT_RC_FILE
    assert argv[argv.index("--rc") + 1] == str(rc_path)
    assert rc_path.stat().st_mode & 0o777 == 0o700
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert f"{tmp_path}:{solswarm_container.STAGE}" in mounts
    for name, dest in solswarm_container.ROLLOUT_MOUNTS:
        assert f"{tmp_path / name}:{dest}" in mounts
    # A read-only rootfs leaves the image's /tmp immutable, so each rollout
    # gets a private writable /tmp, and problem/ must mount AFTER it to land
    # inside it (mounts apply in argv order).
    assert f"{tmp_path / 'tmp'}:/tmp" in mounts
    assert f"{tmp_path / 'problem'}:/tmp/problem" in mounts
    assert mounts.index(f"{tmp_path / 'tmp'}:/tmp") < mounts.index(f"{tmp_path / 'problem'}:/tmp/problem")
    assert (tmp_path / "tmp" / "problem").is_dir()
    script = rc_path.read_text()
    assert "bash /entrypoint.sh" in script
    assert solswarm_container.CONTAINER_START_STAMP in script
    assert "sandbox_probe.py" in script
    # enroot start resets the calling environment; the script's first act must
    # be sourcing the pod-contract env file from the stage mount.
    assert f". {solswarm_container.STAGE}/{solswarm_container.ROLLOUT_ENV_FILE}" in script
    assert script.index(solswarm_container.ROLLOUT_ENV_FILE) < script.index("/entrypoint.sh")
    # With probe=False the script contains no listing invocation.
    solswarm_container.enroot_launch_argv("b", rollout_dir=tmp_path, probe=False)
    assert "sandbox_probe.py" not in rc_path.read_text()


def test_enroot_subprocess_env_is_minimal_and_deterministic(tmp_path, monkeypatch):
    """The launcher env never inherits the pod contract: HOME stays the host's
    (ENROOT_MOUNT_HOME is off regardless), the GPU/IB hooks are voided, and the
    shared base is forced read-only."""
    from responses_api_agents.cuda_agent import solswarm_container

    monkeypatch.setenv("HOME", "/home/hostuser")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0,1")
    env = solswarm_container.enroot_subprocess_env(tmp_path)
    assert env["HOME"] == "/home/hostuser"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert env["MELLANOX_VISIBLE_DEVICES"] == "void"
    assert env["ENROOT_MOUNT_HOME"] == "n"
    assert env["ENROOT_ROOTFS_WRITABLE"] == "n"
    assert env["ENROOT_DATA_PATH"] == str(tmp_path / "data")
    # Nothing from the caller's environment leaks in beyond PATH and HOME.
    assert set(env) == {
        "PATH",
        "HOME",
        "NVIDIA_VISIBLE_DEVICES",
        "MELLANOX_VISIBLE_DEVICES",
        "ENROOT_MOUNT_HOME",
        "ENROOT_ROOTFS_WRITABLE",
        "ENROOT_RUNTIME_PATH",
        "ENROOT_CACHE_PATH",
        "ENROOT_DATA_PATH",
        "ENROOT_TEMP_PATH",
    }


def test_ensure_enroot_base_extracts_exactly_once(tmp_path, monkeypatch):
    """The multi-GB unsquash happens once: a second call finds the rootfs and
    returns without invoking enroot again."""
    import subprocess

    from responses_api_agents.cuda_agent import solswarm_container

    image = tmp_path / "agent.sqsh"
    image.write_bytes(b"sqsh")
    data_root = tmp_path / "er"
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        name = argv[argv.index("--name") + 1]
        (data_root / "data" / name / "etc").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(solswarm_container.subprocess, "run", fake_run)
    name1, rootfs, seconds = solswarm_container.ensure_enroot_base(image, data_root)
    assert seconds is not None
    assert (rootfs / "etc").is_dir()
    # Read-only starts cannot create mountpoints, so extraction must leave
    # every fstab destination in place: the stage, the rollout mounts, and
    # the problem dir.
    assert (rootfs / solswarm_container.STAGE.lstrip("/")).is_dir()
    for _, dest in solswarm_container.ROLLOUT_MOUNTS:
        assert (rootfs / dest.lstrip("/")).is_dir()
    assert (rootfs / "tmp/problem").is_dir()
    name2, _, seconds2 = solswarm_container.ensure_enroot_base(image, data_root)
    assert name2 == name1
    assert seconds2 is None
    assert len(calls) == 1


def test_seed_rollout_from_image_copies_the_mutable_image_paths(tmp_path):
    """Skills libraries and the home template come from the EXTRACTED IMAGE in
    enroot mode (the per-rollout binds would otherwise shadow them with empty
    dirs)."""
    from responses_api_agents.cuda_agent import solswarm_container

    rootfs = tmp_path / "rootfs"
    (rootfs / "skills-library" / "submit").mkdir(parents=True)
    (rootfs / "skills-library" / "submit" / "SKILL.md").write_text("s")
    (rootfs / "skills-library-compat").mkdir(parents=True)
    (rootfs / "home" / "agent" / ".codex").mkdir(parents=True)
    rollout = tmp_path / "rollout"
    for name, _ in solswarm_container.ROLLOUT_MOUNTS:
        (rollout / name).mkdir(parents=True)
    solswarm_container.seed_rollout_from_image(rollout, rootfs)
    assert (rollout / "skills-library" / "submit" / "SKILL.md").read_text() == "s"
    assert (rollout / "home" / ".codex").is_dir()


def test_write_rollout_env_round_trips_through_a_shell(tmp_path):
    """The env contract carries JSON fragments and secrets; quoting must survive
    a real `source`, and values must never need to appear in the argv."""
    import subprocess

    from responses_api_agents.cuda_agent import solswarm_container

    solswarm_container.write_rollout_env(
        tmp_path,
        {"PLAIN": "value", "SPACED": "x y", "JSON": '{"compaction": {"auto": false}}'},
    )
    env_file = tmp_path / solswarm_container.ROLLOUT_ENV_FILE
    text = env_file.read_text()
    assert "export SPACED='x y'" in text
    out = subprocess.run(
        ["bash", "-c", f'. {env_file}; printf %s "$JSON"'],
        capture_output=True,
        text=True,
    )
    assert out.stdout == '{"compaction": {"auto": false}}'


def test_launch_rc_script_forwards_term_to_the_entrypoint(tmp_path):
    """A TERM to the in-container shell (enroot forwards the launcher's group
    kill) must take the backgrounded entrypoint down instead of orphaning it,
    and the shell must wait for the entrypoint to actually exit afterwards."""
    from responses_api_agents.cuda_agent import solswarm_container

    solswarm_container.enroot_launch_argv("base", rollout_dir=tmp_path, probe=True)
    script = (tmp_path / solswarm_container.ROLLOUT_RC_FILE).read_text()
    assert "trap 'kill -TERM $EP 2>/dev/null || true' TERM" in script
    # Installed after the entrypoint pid is captured and before the first wait.
    assert script.index("EP=$!") < script.index("trap '") < script.index("if wait $EP")
    # A trap-interrupted wait returns >128; the script then waits again so the
    # entrypoint has really exited before the listing and the script's exit.
    assert script.count("if wait $EP; then rc=0; else rc=$?; fi") == 2
    assert 'while [ "$rc" -gt 128 ] && kill -0 $EP' in script
    assert script.index("done") < script.index("sandbox_probe.py")


def test_signal_process_group_kills_descendants_and_tolerates_dead_groups():
    """The group signal must reach a backgrounded grandchild (the situation the
    entrypoint is in inside the container), and signalling an exited or reaped
    process must be a no-op rather than an exception."""
    import signal as signal_module
    import subprocess
    import time as time_module
    import types

    from responses_api_agents.cuda_agent import app as cuda_app

    # A session leader whose backgrounded child would survive a plain terminate().
    leader = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 60 & echo $!; wait"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(leader.stdout.readline())
    cuda_app._signal_process_group(types.SimpleNamespace(pid=leader.pid, returncode=None), signal_module.SIGTERM)
    assert leader.wait(timeout=10) != 0
    deadline = time_module.time() + 10
    while time_module.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time_module.sleep(0.05)
    else:
        os.kill(child_pid, signal_module.SIGKILL)
        pytest.fail("the group signal left the backgrounded descendant running")
    # Already-exited process: returncode short-circuits, nothing is signalled.
    cuda_app._signal_process_group(types.SimpleNamespace(pid=leader.pid, returncode=143), signal_module.SIGTERM)
    # Reaped group with a stale handle: ProcessLookupError is swallowed.
    cuda_app._signal_process_group(types.SimpleNamespace(pid=leader.pid, returncode=None), signal_module.SIGTERM)


def test_outer_wait_budget_adds_the_tail_only_in_container_mode():
    """In container mode the inner SOLSWARM_MAX_DURATION clock (which starts
    only after boot and skills install) must be the budget that fires first,
    so the outer wait gets a fixed tail on top; the minimal profile has no
    inner budget and keeps the configured timeout unchanged."""
    from responses_api_agents.cuda_agent import app as cuda_app

    assert cuda_app.CONTAINER_TAIL_SECONDS > 0
    assert cuda_app._outer_wait_budget(3600, container_mode=True) == 3600 + cuda_app.CONTAINER_TAIL_SECONDS
    assert cuda_app._outer_wait_budget(3600, container_mode=False) == 3600


def test_build_env_exports_the_prompt_target_gpu():
    """The optimizer prompt renders ${SOLSWARM_TARGET_GPU}; production sets it
    only on the entrypoint's database path (strategy_config.target_gpu, the
    resolved lowercase GPU class id), which a pre-populated PROBLEM_DIR
    bypasses — so build_env must export the same value itself."""
    from responses_api_agents.cuda_agent import solswarm_container as sc

    kwargs = dict(
        problem_dir="/tmp/problem",
        platform_api_endpoint="e",
        cudagym_url="c",
        rollout_id="r1",
        max_duration=60,
    )
    env = sc.build_env(meta={"definition": {"name": "p"}, "language": "triton", "target_hardware": "B200"}, **kwargs)
    assert env["SOLSWARM_TARGET_GPU"] == "b200"
    # The controller-parity variable stays alongside it.
    assert env["SOLSWARM_GPU_SPEC"] == "b200"
    # No target_hardware: leave it unset so the prompt renders its own
    # default, as production does for a campaign without a gpu_spec.
    env = sc.build_env(meta={"definition": {"name": "p"}, "language": "triton"}, **kwargs)
    assert "SOLSWARM_TARGET_GPU" not in env


def test_rollout_env_file_is_created_owner_only(tmp_path):
    """The env contract carries credentials, so the file must be private from
    creation (no create-then-chmod window)."""
    from responses_api_agents.cuda_agent import solswarm_container

    solswarm_container.write_rollout_env(tmp_path, {"CUDAGYM_AUTH_TOKEN": "hunter2"})
    mode = (tmp_path / solswarm_container.ROLLOUT_ENV_FILE).stat().st_mode & 0o777
    assert mode == 0o600


def test_workspace_root_uses_the_full_uuid_and_refuses_collisions(tmp_path, monkeypatch):
    """A truncated random name with exist_ok=True would silently merge two live
    rollouts' trees on collision; the name must be the full uuid and a
    collision must fail loudly."""
    import types

    from responses_api_agents.cuda_agent import app as ca_app

    stub = types.SimpleNamespace(config=types.SimpleNamespace(workspace_root=str(tmp_path)))
    root = ca_app.CudaAgent._workspace_root(stub)
    assert root.name.startswith("opencode_")
    assert len(root.name) == len("opencode_") + 32

    class _FixedUUID:
        hex = "f" * 32

    monkeypatch.setattr(ca_app, "uuid4", lambda: _FixedUUID())
    ca_app.CudaAgent._workspace_root(stub)
    with pytest.raises(FileExistsError):
        ca_app.CudaAgent._workspace_root(stub)


def test_container_opencode_config_extra_translates_to_the_entrypoints_names():
    """The fragment must re-key the yaml's per-model limit onto the provider and
    model key the registered-endpoint dispatch generates (the agent config's own
    names exist only in our discarded config), and small_model must not pin a
    provider the generated config never defines."""
    from responses_api_agents.cuda_agent import solswarm_container as sc

    model = "nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct"
    model_id = "nvidia/qwen/qwen3-next-80b-a3b-instruct"
    config = {
        "permission": {"bash": "allow"},
        "compaction": {"auto": False, "prune": False},
        "small_model": model,
        "provider": {
            "nvinf": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": "http://x/v1", "apiKey": "TOP-SECRET"},
                "models": {model_id: {"limit": {"context": 131072, "output": 16384}}},
            }
        },
    }
    extra = sc.container_opencode_config_extra(config, model)
    assert extra["compaction"] == {"auto": False, "prune": False}
    assert extra["permission"] == {"bash": "allow"}
    assert extra["small_model"] == f"{sc.CONTAINER_OPENCODE_PROVIDER}/{sc.REGISTRY_POLICY_ID}"
    assert extra["provider"] == {
        sc.CONTAINER_OPENCODE_PROVIDER: {
            "models": {sc.REGISTRY_POLICY_ID: {"limit": {"context": 131072, "output": 16384}}}
        }
    }
    # The entrypoint owns the provider options; ours (with the apiKey) must not travel.
    assert "TOP-SECRET" not in repr(extra)
    # No limit and no small_model configured: no dangling keys in the fragment.
    assert sc.container_opencode_config_extra({"compaction": {"auto": False}}, model) == {
        "compaction": {"auto": False}
    }


def test_reference_tree_launcher_env_matches_the_rollouts(tmp_path, monkeypatch):
    """The reference subprocess must run with exactly the rollout launcher's
    environment: the pod contract reaches the container via the sourced
    .rollout-env file, and stray host variables (ENROOT_* above all) must not
    steer only the reference."""
    import subprocess

    from responses_api_agents.cuda_agent import solswarm_container

    monkeypatch.setenv("ENROOT_CANARY_HOST_VAR", "leaks")
    captured: dict = {}

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "git":
            return subprocess.CompletedProcess(argv, 1, "", "")
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(solswarm_container.subprocess, "run", fake_run)
    launcher_env = {"PATH": "/usr/bin", "ENROOT_DATA_PATH": str(tmp_path / "data")}
    result = solswarm_container.reference_tree(
        tmp_path / "checkout",
        tmp_path / "cache",
        {"definition.json": "{}"},
        argv_builder=lambda rd, probe: ["enroot", "start", str(rd)],
        cache_tag="img",
        extra_env=launcher_env,
    )
    # No listing was produced, so the build reports failure; only the launch
    # environment is under test here.
    assert result is None
    assert captured["env"] == launcher_env
    assert "ENROOT_CANARY_HOST_VAR" not in captured["env"]
    # The contract still reaches the container: it rides the sourced env file,
    # registry routing included (the reference dials an unreachable policy).
    ref_dir = next((tmp_path / "cache").glob("reference_rollout_*"))
    text = (ref_dir / solswarm_container.ROLLOUT_ENV_FILE).read_text()
    assert "SOLSWARM_AGENT_MODEL" in text and "reference-model" in text
    assert (ref_dir / solswarm_container.REGISTRY_MODELS_FILE).is_file()
    assert (ref_dir / solswarm_container.REGISTRY_ENDPOINTS_FILE).is_file()


def test_container_mode_startup_requires_a_resolvable_cudagym_url(tmp_path, monkeypatch):
    """With no address for the row's GPU, CUDAGYM_URL drops out of the container
    contract (build_env filters empty values) and the in-image cudagym CLI falls
    back to its hardcoded public server, so startup must refuse. An address can
    come from cudagym_endpoints or from CUDAGYM_UNIFIED_SERVER_URL."""
    from responses_api_agents.cuda_agent.app import CudaAgent, CudaAgentConfig
    from responses_api_agents.opencode_agent import app as oc_app

    monkeypatch.setattr(oc_app, "ensure_opencode", lambda version: None)
    monkeypatch.setattr(shutil, "which", lambda tool: f"/usr/bin/{tool}")
    image = tmp_path / "agent.sqsh"
    image.write_bytes(b"sqsh")
    config = CudaAgentConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="cuda_agent",
        resources_server={"type": "resources_servers", "name": "cudagym"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        sandbox_profile="solswarm",
        enroot_image=str(image),
    )
    # model_construct itself invokes model_post_init, so build the agent with
    # the URL present and exercise the check by re-running the hook.
    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://cudagym:8000")
    agent = CudaAgent.model_construct(config=config)

    monkeypatch.delenv("CUDAGYM_UNIFIED_SERVER_URL")
    with pytest.raises(RuntimeError, match="CUDAGYM_UNIFIED_SERVER_URL"):
        agent.model_post_init(None)

    # Endpoints that all carry an address need no variable at all.
    agent.config.cudagym_endpoints = {"B200": "http://b200.test", "H100": "http://h100.test"}
    agent.model_post_init(None)
    # One of them left empty means that GPU's rows still need the variable.
    agent.config.cudagym_endpoints = {"B200": "http://b200.test", "H100": ""}
    with pytest.raises(RuntimeError, match="CUDAGYM_UNIFIED_SERVER_URL"):
        agent.model_post_init(None)

    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://cudagym:8000")
    agent.model_post_init(None)


def test_ensure_enroot_base_redoes_a_partial_extraction(tmp_path, monkeypatch):
    """A rootfs left by an extraction that died before the completion marker is
    cleared and re-extracted (`enroot create` refuses an existing directory)."""
    import subprocess

    from responses_api_agents.cuda_agent import solswarm_container

    image = tmp_path / "agent.sqsh"
    image.write_bytes(b"sqsh")
    data_root = tmp_path / "er"
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        name = argv[argv.index("--name") + 1]
        (data_root / "data" / name / "etc").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(solswarm_container.subprocess, "run", fake_run)
    name, rootfs, first = solswarm_container.ensure_enroot_base(image, data_root)
    assert first is not None
    # Simulate the death: marker gone, stale contents present.
    (data_root / f"{name}.complete").unlink()
    (rootfs / "stale-canary").write_text("x")
    name2, rootfs2, second = solswarm_container.ensure_enroot_base(image, data_root)
    assert name2 == name
    assert second is not None
    assert len(calls) == 2
    assert not (rootfs2 / "stale-canary").exists()


def test_a_malformed_target_hardware_is_named_not_crashed_on():
    """A non-string target_hardware must not reach a dict lookup as an unhashable key."""
    with pytest.raises(ValueError, match="must be a string, got list"):
        _cudagym_url({"B200": "http://b200"}, ["B200"])
    with pytest.raises(ValueError, match="must be a string, got int"):
        _cudagym_url({"B200": "http://b200"}, 200)


def test_the_row_endpoint_outranks_inherited_per_service_urls():
    """A stray CUDAGYM_COMPILE/GPU_SERVER_URL must not outrank the row's endpoint.

    The cudagym CLI resolves the per-service names before the unified ones, and
    the sandbox env allowlist passes the whole CUDAGYM_ prefix through, so an
    inherited pair would silently point every row's `cudagym evaluate` at one
    server regardless of the GPU the row declares.
    """
    from responses_api_agents.cuda_agent.app import (
        CUDAGYM_URL_ENV_NAMES,
        _pin_cudagym_urls,
    )

    # What the sandbox would inherit: a stray pair pointing somewhere else.
    env = {"CUDAGYM_COMPILE_SERVER_URL": "http://stray:8000", "CUDAGYM_GPU_SERVER_URL": "http://stray:8000"}
    _pin_cudagym_urls(env, _cudagym_url({"H100": "http://h100:8000"}, "H100"))
    assert {env[name] for name in CUDAGYM_URL_ENV_NAMES} == {"http://h100:8000"}
    # The per-service names must come first, or pinning the unified pair alone
    # would look correct while the CLI still used the stray pair.
    assert CUDAGYM_URL_ENV_NAMES[:2] == ("CUDAGYM_COMPILE_SERVER_URL", "CUDAGYM_GPU_SERVER_URL")


def test_registry_contract_env_points_the_entrypoint_at_the_override_files():
    """Container mode hands the entrypoint the synthetic-catalog override paths."""
    from responses_api_agents.cuda_agent import solswarm_container as sc

    env = sc.registry_contract_env("rl-policy", "operator-key-sentinel")
    assert env["SOLSWARM_AGENT_MODEL"] == "rl-policy"
    assert env["SOLSWARM_AGENT_MODEL_API_KEY"] == "operator-key-sentinel"
    assert env["SOLSWARM_AGENT_MODEL_CATALOG_FILE"] == f"{sc.STAGE}/{sc.REGISTRY_MODELS_FILE}"
    assert env["SOLSWARM_AGENT_MODEL_ENDPOINTS_FILE"] == f"{sc.STAGE}/{sc.REGISTRY_ENDPOINTS_FILE}"


def test_write_registry_catalog_files_routes_the_policy_id_to_the_tokened_url(tmp_path):
    """The synthetic catalogs declare one registered_endpoint whose agent_base_url
    is the tokened model-server URL, so the entrypoint configures OpenCode to
    dial the token-capturing endpoint directly."""
    import json as _json

    from responses_api_agents.cuda_agent import solswarm_container as sc

    url = "http://127.0.0.1:9000/runs/tok123/v1"
    sc.write_registry_catalog_files(tmp_path, "rl-policy", url)

    models = _json.loads((tmp_path / sc.REGISTRY_MODELS_FILE).read_text())
    assert models["version"] == 1
    model = models["models"][0]
    assert model["id"] == "rl-policy"
    assert model["route"]["policy"] == "registered_endpoint"
    assert model["route"]["endpoint"] == "rl-policy"
    assert "opencode" in model["runtimes"]

    endpoints = _json.loads((tmp_path / sc.REGISTRY_ENDPOINTS_FILE).read_text())
    assert endpoints["version"] == 1
    entry = endpoints["endpoints"][0]
    assert entry["id"] == "rl-policy"
    assert entry["agent_base_url"] == url
    assert entry["litellm_provider"] == "openai"


def test_entrypoint_override_mount_is_gated_on_the_env_var(tmp_path, monkeypatch):
    """CUDA_AGENT_ENTRYPOINT_OVERRIDE mounts a host entrypoint over the image's own;
    unset (the normal case) it adds no such mount."""
    from responses_api_agents.cuda_agent import solswarm_container as sc

    monkeypatch.delenv("CUDA_AGENT_ENTRYPOINT_OVERRIDE", raising=False)
    argv = sc.enroot_launch_argv("base-x", rollout_dir=tmp_path, probe=False)
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert not any(m.endswith(":/entrypoint.sh") for m in mounts)

    override = tmp_path / "override-entrypoint.sh"
    override.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CUDA_AGENT_ENTRYPOINT_OVERRIDE", str(override))
    argv = sc.enroot_launch_argv("base-x", rollout_dir=tmp_path, probe=False)
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert f"{override}:/entrypoint.sh" in mounts
