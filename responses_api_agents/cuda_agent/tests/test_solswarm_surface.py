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

"""Unit tests for solswarm_surface (hermetic fixtures): the submission-rule
stitcher (a port of entrypoint.sh's fragment stitcher, ours because a
pre-populated PROBLEM_DIR bypasses it upstream), the problem-dir extras, and
the minimal-profile skills staging."""

from pathlib import Path

from responses_api_agents.cuda_agent import solswarm_surface as ss


def _mk_surface_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "solswarm"
    sub = root / "docker" / "agent" / "prompts" / "submission"
    sub.mkdir(parents=True)
    for name, text in [
        ("00_header.md", "HEADER\n"),
        ("10_forbid_framework_kernels.md", "FK\n"),
        ("20_ban_cuda_graphs.md", "CG\n"),
        ("90_cheating_patterns.md", "CHEAT\n"),
        ("99_footer.md", "FOOTER\n"),
    ]:
        (sub / name).write_text(text)
    skills = root / "skills" / "agent"
    for skill in ("submit", "cudagym", "query", "insights", "hw-blackwell"):
        d = skills / skill / "scripts"
        d.mkdir(parents=True)
        (skills / skill / "SKILL.md").write_text(f"---\nname: {skill}\n---\ndoc\n")
        (d / "main.py").write_text("print('real')\n")
    return root


def test_stitch_submission_rule(tmp_path):
    root = _mk_surface_fixture(tmp_path)
    out = ss.stitch_submission_rule(root, ban_framework_kernels=True, ban_cuda_graphs=False)
    assert out == "HEADER\nFK\nCHEAT\nFOOTER\n"
    out = ss.stitch_submission_rule(
        root, ban_framework_kernels=False, ban_cuda_graphs=True, banned_libraries=["libfoo"]
    )
    assert out.startswith("HEADER\n\n## Campaign-specific banned libraries")
    assert "- `libfoo`" in out
    assert out.index("banned libraries") < out.index("CG")
    assert out.endswith("FOOTER\n")


def test_stage_surface_snapshot_all_and_stubs(tmp_path):
    root = _mk_surface_fixture(tmp_path)
    dest = tmp_path / "staging"
    skills_dir = ss.stage_surface_snapshot(root, dest, None)
    assert skills_dir == dest / "config" / "opencode" / "skills"
    assert sorted(p.name for p in skills_dir.iterdir() if p.is_dir()) == [
        "cudagym",
        "hw-blackwell",
        "insights",
        "query",
        "submit",
    ]
    # Platform-coupled skills keep SKILL.md but get stubbed scripts.
    assert "not available in this RL environment" in (skills_dir / "query" / "scripts" / "main.py").read_text()
    assert "name: query" in (skills_dir / "query" / "SKILL.md").read_text()
    # The real submit skill is untouched.
    assert (skills_dir / "submit" / "scripts" / "main.py").read_text() == "print('real')\n"
    # Idempotent on second call (marker present).
    marker_mtime = (skills_dir / ".staged").stat().st_mtime
    assert ss.stage_surface_snapshot(root, dest, None) == skills_dir
    assert (skills_dir / ".staged").stat().st_mtime == marker_mtime


def test_stage_surface_snapshot_subset_forces_mandatory(tmp_path):
    root = _mk_surface_fixture(tmp_path)
    skills_dir = ss.stage_surface_snapshot(root, tmp_path / "s2", ["hw-blackwell", "nonexistent"])
    staged = sorted(p.name for p in skills_dir.iterdir() if p.is_dir())
    assert "hw-blackwell" in staged
    for mandatory in ("submit", "cudagym", "query", "insights"):
        assert mandatory in staged
    assert "nonexistent" not in staged


def test_write_problem_extras(tmp_path):
    problem = tmp_path / "problem"
    problem.mkdir(parents=True)
    ss.write_problem_extras(problem, user_prompt="optimize it", language="triton", submission_rule="RULES\n")
    assert (problem / "user_prompt.txt").read_text() == "optimize it"
    assert (problem / "language.txt").read_text() == "triton\n"
    assert (problem / "submission_rule.md").read_text() == "RULES\n"
