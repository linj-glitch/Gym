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

"""Problem-dir seeding and skills staging from a SolSwarm checkout.

In container mode SolSwarm's own ``docker/agent/entrypoint.sh`` builds the
sandbox: it installs the skills, renders the role prompt, writes the OpenCode
config and launches the agent. This module deliberately covers only what that
entrypoint does NOT do for an RL rollout; keeping Python duplicates of the
entrypoint's own steps would create a second, drifting definition of the same
sandbox.

Two jobs are genuinely ours, because the entrypoint only performs them on its
database path — which a pre-populated ``PROBLEM_DIR`` bypasses:

  * ``stitch_submission_rule`` / ``banned_libraries_section``: assemble
    ``submission_rule.md`` from the checkout's own prompt fragments.
  * ``write_problem_extras``: write ``user_prompt.txt`` (the ONLY channel
    through which the row's task text reaches the model), ``language.txt``,
    and ``submission_rule.md`` into the problem dir.

``stage_surface_snapshot`` additionally serves ``sandbox_profile: minimal``,
which has no entrypoint and needs the ``submit``/``cudagym`` skills staged by
hand.
"""

import logging
import shutil
import threading
from pathlib import Path
from typing import Optional


LOG = logging.getLogger(__name__)

# Guards first-use staging against concurrent rollouts in one server process.
_STAGE_LOCK = threading.Lock()


# Platform-coupled skills whose live scripts need the SolSwarm platform
# (SOLSWARM_DB_URL / trace DB / bug tracker); their scripts are replaced with
# loud not-available stubs. `summarize` is summarizer-role-only per its
# SKILL.md but its scripts also hit the DB, so it gets the same treatment.
STUB_SKILLS = ("query", "insights", "report-bug", "summarize")
# Always installed regardless of the ``skills`` selection, mirroring
# entrypoint.sh's mandatory harness skills.
MANDATORY_SKILLS = ("query", "submit", "cudagym", "insights")

_STUB_BODY_PY = (
    "#!/usr/bin/env python3\n"
    'import sys\n\nprint("This SolSwarm platform service is not available in this RL '
    'environment; continue optimizing with cudagym evaluate and /submit.")\n'
    "sys.exit(0)\n"
)
_STUB_BODY_SH = (
    "#!/usr/bin/env bash\n"
    'echo "This SolSwarm platform service is not available in this RL environment; '
    'continue optimizing with cudagym evaluate and /submit."\n'
)

# Where the checkout keeps the submission-rule prompt fragments the stitcher
# assembles. In container mode these are the only checkout files that reach the
# sandbox (the agent image supplies everything else), so
# ``solswarm_container.checkout_id`` fingerprints exactly this directory.
SUBMISSION_FRAGMENTS_SUBDIR = Path("docker") / "agent" / "prompts" / "submission"

# The stitcher's fragment universe; anything else appearing in the checkout's
# prompts/submission/ is upstream drift we should hear about.
_KNOWN_FRAGMENTS = {
    "00_header.md",
    "10_forbid_framework_kernels.md",
    "20_ban_cuda_graphs.md",
    "90_cheating_patterns.md",
    "99_footer.md",
}


def banned_libraries_section(banned_libraries: list[str]) -> str:
    """Port of the jq banned-libraries section builder (entrypoint.sh)."""
    if not banned_libraries:
        return ""
    return (
        "## Campaign-specific banned libraries\n\n"
        "The following libraries are banned for this campaign and override generic language permissions:\n\n"
        + "\n".join(f"- `{lib}`" for lib in banned_libraries)
        + "\n\nDo not import, include, link against, or call these libraries from any submitted source. "
        "If another prompt section says one of these is normally allowed for the selected language, "
        "this campaign-specific list wins. "
        "The compliance judge enforces these bans.\n"
    )


def stitch_submission_rule(
    root: Path,
    *,
    ban_framework_kernels: bool,
    ban_cuda_graphs: bool,
    banned_libraries: Optional[list[str]] = None,
) -> str:
    """Port of entrypoint.sh's ``submission_rule.md`` fragment stitcher.

    Warns about fragments in the checkout that the stitcher doesn't know —
    the tell that upstream added a new (possibly conditional) rule section.
    """
    frag_dir = root / SUBMISSION_FRAGMENTS_SUBDIR
    extras = sorted(p.name for p in frag_dir.glob("*.md") if p.name not in _KNOWN_FRAGMENTS)
    if extras:
        LOG.warning(
            "solswarm_surface: unknown submission fragments not stitched (upstream drift?): %s",
            extras,
        )
    fragments = ["00_header.md"]
    if ban_framework_kernels:
        fragments.append("10_forbid_framework_kernels.md")
    if ban_cuda_graphs:
        fragments.append("20_ban_cuda_graphs.md")
    fragments += ["90_cheating_patterns.md", "99_footer.md"]

    banned_section = banned_libraries_section(banned_libraries or [])
    parts: list[str] = []
    for name in fragments:
        path = frag_dir / name
        if not path.is_file():
            LOG.warning("submission rule fragment missing: %s", path)
            continue
        parts.append(path.read_text())
        if name == "00_header.md" and banned_section:
            # entrypoint appends via printf '\n%s\n'.
            parts.append(f"\n{banned_section}\n")
    return "".join(parts)


def _copy_skill_tree(src: Path, dst: Path) -> None:
    """Recursive copy that follows valid symlinks and skips dangling ones."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_symlink() and not entry.resolve(strict=False).exists():
            LOG.warning("solswarm_surface: skipping dangling symlink %s", entry)
            continue
        target = dst / entry.name
        if entry.is_dir():
            _copy_skill_tree(entry, target)
        else:
            shutil.copy2(entry, target)


def _stub_skill_scripts(skill_dir: Path) -> None:
    """Replace every script under the skill's ``scripts/`` dir with a not-available stub."""
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return
    for script in scripts_dir.rglob("*"):
        if not script.is_file():
            continue
        if script.suffix == ".py":
            script.write_text(_STUB_BODY_PY)
        elif script.suffix in (".sh", ""):
            script.write_text(_STUB_BODY_SH)


def stage_surface_snapshot(root: Path, dest: Path, skills: Optional[list[str]] = None) -> Path:
    """Copy the selected skills into ``dest/config/opencode/skills`` (once per process).

    ``skills=None`` mirrors the SolSwarm optimizer default: every baked skill.
    Platform-coupled skills (STUB_SKILLS) keep their SKILL.md verbatim but get
    stubbed scripts. Returns the skills directory. The layout is the one
    OpenCode discovers natively (skills under the opencode config dir), so
    pointing ``XDG_CONFIG_HOME`` at ``dest/config`` also enables OpenCode's
    own skill discovery.
    """
    skills_src = root / "skills" / "agent"
    skills_dest = dest / "config" / "opencode" / "skills"
    marker = skills_dest / ".staged"
    # The marker records a completed stage: reuse it without taking the lock.
    if marker.is_file():
        return skills_dest
    with _STAGE_LOCK:
        # Another rollout may have finished staging while we waited for the lock.
        if marker.is_file():
            return skills_dest
        return _stage_skills_locked(skills_src, skills_dest, marker, skills)


def _stage_skills_locked(skills_src: Path, skills_dest: Path, marker: Path, skills: Optional[list[str]]) -> Path:
    """Copy the selected skills while holding the stage lock; see ``stage_surface_snapshot``."""
    # Resolve the selection: every baked skill by default, else the requested subset.
    available = sorted(p.name for p in skills_src.iterdir() if p.is_dir())
    selected = list(available) if skills is None else [s for s in skills if s in set(available)]
    if skills is not None:
        missing = sorted(set(skills) - set(available))
        if missing:
            LOG.warning("solswarm_surface: unknown skills ignored: %s", missing)
        # The mandatory harness skills are installed even when not requested.
        for name in MANDATORY_SKILLS:
            if name in available and name not in selected:
                selected.append(name)

    # Replace any partial previous stage wholesale.
    if skills_dest.exists():
        shutil.rmtree(skills_dest)
    skills_dest.mkdir(parents=True)
    for name in selected:
        # Follow symlinks so links into the wider repo materialize as content
        # (e.g. kernel-factory-schemas/references -> the monorepo's cudagym/
        # tree), but skip ones that dangle in a partial checkout. Hand-rolled
        # because shutil.copytree's ignore_dangling_symlinks resolves relative
        # link targets against the CWD, misjudging valid in-repo links.
        _copy_skill_tree(skills_src / name, skills_dest / name)
        # Platform-coupled skills keep their SKILL.md but get not-available stub scripts.
        if name in STUB_SKILLS:
            _stub_skill_scripts(skills_dest / name)
    # The marker is written last, so an interrupted copy re-stages on the next call.
    marker.write_text("\n".join(selected) + "\n")
    LOG.info("solswarm_surface: staged %d skills at %s", len(selected), skills_dest)
    return skills_dest


def write_problem_extras(
    problem_dir: Path,
    *,
    user_prompt: str,
    language: str,
    submission_rule: str,
) -> None:
    """SolSwarm problem-dir parity files beyond definition/workloads."""
    (problem_dir / "user_prompt.txt").write_text(user_prompt or "")
    (problem_dir / "language.txt").write_text((language or "") + "\n")
    (problem_dir / "submission_rule.md").write_text(submission_rule)
