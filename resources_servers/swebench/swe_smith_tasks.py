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
"""SWE-smith task facts that need no ``swebench`` import.

The benchmark prepare scripts (benchmarks/swebench/swe_smith/prepare.py and
friends) run in the Gym root environment, which does not install the swebench
resources server's requirements, so everything they need lives here: the
repository profiles (``swe_smith_profiles.py``, generated from SWE-smith's
``swesmith/profiles/python.py``), the lookup of a dataset ``repo`` value, the
registered version (the mirror commit's eight hex digits), the test-file rules
of ``RepoProfile.get_test_files`` and its overrides (which name the F2P files
hidden from the agent) and ``reverse_unified_diff``, which turns the dataset's
bug patch into the fix. ``swe_smith_harness.py`` (server side, imports
swebench) builds on these and re-exports them.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from resources_servers.swebench.swe_smith_profiles import PROFILES, SWESMITH_COMMIT


__all__ = [
    "PROFILES",
    "SWESMITH_COMMIT",
    "profile_for",
    "reverse_unified_diff",
    "swe_smith_hidden_files",
    "swe_smith_test_files",
    "swe_smith_version",
    "test_names",
]

_PROFILES_LOWER = {repo.lower(): repo for repo in PROFILES}


def profile_for(repo: str) -> Optional[Dict[str, Any]]:
    """The SWE-smith profile of a dataset ``repo`` value (``swesmith/<owner>__<repo>.<commit8>``), else None."""
    key = _PROFILES_LOWER.get(repo.lower())
    return None if key is None else PROFILES[key]


def swe_smith_version(repo: str) -> str:
    """The spec version registered for a SWE-smith repository: the mirror commit's eight hex digits."""
    return repo.rsplit(".", 1)[-1]


def test_names(instance: Dict[str, Any], key: str) -> List[str]:
    value = instance.get(key) or []
    if isinstance(value, str):
        value = json.loads(value)
    return [str(name) for name in value]


# --- test files (RepoProfile.get_test_files and its overrides) ---------------------------------------------


def _pytest_files(names: Sequence[str]) -> List[str]:
    return sorted({name.split("::", 1)[0] for name in names})


def _mido_files(names: Sequence[str]) -> List[str]:
    """mido's test ids carry a ``../dev/`` prefix the profile strips."""
    prefix = "../dev/"
    return sorted({name[len(prefix) :] if name.startswith(prefix) else name for name in _pytest_files(names)})


def _tornado_files(names: Sequence[str]) -> List[str]:
    """unittest ids ``test_x (tornado.test.foo_test.Class)`` -> ``tornado/test/foo_test.py``."""
    files = set()
    for name in names:
        match = re.search(r"\s\((.*)\)", name)
        if match:
            files.add("/".join(match.group(1).split(".")[:-1]) + ".py")
    return sorted(files)


_TEST_FILE_RULES: Dict[str, Callable[[Sequence[str]], List[str]]] = {
    "pytest": _pytest_files,
    "mido": _mido_files,
    "single_test_py": lambda names: ["test.py"],
    "tornado": _tornado_files,
}


def swe_smith_test_files(profile: Dict[str, Any], instance: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(F2P test files, P2P test files) as the repository profile's ``get_test_files`` derives them."""
    rule = _TEST_FILE_RULES[profile["test_files"]]
    return rule(test_names(instance, "FAIL_TO_PASS")), rule(test_names(instance, "PASS_TO_PASS"))


def swe_smith_hidden_files(profile: Dict[str, Any], instance: Dict[str, Any]) -> List[str]:
    """The files removed from the agent's checkout: the F2P test files (gather.py's "Remove F2P Tests" commit)."""
    return swe_smith_test_files(profile, instance)[0]


# --- diff reversal -----------------------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@(.*)$")
_INDEX_RE = re.compile(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)(.*)$")
_UNSUPPORTED_HEADERS = (
    "similarity index",
    "dissimilarity index",
    "rename ",
    "copy ",
    "old mode",
    "new mode",
    "Binary files",
)


def _swap_file_header(line: str, new_prefix: str) -> str:
    """``--- a/x`` <-> ``+++ b/x`` (``/dev/null`` stays)."""
    old_prefix = line[:4]
    path = line[4:]
    if path != "/dev/null":
        want = {"--- ": "a/", "+++ ": "b/"}[old_prefix]
        if not path.startswith(want):
            raise ValueError(f"unexpected file header {line!r}")
        path = {"--- ": "b/", "+++ ": "a/"}[old_prefix] + path[len(want) :]
    return new_prefix + path


def reverse_unified_diff(patch: str) -> str:
    """The inverse of a git unified diff, as text: applying it undoes the original.

    Handles what SWE-smith's bug patches contain: ``diff --git`` sections with
    ``index`` lines, new and deleted files, hunks of context, additions,
    deletions and ``\\ No newline at end of file`` markers. Renames, copies, mode
    changes and binary patches raise ``ValueError`` instead of being guessed.
    Hunk bodies are recognised by the line counts of their headers, so a
    removed line that starts with ``--- `` is not mistaken for a file header.
    """
    out: List[str] = []
    lines = patch.split("\n")
    old_left = new_left = 0
    pending_minus: Optional[str] = None
    for line in lines:
        if line.startswith("\\"):
            # "\ No newline at end of file" annotates the preceding line, also when it closes a hunk.
            out.append(line)
            continue
        if old_left > 0 or new_left > 0:
            if line.startswith("-"):
                out.append("+" + line[1:])
                old_left -= 1
            elif line.startswith("+"):
                out.append("-" + line[1:])
                new_left -= 1
            else:
                if line not in ("", " ") and not line.startswith(" "):
                    raise ValueError(f"unexpected hunk line {line!r}")
                out.append(line)
                old_left -= 1
                new_left -= 1
            continue
        if pending_minus is not None:
            if not line.startswith("+++ "):
                raise ValueError(f"file header {pending_minus!r} not followed by +++ but by {line!r}")
            out.append(_swap_file_header(line, "--- "))
            out.append(_swap_file_header(pending_minus, "+++ "))
            pending_minus = None
            continue
        if line.startswith("diff --git a/"):
            match = re.match(r"^diff --git a/(.*) b/(.*)$", line)
            if match is None:
                raise ValueError(f"unexpected diff header {line!r}")
            out.append(f"diff --git a/{match.group(2)} b/{match.group(1)}")
        elif line.startswith("--- "):
            pending_minus = line
        elif line.startswith("@@ "):
            match = _HUNK_RE.match(line)
            if match is None:
                raise ValueError(f"unexpected hunk header {line!r}")
            old_start, old_count, new_start, new_count, tail = match.groups()
            out.append(f"@@ -{new_start}{new_count or ''} +{old_start}{old_count or ''} @@{tail}")
            old_left = int((old_count or ",1")[1:])
            new_left = int((new_count or ",1")[1:])
        elif line.startswith("index "):
            match = _INDEX_RE.match(line)
            if match is None:
                raise ValueError(f"unexpected index line {line!r}")
            out.append(f"index {match.group(2)}..{match.group(1)}{match.group(3)}")
        elif line.startswith("new file mode "):
            out.append("deleted file mode " + line[len("new file mode ") :])
        elif line.startswith("deleted file mode "):
            out.append("new file mode " + line[len("deleted file mode ") :])
        elif line.startswith(_UNSUPPORTED_HEADERS) or line.startswith("GIT binary patch"):
            raise ValueError(f"cannot reverse {line.split(' ')[0]!r} sections textually")
        elif line == "":
            out.append(line)
        else:
            raise ValueError(f"unexpected line outside a hunk {line!r}")
    if pending_minus is not None or old_left or new_left:
        raise ValueError("truncated diff")
    return "\n".join(out)
