# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prefix_turn_count: the replayed prefix is recognised by call_id, not by tool-name coincidence (2026-09-08).

    python3 responses_api_agents/swe_if_agents/tests/test_prefix_turn_count.py
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from responses_api_agents.swe_if_agents.if_constraints.grader import prefix_turn_count  # noqa: E402


def msg(t, i):
    return {"type": "message", "id": f"m{i}", "role": "assistant", "content": [{"type": "output_text", "text": t}]}


def call(i, name="bash", cid=None):
    d = {"type": "function_call", "id": f"fc{i}", "name": name, "arguments": json.dumps({"command": "ls"})}
    if cid is not None:
        d["call_id"] = cid
    return d


def obs(i, cid=None):
    d = {"type": "function_call_output", "output": "ok"}
    if cid is not None:
        d["call_id"] = cid
    return d


SYS = {"type": "message", "role": "system", "content": "neutral"}
USR = {"type": "message", "role": "user", "content": "fix it"}
PREFIX = [msg("one", 1), call(1, "bash", "c1"), obs(1, "c1"), msg("two", 2), call(2, "bash", "c2"), obs(2, "c2")]
INJECT = {"type": "message", "role": "user", "content": "one more rule"}


class TestPrefixTurnCount(unittest.TestCase):
    def test_no_prefix(self):
        self.assertEqual(prefix_turn_count([SYS, USR], [msg("x", 9)]), (0, "no prefix in input", False))

    def test_continuation_with_same_tool_names_is_not_the_prefix(self):
        # live continuation: new call_ids, but the same leading tool names as the prefix (bash, bash)
        cont = [msg("a", 3), call(3, "bash", "c3"), obs(3, "c3"), msg("b", 4), call(4, "bash", "c4"), obs(4, "c4"), msg("final", 5)]
        skip, note, present = prefix_turn_count([SYS, USR] + PREFIX + [INJECT], cont)
        self.assertEqual((skip, present), (0, True))
        self.assertIn("continuation only", note)

    def test_reproduced_prefix_is_skipped(self):
        out = PREFIX + [msg("a", 3), call(3, "grep", "c3"), obs(3, "c3"), msg("final", 5)]
        skip, note, present = prefix_turn_count([SYS, USR] + PREFIX + [INJECT], out)
        self.assertEqual((skip, present), (2, True))
        self.assertIn("call_ids match", note)

    def test_legacy_recordings_without_call_ids_fall_back_to_names(self):
        pre = [msg("one", 1), call(1, "bash"), obs(1), msg("two", 2), call(2, "grep"), obs(2)]
        out = pre + [msg("final", 5)]
        self.assertEqual(prefix_turn_count([SYS, USR] + pre, out)[0], 2)
        cont = [msg("a", 3), call(3, "edit"), obs(3), msg("final", 5)]
        self.assertEqual(prefix_turn_count([SYS, USR] + pre, cont)[0], 0)


if __name__ == "__main__":
    unittest.main()
