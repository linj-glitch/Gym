# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compute_if_reward: the four reward modes over if_constraints records, and the turn_verdicts the trainer reads.

    python3 responses_api_agents/swe_if_agents/tests/test_reward_modes.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from responses_api_agents.swe_if_agents.if_constraints.reward import compute_if_reward, resolve_reward_settings  # noqa: E402


def rec(cid, steps, error=None, continuation_only=False):
    n = len(steps)
    p = sum(1 for s in steps if s >= 1)
    r = {"id": cid, "n_steps": n, "n_pass": p, "step_avg": (p / n) if n else None, "all_pass": bool(n and p == n),
         "graded_turns": max([i for i, _ in enumerate(steps)] + [-1]) + 1, "continuation_only": continuation_only,
         "steps": [{"turn": i, "reward": s, "detail": f"t{i}"} for i, s in enumerate(steps)]}
    if error:
        r["error"] = error
    return r


ALL_PASS = [rec("a", [1, 1, 1]), rec("b", [1])]
ONE_FAIL = [rec("a", [1, 0, 1]), rec("b", [1])]          # a: step_avg 2/3 ; mean fraction (2/3 + 1)/2 = 5/6
NOTHING_APPLICABLE = [rec("a", []), rec("z", [], error="ValueError: unknown matcher 'x'")]


class TestModes(unittest.TestCase):
    def test_outcome_leaves_reward_as_task(self):
        out = compute_if_reward(ONE_FAIL, 1.0, "outcome")
        self.assertEqual(out["reward"], 1.0)
        self.assertAlmostEqual(out["constraint_reward"], 5 / 6)
        self.assertNotIn("constraint", out["reward_components"])

    def test_shaped(self):
        self.assertAlmostEqual(compute_if_reward(ALL_PASS, 1.0, "shaped")["reward"], 2.0)
        self.assertAlmostEqual(compute_if_reward(ONE_FAIL, 1.0, "shaped", alpha=0.6)["reward"], 1.0 + 0.6 * 5 / 6)
        self.assertEqual(compute_if_reward(ALL_PASS, 0.0, "shaped")["reward"], 0.0)          # task failure -> 0
        self.assertEqual(compute_if_reward(NOTHING_APPLICABLE, 1.0, "shaped")["reward"], 1.0)  # not measured -> task

    def test_strict(self):
        self.assertEqual(compute_if_reward(ALL_PASS, 1.0, "strict")["reward"], 1.0)
        self.assertEqual(compute_if_reward(ONE_FAIL, 1.0, "strict")["reward"], 0.0)
        self.assertEqual(compute_if_reward(NOTHING_APPLICABLE, 1.0, "strict")["reward"], 0.0)  # vacuous -> 0
        self.assertEqual(compute_if_reward(ALL_PASS, 0.0, "strict")["reward"], 0.0)

    def test_tiered(self):
        self.assertEqual(compute_if_reward(ALL_PASS, 1.0, "tiered")["reward"], 1.0)
        self.assertEqual(compute_if_reward(ONE_FAIL, 1.0, "tiered")["reward"], 0.5)
        self.assertEqual(compute_if_reward(ONE_FAIL, 1.0, "tiered", partial=0.3)["reward"], 0.3)
        self.assertEqual(compute_if_reward(NOTHING_APPLICABLE, 1.0, "tiered")["reward"], 0.5)
        self.assertEqual(compute_if_reward(ONE_FAIL, 0.5, "tiered")["reward"], 0.25)

    def test_bad_settings_raise(self):
        with self.assertRaises(ValueError):
            compute_if_reward(ALL_PASS, 1.0, "bogus")
        with self.assertRaises(ValueError):
            compute_if_reward(ALL_PASS, 1.0, "tiered", partial=1.5)


class TestDiagnostics(unittest.TestCase):
    def test_turn_verdicts_are_one_based_and_locate_the_violation(self):
        out = compute_if_reward(ONE_FAIL, 1.0, "strict")
        turns = [(v["turn"], v["passed"], v["constraint"]) for v in out["turn_verdicts"]]
        self.assertEqual(turns, [(1, True, "a"), (2, False, "a"), (3, True, "a"), (1, True, "b")])
        self.assertEqual(out["first_violation_turn"], 2)
        self.assertEqual(out["num_graded_turns"], 3)
        self.assertEqual(out["num_violating_turns"], 1)
        self.assertFalse(out["constraint_all_pass"])
        self.assertEqual(out["reward_components"], {"task": 1.0, "constraint": 0.0, "constraint_a": 2 / 3, "constraint_b": 1.0})

    def test_errors_and_inapplicable_records_are_counted_not_graded(self):
        out = compute_if_reward(NOTHING_APPLICABLE, 1.0, "outcome")
        self.assertEqual((out["n_constraints"], out["n_applicable"], out["n_grading_errors"]), (2, 0, 1))
        self.assertFalse(out["constraint_graded"])
        self.assertIsNone(out["constraint_reward"])
        self.assertEqual(out["turn_verdicts"], [])

    def test_no_records(self):
        out = compute_if_reward(None, 1.0, "strict")
        self.assertEqual((out["reward"], out["n_constraints"], out["constraint_graded"]), (0.0, 0, False))
        self.assertEqual(compute_if_reward(None, 1.0, "outcome")["reward"], 1.0)

    def test_continuation_only_propagates(self):
        out = compute_if_reward([rec("a", [1], continuation_only=True)], 1.0, "outcome")
        self.assertTrue(out["continuation_only"])

    def test_row_overrides(self):
        self.assertEqual(resolve_reward_settings({"reward_mode": "tiered", "constraint_alpha": "0.5", "success_partial_reward": "0.7"}, "strict", 1.0, 0.5),
                         ("tiered", 0.5, 0.7))
        self.assertEqual(resolve_reward_settings({}, "strict", 1.0, 0.5), ("strict", 1.0, 0.5))


if __name__ == "__main__":
    unittest.main()
