# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compute_if_reward: the four reward modes over if_constraints records, and the turn_verdicts the trainer reads.

    python3 responses_api_agents/swe_if_agents/tests/test_reward_modes.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from responses_api_agents.swe_if_agents.if_constraints.reward import (  # noqa: E402
    GRADING_ERROR_ID,
    REWARD_MODES,
    compute_if_reward,
    resolve_reward_settings,
)


def rec(cid, steps, error=None, continuation_only=False, turns=None):
    n = len(steps)
    p = sum(1 for s in steps if s >= 1)
    turns = list(range(n)) if turns is None else turns
    r = {"id": cid, "n_steps": n, "n_pass": p, "step_avg": (p / n) if n else None, "all_pass": bool(n and p == n),
         "graded_turns": max([t for t in turns] + [-1]) + 1, "continuation_only": continuation_only,
         "steps": [{"turn": t, "reward": s, "detail": f"t{t}"} for t, s in zip(turns, steps)]}
    if error:
        r["error"] = error
    return r


ALL_PASS = [rec("a", [1, 1, 1]), rec("b", [1])]
ONE_FAIL = [rec("a", [1, 0, 1]), rec("b", [1])]          # a: step_avg 2/3 ; mean fraction (2/3 + 1)/2 = 5/6
NOTHING_APPLICABLE = [rec("a", []), rec("b", [])]         # declared, triggers never fired: vacuous
MATCHER_ERROR = [rec("a", [1, 0]), rec("z", [], error="ValueError: unknown matcher 'x'")]  # one constraint could not be graded
ROW_ERROR = [{"id": GRADING_ERROR_ID, "error": "TypeError: boom"}]                          # the grader's catch-all record
TRAJECTORY_SCOPED = [rec("tc", [0], turns=[-1]), rec("a", [1, 1])]                          # tool_choice step: turn -1


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

    def test_gdpo_two_channels(self):
        out = compute_if_reward(ONE_FAIL, 1.0, "gdpo")
        self.assertAlmostEqual(out["reward"], 1.0 + out["constraint_reward"])
        self.assertEqual(set(out["reward_components"]), {"task", "constraint"})  # no per-constraint channels
        self.assertAlmostEqual(sum(out["reward_components"].values()), out["reward"])
        self.assertTrue(out["constraint_step_avgs"])
        out0 = compute_if_reward(NOTHING_APPLICABLE, 1.0, "gdpo")
        self.assertEqual(out0["reward"], 1.0)
        self.assertEqual(set(out0["reward_components"]), {"task"})  # constraint channel ABSENT, not 0
        self.assertEqual(compute_if_reward(ALL_PASS, 0.0, "gdpo")["reward"], 1.0)  # task fail still earns the constraint channel

    def test_bad_settings_raise(self):
        with self.assertRaises(ValueError):
            compute_if_reward(ALL_PASS, 1.0, "bogus")
        with self.assertRaises(ValueError):
            compute_if_reward(ALL_PASS, 1.0, "tiered", partial=1.5)


class TestPassThrough(unittest.TestCase):
    """The two cases that must never zero or shape a solved task: nothing declared, and a grade that cannot be trusted."""

    def test_no_constraints_declared_passes_the_task_through_in_every_mode(self):
        for mode in REWARD_MODES:
            for records in (None, []):
                out = compute_if_reward(records, 1.0, mode)
                self.assertEqual(out["reward"], 1.0, (mode, records))
                self.assertEqual((out["n_constraints"], out["n_applicable"], out["n_grading_errors"]), (0, 0, 0))
                self.assertFalse(out["constraint_graded"])
                self.assertFalse(out["mask_sample"], (mode, records))          # nothing went wrong: an ordinary task row
                self.assertNotIn("constraint", out["reward_components"])
        # ...whereas a DECLARED constraint whose trigger never fired keeps the vacuous rule
        self.assertEqual(compute_if_reward(NOTHING_APPLICABLE, 1.0, "strict")["reward"], 0.0)
        self.assertEqual(compute_if_reward(NOTHING_APPLICABLE, 1.0, "tiered")["reward"], 0.5)
        self.assertFalse(compute_if_reward(NOTHING_APPLICABLE, 1.0, "strict")["mask_sample"])

    def test_row_level_grading_error_passes_through_and_masks_training_modes(self):
        for mode in REWARD_MODES:
            out = compute_if_reward(ROW_ERROR, 1.0, mode)
            self.assertEqual(out["reward"], 1.0, mode)                        # was 0.0 (strict) / 0.5 (tiered)
            self.assertEqual(out["n_grading_errors"], 1)
            self.assertEqual(out["n_constraints"], 0, "the pseudo-record is not a declared constraint")
            self.assertEqual(out["n_applicable"], 0)
            self.assertEqual(out["mask_sample"], mode != "outcome", mode)
            self.assertNotIn("constraint", out["reward_components"])
            self.assertEqual(out["turn_verdicts"], [])
        # the task failure still gives 0: pass-through is not a floor
        self.assertEqual(compute_if_reward(ROW_ERROR, 0.0, "strict")["reward"], 0.0)

    def test_per_constraint_error_is_not_applicable_and_the_rest_still_folds(self):
        """A retired/unknown matcher on ONE constraint is the recipe's "not applicable, row not lost": the other
        constraints decide the reward and the sample is kept (no mask). MATCHER_ERROR: a fails (1,0), z errored."""
        for mode, expected in (("outcome", 1.0), ("shaped", 1.5), ("strict", 0.0), ("tiered", 0.5)):
            out = compute_if_reward(MATCHER_ERROR, 1.0, mode)
            self.assertAlmostEqual(out["reward"], expected, msg=mode)
            self.assertFalse(out["mask_sample"], mode)
            self.assertEqual((out["n_constraints"], out["n_applicable"], out["n_grading_errors"]), (2, 1, 1))

    def test_turn_verdicts_are_one_based_and_locate_the_violation(self):
        out = compute_if_reward(ONE_FAIL, 1.0, "strict")
        turns = [(v["turn"], v["passed"], v["constraint"]) for v in out["turn_verdicts"]]
        self.assertEqual(turns, [(1, True, "a"), (2, False, "a"), (3, True, "a"), (1, True, "b")])
        self.assertEqual(out["first_violation_turn"], 2)
        self.assertEqual(out["num_graded_turns"], 3)
        self.assertEqual(out["num_violating_turns"], 1)
        self.assertFalse(out["constraint_all_pass"])
        self.assertEqual(out["reward_components"], {"task": 1.0, "constraint": 0.0, "constraint_a": 2 / 3, "constraint_b": 1.0})

    def test_trajectory_scoped_steps_gate_the_reward_but_have_no_turn(self):
        """A tool_choice step (grader turn -1) has no assistant turn: it must not become turn_verdicts turn 0 (NeMo-RL's
        assistant turns are 1-based, so 0 would never match), nor count as a graded/violating turn."""
        out = compute_if_reward(TRAJECTORY_SCOPED, 1.0, "strict")
        self.assertEqual(out["reward"], 0.0)                                  # the failed tool_choice step still gates
        self.assertFalse(out["constraint_all_pass"])
        self.assertEqual([(v["turn"], v["constraint"]) for v in out["turn_verdicts"]], [(1, "a"), (2, "a")])
        self.assertIsNone(out["first_violation_turn"])
        self.assertEqual((out["num_graded_turns"], out["num_violating_turns"]), (2, 0))
        self.assertEqual(out["reward_components"]["constraint_tc"], 0.0)
        self.assertEqual(out["n_applicable"], 2)

    def test_inapplicable_records_are_counted_not_graded(self):
        out = compute_if_reward(NOTHING_APPLICABLE, 1.0, "outcome")
        self.assertEqual((out["n_constraints"], out["n_applicable"], out["n_grading_errors"]), (2, 0, 0))
        self.assertFalse(out["constraint_graded"])
        self.assertIsNone(out["constraint_reward"])
        self.assertEqual(out["turn_verdicts"], [])

    def test_continuation_only_propagates(self):
        out = compute_if_reward([rec("a", [1], continuation_only=True)], 1.0, "outcome")
        self.assertTrue(out["continuation_only"])


class TestRowOverrides(unittest.TestCase):
    def test_valid_overrides_are_applied(self):
        self.assertEqual(
            resolve_reward_settings({"reward_mode": "tiered", "constraint_alpha": "0.5", "success_partial_reward": "0.7"}, "strict", 1.0, 0.5),
            ("tiered", 0.5, 0.7, None),
        )
        self.assertEqual(resolve_reward_settings({}, "strict", 1.0, 0.5), ("strict", 1.0, 0.5, None))
        self.assertEqual(resolve_reward_settings(None, "outcome", 1.0, 0.5), ("outcome", 1.0, 0.5, None))
        self.assertEqual(resolve_reward_settings({"reward_mode": " Strict "}, "outcome", 1.0, 0.5)[0], "strict")  # case-insensitive

    def test_malformed_overrides_fall_back_and_are_reported_not_raised(self):
        m, a, p, err = resolve_reward_settings(
            {"reward_mode": "stric", "constraint_alpha": "", "success_partial_reward": "None"}, "strict", 1.0, 0.5
        )
        self.assertEqual((m, a, p), ("strict", 1.0, 0.5))
        self.assertIn("reward_mode='stric'", err)
        self.assertIn("constraint_alpha=''", err)
        self.assertIn("success_partial_reward='None'", err)
        # a partial outside [0, 1] would make compute_if_reward raise after the episode: rejected here instead
        m, a, p, err = resolve_reward_settings({"success_partial_reward": "1.5"}, "tiered", 1.0, 0.5)
        self.assertEqual(p, 0.5)
        self.assertIn("not in [0, 1]", err)
        # an empty reward_mode string means "no override"
        self.assertEqual(resolve_reward_settings({"reward_mode": ""}, "shaped", 1.0, 0.5), ("shaped", 1.0, 0.5, None))

    def test_row_cannot_switch_on_a_training_mode_without_grading(self):
        m, _, _, err = resolve_reward_settings({"reward_mode": "strict"}, "outcome", 1.0, 0.5, if_grading=False)
        self.assertEqual(m, "outcome")
        self.assertIn("if_grading", err)
        m, _, _, err = resolve_reward_settings({"reward_mode": "outcome"}, "outcome", 1.0, 0.5, if_grading=False)
        self.assertEqual((m, err), ("outcome", None))


if __name__ == "__main__":
    unittest.main()
