# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""grade_row record shape for templates without a matcher (tool_choice) and for outputs without assistant turns.

    python3 responses_api_agents/swe_if_agents/tests/test_grader_records.py

Standard library only (the grader and the verifier package have no gym dependencies).
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from responses_api_agents.swe_if_agents.if_constraints import grade_row  # noqa: E402
from responses_api_agents.swe_if_agents.if_constraints.grader import GRADING_ERROR_ID  # noqa: E402

NEVER_BASH = {"id": "t#tc", "verifier_parameter": {"template": "tool_choice", "trigger": {"mode": "never_call", "tool": "BASH_TOOL_NAME"}}}
FIRST_TURN_TAG = {
    "id": "t#c1",
    "verifier_parameter": {"template": "turn_output", "trigger": {"position": "first_turn"}, "obligation": {"match": "prefix", "value": "[PLAN]"}},
}


def _md(*constraints):
    return {
        "tool_name_overrides": json.dumps({"BASH_TOOL_NAME": "shell"}),
        "sdg_item": json.dumps({"type": "fresh", "constraints": list(constraints)}),
    }


def _msg(text, mid):
    return {"type": "message", "id": mid, "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _call(name, cid):
    return {"type": "function_call", "name": name, "arguments": "{}", "call_id": cid}


TWO_TURNS = [
    _msg("[PLAN] run it", "m1"),
    _call("shell", "c1"),
    {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    _msg("done", "m2"),
]


class TestToolChoice(unittest.TestCase):
    def test_tool_choice_keeps_its_graded_step_and_has_no_no_answer_kind(self):
        """tool_choice has no obligation.match, so there is no matcher to ask for a no-answer policy: the record carries
        no_answer None and the verifier's single trajectory-scoped step (turn -1, no items) — not an error record."""
        (rec,) = grade_row(_md(NEVER_BASH), None, TWO_TURNS)
        self.assertNotIn("error", rec, rec)
        self.assertIsNone(rec["no_answer"])
        self.assertIsNone(rec["match"])
        self.assertEqual((rec["n_steps"], rec["n_pass"], rec["all_pass"], rec["step_avg"]), (1, 0, False, 0.0))
        self.assertEqual([(s["turn"], s["reward"], s["items"]) for s in rec["steps"]], [(-1, 0, [])])
        self.assertIn("never_call", rec["steps"][0]["detail"])
        # ...and the PASS verdict survives too
        (rec_ok,) = grade_row(_md(NEVER_BASH), None, [_msg("no tools needed", "m1")])
        self.assertEqual((rec_ok["n_steps"], rec_ok["n_pass"], rec_ok["all_pass"]), (1, 1, True))
        self.assertEqual(rec_ok["steps"][0]["items"], [])

    def test_turn_output_still_reports_the_matcher_policy_and_its_items(self):
        (rec,) = grade_row(_md(FIRST_TURN_TAG), None, TWO_TURNS)
        self.assertNotIn("error", rec)
        self.assertIsInstance(rec["no_answer"], str)
        self.assertEqual([(s["turn"], s["reward"], s["items"]) for s in rec["steps"]], [(0, 1, ["m1", "c1"])])

    def test_empty_output_does_not_lose_the_row(self):
        """An output without assistant turns (nothing generated / reasoning only) used to raise IndexError on the
        tool_choice step's items join and collapse EVERY constraint into one <grading_error> record."""
        for output in ([], [{"type": "reasoning", "id": "r1", "summary": []}]):
            records = grade_row(_md(FIRST_TURN_TAG, NEVER_BASH), None, output)
            self.assertEqual([r["id"] for r in records], ["t#c1", "t#tc"], records)
            self.assertNotEqual(records[0]["id"], GRADING_ERROR_ID)
            tag, tc = records
            self.assertEqual(tag["n_steps"], 0)                           # the trigger never fired: not applicable
            self.assertNotIn("error", tc)
            self.assertEqual([(s["turn"], s["reward"], s["items"]) for s in tc["steps"]], [(-1, 1, [])])  # never called: ok

    def test_unknown_matcher_is_still_a_per_constraint_error(self):
        bad = {"id": "t#bad", "verifier_parameter": {"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": "no_such", "value": 1}}}
        bad_rec, tc = grade_row(_md(bad, NEVER_BASH), None, TWO_TURNS)
        self.assertIn("unknown matcher", bad_rec["error"])
        self.assertEqual(bad_rec["n_steps"], 0)
        self.assertNotIn("error", tc)


if __name__ == "__main__":
    unittest.main()
