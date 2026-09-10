"""The final-turn rule is per agent class, not per prompt persona: codeact ends with a finish call; every other persona
(opencode, claude_code, ...) runs on the OpenCode toolset and ends on a message without a tool call (2026-09-10)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from responses_api_agents.swe_if_agents.if_constraints.grader import to_tv_turns  # noqa: E402


def test_message_ending_personas_share_the_message_final_rule():
    segs = [{"texts": ["looking"], "calls": [("bash", {"command": "ls"})]}, {"texts": ["done"], "calls": []}]
    for persona in ("opencode", "claude_code", "anything_else"):
        turns = to_tv_turns(segs, persona)
        assert turns[-1].is_final, persona
        assert not turns[0].is_final
    assert not to_tv_turns(segs, "codeact")[-1].is_final


def test_codeact_final_is_the_finish_call():
    segs = [{"texts": [""], "calls": [("finish", {"message": "all done"})]}]
    turns = to_tv_turns(segs, "codeact")
    assert turns[-1].is_final and "all done" in turns[-1].visible_text
    assert not to_tv_turns(segs, "opencode")[-1].is_final
