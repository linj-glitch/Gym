# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_count_sentences: the two misgrades found by the 2026-09-09 blind-judge audit, plus the cases that must not move.

    python3 responses_api_agents/swe_if_agents/tests/test_count_sentences.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from responses_api_agents.swe_if_agents.if_constraints.verifier.core import _count_sentences, _length_count  # noqa: E402


class TestCountSentences(unittest.TestCase):
    def test_closing_fence_after_a_sentence_is_not_a_sentence(self):
        # deepseek#85 (matplotlib-23299#c3): a sibling `fenced yaml` rule forces the wrapper; max 1 sentence must pass
        self.assertEqual(_count_sentences("```yaml\ninfo: The only change is in lib/matplotlib/__init__.py as intended.\n```"), 1)
        self.assertEqual(_count_sentences("Done. :)"), 1)
        self.assertEqual(_count_sentences("Done here.\n```"), 1)

    def test_terminator_followed_by_closing_quote_or_bracket_ends_a_sentence(self):
        # deepseek#195 (xarray-6721#c4): 'encoding."' ends a sentence; the old regex needed whitespace right after '.'
        text = 'I kept the "encoding." Then I ran the tests. (They passed.) Finally I reviewed the diff.'
        self.assertEqual(_count_sentences(text), 4)
        self.assertEqual(_count_sentences('She said "stop." He left.'), 2)

    def test_unchanged_cases(self):
        self.assertEqual(_count_sentences(""), 0)
        self.assertEqual(_count_sentences("   \n"), 0)
        self.assertEqual(_count_sentences("One. Two! Three?"), 3)
        self.assertEqual(_count_sentences("No terminator at all"), 1)
        self.assertEqual(_count_sentences("Wait... what? Really!!"), 3)
        self.assertEqual(_count_sentences("Version 3.5 shipped today. Tests pass."), 2)  # decimal point inside a token is not a boundary
        self.assertEqual(_length_count("A. B. C.", "sentences"), 3)


if __name__ == "__main__":
    unittest.main()
