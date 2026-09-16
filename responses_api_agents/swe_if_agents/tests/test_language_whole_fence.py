# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`language` on a message that is ENTIRELY one fenced block (2026-09-16, Opus 5 200v4 audit).

A language constraint met a fenced / JSON constraint on the same item; the model wrote its compliant Chinese or Russian
message inside the demanded ```status / ```plaintext fence, the prose view stripped the whole block and the check failed
with "0 of 0 alphabetic chars". When the fence IS the message, its body is the prose. A fence that is only PART of the
message is still code and still excluded."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "if_constraints"))
from verifier.core import _whole_fence_body  # noqa: E402
from verifier.matchers import _m_language  # noqa: E402


def test_whole_fence_body():
    assert _whole_fence_body("```status\n好的，我继续。\n```") == "好的，我继续。"
    assert _whole_fence_body("\n```plaintext\nГотово.\nПроблема исправлена.\n```\n") == "Готово.\nПроблема исправлена."
    assert _whole_fence_body("```python\nprint(1)\n```\n好的") is None  # prose after the fence: not a whole-fence message
    assert _whole_fence_body("好的\n```python\nprint(1)\n```") is None
    assert _whole_fence_body("```status\n```") is None  # empty body
    assert _whole_fence_body("plain text") is None


def test_language_all_fenced_message_grades_its_body():
    ok, detail = _m_language("han", "```status\n好的，我继续。文档部分已按上游写法补齐，接下来需要验证代码改动。\n```")
    assert ok, detail
    ok, detail = _m_language("cyrillic", "```plaintext\nГотово. Проблема исправлена. В методе `DataArrayRolling.__iter__` игнорировался аргумент `center`.\n```")
    assert ok, detail
    ok, detail = _m_language("han", "```status\nAll tests pass, nothing else to do here.\n```")
    assert not ok and "0 of 31" in detail, detail  # English inside the fence is still English


def test_language_partial_fence_still_excluded():
    ok, detail = _m_language("han", "```python\nprint('hello world example')\n```\n好的，我继续。")
    assert ok and "5/5" in detail, detail
    ok, detail = _m_language("han", "I'll start by exploring the relevant code.")
    assert not ok, detail


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
