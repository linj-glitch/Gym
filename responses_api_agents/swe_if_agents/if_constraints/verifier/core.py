"""Core of the constraint verifier: data model, resolver, text helpers, no-answer policies.

No registry lives here. `matchers.py`, `triggers.py` and `templates.py` build on this module; `__init__.py` exposes the
public API (`grade`, `grade_ext`, ...). Stdlib only; python 3.9 compatible (no match statements, no `X | None`).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Pseudo tool identifiers (never resolved through the resolver)
# --------------------------------------------------------------------------- #
NO_TOOL = "NO_TOOL"  # trigger: a turn with zero tool calls
ANY_TOOL = "ANY_TOOL"  # trigger: a turn with >= 1 tool call

# --------------------------------------------------------------------------- #
# DEFAULT_RESOLVER: tool IDENTIFIER -> default concrete emitted name.
#
# Extracted verbatim from the PINNED clone
#   /lustre/fsw/portfolios/llmservice/users/charlwang/cluster/gym_workdir/
#     nv-OpenHands/openhands/llm/tool_names.py
# at pinned commit 7466868e2, with diversification OFF (DIVERSIFY_TOOL_NAMES
# unset -> every identifier takes its default, i.e. the first `_pick` argument)
# and camel-casing OFF. Covers the CodeAct section and the two OpenCode
# sections ("OpenCode-inspired" + "OpenCode additional") per the spec's
# "opencode and CodeAct identifiers"; the Codex-inspired and readonly-agent
# identifiers in that file are deliberately not included. A trigger value not
# present here is treated as a literal concrete name (covers never-diversified
# tools like `think`).
# --------------------------------------------------------------------------- #
DEFAULT_RESOLVER = {
    # ---------- CodeAct tools ----------
    "EXECUTE_BASH_TOOL_NAME": "execute_bash",
    "STR_REPLACE_EDITOR_TOOL_NAME": "str_replace_editor",
    "BROWSER_TOOL_NAME": "browser",
    "FINISH_TOOL_NAME": "finish",
    "LLM_BASED_EDIT_TOOL_NAME": "edit_file",
    "TASK_TRACKER_TOOL_NAME": "task_tracker",
    # ---------- OpenCode-inspired tools ----------
    "BASH_TOOL_NAME": "bash",
    "GLOB_TOOL_NAME": "glob",
    "GREP_TOOL_NAME": "grep",
    "LIST_DIR_TOOL_NAME": "list_dir",
    "READ_TOOL_NAME": "read",
    "WRITE_TOOL_NAME": "write",
    "EDIT_TOOL_NAME": "edit",
    # ---------- OpenCode additional tools ----------
    "OPENCODE_APPLY_PATCH_TOOL_NAME": "apply_patch",
    "QUESTION_TOOL_NAME": "question",
    "TODO_READ_TOOL_NAME": "todo_read",
    "TODO_WRITE_TOOL_NAME": "todo_write",
}


# --------------------------------------------------------------------------- #
# Data model (per VERIFIER_SPEC.md)
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    name: str  # concrete emitted tool name (post-diversification)
    args: Dict[str, Any] = field(default_factory=dict)  # parsed args; {} if unparseable


@dataclass
class Turn:
    index: int  # 0-based assistant-turn index
    visible_text: str = ""  # model-authored visible text; reasoning EXCLUDED
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_final: bool = False  # last assistant turn of the episode
    # (role, text) messages between the previous assistant turn and this one
    preceding_messages: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class GradedStep:
    turn: int  # assistant-turn index; -1 for trajectory-scoped
    reward: int  # 1 or 0
    detail: str  # short reason, human-readable


# --------------------------------------------------------------------------- #
# No-answer policies (what a silent in-scope turn means for a matcher)
# --------------------------------------------------------------------------- #
# Owner ruling 2026-09-09 (Charles): "a tool call is a tool call, a message is model output text" — a silent turn is
# not a message and is NEVER a graded step, whatever the matcher. The two kinds below now say only what a MISSING FINAL
# MESSAGE means for the rule (turn_output, position=final, episode ended on a tool call or an error), per the 2026-09-03
# ruling: a rule that needs an answer has failed once; a ban / maximum / sentinel is not gradable.
SILENT_TURN_FAILS = "fail"  # the rule needs an answer: a missing final message is a graded step with reward 0
SILENT_TURN_NOT_GRADABLE = "ungradable"  # absence cannot violate the rule: a missing final message is not a step
NO_ANSWER_POLICIES = (SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE)

SILENT_DETAIL = "silent turn: no visible text (a required shape needs an answer)"  # retired 2026-09-09; kept for readers of old records
NO_FINAL_DETAIL = "no final message: the episode ended with a tool call or an error"


def is_silent_step(step):
    """True when a graded step failed only for lack of an answer: no final message (current rule), or — in records
    graded before 2026-09-09 — a silent turn."""
    return step.detail.endswith(SILENT_DETAIL) or step.detail.endswith(NO_FINAL_DETAIL)


# --------------------------------------------------------------------------- #
# Text helpers shared by matchers
# --------------------------------------------------------------------------- #
# Width-variant forms (FULLWIDTH LATIN ..., HALFWIDTH KATAKANA ...,
# HALFWIDTH HANGUL ...) belong to the same SCRIPT; their unicodedata names
# start with FULLWIDTH/HALFWIDTH, so those prefixes are listed explicitly.
_SCRIPT_NAME_PREFIXES = {
    "latin": ("LATIN", "FULLWIDTH LATIN"),
    "han": ("CJK",),  # CJK UNIFIED IDEOGRAPH, CJK COMPATIBILITY ...
    "kana": ("HIRAGANA", "KATAKANA", "HALFWIDTH KATAKANA"),
    "hangul": ("HANGUL", "HALFWIDTH HANGUL"),
    "cyrillic": ("CYRILLIC",),
}


def _script_matches(ch, prefixes):
    name = unicodedata.name(ch, "")
    for p in prefixes:
        if name.startswith(p):
            return True
    return False


def _count_paired_fences(s, info_pattern):
    """Number of properly PAIRED fences whose info-string matches info_pattern."""
    open_info = None  # info-string of the currently open fence, else None
    matched = 0
    for line in s.splitlines():
        stripped = line.strip()
        if open_info is None:
            if stripped.startswith("```"):
                info = stripped[3:].strip()
                if info:  # opener requires a NON-EMPTY info-string
                    open_info = info
                # bare ``` outside a fence: neither opener nor a countable match
        else:
            if stripped == "```":  # closing = line of exactly three backticks
                if re.search(info_pattern, open_info):
                    matched += 1
                open_info = None
            # any other line (including a ```lang line) is fence CONTENT
    # an unpaired opener left at EOF never counts
    return matched


def _whole_text_is_one_fence(s, info_pattern):
    """(ok, detail): the ENTIRE stripped text is exactly one paired fence whose info-string matches info_pattern.

    2026-09-09 (blind-judge audit of the 200v4 benchmark, 18 misgrades): every `fenced` instruction the phrasing layer
    emits says "entirely inside a single ... block" / "nothing outside the fence", but the old check only asked for
    at least one paired fence anywhere, so a message with prose, headings and other blocks around a small matching
    fence passed. Now: first line = opener (three backticks + a non-empty info-string matching the pattern), last line
    = a closing line of exactly three backticks, and no closing line in between (that would end the fence early and
    leave text outside it). Bare content lines inside may be anything, including lines that start with backticks and
    an info-string (nested openers are content, as before).
    """
    lines = [ln for ln in s.splitlines()]
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if not lines:
        return False, "no fence: the text is empty"
    first, last = lines[0].strip(), lines[-1].strip()
    if not first.startswith("```") or not first[3:].strip():
        return False, "text does not start with a ```<info> fence opener (text outside the fence)"
    info = first[3:].strip()
    if not re.search(info_pattern, info):
        return False, "fence info-string %r does not match %r" % (info, info_pattern)
    if len(lines) < 2 or last != "```":
        return False, "text does not end with a closing ``` line (fence unpaired or text outside the fence)"
    if any(ln.strip() == "```" for ln in lines[1:-1]):
        return False, "the fence closes before the end of the text (text outside the fence)"
    return True, "ok (the whole text is one ```%s fence)" % info


_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_FENCE_BLOCK_RE = re.compile(r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", re.S | re.M)


def _strip_code(s):
    """The text with fenced code blocks and inline code spans replaced by a space (prose only).

    2026-09-09 (blind-judge audit): a ban on first-person pronouns matched the imaginary unit `I` inside a code span,
    a ban on parenthetical remarks matched the `()` of a function name in backticks. Banned words in code are code,
    not prose; the `forbidden` matcher searches the prose view unless the pattern itself targets backticks."""
    s = _FENCE_BLOCK_RE.sub(" ", s)
    return _INLINE_CODE_RE.sub(" ", s)


_ASCII_RUN_RE = re.compile(r"[\x21-\x7e]+")
_IDENTIFIER_LIKE_RE = re.compile(r"[_./\\()\[\]{}<>=:#@$%&*+|~^]|\d|[a-z][A-Z]|^[A-Z]{2,}$")
_TOKEN_PUNCT = ",;:.!?\"'`()[]{}\u3001\u3002\u300c\u300d\uff08\uff09\uff1a\uff1b\uff01\uff1f"


def _prose_view(s):
    """The text a reader would call prose: code blocks and inline spans removed (`_strip_code`) and ASCII runs that
    look like code — identifiers with `_`/`.`/`/`, paths, calls, CamelCase, digits, ALLCAPS — dropped. Runs are maximal
    sequences of printable ASCII, so an identifier glued to CJK text ("ConditionSet。检查") is still found; non-ASCII
    text is always kept.

    2026-09-09 (blind-judge audit, 26 language false fails): "write in Korean" messages carried `PyMethod.get_index_text()`,
    `django/forms/models.py`, `help_text` in the running prose (not in backticks) and lost the character majority to
    Latin; the judge calls code identifiers unavoidable and not prose. With this view the strict-majority rule agrees
    with the judge on 104 of 110 scored language rows (72 % before); higher thresholds only lose agreement."""

    def _keep_or_drop(m):
        run = m.group(0)
        core = run.strip(_TOKEN_PUNCT) or run
        return " " if _IDENTIFIER_LIKE_RE.search(core) else run

    return _ASCII_RUN_RE.sub(_keep_or_drop, _strip_code(s))


def _whole_fence_body(s):
    """The body of a message that is ENTIRELY one fenced block (opener line ... closing line), else None.

    2026-09-16 (Opus 5 200v4 audit): a language constraint met a fenced/JSON constraint on the same item; the model
    wrote its (compliant, Chinese/Russian) message inside the demanded ```status / ```plaintext fence, `_prose_view`
    stripped the whole block and the language check failed with "0 of 0 alphabetic chars". When the fence IS the
    message, its body is the prose."""
    lines = [ln for ln in s.splitlines()]
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if len(lines) < 2 or not lines[0].strip().startswith("```") or lines[-1].strip() != "```":
        return None
    body = "\n".join(lines[1:-1])
    return body if body.strip() else None


def _script_share(s, script):
    """(in_script, alphabetic) over the prose view. Japanese (`kana`) counts kana AND han characters, provided at least
    one kana character is present (kanji-heavy Japanese lost the kana majority to its own kanji; Chinese text has no
    kana and stays out)."""
    alpha = [ch for ch in _prose_view(s) if ch.isalpha()]
    if script == "kana":
        if not any(_script_matches(ch, _SCRIPT_NAME_PREFIXES["kana"]) for ch in alpha):
            return 0, len(alpha)
        n = sum(1 for ch in alpha if _script_matches(ch, _SCRIPT_NAME_PREFIXES["kana"]) or _script_matches(ch, _SCRIPT_NAME_PREFIXES["han"]))
        return n, len(alpha)
    prefixes = _SCRIPT_NAME_PREFIXES[script]
    return sum(1 for ch in alpha if _script_matches(ch, prefixes)), len(alpha)


_LEADING_TAG_RE = re.compile(r"^[\[(][A-Za-z][\w -]*[\])]:?$")


def _count_words(s):
    """Words = whitespace-separated tokens that carry at least one letter or digit, minus markup: fence marker
    tokens (``` or ```info), pure punctuation (list bullets `-` `*`, dashes, arrows) and ONE leading bracketed tag
    such as `[PLAN]` or `(edit)` (a label the tag-opener constraints of this very pool require, not a word).
    2026-09-09 (blind-judge audit): "[PLAN] Let me search the doc for usage examples." was 9 words to the old
    `len(s.split())`, 8 to the judge; "done ```python ... ```" counted its two fence markers."""
    tokens = s.split()
    if tokens and _LEADING_TAG_RE.match(tokens[0]):
        tokens = tokens[1:]
    return len([t for t in tokens if not t.startswith("```") and any(ch.isalnum() for ch in t)])


def _count_sentences(s):
    """Sentences = segments ended by . ! or ? (optionally followed by closing quotes/brackets) and containing at least
    one word character. 2026-09-09 (blind-judge audit of the 200v4 benchmark, two confirmed misgrades): a terminator
    followed by a closing quote or bracket ('encoding."', 'done.)') is a sentence end (the old regex demanded whitespace
    right after the terminator); a trailing segment with no word character (a closing code fence '```' after a
    period, a bare emoticon) is not a sentence."""
    if not s.strip():
        return 0
    parts = re.split(r"[.!?]+[\"')\]\}]*(?:\s+|$)", s)
    return len([p for p in parts if re.search(r"\w", p)])


def _length_count(s, unit):
    if unit == "lines":
        return len([ln for ln in s.splitlines() if ln.strip()])
    if unit == "words":
        return _count_words(s)
    if unit == "sentences":
        return _count_sentences(s)
    if unit == "chars":
        return len(s)
    raise ValueError("length_bound: unknown unit %r" % (unit,))


# --------------------------------------------------------------------------- #
# Tool-call helpers shared by triggers and templates
# --------------------------------------------------------------------------- #
def _resolve(identifier, resolver):
    """Identifier -> concrete name; pseudo-identifiers and unknown identifiers
    pass through (unknown == literal concrete name, e.g. `think`)."""
    if identifier in (NO_TOOL, ANY_TOOL):
        return identifier
    return resolver.get(identifier, identifier)


def _matching_calls(turn, resolved):
    if resolved == ANY_TOOL:
        return list(turn.tool_calls)
    return [c for c in turn.tool_calls if c.name == resolved]


def _tool_trigger_fires(turn, resolved, arg_predicate):
    if resolved == NO_TOOL:
        # no call exists, so an arg_predicate can never be satisfied
        return (len(turn.tool_calls) == 0) and (arg_predicate is None)
    calls = _matching_calls(turn, resolved)
    if not calls:
        return False
    if arg_predicate is None:
        return True
    fld = arg_predicate["field"]
    pattern = arg_predicate["regex"]
    for c in calls:
        if fld in c.args and re.search(pattern, str(c.args[fld])):
            return True
    return False


def _flatten_calls(turns):
    flat = []
    for turn in turns:
        for c in turn.tool_calls:
            flat.append((turn.index, c))
    return flat
