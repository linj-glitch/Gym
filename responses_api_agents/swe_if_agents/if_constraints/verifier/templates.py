"""Constraint templates and their registry: which surface of the trajectory an obligation is graded on.

`turn_output`  the visible text of the selected turns; a silent turn (tool call, no text) is not a message and is never
               a graded step (owner ruling 2026-09-09: "a tool call is a tool call, a message is model output text");
               silent in-scope turns are counted; a missing final message follows the matcher's no-answer kind
`reply_output` the visible text of the turn that follows a tool or a message (legacy: graded like `fail`, no silent count)
`tool_args`    one argument of each call of the tool (argument-format contract)
`tool_choice`  which tools were called, how often, in what order (trajectory-scoped: always exactly one step)
EVERY template is declared exactly once in TEMPLATES as a Template(...): its grade function, which trigger kinds it
accepts, and one line of documentation. `grade(turns, trigger, obligation, resolver, policy) -> (steps, n_silent)`.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import ANY_TOOL, NO_TOOL, GradedStep, Turn, _flatten_calls, _resolve
from .matchers import _apply_matcher
from .triggers import missing_target, select_turns

import re


def _require_visible_target(obligation, template):
    target = obligation.get("target", "visible_message")
    if target != "visible_message":
        raise ValueError(
            "%s obligations only support target 'visible_message' (tool_arg is "
            "tool_args only); got %r" % (template, target))


def _grade_visible(turn, obligation, prefix, policy="fail"):
    """Grade one in-scope turn. Returns a GradedStep, or None when the turn is silent.

    Owner ruling 2026-09-09 (Charles): a tool call is a tool call, a message is model output text. A turn with no
    visible text is not a message, so no message rule — required shape or ban — applies to it; it is counted as a
    no-answer (`n_silent`) and never graded. Before this ruling (2026-09-03) a required shape failed on silence; that
    kind (`Matcher.silent_turn`, the `policy` argument) now governs only the missing final message
    (`_grade_visible_turns`).
    """
    if not turn.visible_text.strip():
        return None
    ok, why = _apply_matcher(obligation["match"], obligation.get("value"),
                             turn.visible_text)
    return GradedStep(turn=turn.index, reward=1 if ok else 0,
                      detail=prefix + why)


def _grade_visible_turns(turns, trigger, obligation, resolver, policy, allowed, count_silent):
    """The generic loop: select the turns with the trigger registry, grade each visible text with the matcher registry."""
    out = []
    n_silent = 0
    for turn, prefix in select_turns(turns, trigger, resolver, allowed):
        if count_silent and not turn.visible_text.strip():
            n_silent += 1
        step = _grade_visible(turn, obligation, prefix, policy)
        if step is not None:
            out.append(step)
    # A missing final message (the episode ended on a tool call or an error) keeps the 2026-09-03 ruling: a rule that
    # needs an answer (`policy == "fail"`) has FAILED once; a ban / maximum / sentinel is not gradable.
    detail = missing_target(turns, trigger, resolver, allowed)
    if detail is not None and count_silent:
        n_silent += 1
        if policy == "fail":
            out.append(GradedStep(turn=turns[-1].index, reward=0, detail=detail))
    return out, n_silent


def _grade_turn_output(turns, trigger, obligation, resolver, policy):
    _require_visible_target(obligation, "turn_output")
    return _grade_visible_turns(turns, trigger, obligation, resolver, policy, ("tool", "position", "all_of"), True)


def _grade_reply_output(turns, trigger, obligation, resolver, policy):
    # Legacy template: the no-answer policy is not applied (a silent reply is a failed step) and silence is not counted.
    _require_visible_target(obligation, "reply_output")
    return _grade_visible_turns(turns, trigger, obligation, resolver, "fail", ("prev_tool", "prev_message"), False)


def _grade_tool_args(turns, trigger, obligation, resolver, policy):
    """tool_args: argument-format contract. One GradedStep per (gradable) call of
    the triggering tool; empty list if it is never called."""
    resolved = _resolve(trigger["tool"], resolver)
    if resolved == NO_TOOL:
        raise ValueError("tool_args cannot target NO_TOOL")
    target = obligation.get("target")
    if not (isinstance(target, dict) and "tool_arg" in target):
        raise ValueError("tool_args requires obligation target {'tool_arg': <field>}")
    fld = target["tool_arg"]
    # Optional trigger filter: only calls whose predicate arg matches are
    # gradable (added after real-trace validation finding F3 -- without it a
    # path constraint on summary-writing exec calls over-fires on every exec
    # call in the episode). A call whose predicate field is MISSING does not
    # fire (unlike the obligation field, whose absence grades 0).
    arg_pred = trigger.get("arg_predicate")
    out = []
    for turn_index, call in _flatten_calls(turns):
        if resolved != ANY_TOOL and call.name != resolved:
            continue
        if arg_pred is not None:
            pf = arg_pred["field"]
            if pf not in call.args or not re.search(arg_pred["regex"],
                                                    str(call.args[pf])):
                continue
        if fld not in call.args:
            out.append(GradedStep(
                turn=turn_index, reward=0,
                detail="call of %s has no argument %r" % (call.name, fld)))
            continue
        ok, why = _apply_matcher(obligation["match"], obligation.get("value"),
                                 str(call.args[fld]))
        out.append(GradedStep(
            turn=turn_index, reward=1 if ok else 0,
            detail="call of %s arg %r: %s" % (call.name, fld, why)))
    return out, 0  # empty list if T never called


def _grade_tool_choice(turns, trigger, obligation, resolver, policy):
    """tool_choice: which tools, how many times, in what order. ALWAYS exactly
    one trajectory-scoped GradedStep (turn=-1) -- never an empty list."""
    mode = trigger.get("mode")
    if mode == "only_call":
        # Allowlist over the whole call stream (added after real-trace
        # validation finding F2 -- "you may only call these tools").
        allowed = {_resolve(t, resolver) for t in trigger["tools"]}
        offenders = sorted({c.name for _, c in _flatten_calls(turns)
                            if c.name not in allowed})
        ok = not offenders
        why = ("ok (all calls within allowlist %s)" % sorted(allowed) if ok
               else "calls outside allowlist %s: %s"
               % (sorted(allowed), offenders))
        return [GradedStep(turn=-1, reward=1 if ok else 0, detail=why)], 0

    if mode == "order":
        t1 = _resolve(trigger["first"], resolver)
        t2 = _resolve(trigger["then"], resolver)
        names = [c.name for _, c in _flatten_calls(turns)]
        t2_positions = [i for i, n in enumerate(names) if n == t2]
        if not t2_positions:
            return [GradedStep(turn=-1, reward=1,
                               detail="vacuous pass: %s never called" % (t2,))], 0
        for pos in t2_positions:
            if not any(names[i] == t1 for i in range(pos)):
                return [GradedStep(
                    turn=-1, reward=0,
                    detail="call #%d of %s has no earlier call of %s"
                    % (pos, t2, t1))], 0
        return [GradedStep(turn=-1, reward=1,
                           detail="ok: every %s call preceded by a %s call"
                           % (t2, t1))], 0

    if mode in ("must_call", "never_call", "exactly_n"):
        resolved = _resolve(trigger["tool"], resolver)
        count = len([1 for _, c in _flatten_calls(turns)
                     if resolved == ANY_TOOL or c.name == resolved])
        if mode == "must_call":
            ok = count >= 1
            why = ("ok (%d call(s) of %s)" % (count, resolved) if ok
                   else "%s never called (must_call)" % (resolved,))
        elif mode == "never_call":
            ok = count == 0
            why = ("ok (%s never called)" % (resolved,) if ok
                   else "%s called %d time(s) (never_call)" % (resolved, count))
        else:
            n = int(trigger["n"])
            ok = count == n
            why = ("ok (exactly %d call(s) of %s)" % (n, resolved) if ok
                   else "%s called %d time(s), required exactly %d"
                   % (resolved, count, n))
        return [GradedStep(turn=-1, reward=1 if ok else 0, detail=why)], 0

    raise ValueError("unknown tool_choice mode %r" % (mode,))


@dataclass(frozen=True)
class Template:
    """One constraint template. `grade(turns, trigger, obligation, resolver, policy) -> (steps, n_silent)`;
    `applies_policy` says whether the matcher's no-answer policy governs silent turns (turn_output) or the template
    has its own fixed rule; `doc`, one line."""
    name: str
    grade: Callable[[List[Turn], Dict[str, Any], Dict[str, Any], Dict[str, str], str], Tuple[List[GradedStep], int]]
    doc: str
    applies_policy: bool = False


TEMPLATES: Dict[str, Template] = {t.name: t for t in (
    Template("turn_output", _grade_turn_output, "visible text of the turns a tool/position/all_of trigger selects; silent turns are not steps; a missing final follows the no-answer kind", applies_policy=True),
    Template("reply_output", _grade_reply_output, "visible text of the turn after a tool call or a matching message (legacy; silent reply = failed step)"),
    Template("tool_args", _grade_tool_args, "one argument of every call of the tool, graded with the matcher; empty when never called"),
    Template("tool_choice", _grade_tool_choice, "which tools / how many / in what order; always exactly one trajectory-scoped step"),
)}
