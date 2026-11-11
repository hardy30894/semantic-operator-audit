"""Two extension prompt styles, `explain` and `format`, and their shared parser.

`explain` asks the model to reason first and then give the answer on the last line in LOTUS's
answer format ("Answer: True"). Compared with semaudit's `lotus` style it changes two things at
once: the answer format (an answer line instead of the bare word) and the instruction to reason.
`format` is the control that changes only the first: the same answer line, with no instruction to
reason. Comparing base -> format measures the format; format -> explain measures the reasoning.
Both styles keep the `lotus` style's user prompts unchanged.

The styles are registered in `semaudit.prompts.STYLES` at import time (a runtime addition; no file of
semaudit changes), so semaudit's prompt and identity builders produce their prompts with
`style="explain"` or `style="format"`. `style_hash` fingerprints a style's prompt text; extension
designs record it, so a run is never resumed with edited prompts.

Parsing reuses semaudit's frozen rules (`semaudit.prompts`) on the answer segment, with v1's
conventions:

- filter / match: the last non-empty line must be `Answer: True` or `Answer: False`, or the bare
  word; markdown emphasis around the whole line is allowed. The word must match parse_bool_rule's
  "exact" rule.
- count: the last line must be `Answer: <integer>` or a bare integer, read by parse_int_rule: a
  number after a list bullet ("* 9", "- 9") or a minus sign is INVALID, and so is a count above
  the batch size.
- map: the last line must be `Answer: <value>`; parse_text_rule reads the value (so a trailing
  underscore is kept, as in v1).
- filter_batch: the lines after the last line that reads `Answers:` (or `Answer:`), read with
  parse_batch_rules.
- Self-correction: an earlier answer line whose answer differs from the final one makes the reply
  INVALID ("conflict"), as in v1.
- Replies stopped early (length, refusal, ...) are INVALID.

A third style, `unknown` (Addendum G), offers a third answer word: the system prompt asks for "True",
"False" or "Unknown" (its first two lines are lotus's), and the user prompt is lotus's. Only
filter_algebra's single-row filters use it (extension/placebo_segments.py). Its parser
(`parse_unknown_rule`) is a rule-for-rule port of semaudit's parse_bool_rule to three answer words, so
that two-valued and three-valued answers are read with the same strictness: the same rules (exact,
verdict, leading, trailing), the same conflict check and negation guard, the same normalization of case,
punctuation and markdown, and the same rule names; "Unknown" parses to UNKNOWN_ANSWER, a value distinct
from INVALID, True and False. On a reply without the word "unknown" it returns exactly what
parse_bool_rule(text, "lotus") returns (tested on every lotus filter reply logged in the study's runs).
Replies stopped early are INVALID, as in semaudit. The style is registered like the other two, and
extension designs record its hash and template version.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

from semaudit import prompts
from semaudit.core import INVALID
from semaudit.identities import other_host, parse_record
from semaudit.prompts import (STOP_INVALID, STYLES, Style, parse_batch_rules, parse_bool_rule, parse_int_rule,
                              parse_text_rule)

EXPLAIN = "explain"
FORMAT = "format"
EXPLAIN_TEMPLATE_VERSION = "ext-2026-10-21.explain-1"
FORMAT_TEMPLATE_VERSION = "ext-2026-10-21.format-1"
EXPLAIN_PARSER_VERSION = "ext-2026-10-21.answer-line-rules-2"
TEMPLATE_VERSIONS = {EXPLAIN: EXPLAIN_TEMPLATE_VERSION, FORMAT: FORMAT_TEMPLATE_VERSION}

_LOTUS = STYLES["lotus"]

EXPLAIN_STYLE = Style(
    filter_system=(
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        "First explain your reasoning in a few sentences. Then, on the last line, write "
        '"Answer: True" or "Answer: False" and nothing else.'
    ),
    filter_user=_LOTUS.filter_user,
    match_user=_LOTUS.match_user,
    batch_system=(
        "The user will provide a claim and a numbered list of records.\n"
        "For each record, determine whether the claim is true for that record.\n"
        "First explain your reasoning briefly. Then write a line that reads only \"Answers:\", "
        'followed by exactly one line per record in the form "<number>: True" or "<number>: False", '
        "and nothing after them."
    ),
    batch_user=_LOTUS.batch_user,
    count_system=(
        "The user will provide a claim and a numbered list of records.\n"
        "Count the records for which the claim is true.\n"
        "First explain your reasoning briefly. Then, on the last line, write \"Answer: \" followed by a "
        "single integer and nothing else."
    ),
    count_user=_LOTUS.count_user,
    map_system=(
        "The user will provide an instruction and a value.\n"
        "Apply the instruction to the value.\n"
        "First explain your reasoning briefly. Then, on the last line, write \"Answer: \" followed only "
        "by the resulting value, with no quotation marks."
    ),
    map_user=_LOTUS.map_user,
    yes="true",
    no="false",
)

# The explain style's answer format, with no instruction to reason.
FORMAT_STYLE = Style(
    filter_system=(
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'Answer with a single line, "Answer: True" or "Answer: False", and nothing else.'
    ),
    filter_user=_LOTUS.filter_user,
    match_user=_LOTUS.match_user,
    batch_system=(
        "The user will provide a claim and a numbered list of records.\n"
        "For each record, determine whether the claim is true for that record.\n"
        "Write a line that reads only \"Answers:\", followed by exactly one line per record in the form "
        '"<number>: True" or "<number>: False", and nothing else.'
    ),
    batch_user=_LOTUS.batch_user,
    count_system=(
        "The user will provide a claim and a numbered list of records.\n"
        "Count the records for which the claim is true.\n"
        "Answer with a single line, \"Answer: \" followed by a single integer, and nothing else."
    ),
    count_user=_LOTUS.count_user,
    map_system=(
        "The user will provide an instruction and a value.\n"
        "Apply the instruction to the value.\n"
        "Answer with a single line, \"Answer: \" followed only by the resulting value, with no quotation "
        "marks, and nothing else."
    ),
    map_user=_LOTUS.map_user,
    yes="true",
    no="false",
)

EXT_STYLES = {EXPLAIN: EXPLAIN_STYLE, FORMAT: FORMAT_STYLE}

# --- the three-valued style (Addendum G) ------------------------------------------------------------

UNKNOWN = "unknown"
UNKNOWN_TEMPLATE_VERSION = "ext-2026-10-21.unknown-1"
UNKNOWN_PARSER_VERSION = "ext-2026-10-21.unknown-rules-1"
UNKNOWN_ANSWER = "UNKNOWN"  # the parsed third answer; never equal to INVALID, True or False

UNKNOWN_STYLE = Style(
    filter_system=(
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'You must answer with a single word, "True", "False" or "Unknown". Answer "Unknown" if the context does '
        "not let you determine whether the claim is true or false."
    ),
    filter_user=_LOTUS.filter_user,
    # Only single-row filters use this style. The other operators' templates are lotus's.
    match_user=_LOTUS.match_user,
    batch_system=_LOTUS.batch_system,
    batch_user=_LOTUS.batch_user,
    count_system=_LOTUS.count_system,
    count_user=_LOTUS.count_user,
    map_system=_LOTUS.map_system,
    map_user=_LOTUS.map_user,
    yes="true",
    no="false",
)

THREE_VALUED_STYLES = {UNKNOWN: UNKNOWN_STYLE}
# Styles whose prompt text an extension design fingerprints (style_hashes), with their template versions.
DESIGN_STYLES = {**EXT_STYLES, **THREE_VALUED_STYLES}
DESIGN_TEMPLATE_VERSIONS = {**TEMPLATE_VERSIONS, UNKNOWN: UNKNOWN_TEMPLATE_VERSION}


def register() -> None:
    """Add the extension styles to semaudit's style registry (idempotent)."""
    for name, style in DESIGN_STYLES.items():
        current = prompts.STYLES.get(name)
        if current is None:
            prompts.STYLES[name] = style
        elif current != style:
            raise RuntimeError(f"a different prompt style is already registered as {name!r}")


register()


def style_hash(name: str) -> str:
    """SHA-256 of a style's prompt text and answer words."""
    blob = json.dumps(asdict(STYLES[name]), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Markdown emphasis around a whole final line ("**Answer: True**", "`Answer: 3`").
_WRAP = re.compile(r"^(\*\*|__|\*|`)(.+)\1$")
_ANSWER = re.compile(r"^(?:>\s*|#+\s*)?answer\s*:\s*(.*)$", re.IGNORECASE)
_ANSWERS_HEADER = re.compile(r"^[\s*_`>#]*answers?\s*:?[\s*_`]*$", re.IGNORECASE)


def _unwrap(line: str) -> str:
    line = line.strip()
    m = _WRAP.match(line)
    return m.group(2).strip() if m else line


def _answer_segment(line: str) -> str | None:
    """The answer after "Answer:" on an answer line, else None."""
    m = _ANSWER.match(_unwrap(line))
    return m.group(1).strip() if m else None


def _single(op: str, segment: str, count_max: int | None, answer_line: bool) -> tuple[Any, str]:
    if op in ("filter", "match"):
        value, rule = parse_bool_rule(segment, "lotus")
        return (value, "ok") if value != INVALID and rule == "exact" else (INVALID, "unparsed")
    if op == "count":
        value, rule = parse_int_rule(segment, count_max)
        if value == INVALID:
            return INVALID, rule if rule == "out of range" else "unparsed"
        return (value, "ok") if rule == "exact" else (INVALID, "unparsed")
    if op == "map":
        if not answer_line:  # a bare last line could be the end of the reasoning
            return INVALID, "no answer line"
        value, rule = parse_text_rule(segment)
        return (INVALID, rule) if value == INVALID else (value, "ok")
    raise ValueError(f"unknown op: {op}")


def parse_explain(op: str, text: str | None, n_items: int, stop_reason: str | None = None,
                  count_max: int | None = None) -> tuple[list[Any], list[str]]:
    """Outputs and the rule that produced each one, for a reply in the answer-line format."""
    if stop_reason in STOP_INVALID:
        return [INVALID] * n_items, [f"stop:{stop_reason}"] * n_items
    if not text or not text.strip():
        return [INVALID] * n_items, ["empty"] * n_items
    if op == "filter_batch":
        lines = text.splitlines()
        headers = [i for i, line in enumerate(lines) if _ANSWERS_HEADER.match(line.strip())]
        if not headers:
            return [INVALID] * n_items, ["no answer block"] * n_items
        values, rules = parse_batch_rules("\n".join(lines[headers[-1] + 1:]), n_items, "lotus")
        if len(headers) > 1:  # an earlier answer block that disagrees is a self-correction
            earlier, _ = parse_batch_rules("\n".join(lines[headers[-2] + 1:headers[-1]]), n_items, "lotus")
            if any(e != INVALID and v != INVALID and e != v for e, v in zip(earlier, values)):
                return [INVALID] * n_items, ["conflict"] * n_items
        return values, [f"answer-line:{r}" for r in rules]
    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1]
    segment = _answer_segment(last)
    answer_line = segment is not None
    value, rule = _single(op, segment if answer_line else _unwrap(last), count_max, answer_line)
    if value == INVALID:
        return [INVALID], [rule]
    for line in lines[:-1]:  # v1: an earlier answer that differs from the final one is a self-correction
        earlier = _answer_segment(line)
        if earlier is None:
            continue
        other, _ = _single(op, earlier, count_max, True)
        if other != INVALID and other != value:
            return [INVALID], ["conflict"]
    return [value], ["answer-line:" + ("answer line" if answer_line else "bare last line")]


_VALUES3 = {"true": True, "false": False, "unknown": UNKNOWN_ANSWER}  # answer word -> label, in regex order


def parse_bool3_rule(text: str | None) -> tuple[Any, str]:
    """(label, rule): semaudit.prompts.parse_bool_rule ported rule for rule to the answer words True,
    False and Unknown (labels True, False and UNKNOWN_ANSWER). The port changes only the vocabulary:

    exact     the whole reply is an answer word, apart from punctuation and markdown (semaudit's _bare)
    verdict   the reply opens with a verdict sentence ("The claim is unknown.", "Answer: Unknown")
    leading   the reply opens with an answer word ("Unknown. The record does not say ...")
    trailing  the last line is only an answer word or a verdict ("**Unknown**", "Final answer: False")

    A line that holds two or more distinct answer words never counts (parse_bool_rule: both words). A reply
    whose opening answer differs from an answer word ending its last line, unless that word is negated ("is
    not unknown"), is a self-correction and INVALID ("conflict"). Anything else is INVALID ("unparsed"), an
    empty reply INVALID ("empty"). Case is ignored throughout. On a reply without the word "unknown" the
    result is exactly parse_bool_rule(text, "lotus")'s.
    """
    if not text or not text.strip():
        return INVALID, "empty"
    t = text.strip()
    bare = prompts._bare(t)
    if bare in _VALUES3:
        return _VALUES3[bare], "exact"
    lines = [line for line in t.splitlines() if line.strip()]
    first, last = lines[0], lines[-1]
    answer = "(" + "|".join(_VALUES3) + ")"
    verdict = rf"^\W*(?:the\s+)?(?:final\s+)?(?:claim|statement|answer)(?:\s+is)?\W*{answer}\b"

    def several(line: str) -> bool:
        words = set(re.findall(rf"\b{answer}\b", line, re.IGNORECASE))
        return len({w.lower() for w in words}) > 1

    head = rule = None
    if not several(first):
        for name, pattern in (("verdict", verdict), ("leading", rf"^\W*{answer}\b")):
            m = re.match(pattern, first, re.IGNORECASE)
            if m:
                head, rule = _VALUES3[m.group(1).lower()], name
                break
    end_word = None  # an answer word ending the last line, not negated
    if not several(last):
        m = re.search(rf"\b{answer}\W*$", last, re.IGNORECASE)
        if m and not re.search(rf"(\bnot|\bnever|n't)\W+{m.group(1)}\W*$", last, re.IGNORECASE):
            end_word = _VALUES3[m.group(1).lower()]
    if head is not None:
        if end_word is not None and end_word != head:
            return INVALID, "conflict"
        return head, rule
    if end_word is not None and (prompts._bare(last) in _VALUES3 or re.match(verdict + r"\W*$", last, re.IGNORECASE)):
        return end_word, "trailing"
    return INVALID, "unparsed"


def parse_unknown_rule(text: str | None, stop_reason: str | None = None) -> tuple[Any, str]:
    """(label, rule) for a filter reply in the `unknown` style: True, False, UNKNOWN_ANSWER or INVALID.
    A reply stopped early is INVALID ("stop:<reason>"), as in semaudit's parse_with_rules; any other reply is
    read by parse_bool3_rule."""
    if stop_reason in STOP_INVALID:
        return INVALID, f"stop:{stop_reason}"
    return parse_bool3_rule(text)


def parse_unknown(op: str, text: str | None, n_items: int,
                  stop_reason: str | None = None) -> tuple[list[Any], list[str]]:
    """Outputs and rules for a reply in the `unknown` style. Only filters use the style; any other
    operator's reply is INVALID (never an exception, because the runner parses every reply as it logs it)."""
    if op != "filter":
        return [INVALID] * n_items, [f"unsupported op: {op}"] * n_items
    value, rule = parse_unknown_rule(text, stop_reason)
    return [value], [rule]


def parse_record_ext(rec: dict[str, Any]) -> tuple[list[Any], list[str]]:
    """semaudit's parse_record for its own styles; the answer-line parser for the extension styles; the
    three-valued parser for the `unknown` style. Failed calls and calls served by another host than the
    pinned one are INVALID, as in semaudit."""
    context = rec.get("context") or {}
    if context.get("style") in THREE_VALUED_STYLES:
        n = len(rec["items"])
        if rec.get("error") or rec.get("text") is None:
            return [INVALID] * n, ["error"] * n
        if other_host(rec):
            return [INVALID] * n, ["other host"] * n
        return parse_unknown(rec["op"], rec["text"], n, rec.get("stop_reason"))
    if context.get("style") not in EXT_STYLES:
        return parse_record(rec)
    n = len(rec["items"])
    if rec.get("error") or rec.get("text") is None:
        return [INVALID] * n, ["error"] * n
    if other_host(rec):
        return [INVALID] * n, ["other host"] * n
    count_max = len(context["batch"]) if rec["op"] == "count" and context.get("batch") else None
    return parse_explain(rec["op"], rec["text"], n, rec.get("stop_reason"), count_max)


def reparse_ext(rec: dict[str, Any]) -> list[Any]:
    return parse_record_ext(rec)[0]
