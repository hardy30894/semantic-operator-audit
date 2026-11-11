"""Prompt templates and output parsers.

Two prompt styles exist so that results cannot be an artifact of one wording:

- `lotus` (primary) follows the shape of LOTUS's semantic-operator prompts: a fixed system
  instruction, the serialized record(s) as context, and the predicate as a claim answered with
  True or False.
- `plain` uses a different instruction, puts the statement before the record, asks a question, and
  expects Yes or No.

Any change to the text below must bump TEMPLATE_VERSION, which is logged with every call together
with the style.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .core import INVALID

TEMPLATE_VERSION = "2026-10-14.1"

FILTER_SYSTEM = (
    "The user will provide a claim and some relevant context.\n"
    "Your job is to determine whether the claim is true for the given context.\n"
    'You must answer with a single word, "True" or "False".'
)
BATCH_SYSTEM = (
    "The user will provide a claim and a numbered list of records.\n"
    "For each record, determine whether the claim is true for that record.\n"
    'Answer with exactly one line per record, in the form "<number>: True" or '
    '"<number>: False", and nothing else.'
)
COUNT_SYSTEM = (
    "The user will provide a claim and a numbered list of records.\n"
    "Count the records for which the claim is true.\n"
    "Answer with a single integer and nothing else."
)
MAP_SYSTEM = (
    "The user will provide an instruction and a value.\n"
    "Apply the instruction to the value.\n"
    "Output only the resulting value, with no explanation and no quotation marks."
)


@dataclass(frozen=True)
class Style:
    filter_system: str
    filter_user: str  # {context} {claim}
    match_user: str  # {left} {right} {claim}
    batch_system: str
    batch_user: str  # {claim} {records}
    count_system: str
    count_user: str  # {claim} {records}
    map_system: str
    map_user: str  # {instruction} {value}
    yes: str
    no: str


STYLES = {
    "lotus": Style(
        filter_system=FILTER_SYSTEM,
        filter_user="Context:\n{context}\n\nClaim: {claim}",
        match_user="Context:\nRecord 1:\n{left}\n\nRecord 2:\n{right}\n\nClaim: {claim}",
        batch_system=BATCH_SYSTEM,
        batch_user="Claim: {claim}\n\nRecords:\n\n{records}",
        count_system=COUNT_SYSTEM,
        count_user="Claim: {claim}\n\nRecords:\n\n{records}",
        map_system=MAP_SYSTEM,
        map_user="Instruction: {instruction}\n\nValue: {value}",
        yes="true",
        no="false",
    ),
    "plain": Style(
        filter_system="You check statements about data records. Reply with Yes or No only.",
        filter_user="Statement: {claim}\n\nRecord:\n{context}\n\nIs the statement true for this record?",
        match_user="Statement: {claim}\n\nRecord 1:\n{left}\n\nRecord 2:\n{right}\n\nIs the statement true?",
        batch_system=("You check one statement against each record in a numbered list. Reply with one "
                      'line per record in the form "<number>: Yes" or "<number>: No", and nothing else.'),
        batch_user="Statement: {claim}\n\n{records}\n\nFor each record, is the statement true?",
        count_system="You count data records. Reply with a single integer only.",
        count_user="Statement: {claim}\n\n{records}\n\nFor how many of these records is the statement true?",
        map_system="You rewrite data values. Reply with the rewritten value only.",
        map_user="Value: {value}\n\nRewrite it as follows: {instruction}",
        yes="yes",
        no="no",
    ),
}


def serialize(record: dict[str, str], fields: list[str] | tuple[str, ...], prefix: str = "") -> str:
    return "\n".join(f"{prefix}{f}: {record.get(f, '')}" for f in fields)


def filter_prompt(context: str, claim: str, style: str = "lotus") -> tuple[str, str]:
    s = STYLES[style]
    return s.filter_system, s.filter_user.format(context=context, claim=claim)


def match_prompt(left: str, right: str, claim: str, style: str = "lotus") -> tuple[str, str]:
    s = STYLES[style]
    return s.filter_system, s.match_user.format(left=left, right=right, claim=claim)


def _numbered(contexts: list[str]) -> str:
    return "\n\n".join(f"Record {i}:\n{c}" for i, c in enumerate(contexts, 1))


def batch_prompt(contexts: list[str], claim: str, style: str = "lotus") -> tuple[str, str]:
    s = STYLES[style]
    return s.batch_system, s.batch_user.format(claim=claim, records=_numbered(contexts))


def count_prompt(contexts: list[str], claim: str, style: str = "lotus") -> tuple[str, str]:
    s = STYLES[style]
    return s.count_system, s.count_user.format(claim=claim, records=_numbered(contexts))


def map_prompt(instruction: str, value: str, style: str = "lotus") -> tuple[str, str]:
    s = STYLES[style]
    return s.map_system, s.map_user.format(instruction=instruction, value=value)


PARSER_VERSION = "2026-10-14.rules-3"

# Replies that ended for these reasons are never parsed: their text is cut off or not an answer.
STOP_INVALID = {"length", "max_tokens", "content_filter", "refusal", "error"}

_DECORATION = re.compile(r"[\s*_`\"'.,:;!()\[\]#>-]+")


def _words(style: str) -> tuple[str, str]:
    s = STYLES[style]
    return s.yes, s.no


def _bare(text: str) -> str:
    return _DECORATION.sub(" ", text).strip().lower()


def parse_bool_rule(text: str | None, style: str = "lotus") -> tuple[bool | str, str]:
    """(label, rule). Rules, tried in order:

    exact     the whole reply is the answer word, apart from punctuation and markdown
    verdict   the reply opens with a verdict sentence ("The claim is false.", "Answer: yes")
    leading   the reply opens with the answer word ("False. The product ...")
    trailing  the last line is only the answer word or a verdict ("**True**", "Final answer: False")

    A line that holds both answer words never counts. A reply whose opening answer differs from an
    answer word ending its last line (unless negated, as in "is not true") is a self-correction and
    INVALID ("conflict"). Anything else, including hedged sentences such as "I don't think the claim
    is true", is INVALID ("unparsed").
    """
    yes, no = _words(style)
    if not text or not text.strip():
        return INVALID, "empty"
    t = text.strip()
    bare = _bare(t)
    if bare in (yes, no):
        return bare == yes, "exact"
    lines = [line for line in t.splitlines() if line.strip()]
    first, last = lines[0], lines[-1]
    answer = rf"({yes}|{no})"
    verdict = rf"^\W*(?:the\s+)?(?:final\s+)?(?:claim|statement|answer)(?:\s+is)?\W*{answer}\b"

    def both(line: str) -> bool:
        words = set(re.findall(rf"\b{answer}\b", line, re.IGNORECASE))
        return len({w.lower() for w in words}) > 1

    head = rule = None
    if not both(first):
        for name, pattern in (("verdict", verdict), ("leading", rf"^\W*{answer}\b")):
            m = re.match(pattern, first, re.IGNORECASE)
            if m:
                head, rule = m.group(1).lower() == yes, name
                break
    end_word = None  # an answer word ending the last line, not negated
    if not both(last):
        m = re.search(rf"\b{answer}\W*$", last, re.IGNORECASE)
        if m and not re.search(rf"(\bnot|\bnever|n't)\W+{m.group(1)}\W*$", last, re.IGNORECASE):
            end_word = m.group(1).lower() == yes
    if head is not None:
        if end_word is not None and end_word != head:
            return INVALID, "conflict"
        return head, rule
    if end_word is not None and (_bare(last) in (yes, no) or re.match(verdict + r"\W*$", last, re.IGNORECASE)):
        return end_word, "trailing"
    return INVALID, "unparsed"


def parse_bool(text: str | None, style: str = "lotus") -> bool | str:
    return parse_bool_rule(text, style)[0]


def parse_batch_rules(text: str | None, n: int, style: str = "lotus") -> tuple[list[bool | str], list[str]]:
    """One label per numbered line ("3: True", "Record 3: False"); nothing may follow the answer word
    except punctuation. Missing, malformed or contradictory lines are INVALID."""
    yes, no = _words(style)
    line_re = re.compile(rf"^\W*(?:record\s*)?(\d+)\W*({yes}|{no})\W*$", re.IGNORECASE)
    labels: dict[int, bool | str] = {}
    for line in (text or "").splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        i, value = int(m.group(1)), m.group(2).lower() == yes
        if 1 <= i <= n:
            labels[i] = value if labels.get(i, value) == value else INVALID
    outputs = [labels.get(i, INVALID) for i in range(1, n + 1)]
    return outputs, ["line" if v != INVALID else "missing" for v in outputs]


def parse_batch(text: str | None, n: int, style: str = "lotus") -> list[bool | str]:
    return parse_batch_rules(text, n, style)[0]


# Minus signs and dashes (ASCII hyphen, U+2212, en and em dash) and list bullets (* + •) never introduce a count.
_NEG = "\\-\u2212\u2013\u2014"
_COUNT_ONLY = re.compile(rf"(?!\s*[*+\u2022]\s)[^\w{_NEG}]*(\d+)\W*")
_COUNT_LABEL = re.compile(rf"\b(?:count|total|answer|number)\b[^\d\n{_NEG}]*?(\d+)\W*$", re.IGNORECASE)


def parse_int_rule(text: str | None, count_max: int | None = None) -> tuple[int | str, str]:
    """(count, rule): the whole reply is an integer ("exact"), or the last line is an integer or a
    labelled count such as "Total: 3" ("final"). The first number in a sentence never counts. A
    number after a minus sign or dash ("-1", "−1") or a list bullet ("- 9", "* 9", "• 9") is INVALID,
    and so is a count above `count_max`, the number of records in the batch ("out of range")."""
    if not text or not text.strip():
        return INVALID, "empty"
    t = text.strip()
    m = _COUNT_ONLY.fullmatch(t)
    rule = "exact"
    if not m:
        last = [line for line in t.splitlines() if line.strip()][-1]
        m = _COUNT_ONLY.fullmatch(last) or _COUNT_LABEL.search(last)
        rule = "final"
    if not m:
        return INVALID, "unparsed"
    value = int(m.group(1))
    if count_max is not None and value > count_max:
        return INVALID, "out of range"
    return value, rule


def parse_int(text: str | None) -> int | str:
    return parse_int_rule(text)[0]


_PREFIX = re.compile(r"^(?:here is[^:]*|canonical(?: form)?|rewritten(?: value| title)?|result|output|answer)\s*:\s*",
                     re.IGNORECASE)


def parse_text_rule(text: str | None) -> tuple[str, str]:
    """(value, rule): a single line ("exact"), a known label prefix removed ("prefix"), or a lone
    line after a preamble ending in a colon ("preamble"). Several lines of content are INVALID."""
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    if not lines:
        return INVALID, "empty"
    rule = "exact"
    if len(lines) == 2 and lines[0].endswith(":"):
        lines, rule = lines[1:], "preamble"
    if len(lines) != 1:
        return INVALID, "multiline"
    value = lines[0]
    stripped = _PREFIX.sub("", value)
    if stripped != value:
        value, rule = stripped, "prefix"
    value = " ".join(value.split()).strip("\"'`*")
    return (value, rule) if value else (INVALID, "empty")


def parse_text(text: str | None) -> str:
    return parse_text_rule(text)[0]


def parse_with_rules(op: str, text: str | None, n_items: int, style: str = "lotus",
                     stop_reason: str | None = None, count_max: int | None = None) -> tuple[list[Any], list[str]]:
    """Outputs plus the rule that produced each one. Stopped-early replies are INVALID. `count_max`
    is the number of records a COUNT prompt shows."""
    if stop_reason in STOP_INVALID:
        return [INVALID] * n_items, [f"stop:{stop_reason}"] * n_items
    if op in ("filter", "match"):
        value, rule = parse_bool_rule(text, style)
        return [value], [rule]
    if op == "filter_batch":
        return parse_batch_rules(text, n_items, style)
    if op == "count":
        value, rule = parse_int_rule(text, count_max)
        return [value], [rule]
    if op == "map":
        value, rule = parse_text_rule(text)
        return [value], [rule]
    raise ValueError(f"unknown op: {op}")


def parse_outputs(op: str, text: str | None, n_items: int, style: str = "lotus",
                  stop_reason: str | None = None, count_max: int | None = None) -> list[Any]:
    return parse_with_rules(op, text, n_items, style, stop_reason, count_max)[0]
