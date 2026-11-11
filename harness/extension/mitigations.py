"""Mitigation conditions, issued as new Task segments next to semaudit's identities.

An extension run is a set of *segments*. Each segment is semaudit's identity builders
(`semaudit.identities.build`) applied with the same seed and unit sizes, so every segment measures
the same units, and then transformed:

| Segment | Identity prefix | What it sends | How analysis reads it |
|---|---|---|---|
| base | (none) | semaudit's requests, unchanged (same task ids and request bytes as v1) | as semaudit does |
| explain | `explain/` | the `explain` style: reason, then the answer on a final "Answer:" line | answer-line parser |
| format | `format/` | the `format` style: the same answer line, no instruction to reason | answer-line parser |
| negation_normalized | `negnorm/` | nothing by default (reuses the base segment's F(p) calls); with `negnorm = "fresh"`, p's request again for each not_p draw | the not_p answer is NOT F(p): NOT p is never sent |
| symmetric | `sym/` | nothing when the base segment has both operand orders, else both orders | sym(a,b) = combine(match(a,b), match(b,a)) |
| vote | `vote<v>/` | every request `repeats * v` times | majority (labels), lower median (counts) or plurality (text) of draws g*v .. g*v+v-1; map(map(x)) is applied to the voted map(x) |

base -> format measures the answer format; format -> explain measures the instruction to reason.

Reasoning on is a backend setting, so it is a separate arm in the roster that runs the base segment;
analysis pairs each such arm with the arm named by its `pairs_with` field, and the runner
interleaves the arms of one run in time.

The config's `[ext]` table selects segments:

    [ext]
    include_base = true                   # also issue the unmitigated requests, interleaved with the rest
    mitigations = ["explain", "format", "negation_normalized", "symmetric"]
    combine = "and"                       # symmetric matching: "and" or "or"
    negnorm = "reuse"                     # or "fresh" (default "reuse" when the base segment has filter_algebra)
    votes = 3                             # vote segment: draws per voted answer; odd, so labels cannot tie
    [ext.identities]                      # optional: limit a mitigation to some identities
    explain = ["symmetry", "filter_algebra"]

All tasks of all segments are shuffled together by the runner, so segments are interleaved in time.
The design records a hash of each extension style's prompt text, so a resume with edited prompts is
refused.

Addendum G adds two keys (absent from earlier configs, whose designs and requests are unchanged):

    [ext]
    base_count = false                    # no COUNT requests in the base segment (default true)
    segments = ["reword", "unknown"]      # filter_algebra's rewording and three-valued segments

`segments` is a key of its own rather than a mitigation: those segments send new conditions (reworded
claims; three-valued answers) that semaudit's comparisons do not read, and they are analyzed by
extension/placebo.py (extension/placebo_segments.py builds them). With base_count = false the design's
COUNT batches are empty, so no comparison expects COUNT answers; a mitigation that would send
filter_algebra's COUNT requests (explain, format, vote) cannot be combined with it.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from typing import Any, Iterable

from semaudit.core import INVALID, Task
from semaudit.data import Dataset
from semaudit.identities import BUILDERS, Outputs, build, idempotence_followups, other_host
from semaudit.prompts import map_prompt

from . import placebo_segments, styles

EXT_VERSION = "ext-2026-10-21.2"

MITIGATIONS = ("vote", "negation_normalized", "explain", "format", "symmetric")
# Which identities each mitigation applies to (by default all of them that are enabled).
APPLIES_TO = {
    "vote": tuple(BUILDERS),
    "negation_normalized": ("filter_algebra",),
    "explain": tuple(BUILDERS),
    "format": tuple(BUILDERS),
    "symmetric": ("symmetry",),
}
STYLE_OF = {"explain": styles.EXPLAIN, "format": styles.FORMAT}
COMBINE = ("and", "or")


def segment_prefix(mitigation: str, votes: int = 0) -> str:
    return {"vote": f"vote{votes}/", "negation_normalized": "negnorm/", "explain": "explain/",
            "format": "format/", "symmetric": "sym/"}[mitigation]


def _enabled(identities_cfg: dict[str, Any]) -> list[str]:
    return [name for name in BUILDERS if identities_cfg.get(name, {}).get("enabled", False)]


def _subset(identities_cfg: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    keep = set(names)
    return {name: ({**cfg, "enabled": True} if name in keep else {**cfg, "enabled": False})
            for name, cfg in identities_cfg.items()}


def _prefixed(task: Task, prefix: str, extra_context: dict[str, Any], condition: str | None = None) -> Task:
    cond = condition or task.condition
    identity = prefix + task.identity
    return replace(task, task_id=f"{identity}:{cond}:{task.unit}:{task.repeat}", identity=identity,
                   condition=cond, context={**task.context, **extra_context})


def _units(design: dict[str, Any]) -> dict[str, Any]:
    return {name: design[name] for name in BUILDERS if name in design}


def combine_rule(ext: dict[str, Any]) -> str:
    combine = str(ext.get("combine", "and")).strip().lower()
    if combine not in COMBINE:
        raise ValueError(f"[ext] combine must be one of {COMBINE} (any case), not {ext.get('combine')!r}")
    return combine


def plan_segments(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The segments a config asks for, in a fixed order."""
    ext = config.get("ext", {})
    run = config["run"]
    base_style = run.get("prompt_style", "lotus")
    enabled = _enabled(config["identities"])
    repeats = run["repeats"]
    votes = int(ext.get("votes", 3))
    unknown = set(ext.get("mitigations", [])) - set(MITIGATIONS)
    if unknown:
        raise ValueError(f"unknown mitigations {sorted(unknown)}; known: {list(MITIGATIONS)}")
    limits = ext.get("identities", {})
    segments: list[dict[str, Any]] = []
    include_base = bool(ext.get("include_base", True))
    base_count = ext.get("base_count", True)
    if not isinstance(base_count, bool):
        raise ValueError(f"[ext] base_count must be true or false, not {base_count!r}")
    if include_base:
        segments.append({"name": "base", "prefix": "", "mitigation": "none", "style": base_style,
                         "repeats": repeats, "identities": enabled})
        if not base_count:  # recorded only when set, so earlier designs are unchanged
            if "filter_algebra" not in enabled:
                raise ValueError("[ext] base_count = false needs [identities.filter_algebra] enabled")
            segments[-1]["count"] = False
    elif not base_count:
        raise ValueError("[ext] base_count = false applies to the base segment, which include_base = false leaves out")
    for mitigation in ext.get("mitigations", []):
        names = [n for n in enabled if n in APPLIES_TO[mitigation]]
        if mitigation in limits:
            names = [n for n in names if n in set(limits[mitigation])]
        if not names:
            raise ValueError(f"mitigation {mitigation} applies to none of the enabled identities")
        seg = {"name": segment_prefix(mitigation, votes).rstrip("/"), "prefix": segment_prefix(mitigation, votes),
               "mitigation": mitigation, "style": STYLE_OF.get(mitigation, base_style), "repeats": repeats,
               "identities": names}
        if mitigation == "vote":
            if votes < 2:
                raise ValueError("votes must be at least 2 (odd values avoid ties between labels)")
            seg.update(votes=votes, repeats=repeats * votes, voted_repeats=repeats)
        elif mitigation == "symmetric":
            seg["combine"] = combine_rule(ext)
            if include_base and "symmetry" in enabled:
                # The base segment already sends both operand orders of every pair: sym(a,b) is computed
                # from those calls, at no extra cost.
                seg["issues_tasks"] = False
        elif mitigation == "negation_normalized":
            can_reuse = include_base and "filter_algebra" in enabled
            mode = str(ext.get("negnorm", "reuse" if can_reuse else "fresh")).lower()
            if mode not in ("reuse", "fresh"):
                raise ValueError(f"[ext] negnorm must be 'reuse' or 'fresh', not {ext.get('negnorm')!r}")
            if mode == "reuse" and not can_reuse:
                raise ValueError("negnorm = 'reuse' needs the base segment with filter_algebra")
            seg["negnorm"] = mode
            if mode == "reuse":
                seg["issues_tasks"] = False  # NOT F(p) is read off the base segment's F(p) calls
        if not base_count and mitigation in ("explain", "format", "vote") and "filter_algebra" in names:
            raise ValueError(f"[ext] base_count = false leaves the {mitigation} segment's COUNT requests without a "
                             "base to compare with; limit it to other identities with [ext.identities]")
        segments.append(seg)
    extra = ext.get("segments", [])
    if not isinstance(extra, list) or len(set(extra)) != len(extra):
        raise ValueError(f"[ext] segments must be a list of distinct names, not {extra!r}")
    if extra and not include_base:
        raise ValueError("[ext] segments are compared with the base segment's requests; they need include_base = true")
    for kind in extra:
        segments.append(placebo_segments.plan_segment(kind, config, enabled))
    names_seen = [seg["name"] for seg in segments]
    if len(set(names_seen)) != len(names_seen):
        raise ValueError(f"segment names collide: {names_seen}")
    return segments


def build_ext(ds: Dataset, config: dict[str, Any]) -> tuple[list[Task], dict[str, Any]]:
    """Stage-1 and stage-0 tasks of every segment, and the design analysis needs."""
    run = config["run"]
    seed, repeats = run["seed"], run["repeats"]
    base_style = run.get("prompt_style", "lotus")
    base_tasks, design = build(ds, config["identities"], repeats, seed, base_style)
    units = _units(design)
    segments = plan_segments(config)
    tasks: list[Task] = []
    for seg in segments:
        if seg["mitigation"] == "none":
            if seg.get("count") is False:  # base_count = false: filter_algebra's COUNT requests are not sent
                base_tasks = [t for t in base_tasks if not (t.identity == "filter_algebra" and t.condition == "count")]
            tasks += base_tasks
            continue
        if seg.get("issues_tasks") is False:
            continue
        if seg["mitigation"] in placebo_segments.KINDS:
            tasks += placebo_segments.segment_tasks(ds, config, seg, units["filter_algebra"]["rows"], EXT_VERSION)
            continue
        cfg = _subset(config["identities"], seg["identities"])
        seg_tasks, seg_design = build(ds, cfg, seg["repeats"], seed, seg["style"])
        for name in seg["identities"]:  # same seed and sizes: the segment measures the base units
            assert seg_design[name] == units[name], f"segment {seg['name']} sampled other units for {name}"
        marker = {"segment": seg["name"], "mitigation": seg["mitigation"], "ext": EXT_VERSION}
        if seg["style"] in styles.TEMPLATE_VERSIONS:
            marker["template"] = styles.TEMPLATE_VERSIONS[seg["style"]]
        if seg["mitigation"] == "vote":
            v = seg["votes"]
            tasks += [_prefixed(t, seg["prefix"], {**marker, "votes": v, "vote_group": t.repeat // v,
                                                   "vote_draw": t.repeat % v}) for t in seg_tasks]
        elif seg["mitigation"] == "negation_normalized":  # "fresh": p's request again for each not_p draw
            for t in seg_tasks:
                if t.identity == "filter_algebra" and t.condition == "p":
                    tasks.append(_prefixed(t, seg["prefix"], {**marker, "claim": "not_p", "sent_claim": "p",
                                                              "negate_answer": True}, condition="not_p"))
        else:  # explain, format, symmetric
            tasks += [_prefixed(t, seg["prefix"], marker) for t in seg_tasks]
    used = sorted({seg["style"] for seg in segments if seg["style"] in styles.DESIGN_STYLES})
    design["ext"] = {"version": EXT_VERSION, "segments": segments, "base_style": base_style,
                     "style_hashes": {name: styles.style_hash(name) for name in used},
                     "template_versions": {name: styles.DESIGN_TEMPLATE_VERSIONS[name] for name in used}}
    if any(seg.get("count") is False for seg in segments) and "filter_algebra" in design:
        # No COUNT requests were sent: the design's COUNT batches are empty, so no analysis expects their answers.
        design["filter_algebra"] = {**design["filter_algebra"], "count_batches": {}}
    return tasks, design


# --- follow-up requests (map(map(x))) ------------------------------------------------------------

def _norm_text(value: Any) -> str:
    return " ".join(str(value).split())


def plurality(values: list[Any]) -> Any:
    """The most frequent value; ties go to the earliest draw. Text is compared after whitespace
    normalization; the earliest draw's original text is returned."""
    keys = [_norm_text(v) for v in values]
    counts = Counter(keys)
    best = max(counts.values())
    for key, value in zip(keys, values):
        if counts[key] == best:
            return value
    raise AssertionError("unreachable")


def _strip_prefix(rec: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = dict(rec)
    out["identity"] = rec["identity"][len(prefix):]
    return out


def _segment_marker(seg: dict[str, Any]) -> dict[str, Any]:
    marker = {"segment": seg["name"], "mitigation": seg["mitigation"], "ext": EXT_VERSION}
    if seg["style"] in styles.TEMPLATE_VERSIONS:
        marker["template"] = styles.TEMPLATE_VERSIONS[seg["style"]]
    return marker


def vote_followups(ds: Dataset, records: Iterable[dict[str, Any]], seg: dict[str, Any]) -> list[Task]:
    """map(y_g) for voted repeat g, drawn `votes` times, where y_g is the plurality of the voted
    group's stage-0 draws of map(x). A group is used once all its stage-0 draws are valid."""
    v, prefix = seg["votes"], seg["prefix"]
    identity = prefix + "idempotence"
    groups: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for rec in records:
        if rec["identity"] != identity or rec["condition"] != "x_in" or rec.get("error"):
            continue
        if rec["outputs"][0] == INVALID or other_host(rec):
            continue
        groups.setdefault((rec["unit"], rec["repeat"] // v), {})[rec["repeat"] % v] = rec
    tasks = []
    for (unit, g), draws in sorted(groups.items()):
        if len(draws) < v:
            continue
        ordered = [draws[j] for j in range(v)]
        y = plurality([r["outputs"][0] for r in ordered])
        prompt = map_prompt(ds.map_spec.instruction, y, seg["style"])
        parents = [r["task_id"] for r in ordered]
        for j in range(v):
            repeat = g * v + j
            tasks.append(Task(
                task_id=f"{identity}:fx:{unit}:{repeat}", identity=identity, condition="fx", unit=unit,
                repeat=repeat, op="map", items=(unit,), system=prompt[0], user=prompt[1], payload={"value": y},
                context={"template": ordered[0]["context"].get("template"), "style": seg["style"], "op": "map",
                         "input": "voted stage-0 output", "parents": parents, **_segment_marker(seg),
                         "votes": v, "vote_group": g, "vote_draw": j},
            ))
    return tasks


def ext_followups(ds: Dataset, records: Iterable[dict[str, Any]], design: dict[str, Any]) -> list[Task]:
    """Condition B of idempotence (map(map(x))) for every segment that measures it."""
    records = list(records)
    tasks = []
    for seg in design["ext"]["segments"]:
        if "idempotence" not in seg["identities"] or seg.get("issues_tasks") is False:
            continue
        if seg["mitigation"] == "none":
            tasks += idempotence_followups(ds, records)
        elif seg["mitigation"] == "vote":
            tasks += vote_followups(ds, records, seg)
        else:
            prefix = seg["prefix"]
            mine = [_strip_prefix(r, prefix) for r in records if r["identity"] == prefix + "idempotence"]
            tasks += [_prefixed(t, prefix, _segment_marker(seg)) for t in idempotence_followups(ds, mine)]
    return tasks


def planned_followups(tasks: list[Task]) -> int:
    """Follow-up calls a run will add: one map(map(x)) per stage-0 draw (voted groups too)."""
    return sum(t.condition == "x_in" for t in tasks)


# --- reading a run back -------------------------------------------------------------------------

def segment_outputs(out: Outputs, seg: dict[str, Any]) -> Outputs:
    """The outputs of one segment, under semaudit's identity names."""
    prefix = seg["prefix"]
    if not prefix:
        return {key: slot for key, slot in out.items() if "/" not in key[0]}
    return {(identity[len(prefix):], condition, item): slot
            for (identity, condition, item), slot in out.items() if identity.startswith(prefix)}


def vote_value(draws: list[Any]) -> Any:
    """The voted answer of one group of draws: majority for labels (INVALID on a tie or without a
    strict majority of all draws), lower median for counts (at least half the draws valid), plurality
    for text (ties to the earliest draw)."""
    valid = [d for d in draws if d != INVALID and d is not None]
    if not valid:
        return INVALID
    if all(isinstance(d, bool) for d in valid):
        trues = sum(valid)
        if 2 * trues > len(draws):
            return True
        if 2 * (len(valid) - trues) > len(draws):
            return False
        return INVALID
    if all(isinstance(d, int) for d in valid):
        if 2 * len(valid) < len(draws):
            return INVALID
        ordered = sorted(valid)
        return ordered[(len(ordered) - 1) // 2]
    return plurality(valid)


def vote_outputs(out: Outputs, votes: int, voted_repeats: int) -> Outputs:
    voted: Outputs = {}
    for key, slot in out.items():
        voted[key] = {g: vote_value([slot.get(g * votes + j, INVALID) for j in range(votes)])
                      for g in range(voted_repeats)}
    return voted


def negation_normalized_outputs(base: Outputs, negnorm: Outputs) -> Outputs:
    """The base segment's outputs with not_p replaced by NOT F(p) from the negnorm segment's fresh
    F(p) draws (negnorm = "fresh")."""
    out = {key: slot for key, slot in base.items() if not (key[0] == "filter_algebra" and key[1] == "not_p")}
    for (identity, condition, item), slot in negnorm.items():
        if identity == "filter_algebra" and condition == "not_p":
            out[(identity, condition, item)] = {r: (INVALID if v == INVALID else (not v)) for r, v in slot.items()}
    return out


def symmetric_value(ab: Any, ba: Any, combine: str = "and") -> Any:
    if combine not in COMBINE:
        raise ValueError(f"combine must be one of {COMBINE}, not {combine!r}")
    if INVALID in (ab, ba):
        return INVALID
    return (ab and ba) if combine == "and" else (ab or ba)


def symmetric_outputs(out: Outputs, design: dict[str, Any], combine: str = "and") -> Outputs:
    """Outputs in which both operand orders answer with sym(a,b) = combine(match(a,b), match(b,a))
    from the same repeat. H1 then holds by construction; transitivity and accuracy can be read off."""
    sym = copy.copy(out)
    for pid in design.get("symmetry", {}).get("pairs", []):
        ab = out.get(("symmetry", "ab", pid), {})
        ba = out.get(("symmetry", "ba", pid), {})
        values = {r: symmetric_value(ab.get(r, INVALID), ba.get(r, INVALID), combine) for r in range(design["repeats"])}
        sym[("symmetry", "ab", pid)] = dict(values)
        sym[("symmetry", "ba", pid)] = dict(values)
    return sym


def segment_design(design: dict[str, Any], seg: dict[str, Any]) -> dict[str, Any]:
    """The design one segment's outputs are compared under: the base units, the segment's repeats
    (voted repeats for a vote segment), and only the identities the segment measured."""
    out = {k: v for k, v in design.items() if k not in BUILDERS and k != "ext"}
    for name in seg["identities"]:
        if name in design:
            out[name] = design[name]
    out["repeats"] = seg.get("voted_repeats", seg["repeats"])
    out["style"] = seg["style"]
    return out
