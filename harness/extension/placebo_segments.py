"""Addendum G's segments for filter_algebra: rewordings of the claims, and a three-valued prompt.

Both are issued next to the base segment, on the base segment's rows, and shuffled with every other
request of the run (extension/runner.py). Like the mitigation segments (extension/mitigations.py), each
one re-runs semaudit's filter_algebra builder with the run's seed and sizes, checks that it samples the
same rows, and transforms the requests:

| Segment | Identity prefix | Condition | What is sent | Payload claim and rule |
|---|---|---|---|---|
| reword | `reword/` | p_para1, p_para2 | p's request with the claim replaced by the config's two paraphrases of p | p |
| | | pq_para | p_and_q's request with the claim replaced by the config's paraphrase of it | p_and_q |
| | | p_tmpl | p's claim and record in another of semaudit's prompt styles (`template_style`, e.g. plain), read with that style's parser | p |
| | | p_order | p's request with the record's filter fields serialized in reversed order | p |
| unknown | `unknown/` | p, not_p, q, p_and_q | those requests in the `unknown` style (True, False or Unknown); no q_and_p, no COUNT | as in the base segment |

Every reworded request carries the payload of the claim it restates (its name and its truth rule), so the
offline backends and the accuracy checks treat it as that claim. The config's `[ext]` table:

    [ext]
    include_base = true
    base_count = false              # no COUNT requests in the base segment (default true)
    segments = ["reword", "unknown"]
    [ext.reword]
    p = ["<paraphrase 1 of p>", "<paraphrase 2 of p>"]
    pq = "<paraphrase of the conjunction p and q, p first>"
    template_style = "plain"
    field_order = "reversed"

The design records each segment's settings: the paraphrases as sent, the template style, the field order,
which claim each condition restates, and the builder version; extension.mitigations.build_ext adds the
`unknown` style's hash and template version, so a resume with other texts or prompts is refused.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from semaudit.core import Task
from semaudit.data import Dataset
from semaudit.identities import _filter_payload, build
from semaudit.prompts import filter_prompt, serialize

from . import styles

SEGMENTS_VERSION = "ext-2026-10-21.g-segments-1"
FA = "filter_algebra"
KINDS = ("reword", "unknown")
PREFIXES = {"reword": "reword/", "unknown": "unknown/"}
# Condition -> the claim it restates (its payload claim and truth rule).
REWORD_RESTATES = {"p_para1": "p", "p_para2": "p", "pq_para": "p_and_q", "p_tmpl": "p", "p_order": "p"}
REWORD_CONDITIONS = tuple(REWORD_RESTATES)
UNKNOWN_CONDITIONS = ("p", "not_p", "q", "p_and_q")
SEMAUDIT_STYLES = ("lotus", "plain")  # the v1 styles with semaudit's own parsers
FIELD_ORDERS = ("reversed",)


def _texts(reword: dict[str, Any]) -> dict[str, str]:
    p = reword.get("p")
    if not isinstance(p, list) or len(p) != 2 or not all(isinstance(t, str) and t.strip() for t in p):
        raise ValueError("[ext.reword] p must list exactly two non-empty paraphrases of p")
    pq = reword.get("pq")
    if not isinstance(pq, str) or not pq.strip():
        raise ValueError("[ext.reword] pq must be a non-empty paraphrase of the conjunction p and q")
    if p[0] == p[1]:
        raise ValueError("[ext.reword] the two paraphrases of p must differ")
    return {"p_para1": p[0], "p_para2": p[1], "pq_para": pq}


def plan_segment(kind: str, config: dict[str, Any], enabled: list[str]) -> dict[str, Any]:
    """The design entry of one Addendum G segment, from the config alone (no dataset)."""
    if kind not in KINDS:
        raise ValueError(f"unknown segments [{kind!r}]; known: {list(KINDS)}")
    if FA not in enabled:
        raise ValueError(f"segment {kind} needs [identities.filter_algebra] enabled")
    run = config["run"]
    base_style = run.get("prompt_style", "lotus")
    seg: dict[str, Any] = {"name": kind, "prefix": PREFIXES[kind], "mitigation": kind, "repeats": run["repeats"],
                           "identities": [FA]}
    if kind == "unknown":
        seg.update(style=styles.UNKNOWN, conditions=list(UNKNOWN_CONDITIONS), builder=SEGMENTS_VERSION)
        return seg
    reword = config.get("ext", {}).get("reword")
    if not isinstance(reword, dict):
        raise ValueError("segment reword needs an [ext.reword] table with p, pq, template_style and field_order")
    unknown_keys = set(reword) - {"p", "pq", "template_style", "field_order"}
    if unknown_keys:
        raise ValueError(f"[ext.reword] has unknown keys {sorted(unknown_keys)}")
    template = reword.get("template_style")
    if template not in SEMAUDIT_STYLES or template == base_style:
        raise ValueError(f"[ext.reword] template_style must be one of semaudit's styles {list(SEMAUDIT_STYLES)} "
                         f"other than the run's prompt_style {base_style!r}, not {template!r}")
    order = reword.get("field_order")
    if order not in FIELD_ORDERS:
        raise ValueError(f"[ext.reword] field_order must be one of {list(FIELD_ORDERS)}, not {order!r}")
    seg.update(style=base_style, conditions=list(REWORD_CONDITIONS), restates=dict(REWORD_RESTATES),
               texts=_texts(reword), template_style=template, field_order=order, builder=SEGMENTS_VERSION)
    return seg


def _rerun(ds: Dataset, config: dict[str, Any], seg: dict[str, Any], style: str,
           rows: list[str]) -> list[Task]:
    """semaudit's filter_algebra builder, re-run with the run's seed and sizes, in `style`."""
    cfg = {FA: {**copy.deepcopy(config["identities"][FA]), "enabled": True}}
    tasks, design = build(ds, cfg, seg["repeats"], config["run"]["seed"], style)
    if design[FA]["rows"] != rows:
        raise AssertionError(f"segment {seg['name']} sampled other rows than the base segment")
    return tasks


def _marker(seg: dict[str, Any], ext_version: str) -> dict[str, Any]:
    return {"segment": seg["name"], "mitigation": seg["mitigation"], "ext": ext_version}


def _with(task: Task, seg: dict[str, Any], condition: str, prompt: tuple[str, str], payload: dict[str, Any],
          context: dict[str, Any]) -> Task:
    identity = seg["prefix"] + task.identity
    return replace(task, task_id=f"{identity}:{condition}:{task.unit}:{task.repeat}", identity=identity,
                   condition=condition, system=prompt[0], user=prompt[1], payload=payload,
                   context={**task.context, **context})


def _rule(ds: Dataset, claim: str) -> dict[str, Any] | None:
    p, q = ds.p.rule, ds.q.rule
    if claim == "p":
        return p
    if claim == "p_and_q":
        return {"and": [p, q]} if p and q else None
    raise ValueError(f"no rewording restates {claim}")


def reword_tasks(ds: Dataset, config: dict[str, Any], seg: dict[str, Any], rows: list[str],
                 ext_version: str) -> list[Task]:
    """Five reworded requests per base request of p (four, or three when the filter shows one field) or p_and_q
    (one), same rows and repeats."""
    texts = seg["texts"]
    originals = {"p_para1": ds.p.text, "p_para2": ds.p.text, "pq_para": ds.conj_pq}
    for cond, text in texts.items():
        if text.strip() == originals[cond].strip():
            raise ValueError(f"[ext.reword] {cond} repeats the dataset's own claim text; a rewording must differ")
    base_style = config["run"].get("prompt_style", "lotus")
    fields = list(ds.filter_fields)
    # A filter that shows one field has no other order, so p_order is not built and the design says so
    # (Addendum H: four contract pairs). Designs whose filters show two or more fields are unchanged.
    reversed_fields = fields[::-1] if len(fields) >= 2 else None
    if reversed_fields is None:
        seg["conditions"] = [c for c in seg["conditions"] if c != "p_order"]
        seg["not_built"] = {"p_order": "the filter shows one field, so there is no other order"}
    marker = _marker(seg, ext_version)
    out = []
    for task in _rerun(ds, config, seg, base_style, rows):
        if task.condition not in ("p", "p_and_q"):
            continue
        record = ds.records[task.unit]
        context = serialize(record, fields)
        if task.condition == "p":
            variants = [("p_para1", filter_prompt(context, texts["p_para1"], base_style), base_style, fields),
                        ("p_para2", filter_prompt(context, texts["p_para2"], base_style), base_style, fields),
                        ("p_tmpl", filter_prompt(context, ds.p.text, seg["template_style"]), seg["template_style"],
                         fields)]
            if reversed_fields is not None:
                variants.append(("p_order", filter_prompt(serialize(record, reversed_fields), ds.p.text, base_style),
                                 base_style, reversed_fields))
        else:
            variants = [("pq_para", filter_prompt(context, texts["pq_para"], base_style), base_style, fields)]
        for cond, prompt, style, used_fields in variants:
            claim = REWORD_RESTATES[cond]
            payload = _filter_payload(ds, task.unit, claim, _rule(ds, claim), rewording=cond)
            out.append(_with(task, seg, cond, prompt, payload,
                             {"style": style, "claim": claim, "rewording": cond, "fields": list(used_fields),
                              **marker}))
    order = {c: i for i, c in enumerate(REWORD_CONDITIONS)}
    # One fixed order: by row, then condition, then repeat (the runner shuffles all requests with a seed).
    row_pos = {r: i for i, r in enumerate(rows)}
    out.sort(key=lambda t: (row_pos[t.unit], order[t.condition], t.repeat))
    return out


def unknown_tasks(ds: Dataset, config: dict[str, Any], seg: dict[str, Any], rows: list[str],
                  ext_version: str) -> list[Task]:
    """p, not_p, q and p_and_q in the `unknown` style; the builder's q_and_p and COUNT requests are dropped."""
    marker = {**_marker(seg, ext_version), "template": styles.UNKNOWN_TEMPLATE_VERSION}
    keep = set(seg["conditions"])
    return [_with(t, seg, t.condition, (t.system, t.user), t.payload, marker)
            for t in _rerun(ds, config, seg, styles.UNKNOWN, rows) if t.condition in keep]


def segment_tasks(ds: Dataset, config: dict[str, Any], seg: dict[str, Any], rows: list[str],
                  ext_version: str) -> list[Task]:
    if seg["mitigation"] == "reword":
        return reword_tasks(ds, config, seg, rows, ext_version)
    if seg["mitigation"] == "unknown":
        return unknown_tasks(ds, config, seg, rows, ext_version)
    raise ValueError(f"not an Addendum G segment: {seg['mitigation']}")
