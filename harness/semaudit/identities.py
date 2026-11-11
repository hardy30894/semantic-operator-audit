"""Metamorphic identities: which requests to issue, and which outputs must agree.

Each builder samples its analysis units, emits `repeats` identical requests per (condition, unit),
and records what it sampled in the design. `comparisons` later pairs the outputs of two conditions
that a function would have to agree on; the statistics decide whether they disagree by more than
the operator disagrees with itself.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from .core import INVALID, Task, stable_seed
from .data import Dataset, eval_rule, sample_blocks, sample_rows, sample_tuples
from .prompts import (TEMPLATE_VERSION, batch_prompt, count_prompt, filter_prompt, map_prompt,
                      match_prompt, parse_with_rules, serialize)

# (identity, condition, item) -> {repeat: parsed output}
Outputs = dict[tuple[str, str, str], dict[int, Any]]


def _task(identity: str, condition: str, unit: str, repeat: int, op: str, items: list[str],
          prompt: tuple[str, str], payload: dict[str, Any], context: dict[str, Any],
          stage: int = 1, style: str = "lotus") -> Task:
    system, user = prompt
    return Task(
        task_id=f"{identity}:{condition}:{unit}:{repeat}",
        identity=identity, condition=condition, unit=unit, repeat=repeat, op=op,
        items=tuple(items), system=system, user=user, payload=payload,
        context={"template": TEMPLATE_VERSION, "style": style, "op": op, **context}, stage=stage,
    )


def _claims(ds: Dataset) -> dict[str, tuple[str, dict[str, Any] | None]]:
    p, q = ds.p.rule, ds.q.rule
    both = {"and": [p, q]} if p and q else None
    return {
        "p": (ds.p.text, p),
        "not_p": (ds.p.negation, {"not": p} if p else None),
        "q": (ds.q.text, q),
        "p_and_q": (ds.conj_pq, both),
        "q_and_p": (ds.conj_qp, both),
    }


def _filter_payload(ds: Dataset, rid: str, claim: str, rule: dict[str, Any] | None,
                    **extra: Any) -> dict[str, Any]:
    return {"record": ds.records[rid], "record_id": rid, "claim": claim, "rule": rule,
            "rules": {"p": ds.p.rule, "q": ds.q.rule}, **extra}


def _batch_payload(ds: Dataset, members: list[str]) -> dict[str, Any]:
    return {"records": [ds.records[r] for r in members], "record_ids": members, "rule": ds.p.rule,
            "rules": {"p": ds.p.rule, "q": ds.q.rule}}


def symmetry(ds: Dataset, cfg: dict[str, Any], repeats: int, rng: random.Random):
    """match(a, b) vs match(b, a) for every pair inside each block; blocks also yield triangles."""
    style = cfg.get("style", "lotus")
    blocks = sample_blocks(ds, cfg["n_blocks"], cfg["block_size"], cfg.get("positives", 3), rng)
    pairs = [(a, b) for block in blocks for a, b in combinations(block, 2)]
    tasks = []
    for a, b in pairs:
        pid = f"{a}|{b}"
        for condition, (left, right) in (("ab", (a, b)), ("ba", (b, a))):
            prompt = match_prompt(serialize(ds.records[left], ds.text_fields),
                                  serialize(ds.records[right], ds.text_fields), ds.match_claim, style)
            payload = {"left": ds.records[left], "right": ds.records[right],
                       "left_id": left, "right_id": right}
            for k in range(repeats):
                tasks.append(_task("symmetry", condition, pid, k, "match", [pid], prompt, payload,
                                   {"operand_order": [left, right]}, style=style))
    return tasks, {"blocks": blocks, "pairs": [f"{a}|{b}" for a, b in pairs]}


def filter_algebra(ds: Dataset, cfg: dict[str, Any], repeats: int, rng: random.Random):
    """p, not-p, q and both conjunctions on the same rows and context, plus batch COUNTs of p."""
    style = cfg.get("style", "lotus")
    rows = sample_rows(ds, cfg["n_rows"], rng)
    fields = ds.filter_fields
    tasks = []
    for rid in rows:
        context = serialize(ds.records[rid], fields)
        for claim, (text, rule) in _claims(ds).items():
            prompt = filter_prompt(context, text, style)
            payload = _filter_payload(ds, rid, claim, rule)
            for k in range(repeats):
                tasks.append(_task("filter_algebra", claim, rid, k, "filter", [rid], prompt, payload,
                                   {"claim": claim, "fields": fields}, style=style))
    size = cfg.get("count_batch", 10)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    batches = {f"count{i:03d}": shuffled[i * size:(i + 1) * size] for i in range(len(shuffled) // size)}
    for bid, members in batches.items():
        prompt = count_prompt([serialize(ds.records[r], fields) for r in members], ds.p.text, style)
        for k in range(repeats):
            tasks.append(_task("filter_algebra", "count", bid, k, "count", [bid], prompt,
                               _batch_payload(ds, members), {"batch": members}, style=style))
    return tasks, {"rows": rows, "count_batches": batches}


def placement(ds: Dataset, cfg: dict[str, Any], repeats: int, rng: random.Random):
    """A filter on r before a join vs the same filter on the joined tuple (r, s).

    With referenced-column serialization and unchanged column names the two requests are byte-
    identical, so that rewrite cannot change answers beyond duplicate-request noise. The conditions
    here change the context: renamed columns (joined_names) and the full joined row (joined_full).
    """
    style = cfg.get("style", "lotus")
    tuples = sample_tuples(ds, cfg["n_tuples"], rng)
    fields = ds.filter_fields
    tasks = []
    for r, s in tuples:
        tid = f"{r}|{s}"
        rec_r, rec_s = ds.records[r], ds.records[s]
        contexts = {
            "r_only": serialize(rec_r, fields),
            "joined_names": serialize(rec_r, fields, prefix="left."),
            "joined_full": serialize(rec_r, ds.text_fields, prefix="left.") + "\n"
                           + serialize(rec_s, ds.text_fields, prefix="right."),
        }
        for condition, context in contexts.items():
            prompt = filter_prompt(context, ds.p.text, style)
            payload = _filter_payload(ds, r, "p", ds.p.rule, serialization=condition)
            for k in range(repeats):
                tasks.append(_task("placement", condition, tid, k, "filter", [tid], prompt, payload,
                                   {"plan": condition, "join_partner": s}, style=style))
    return tasks, {"tuples": [f"{r}|{s}" for r, s in tuples]}


def batching(ds: Dataset, cfg: dict[str, Any], repeats: int, rng: random.Random):
    """Multi-row prompts. N1 and N2 keep every row at the same position with all-new neighbours;
    rot keeps the neighbours and moves every row one position; single is one row per prompt."""
    style = cfg.get("style", "lotus")
    size = cfg.get("batch_size", 10)
    n_batches = cfg["n_rows"] // size
    if n_batches < size:
        raise ValueError("batch identity needs n_rows >= batch_size**2 so every row gets new neighbours")
    rows = sample_rows(ds, n_batches * size, rng)
    grid = [rows[b * size:(b + 1) * size] for b in range(n_batches)]
    partitions = {
        "N1": {f"N1:{b:03d}": grid[b][:] for b in range(n_batches)},
        "N2": {f"N2:{b:03d}": [grid[(b + j) % n_batches][j] for j in range(size)]
               for b in range(n_batches)},
        "rot": {f"rot:{b:03d}": grid[b][-1:] + grid[b][:-1] for b in range(n_batches)},
    }
    fields = ds.filter_fields
    tasks = []
    for condition, batches in partitions.items():
        for bid, members in batches.items():
            prompt = batch_prompt([serialize(ds.records[r], fields) for r in members], ds.p.text, style)
            for k in range(repeats):
                tasks.append(_task("batch", condition, bid, k, "filter_batch", members, prompt,
                                   _batch_payload(ds, members), {"batch_id": bid, "batch": members}, style=style))
    for rid in rows:
        prompt = filter_prompt(serialize(ds.records[rid], fields), ds.p.text, style)
        payload = _filter_payload(ds, rid, "p", ds.p.rule)
        for k in range(repeats):
            tasks.append(_task("batch", "single", rid, k, "filter", [rid], prompt, payload, {}, style=style))
    return tasks, {"rows": rows, "partitions": partitions}


def idempotence(ds: Dataset, cfg: dict[str, Any], repeats: int, rng: random.Random):
    """map(x) vs map(map(x)), with the two conditions drawn independently and interleaved in time.

    Stage 0 draws `repeats` outputs of map(x) that serve only as inputs ("x_in"). The main phase draws
    another `repeats` outputs of map(x) (condition A, "x") and map(y) for each stage-0 output y
    (condition B, "fx", built by `idempotence_followups`). B never reuses an A draw: a map that is
    idempotent but noisy would otherwise show a spurious negative excess.
    """
    style = cfg.get("style", "lotus")
    rows = sample_rows(ds, cfg["n_rows"], rng)
    spec = ds.map_spec
    tasks = []
    for rid in rows:
        value = ds.records[rid][spec.field]
        prompt = map_prompt(spec.instruction, value, style)
        for k in range(repeats):
            tasks.append(_task("idempotence", "x", rid, k, "map", [rid], prompt, {"value": value},
                               {"input": "original"}, style=style))
            tasks.append(_task("idempotence", "x_in", rid, k, "map", [rid], prompt, {"value": value},
                               {"input": "original", "role": "input to map(map(x))"}, stage=0, style=style))
    return tasks, {"rows": rows}


def idempotence_followups(ds: Dataset, stage0: Iterable[dict[str, Any]]) -> list[Task]:
    """Condition B: map(y) for every stage-0 output y, run in the main phase."""
    tasks = []
    for rec in stage0:
        if rec["identity"] != "idempotence" or rec["condition"] != "x_in" or rec["outputs"][0] == INVALID:
            continue
        if other_host(rec):  # an input that analysis would treat as invalid
            continue
        value = rec["outputs"][0]
        style = (rec.get("context") or {}).get("style", "lotus")
        prompt = map_prompt(ds.map_spec.instruction, value, style)
        tasks.append(_task("idempotence", "fx", rec["unit"], rec["repeat"], "map", [rec["unit"]], prompt,
                           {"value": value}, {"input": "stage-0 output", "parent": rec["task_id"]},
                           style=style))
    return tasks


BUILDERS = {
    "symmetry": symmetry,
    "filter_algebra": filter_algebra,
    "placement": placement,
    "batch": batching,
    "idempotence": idempotence,
}


def build(ds: Dataset, cfg: dict[str, Any], repeats: int, seed: int,
          style: str = "lotus") -> tuple[list[Task], dict[str, Any]]:
    """Stage-1 tasks for every enabled identity, plus the design that analysis needs."""
    tasks: list[Task] = []
    design: dict[str, Any] = {"dataset": ds.name, "repeats": repeats, "style": style}
    for name, builder in BUILDERS.items():
        if not cfg.get(name, {}).get("enabled", False):
            continue
        # Each identity samples from its own stream, so enabling one never changes another's units.
        rng = random.Random(stable_seed(seed, name))
        new, units = builder(ds, {**cfg[name], "style": style}, repeats, rng)
        tasks += new
        design[name] = units
    design["gold"] = {rid: rec["cluster"] for rid, rec in ds.records.items() if rec.get("cluster")}
    if ds.p.rule and ds.q.rule:
        design["truth"] = {
            claim: {rid: eval_rule(rule, rec) for rid, rec in ds.records.items()}
            for claim, (_, rule) in _claims(ds).items()
        }
    return tasks, design


# --- pairing outputs ----------------------------------------------------------------------------

@dataclass
class Comparison:
    name: str
    hypothesis: str
    primary: bool | None  # True: primary family; False: secondary family; None: exploratory, no Holm
    kind: str  # categorical | numeric
    a: str
    b: str
    draws: dict[str, tuple[list[Any], list[Any]]]
    clusters: dict[str, Any] | None = None  # items that share records or calls, for the bootstrap
    # Items answered by the same calls (rows of one batched prompt): per item, the group of calls
    # behind condition A's draws and, if different, the group behind condition B's. The test then
    # guards against within-call dependence (stats.compare).
    call_clusters: dict[str, tuple[Any, ...]] | None = None


# OpenRouter reports the serving host by display name. Its endpoint list (GET /api/v1/models/{model}/
# endpoints, 2026-10-13) maps the names of the hosts the study pins to their slugs; note that Google
# Vertex is reported as "Google". Other names fall back to a normalized slug.
HOST_SLUGS = {"Anthropic": "anthropic", "OpenAI": "openai", "Google AI Studio": "google-ai-studio",
              "Google": "google-vertex", "Google Vertex": "google-vertex", "Alibaba": "alibaba"}


def _host_slug(name: str) -> str:
    """OpenRouter's display name of a host as its slug: "Google AI Studio" -> "google-ai-studio",
    "Google (Europe)" -> "google-vertex". A parenthetical (region, tier) is ignored."""
    base = re.sub(r"\s*\(.*?\)", "", name).strip()
    return HOST_SLUGS.get(base, base.lower().replace(" ", "-"))


def other_host(rec: dict[str, Any]) -> bool:
    """True if the call was pinned to one host but OpenRouter reports that another one served it."""
    order = ((rec.get("params") or {}).get("provider") or {}).get("order") or []
    served = (rec.get("extra") or {}).get("served_provider")
    if not order or not served:
        return False
    return _host_slug(str(served)) != str(order[0]).split("/")[0].lower()


def parse_record(rec: dict[str, Any]) -> tuple[list[Any], list[str]]:
    """Outputs and parse rules under the current parser, from the logged raw text. Failed calls, and
    calls served by a host other than the pinned one, are INVALID."""
    n = len(rec["items"])
    if rec.get("error") or rec.get("text") is None:
        return [INVALID] * n, ["error"] * n
    if other_host(rec):
        return [INVALID] * n, ["other host"] * n
    context = rec.get("context") or {}
    count_max = len(context["batch"]) if rec["op"] == "count" and context.get("batch") else None
    return parse_with_rules(rec["op"], rec["text"], n, context.get("style", "lotus"), rec.get("stop_reason"),
                            count_max)


def reparse(rec: dict[str, Any]) -> list[Any]:
    """Outputs under the current parser (see parse_record)."""
    return parse_record(rec)[0]


def collect(records: Iterable[dict[str, Any]]) -> Outputs:
    """Latest output per (identity, condition, item, repeat), re-parsed from the logged text with the
    current parser. A failed call never overwrites a success."""
    out: Outputs = {}
    for rec in records:
        for item, value in zip(rec["items"], reparse(rec)):
            slot = out.setdefault((rec["identity"], rec["condition"], item), {})
            if rec.get("error") and rec["repeat"] in slot:
                continue
            slot[rec["repeat"]] = value
    return out


def draws(out: Outputs, identity: str, condition: str, item: str, repeats: int) -> list[Any]:
    slot = out.get((identity, condition, item), {})
    return [slot.get(k, INVALID) for k in range(repeats)]


def _not(v: Any) -> Any:
    return INVALID if v == INVALID else (not v)


def _and(x: Any, y: Any) -> Any:
    return INVALID if INVALID in (x, y) else (x and y)


def _count(values: list[Any]) -> Any:
    return INVALID if INVALID in values else sum(bool(v) for v in values)


def _text(v: Any) -> Any:
    return v if v == INVALID else " ".join(str(v).split())


def _content(v: Any) -> Any:
    """Order-, case- and punctuation-insensitive content: the sorted letter and digit runs."""
    return v if v == INVALID else " ".join(sorted(re.findall(r"[a-z]+|[0-9]+", str(v).lower())))


def comparisons(out: Outputs, design: dict[str, Any]) -> list[Comparison]:
    k = design["repeats"]

    def d(identity: str, condition: str, item: str) -> list[Any]:
        return draws(out, identity, condition, item, k)

    comps: list[Comparison] = []
    if "symmetry" in design:
        block_of = {r: n for n, block in enumerate(design["symmetry"]["blocks"]) for r in block}
        comps.append(Comparison(
            "symmetry", "H1", True, "categorical", "match(a,b)", "match(b,a)",
            {pid: (d("symmetry", "ab", pid), d("symmetry", "ba", pid)) for pid in design["symmetry"]["pairs"]},
            clusters={pid: f"block{block_of[pid.split('|')[0]]}" for pid in design["symmetry"]["pairs"]}))
    if "filter_algebra" in design:
        fa, rows = "filter_algebra", design["filter_algebra"]["rows"]
        comps.append(Comparison(
            "negation", "H2", True, "categorical", "F(p)", "NOT F(not p)",
            {r: (d(fa, "p", r), [_not(v) for v in d(fa, "not_p", r)]) for r in rows}))
        comps.append(Comparison(
            "fusion", "H3", True, "categorical", "F(p and q)", "F(p) AND F(q)",
            {r: (d(fa, "p_and_q", r), [_and(x, y) for x, y in zip(d(fa, "p", r), d(fa, "q", r))])
             for r in rows}))
        comps.append(Comparison(
            "count", "H7", True, "numeric", "LLM COUNT over a batch", "SUM of F(p) over the batch",
            {bid: (d(fa, "count", bid), [_count([d(fa, "p", r)[i] for r in members]) for i in range(k)])
             for bid, members in design["filter_algebra"]["count_batches"].items()}))
        comps.append(Comparison(
            "conjunct_order", "S3", False, "categorical", "F(p and q)", "F(q and p)",
            {r: (d(fa, "p_and_q", r), d(fa, "q_and_p", r)) for r in rows}))
    if "batch" in design:
        rows = design["batch"]["rows"]
        parts = design["batch"]["partitions"]
        n1_batch = {r: bid for bid, members in parts["N1"].items() for r in members}
        n2_batch = {r: bid for bid, members in parts["N2"].items() for r in members}
        # rot batch b holds the same rows as N1 batch b, so one grouping covers both conditions of S2.
        comps.append(Comparison(
            "batch_neighbours", "H4", True, "categorical", "batch N1", "batch N2 (same position)",
            {r: (d("batch", "N1", r), d("batch", "N2", r)) for r in rows}, clusters=n1_batch,
            call_clusters={r: (n1_batch[r], n2_batch[r]) for r in rows}))
        comps.append(Comparison(
            "batch_position", "S2", False, "categorical", "batch N1", "batch rot (same neighbours)",
            {r: (d("batch", "N1", r), d("batch", "rot", r)) for r in rows}, clusters=n1_batch,
            call_clusters={r: (n1_batch[r],) for r in rows}))
        comps.append(Comparison(
            "batch_vs_single", "S4", False, "categorical", "single-row prompt", "batch N1",
            {r: (d("batch", "single", r), d("batch", "N1", r)) for r in rows}, clusters=n1_batch,
            call_clusters={r: (n1_batch[r],) for r in rows}))
    if "placement" in design:
        tuples = design["placement"]["tuples"]
        comps.append(Comparison(
            "placement_full", "H5", True, "categorical", "F(r)", "F(r JOIN s), full joined row",
            {t: (d("placement", "r_only", t), d("placement", "joined_full", t)) for t in tuples}))
        comps.append(Comparison(
            "placement_names", "S5", False, "categorical", "F(r)", "F(r JOIN s), renamed columns",
            {t: (d("placement", "r_only", t), d("placement", "joined_names", t)) for t in tuples}))
    if "idempotence" in design:
        comps.append(Comparison(
            "idempotence", "H6", True, "categorical", "map(x)", "map(map(x))",
            {r: ([_text(v) for v in d("idempotence", "x", r)], [_text(v) for v in d("idempotence", "fx", r)])
             for r in design["idempotence"]["rows"]}))
        comps.append(Comparison(
            "idempotence_content", "X1", None, "categorical", "content of map(x)", "content of map(map(x))",
            {r: ([_content(v) for v in d("idempotence", "x", r)], [_content(v) for v in d("idempotence", "fx", r)])
             for r in design["idempotence"]["rows"]}))
        # Identical requests issued in stage 0 (H6's inputs) and in the main phase: a within-run drift
        # check on which H6's condition B depends.
        comps.append(Comparison(
            "idempotence_phase", "X2", None, "categorical", "map(x), stage 0", "map(x), main phase",
            {r: ([_text(v) for v in d("idempotence", "x_in", r)], [_text(v) for v in d("idempotence", "x", r)])
             for r in design["idempotence"]["rows"]}))
    return comps
