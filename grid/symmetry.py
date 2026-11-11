"""The grid's symmetry units: 200 record pairs per matching dataset, 100 true matches and 100 look-alikes.

A look-alike pair is a record of one product with one of the most similar records of another product,
the hard negatives an entity-matching benchmark evaluates on. On WDC Products and IBM ArgKP the pairs
are drawn from the 40 blocks that already ran (each block holds offers of one product and the most
similar records of others), so their calls are reused; Abt-Buy gets new pairs built the same way.

`tasks` issues each pair in both orders with the harness's own prompt and task format
(semaudit.identities.symmetry), so a pair that already ran is the same request.
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import Any

from semaudit.core import stable_seed
from semaudit.data import jaccard, tokens
from semaudit.identities import _task
from semaudit.prompts import match_prompt, serialize

N_MATCH = 100
N_LOOKALIKE = 100


def from_blocks(ds: Any, blocks: list[list[str]], seed: int) -> dict[str, list[str]]:
    """100 matches and 100 look-alikes among the pairs of blocks that already ran."""
    cluster = {rid: rec.get("cluster") for rid, rec in ds.records.items()}
    matches, lookalikes, group = [], [], {}
    for n, block in enumerate(blocks):
        product = Counter(cluster[r] for r in block).most_common(1)[0][0]
        for a, b in combinations(block, 2):  # the orientation the blocks ran in
            group[f"{a}|{b}"] = f"block{n}"
            if cluster[a] == cluster[b]:
                matches.append(f"{a}|{b}")
            elif product in (cluster[a], cluster[b]):  # one of the product's records and a look-alike
                lookalikes.append(f"{a}|{b}")
    rng = random.Random(stable_seed(seed, "grid-symmetry"))
    if len(matches) < N_MATCH or len(lookalikes) < N_LOOKALIKE:
        raise ValueError(f"{len(matches)} matches and {len(lookalikes)} look-alikes available")
    m = sorted(rng.sample(sorted(matches), N_MATCH))
    lk = sorted(rng.sample(sorted(lookalikes), N_LOOKALIKE))
    # pairs from one block share records and look-alikes: they are one cluster for intervals
    return {"matches": m, "lookalikes": lk, "cluster_of": {pid: group[pid] for pid in m + lk}}


def fresh(ds: Any, seed: int) -> dict[str, list[str]]:
    """100 products: one matched pair each, and one record of the product with its most similar record of
    another product. No record is used twice."""
    rng = random.Random(stable_seed(seed, "grid-symmetry"))
    groups = ds.clusters()
    products = sorted(c for c, ids in groups.items() if len(ids) >= 2)
    rng.shuffle(products)
    toks = {rid: tokens(rec.get("title", "")) for rid, rec in ds.records.items()}
    used: set[str] = set()
    matches, lookalikes, cluster_of = [], [], {}
    for c in products:
        if len(matches) == N_MATCH:
            break
        members = sorted(r for r in groups[c] if r not in used)
        if len(members) < 2:
            continue
        a, b = sorted(rng.sample(members, 2))
        others = [r for r in ds.records if r not in used and r not in (a, b)
                  and ds.records[r].get("cluster") != c]
        x = max(others, key=lambda r: (jaccard(toks[a], toks[r]), r))
        used.update((a, b, x))
        matches.append(f"{a}|{b}")
        lookalikes.append("|".join(sorted((a, x))))
        cluster_of[matches[-1]] = cluster_of[lookalikes[-1]] = f"product{c}"  # the two pairs share record a
    if len(matches) < N_MATCH:
        raise ValueError(f"only {len(matches)} products available")
    return {"matches": sorted(matches), "lookalikes": sorted(lookalikes), "cluster_of": cluster_of}


def design(unit: dict[str, Any]) -> dict[str, Any]:
    """The symmetry part of a run's design, in the form semaudit's analysis reads: each pair is its own
    two-record block (no triangles: transitivity is not part of the grid), and `cluster_of` groups the pairs
    that share records for the intervals."""
    sym = unit["symmetry"]
    return {"pairs": sym["pairs"], "blocks": [pid.split("|") for pid in sym["pairs"]],
            "matches": sym["matches"], "lookalikes": sym["lookalikes"], "cluster_of": sym["cluster_of"]}


def tasks(ds: Any, pairs: list[str], repeats: int = 5, style: str = "lotus") -> list:
    """match(a, b) and match(b, a) for each pair, exactly as semaudit.identities.symmetry issues them."""
    out = []
    for pid in pairs:
        a, b = pid.split("|")
        for condition, (left, right) in (("ab", (a, b)), ("ba", (b, a))):
            prompt = match_prompt(serialize(ds.records[left], ds.text_fields),
                                  serialize(ds.records[right], ds.text_fields), ds.match_claim, style)
            payload = {"left": ds.records[left], "right": ds.records[right], "left_id": left, "right_id": right}
            for k in range(repeats):
                out.append(_task("symmetry", condition, pid, k, "match", [pid], prompt, payload,
                                 {"operand_order": [left, right]}, style=style))
    return out
