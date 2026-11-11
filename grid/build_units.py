"""Curate the consistent grid's units: 200 per test on 10 predicate pairs, plus the matching datasets.

Every unit comes from the pair's frozen seed and the harness's own builders (semaudit.identities), so a
unit that already ran is the same unit here. Writes, under grid/:
  units/<pair>.json        filter rows, COUNT batches, batching batches, idempotence values, rewordings
  units/match-<ds>.json    symmetry blocks and placement tuples (WDC, Abt-Buy, ArgKP)
  rows/<pair>-<test>.csv   the rows themselves (id, the fields the prompt shows, labels where they exist)
  UNITS.md                 counts, and checks against the designs that already ran

Makes no model calls. Run from the project root:  python3 grid/build_units.py
WDC Products and Abt-Buy records are not redistributable; rows/ stays local (not in the release).
"""
from __future__ import annotations

import csv
import json
import random
import sys
import tomllib
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"
sys.path.insert(0, str(H))
from semaudit.core import stable_seed  # noqa: E402
from semaudit.data import load_dataset_dir  # noqa: E402
from semaudit.identities import build  # noqa: E402

sys.path.insert(0, str(ROOT / "grid"))
import symmetry  # noqa: E402

PAIRS = ["wdc-A", "wdc-B", "abt-A", "abt-B", "cuad-A", "cuad-B", "argq-A", "argq-B", "arxiv-A", "arxiv-B", "argkp-A"]
N = 200            # units per test
BATCH = 10         # rows per batched prompt and per COUNT batch
OUT = ROOT / "grid"


def seed_of(config: str) -> int:
    return tomllib.loads((H / "configs" / config).read_text(encoding="utf-8"))["run"]["seed"]


def old_design(run: str) -> dict:
    p = H / "runs" / run / "design.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def reword_conditions(pair: str, ds) -> list[str]:
    own = OUT / "configs" / f"ext-placebo-{pair}.toml"  # ArgKP's is the grid's own
    path = own if own.exists() else H / "configs" / f"ext-placebo-{pair}.toml"
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))["ext"]["reword"]
    conds = ["p_para1", "p_para2", "p_tmpl", "pq_para"]
    if cfg.get("field_order") and len(ds.filter_fields) > 1:  # a one-field prompt has no order to reverse
        conds.append("p_order")
    return conds


def write_rows(path: Path, ds, ids: list[str], truth: dict | None) -> None:
    fields = ds.filter_fields
    claims = sorted(truth) if truth else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + fields + [f"truth_{c}" for c in claims])
        for rid in ids:
            rec = ds.records[rid]
            w.writerow([rid] + [rec.get(f, "") for f in fields] + [truth[c].get(rid) for c in claims])


def self_join_tuples(ds, n: int, rng: random.Random) -> list[list[str]]:
    """sample_tuples with every record its own source: two offers of one product, at most one tuple per
    product (WDC Products records carry no shop, so cross-source tuples do not exist there)."""
    pairs = []
    for _, ids in sorted(ds.clusters().items()):
        cross = list(combinations(sorted(ids), 2))
        if cross:
            pairs.append(rng.choice(cross))
    rng.shuffle(pairs)
    if len(pairs) < n:
        raise ValueError(f"only {len(pairs)} tuples; {n} requested")
    return ["|".join(t) for t in pairs[:n]]


def main() -> int:
    (OUT / "units").mkdir(parents=True, exist_ok=True)
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    lines = ["# Grid units", "",
             f"Built by `grid/build_units.py` from each pair's frozen seed with the harness's own builders. "
             f"{N} units per test; COUNT and batched prompts hold {BATCH} rows.", "",
             "| Pair | Filter rows | COUNT batches | Batching rows analysed (prompts: N1, rot, N2) | Idempotence values | Rewordings | Checks against what already ran |",
             "|---|---|---|---|---|---|---|"]
    for pair in PAIRS:
        seed = seed_of(f"study-{pair}.toml")
        ds = load_dataset_dir(H / ".." / "data" / pair)
        old = old_design(f"study-{pair}")
        cfg = {"filter_algebra": {"enabled": True, "n_rows": N, "count_batch": BATCH},
               # batching keeps the frozen 30-batch grid; the analysed rows are those of its first 20 batches
               "batch": {"enabled": True, "n_rows": 300, "batch_size": BATCH}}
        if pair.endswith("-A"):
            cfg["idempotence"] = {"enabled": True, "n_rows": N}
        _, d = build(ds, cfg, 5, seed, "lotus")
        rows = d["filter_algebra"]["rows"]
        counts = d["filter_algebra"]["count_batches"]
        grid_rows = d["batch"]["rows"]
        analysed = grid_rows[:N]
        parts = d["batch"]["partitions"]
        aset = set(analysed)
        n1 = {k: v for k, v in parts["N1"].items() if set(v) <= aset}
        rot = {k: v for k, v in parts["rot"].items() if set(v) <= aset}
        n2 = {k: v for k, v in parts["N2"].items() if aset & set(v)}
        conds = reword_conditions(pair, ds)
        unit = {"pair": pair, "seed": seed, "dataset": ds.name, "filter_fields": ds.filter_fields,
                "filter": {"rows": rows, "count_batches": counts},
                "batch": {"grid_rows": grid_rows, "analysed_rows": analysed,
                          "prompts": {"N1": n1, "rot": rot, "N2": n2}},
                "rewording": {"rows": rows, "conditions": conds}}
        if "idempotence" in d:
            unit["idempotence"] = {"rows": d["idempotence"]["rows"]}
        (OUT / "units" / f"{pair}.json").write_text(json.dumps(unit, indent=1) + "\n", encoding="utf-8")
        truth = old.get("truth")
        write_rows(OUT / "rows" / f"{pair}-filter.csv", ds, rows, truth)
        write_rows(OUT / "rows" / f"{pair}-batching.csv", ds, analysed, truth)
        if "idempotence" in unit:
            write_rows(OUT / "rows" / f"{pair}-idempotence.csv", ds, unit["idempotence"]["rows"], None)

        checks = []
        of = old.get("filter_algebra", {})
        if not of:
            checks.append("filter tests not run before")
        else:
            checks.append("filter rows = first 200 that ran" if of.get("rows", [])[:N] == rows else "FILTER ROWS DIFFER")
        oc = {tuple(v) for v in of.get("count_batches", {}).values()}
        same_c = sum(tuple(v) in oc for v in counts.values())
        checks.append(f"COUNT batches that ran: {same_c}/{len(counts)}")
        ob = old.get("batch")
        if ob:
            same = all({tuple(v) for v in parts[p].values()} == {tuple(v) for v in ob["partitions"][p].values()}
                       for p in ("N1", "N2", "rot")) and ob["rows"] == grid_rows
            checks.append("batching grid = the one that ran" if same else "BATCH GRID DIFFERS")
        else:
            checks.append("batching not run before")
        if "idempotence" in unit:
            oi = old.get("idempotence", {}).get("rows")
            checks.append("idempotence not run before" if oi is None else
                          "idempotence = what ran" if oi == unit["idempotence"]["rows"] else "IDEMPOTENCE DIFFERS")
        g = old_design(f"ext-placebo-{pair}").get("filter_algebra", {}).get("rows")
        checks.append("rewordings not run before" if g is None else
                      "rewording rows = what ran" if g == rows else "REWORDING ROWS DIFFER")
        lines.append(f"| {pair} | {len(rows)} | {len(counts)} | {len(analysed)} ({len(n1)}, {len(rot)}, {len(n2)}) | "
                     f"{len(unit.get('idempotence', {}).get('rows', [])) or '-'} | {', '.join(conds)} | {'; '.join(checks)} |")

    # matching datasets
    lines += ["", "| Matching dataset | Symmetry | Placement tuples | Checks |", "|---|---|---|---|"]
    match = [("wdc", "wdc-A", "study-wdc-A.toml", "study-wdc-A"),
             ("abt", "abt-A", "study-abt-A.toml", "study-abt-A"),
             ("argkp", "argkp-A", "study-argkp-A.toml", "study-argkp-A")]
    for name, pair, config, run in match:
        seed = seed_of(config)
        ds = load_dataset_dir(H / ".." / "data" / pair)
        old = old_design(run)
        n_tuples = 120 if name == "argkp" else N
        cfg = {} if name == "wdc" else {"placement": {"enabled": True, "n_tuples": n_tuples}}
        d = build(ds, cfg, 5, seed, "lotus")[1] if cfg else {}
        if name == "wdc":
            tuples = self_join_tuples(ds, n_tuples, random.Random(stable_seed(seed, "placement")))
        else:
            tuples = list(d["placement"]["tuples"])  # "r|s", the harness's own format
        sym = symmetry.from_blocks(ds, old["symmetry"]["blocks"], seed) if "symmetry" in old \
            else symmetry.fresh(ds, seed)
        pairs = sym["matches"] + sym["lookalikes"]
        unit = {"dataset": name, "pair_config": pair, "seed": seed,
                "symmetry": {"pairs": pairs, "matches": sym["matches"], "lookalikes": sym["lookalikes"],
                             "cluster_of": sym["cluster_of"],
                             "source": "pairs of the 40 blocks that ran" if "symmetry" in old else "new"},
                "placement": {"tuples": tuples,
                              "kind": "self-join of two offers of one product" if name == "wdc" else
                                      ("Abt record joined with its matching Buy record" if name == "abt" else
                                       "self-join of two arguments of one key point")}}
        (OUT / "units" / f"match-{name}.json").write_text(json.dumps(unit, indent=1) + "\n", encoding="utf-8")
        checks = []
        if "symmetry" in old:
            checks.append("symmetry pairs all among the pairs that ran" if set(pairs) <= set(old["symmetry"]["pairs"])
                          else "SYMMETRY PAIRS NOT AMONG THOSE THAT RAN")
        else:
            checks.append("symmetry not run before")
        if "placement" in old:
            checks.append("placement = what ran" if list(old["placement"]["tuples"]) == tuples
                          else "PLACEMENT DIFFERS")
        else:
            checks.append("placement not run before")
        lines.append(f"| {name} | {len(sym['matches'])} matches + {len(sym['lookalikes'])} look-alikes = {len(pairs)} "
                     f"record pairs | {len(tuples)} | {'; '.join(checks)} |")
    lines += ["", "ArgKP placement holds 120 tuples: one per key point, and only 122 key points have two arguments.",
              "", "argkp-A was added as the 11th condition pair after the first grid runs: ArgKP's stance condition (p, "
              "word for word argq-A's p) and its length condition (q, argq-D's p), with the rewording config in "
              "grid/configs/ext-placebo-argkp-A.toml. Its filter tests, batching, COUNT, idempotence and rewordings had "
              "not run before."]
    (OUT / "UNITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
