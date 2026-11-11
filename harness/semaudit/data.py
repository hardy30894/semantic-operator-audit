"""Datasets, predicates, and sampling of analysis units."""

from __future__ import annotations

import csv
import random
import re
import tomllib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

Record = dict[str, str]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Predicate:
    text: str
    negation: str
    fields: tuple[str, ...]
    rule: dict[str, Any] | None = None  # machine-checkable truth, when the data has the attributes


@dataclass(frozen=True)
class MapSpec:
    field: str
    instruction: str


@dataclass
class Dataset:
    name: str
    records: dict[str, Record]
    text_fields: list[str]
    p: Predicate
    q: Predicate
    conj_pq: str
    conj_qp: str
    match_claim: str
    map_spec: MapSpec

    @property
    def filter_fields(self) -> list[str]:
        """Columns every filter prompt shows: the union of what p and q reference.

        Using one context for p, not-p, q and both conjunctions means those prompts differ only in
        the claim, so fusion and negation comparisons are not confounded by different columns.
        """
        wanted = set(self.p.fields) | set(self.q.fields)
        return [f for f in self.text_fields if f in wanted]

    def clusters(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for rid, rec in self.records.items():
            if rec.get("cluster"):
                groups.setdefault(rec["cluster"], []).append(rid)
        return groups


def _number(value: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(m.group()) if m else float("nan")


def eval_rule(rule: dict[str, Any], record: Record) -> bool:
    """Evaluate a predicate rule: and / or / not over field tests."""
    if "and" in rule:
        return all(eval_rule(r, record) for r in rule["and"])
    if "or" in rule:
        return any(eval_rule(r, record) for r in rule["or"])
    if "not" in rule:
        return not eval_rule(rule["not"], record)
    value = record.get(rule["field"], "")
    if "equals" in rule:
        return value.lower() == str(rule["equals"]).lower()
    if "in" in rule:
        return value.lower() in {str(v).lower() for v in rule["in"]}
    if "contains_any" in rule:
        low = value.lower()
        return any(str(v).lower() in low for v in rule["contains_any"])
    if "gt" in rule:
        return _number(value) > float(rule["gt"])
    if "lt" in rule:
        return _number(value) < float(rule["lt"])
    raise ValueError(f"unknown rule: {rule}")


# --- synthetic product offers -------------------------------------------------------------------

# category: (display name, brands, price range, specifications)
_CATEGORIES: dict[str, tuple[str, list[str], tuple[int, int], list[str]]] = {
    "headphones": ("Headphones", ["Sony", "Bose", "JBL", "Sennheiser"], (49, 399),
                   ["Wireless", "Noise Cancelling", "Over-Ear", "In-Ear"]),
    "speaker": ("Speaker", ["Sony", "Bose", "JBL", "Marshall"], (39, 499),
                ["Portable", "Bluetooth", "Waterproof", "Smart"]),
    "camera": ("Mirrorless Camera", ["Canon", "Nikon", "Sony", "Fujifilm"], (449, 2499),
               ["24MP", "33MP", "45MP", "4K Video"]),
    "laptop": ("Laptop", ["Dell", "HP", "Lenovo", "Asus"], (499, 2299),
               ["16GB RAM", "8GB RAM", "512GB SSD", "1TB SSD"]),
    "monitor": ("Monitor", ["Dell", "LG", "Samsung", "Asus"], (129, 999),
                ["27-inch", "24-inch", "32-inch", "4K UHD"]),
    "mouse": ("Mouse", ["Logitech", "Razer", "Microsoft", "Corsair"], (15, 149),
              ["Wireless", "Ergonomic", "Gaming", "Bluetooth"]),
    "keyboard": ("Keyboard", ["Logitech", "Corsair", "Razer", "Keychron"], (25, 229),
                 ["Mechanical", "Wireless", "Compact", "Backlit"]),
    "tablet": ("Tablet", ["Apple", "Samsung", "Lenovo", "Amazon"], (99, 1099),
               ["64GB", "128GB", "256GB", "Wi-Fi"]),
}
_COLORS = ["Black", "White", "Silver", "Blue", "Red", "Gray"]
_MARKETING = ["New", "Genuine", "Latest", "Brand New", "Official"]
_SOURCES = ["shop_a", "shop_b", "shop_c", "shop_d"]
_LETTERS = "ABCDEFGHJKLMNPRSTUVWXZ"


def _model_code(rng: random.Random) -> str:
    suffix = rng.choice(["", "X", "S", "M"])
    return f"{rng.choice(_LETTERS)}{rng.choice(_LETTERS)}-{rng.randint(100, 999)}{suffix}"


def _sibling(ent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """A near-duplicate that is a different product: exactly one identifying attribute differs."""
    sib = dict(ent)
    specs = _CATEGORIES[ent["category"]][3]
    change = rng.choice(["model", "spec", "color"])
    if change == "model":
        head, tail = ent["model"].split("-")
        digits = re.match(r"\d+", tail).group()
        sib["model"] = f"{head}-{int(digits) + rng.choice([1, 10, 100])}{tail[len(digits):]}"
    elif change == "spec":
        sib["spec"] = rng.choice([s for s in specs if s != ent["spec"]])
    else:
        sib["color"] = rng.choice([c for c in _COLORS if c != ent["color"]])
    sib["price"] = ent["price"] * rng.uniform(0.85, 1.15)
    return sib


def _offer(ent: dict[str, Any], cluster: str, source: str, rng: random.Random) -> Record:
    display = _CATEGORIES[ent["category"]][0]
    model = ent["model"]
    if rng.random() < 0.3:
        model = model.replace("-", "")
    if rng.random() < 0.2:
        model = model.lower()
    title = rng.choice([
        "{brand} {model} {spec} {display} - {color}",
        "{brand} {display} {model} ({spec}, {color})",
        "{marketing} {brand} {model} {display}, {color}",
        "{brand_upper} {model} {spec} {display}",
    ]).format(brand=ent["brand"], brand_upper=ent["brand"].upper(), model=model, spec=ent["spec"],
              display=display, color=ent["color"], marketing=rng.choice(_MARKETING))
    description = rng.choice([
        "{brand} {noun} with {spec_l} design. Color: {color}.",
        "The {brand} {model} is a {spec_l} {noun} in {color_l}.",
        "{spec} {noun} from {brand}. Finish: {color}.",
    ]).format(brand=ent["brand"], noun=display.lower(), spec=ent["spec"], spec_l=ent["spec"].lower(),
              model=ent["model"], color=ent["color"], color_l=ent["color"].lower())
    price = ent["price"] * rng.uniform(0.94, 1.06)
    return {
        "title": title, "description": description, "price": f"{price:.2f}",
        "cluster": cluster, "source": source,
        # Hidden attributes: used by predicate rules and the offline backends, never shown to models.
        "brand": ent["brand"], "category": ent["category"], "spec": ent["spec"],
        "color": ent["color"], "model": ent["model"],
    }


SYNTHETIC_P = Predicate(
    text="The product is an audio device (for example headphones or a speaker).",
    negation="The product is not an audio device (for example headphones or a speaker).",
    fields=("title", "description"),
    rule={"field": "category", "in": ["headphones", "speaker"]},
)
SYNTHETIC_Q = Predicate(
    text="The product's price is above $200.",
    negation="The product's price is $200 or less.",
    fields=("title", "price"),
    rule={"field": "price", "gt": 200},
)


def synthetic_dataset(n_entities: int = 240, seed: int = 0, sibling_rate: float = 0.5) -> Dataset:
    """Product offers from several shops, with near-duplicate siblings as hard negatives.

    Used to validate the harness (control and simulated backends) and as a cheap pilot domain.
    """
    rng = random.Random(seed)
    entities: list[dict[str, Any]] = []
    while len(entities) < n_entities:
        category = rng.choice(sorted(_CATEGORIES))
        _, brands, (lo, hi), specs = _CATEGORIES[category]
        ent = {"category": category, "brand": rng.choice(brands), "spec": rng.choice(specs),
               "color": rng.choice(_COLORS), "model": _model_code(rng), "price": rng.uniform(lo, hi)}
        entities.append(ent)
        if len(entities) < n_entities and rng.random() < sibling_rate:
            entities.append(_sibling(ent, rng))
    records: dict[str, Record] = {}
    for idx, ent in enumerate(entities):
        for source in rng.sample(_SOURCES, rng.choice([2, 2, 3, 3, 4])):
            records[f"r{len(records):05d}"] = _offer(ent, f"e{idx:04d}", source, rng)
    return Dataset(
        name=f"synthetic-{seed}",
        records=records,
        text_fields=["title", "description", "price"],
        p=SYNTHETIC_P,
        q=SYNTHETIC_Q,
        conj_pq="The product is an audio device (for example headphones or a speaker), "
                "and its price is above $200.",
        conj_qp="The product's price is above $200, "
                "and the product is an audio device (for example headphones or a speaker).",
        match_claim="Record 1 and Record 2 refer to the same real-world product.",
        map_spec=MapSpec(
            field="title",
            instruction="Rewrite this product title in canonical form: the brand name, then the model "
                        "number, then the main specification, separated by single spaces. Remove "
                        "colors, marketing words and punctuation.",
        ),
    )


def load_dataset_dir(path: Path) -> Dataset:
    """Load `dataset.toml` (predicates etc.) and its records CSV (id, optional cluster/source, fields).

    The CSV defaults to `records.csv` beside the spec; `records = "<relative path>"` lets several
    predicate configurations share one set of records.
    """
    spec = tomllib.loads((path / "dataset.toml").read_text(encoding="utf-8"))
    with open(path / spec.get("records", "records.csv"), newline="", encoding="utf-8") as fh:
        records = {row["id"]: {k: (v or "") for k, v in row.items() if k != "id"}
                   for row in csv.DictReader(fh)}

    def predicate(d: dict[str, Any]) -> Predicate:
        return Predicate(text=d["text"], negation=d["negation"], fields=tuple(d["fields"]),
                         rule=d.get("rule"))

    return Dataset(
        name=spec["name"],
        records=records,
        text_fields=list(spec["text_fields"]),
        p=predicate(spec["predicates"]["p"]),
        q=predicate(spec["predicates"]["q"]),
        conj_pq=spec["conjunction"]["pq"],
        conj_qp=spec["conjunction"]["qp"],
        match_claim=spec["match_claim"],
        map_spec=MapSpec(**spec["map"]),
    )


def load_data(cfg: dict[str, Any]) -> Dataset:
    if cfg["kind"] == "synthetic":
        return synthetic_dataset(cfg.get("n_entities", 240), cfg.get("seed", 0),
                                 cfg.get("sibling_rate", 0.5))
    if cfg["kind"] == "dir":
        return load_dataset_dir(Path(cfg["path"]))
    raise ValueError(f"unknown data kind: {cfg['kind']}")


# --- sampling -----------------------------------------------------------------------------------

def sample_rows(ds: Dataset, n: int, rng: random.Random) -> list[str]:
    return rng.sample(sorted(ds.records), n)


def sample_blocks(ds: Dataset, n_blocks: int, block_size: int, positives: int,
                  rng: random.Random) -> list[list[str]]:
    """Disjoint blocks for matching: a few offers of one product plus its most similar non-matches.

    Every pair inside a block is evaluated, so every triple inside a block is a checkable triangle.
    """
    groups = ds.clusters()
    seeds = sorted(c for c, ids in groups.items() if len(ids) >= 2)
    rng.shuffle(seeds)
    toks = {rid: tokens(rec.get("title", "")) for rid, rec in ds.records.items()}
    used: set[str] = set()
    blocks: list[list[str]] = []
    for cluster in seeds:
        if len(blocks) == n_blocks:
            break
        members = sorted(r for r in groups[cluster] if r not in used)
        if len(members) < 2:
            continue
        rng.shuffle(members)
        block = members[:positives]
        others = [r for r in ds.records if r not in used and ds.records[r].get("cluster") != cluster]
        others.sort(key=lambda r: (-max(jaccard(toks[r], toks[b]) for b in block), r))
        block += others[: block_size - len(block)]
        used.update(block)
        blocks.append(sorted(block))
    if len(blocks) < n_blocks:
        raise ValueError(f"only {len(blocks)} blocks available; {n_blocks} requested")
    return blocks


def sample_tuples(ds: Dataset, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Gold-matched (r, s) pairs from different sources, at most one per product."""
    pairs = []
    for _, ids in sorted(ds.clusters().items()):
        cross = [(a, b) for a, b in combinations(sorted(ids), 2)
                 if ds.records[a].get("source") != ds.records[b].get("source")]
        if cross:
            pairs.append(rng.choice(cross))
    rng.shuffle(pairs)
    if len(pairs) < n:
        raise ValueError(f"only {len(pairs)} matched tuples available; {n} requested")
    return pairs[:n]
