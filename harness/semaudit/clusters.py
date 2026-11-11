"""Transitivity of pairwise matching, and what violations do to deduplication clusters.

Every pair inside a block was matched in both operand orders, so every triple inside a block is a
triangle whose three edges were all judged. An equivalence relation never has exactly two positive
edges in a triangle; a matcher that does forces the clustering step (transitive closure) to either
merge records it judged different or split records it judged the same.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .core import INVALID
from .identities import Outputs

POLICIES = ("ab", "ba", "or", "and")


class UnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {n: n for n in nodes}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def pair_id(a: str, b: str) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def _combine(ab: Any, ba: Any, policy: str) -> Any:
    if policy == "ab":
        return ab
    if policy == "ba":
        return ba
    if INVALID in (ab, ba):
        return INVALID
    return (ab or ba) if policy == "or" else (ab and ba)


def edge_labels(out: Outputs, design: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    """One {pair id: label} map per repeat, combining the two operand orders by `policy`."""
    labels = []
    for r in range(design["repeats"]):
        labels.append({
            pid: _combine(out.get(("symmetry", "ab", pid), {}).get(r, INVALID),
                          out.get(("symmetry", "ba", pid), {}).get(r, INVALID), policy)
            for pid in design["symmetry"]["pairs"]
        })
    return labels


def stable_labels(per_repeat: list[dict[str, Any]]) -> dict[str, Any]:
    """A label that every repeat agrees on, else INVALID.

    Unanimity over k draws is not determinism: an ambiguous pair answered at random agrees with
    itself three times in a row a quarter of the time. Use `modal_labels` for the confident view.
    """
    stable = {}
    for pid in per_repeat[0]:
        values = [lab[pid] for lab in per_repeat]
        stable[pid] = values[0] if INVALID not in values and len(set(values)) == 1 else INVALID
    return stable


def modal_labels(out: Outputs, design: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Majority label over every draw in both operand orders (ties INVALID), and its agreement share.

    This is the matcher's best single answer per pair. A triangle that violates transitivity even
    when all three edges are confident (agreement >= 5/6) is very unlikely to be sampling noise.
    """
    k = design["repeats"]
    labels: dict[str, Any] = {}
    agreement: dict[str, float] = {}
    for pid in design["symmetry"]["pairs"]:
        values = [out.get(("symmetry", cond, pid), {}).get(r, INVALID) for cond in ("ab", "ba") for r in range(k)]
        valid = [bool(v) for v in values if v != INVALID]
        positives = sum(valid)
        if not valid or 2 * positives == len(valid):
            labels[pid], agreement[pid] = INVALID, 0.0
        else:
            labels[pid] = 2 * positives > len(valid)
            agreement[pid] = max(positives, len(valid) - positives) / len(valid)
    return labels, agreement


def triangle_counts(blocks: list[list[str]], labels: dict[str, Any]) -> list[tuple[int, int]]:
    """Per block: (triangles with exactly two positive edges, triangles with all edges valid)."""
    counts = []
    for block in blocks:
        bad = valid = 0
        for a, b, c in combinations(block, 3):
            edges = [labels[pair_id(a, b)], labels[pair_id(a, c)], labels[pair_id(b, c)]]
            if INVALID in edges:
                continue
            valid += 1
            bad += sum(bool(e) for e in edges) == 2
        counts.append((bad, valid))
    return counts


def positive_rate(blocks: list[list[str]], labels: dict[str, Any]) -> float:
    values = [labels[pair_id(a, b)] for block in blocks for a, b in combinations(block, 2)]
    valid = [bool(v) for v in values if v != INVALID]
    return sum(valid) / len(valid) if valid else float("nan")


def cluster_metrics(blocks: list[list[str]], labels: dict[str, Any], gold: dict[str, str]) -> dict[str, Any]:
    """Transitive closure of positive edges, scored against gold entities within blocks."""
    nodes = [n for block in blocks for n in block]
    uf = UnionFind(nodes)
    for block in blocks:
        for a, b in combinations(block, 2):
            if labels[pair_id(a, b)] is True:
                uf.union(a, b)
    tp = fp = fn = conflicts = 0
    for block in blocks:
        for a, b in combinations(block, 2):
            same_pred = uf.find(a) == uf.find(b)
            same_gold = gold.get(a) is not None and gold.get(a) == gold.get(b)
            tp += same_pred and same_gold
            fp += same_pred and not same_gold
            fn += same_gold and not same_pred
            conflicts += same_pred and labels[pair_id(a, b)] is False
    members: dict[str, set[Any]] = {}
    for n in nodes:
        members.setdefault(uf.find(n), set()).add(gold.get(n))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "n_clusters": len(members),
        "merged_clusters": sum(len(g) > 1 for g in members.values()),
        "closure_conflicts": conflicts,  # same cluster after closure, but judged different directly
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def random_error_baseline(blocks: list[list[str]], gold_labels: dict[str, bool],
                          label_sets: list[dict[str, Any]], n_sims: int,
                          rng: np.random.Generator) -> dict[str, Any]:
    """Violation rate of a matcher with the same false-positive and false-negative rates as the
    observed labels (pooled over `label_sets`, measured against gold), but whose errors are
    independent coin flips.

    Random errors produce violations at this rate. Fewer violations means the errors agree with each
    other (e.g. two products always merged); more means the errors contradict each other, which no
    consistent matcher can produce.
    """
    fn = n_pos = fp = n_neg = 0
    for labels in label_sets:
        for pid, v in labels.items():
            if v == INVALID:
                continue
            if gold_labels[pid]:
                n_pos += 1
                fn += v is False
            else:
                n_neg += 1
                fp += v is True
    fnr = fn / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    rates = []
    for _ in range(n_sims):
        per_labelling = []
        for observed in label_sets:  # as many simulated labellings as observed ones, averaged alike
            sim = {}
            for pid, truth in gold_labels.items():
                if observed.get(pid, INVALID) == INVALID:
                    sim[pid] = INVALID  # evaluate exactly the triangles the observed data allows
                    continue
                flip = rng.random() < (fnr if truth else fpr)
                sim[pid] = truth != flip
            counts = triangle_counts(blocks, sim)
            per_labelling.append(sum(b for b, _ in counts) / max(1, sum(v for _, v in counts)))
        rates.append(float(np.mean(per_labelling)))
    return {"false_negative_rate": fnr, "false_positive_rate": fpr,
            "violation_rate": float(np.mean(rates)),
            "interval": [float(x) for x in np.quantile(rates, [0.025, 0.975])]}


def _rate_ci(per_repeat: list[list[tuple[int, int]]], n_boot: int,
             rng: np.random.Generator) -> tuple[float, list[float]]:
    """Violation rate averaged over repeats, with a block bootstrap (triangles share edges)."""
    bad = np.array([[b for b, _ in rep] for rep in per_repeat], dtype=float)
    valid = np.array([[v for _, v in rep] for rep in per_repeat], dtype=float)
    rate = float((bad.sum(1) / np.maximum(valid.sum(1), 1)).mean())
    idx = rng.integers(0, bad.shape[1], size=(n_boot, bad.shape[1]))
    boot = (bad[:, idx].sum(-1) / np.maximum(valid[:, idx].sum(-1), 1)).mean(0)
    return rate, [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def transitivity_report(out: Outputs, design: dict[str, Any], n_boot: int = 2000, seed: int = 0,
                        confident: float = 5 / 6) -> dict[str, Any]:
    if "symmetry" not in design:
        return {}
    rng = np.random.default_rng(seed)
    blocks, gold = design["symmetry"]["blocks"], design.get("gold", {})
    gold_labels = {}
    for pid in design["symmetry"]["pairs"]:
        a, b = pid.split("|")
        gold_labels[pid] = gold.get(a) is not None and gold.get(a) == gold.get(b)
    report: dict[str, Any] = {
        "n_blocks": len(blocks),
        "n_triangles": sum(len(list(combinations(b, 3))) for b in blocks),
        "gold_violations": sum(b for b, _ in triangle_counts(blocks, gold_labels)),  # 0 by construction
    }
    for policy in POLICIES:
        per_repeat = edge_labels(out, design, policy)
        counts = [triangle_counts(blocks, lab) for lab in per_repeat]
        rate, ci = _rate_ci(counts, n_boot, rng)
        stable = triangle_counts(blocks, stable_labels(per_repeat))
        rho = float(np.nanmean([positive_rate(blocks, lab) for lab in per_repeat]))
        clusters = [cluster_metrics(blocks, lab, gold) for lab in per_repeat]
        report[policy] = {
            "violation_rate": rate,
            "violation_ci": ci,
            "chance_rate": 3 * rho * rho * (1 - rho),  # a random labeler with the same positive rate
            "random_error_baseline": random_error_baseline(blocks, gold_labels, per_repeat, 200, rng),
            "positive_rate": rho,
            "unanimous_violations": sum(b for b, _ in stable),  # edges unanimous across the k repeats
            "unanimous_triangles": sum(v for _, v in stable),
            "clusters": {key: float(np.mean([c[key] for c in clusters])) for key in clusters[0]},
            "clusters_per_repeat": clusters,
        }
    modal, agreement = modal_labels(out, design)
    sure = {pid: (label if agreement[pid] >= confident else INVALID) for pid, label in modal.items()}
    modal_counts = triangle_counts(blocks, modal)
    sure_counts = triangle_counts(blocks, sure)
    rate, ci = _rate_ci([modal_counts], n_boot, rng)
    report["modal"] = {
        "violation_rate": rate,
        "violation_ci": ci,
        "violations": sum(b for b, _ in modal_counts),
        "triangles": sum(v for _, v in modal_counts),
        "confident_threshold": confident,
        "confident_violations": sum(b for b, _ in sure_counts),
        "confident_triangles": sum(v for _, v in sure_counts),
        "random_error_baseline": random_error_baseline(blocks, gold_labels, [modal], 200, rng),
        "tied_edges": sum(1 for pid, label in modal.items() if label == INVALID),
        "clusters": cluster_metrics(blocks, modal, gold),
    }
    # Does operand order change the clustering by more than re-running the same order does?
    ab = report["ab"]["clusters_per_repeat"]
    ba = report["ba"]["clusters_per_repeat"]
    k = len(ab)
    report["order_effect_on_clusters"] = {
        key: {
            "ab_vs_ba": float(np.mean([abs(ab[r][key] - ba[r][key]) for r in range(k)])),
            "ab_vs_ab": float(np.mean([abs(ab[r][key] - ab[s][key])
                                       for r in range(k) for s in range(k) if r < s])) if k > 1 else float("nan"),
        }
        for key in ("n_clusters", "merged_clusters", "f1")
    }
    return report
