"""Paired comparisons of two plans (or two segments, or two arms) over shared items.

Used by analysis.py (difference in excess divergence) and accuracy.py (difference in accuracy).
The items' dependence follows semaudit's rules for the comparison (`semaudit.stats.compare`):

- H4 (batch_neighbours): two crossed groupings, the N1 batch and the N2 batch of each row;
- S2, S4: the N1 batch;
- H1: the matching block;
- otherwise each item is its own cluster.

Primary interval and test: the mean difference with a t interval, mean +/- t(df) x SE, and a
two-sided t-test. SE is semaudit's cluster-robust standard error (Cameron, Gelbach and Miller's
two-way estimator for crossed groupings) and df = min(G) - 1, the number of clusters in the smaller
grouping minus one. With df < 1 (every item in one cluster) there is no test: p = 1, marked
"insufficient clusters". A cluster bootstrap interval (Owen's pigeonhole bootstrap for crossed
groupings) is kept as a secondary column only: with few clusters it under-covers (a null
simulation with 10 N1 batches excluded 0 in 10-15% of runs).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from semaudit.stats import bootstrap_ci, cluster_robust_test, t_sf

from .meta import t_quantile


def mapping_ways(items: list[str], groups: dict[str, Any] | None) -> list[np.ndarray]:
    """Grouping from a mapping: no mapping (each item its own cluster), one label per item, or a tuple
    of labels per item (crossed groupings, as for H4)."""
    if not groups:
        return [np.array([str(i) for i in items])]
    first = groups.get(items[0])
    if isinstance(first, tuple):
        if len(first) == 1:
            return [np.array([str(groups[i][0]) for i in items])]
        return [np.array([str(groups[i][w]) for i in items]) for w in range(len(first))]
    return [np.array([str(groups.get(i, i)) for i in items])]


def comparison_ways(comp: Any, items: list[str]) -> list[np.ndarray]:
    """semaudit's grouping of a comparison's items: crossed call groups, one call group, clusters, or none.
    `comp` is a semaudit Comparison, or a mapping from item to its cluster label (or tuple of labels)."""
    if comp is None or isinstance(comp, dict):
        return mapping_ways(items, comp)
    if comp.call_clusters:
        n_ways = len(comp.call_clusters[items[0]])
        return [np.array([str(comp.call_clusters[i][w]) for i in items]) for w in range(n_ways)]
    if comp is not None and comp.clusters:
        return [np.array([str(comp.clusters.get(i, i)) for i in items])]
    return [np.array([str(i) for i in items])]


def paired_inference(d: np.ndarray, ways: list[np.ndarray], n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """Mean of the per-item differences d, its t interval and two-sided p, and a bootstrap interval."""
    mean = float(d.mean())
    robust = cluster_robust_test(d, ways)
    out: dict[str, Any] = {"difference": mean, "n_clusters": robust["n_clusters"], "df": robust["df"],
                           "clustering": "crossed" if len(ways) == 2 else "one-way"}
    if robust["df"] < 1:
        out.update(status="insufficient clusters", p_two_sided=1.0, difference_ci=None, se=None,
                   difference_ci_bootstrap=None)
        return out
    se = robust["se"]
    if se == 0:
        p, ci = (1.0 if mean == 0 else 0.0), [mean, mean]
    else:
        p = min(1.0, 2 * t_sf(abs(mean) / se, robust["df"]))
        half = t_quantile(0.975, robust["df"]) * se
        ci = [mean - half, mean + half]
    if len(ways) == 2:
        boot = bootstrap_ci(d, n_boot, rng, crossed=(ways[0], ways[1]))
    else:
        boot = bootstrap_ci(d, n_boot, rng, clusters=ways[0])
    out.update(status="ok", se=se, p_two_sided=p, difference_ci=ci, difference_ci_bootstrap=list(boot))
    return out


def fmt_ci(ci: list[float] | None, digits: int = 4, sign: bool = True) -> str:
    if not ci:
        return "n/a"
    spec = f"+.{digits}f" if sign else f".{digits}f"
    return f"[{format(ci[0], spec)}, {format(ci[1], spec)}]"


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))
