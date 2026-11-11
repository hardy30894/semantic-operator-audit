"""Noise-calibrated comparison of two conditions that a function would have to agree on.

For each item we observe k draws under condition A and k under condition B. The per-item statistic

    delta_i = mean_{r,s} K(A_r, B_s) - (1/2) [ mean_{r!=r'} K(A_r, A_r') + mean_{s!=s'} K(B_s, B_s') ]

with K(x, y) = 1[x != y] (categorical) or (x - y)^2 (numeric) is an unbiased estimate of half the
squared L2 distance between the two output distributions (categorical; for binary outputs this is
(pi_A - pi_B)^2) or of (mu_A - mu_B)^2 (numeric). It is zero in expectation exactly when A and B
are exchangeable, i.e. when the rewrite changes nothing beyond the operator's own run-to-run
noise. The mean over items is tested with an exact permutation test that reshuffles each item's
pooled draws between A and B.

A "stable violation" is an item whose A draws all agree, whose B draws all agree, and A != B: a
disagreement that resampling did not produce. Its expectation under the permutation null is exact.

The permutation test treats items as independent. When several items are answered by one call (the
rows of a batched prompt), their draws are dependent and the permutation test is too liberal. For
those comparisons the reported p is max(p_perm, p_cluster_robust): the second is a one-sided t-test
of the mean with a variance that is robust to dependence within calls (one-way, or two-way when an
item's two conditions are answered by different groups of calls). The larger p is valid whenever
either test is.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np

from .core import INVALID


def _kernel(x: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "categorical":
        return (x != y).astype(float)
    return (x - y) ** 2


def item_deltas(A: np.ndarray, B: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-item (delta, cross term, within term). Leading batch dimensions are allowed."""
    ka, kb = A.shape[-1], B.shape[-1]
    cross = _kernel(A[..., :, None], B[..., None, :], kind).mean(axis=(-2, -1))
    within_a = _kernel(A[..., :, None], A[..., None, :], kind).sum(axis=(-2, -1)) / (ka * (ka - 1))
    within_b = _kernel(B[..., :, None], B[..., None, :], kind).sum(axis=(-2, -1)) / (kb * (kb - 1))
    within = 0.5 * (within_a + within_b)
    return cross - within, cross, within


def stable_count(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    stable_a = (A == A[..., :1]).all(axis=-1)
    stable_b = (B == B[..., :1]).all(axis=-1)
    return (stable_a & stable_b & (A[..., 0] != B[..., 0])).sum(axis=-1)


def expected_stable(A: np.ndarray, B: np.ndarray) -> float:
    """Exact expected number of stable violations if each item's draws were split at random."""
    ka, kb = A.shape[1], B.shape[1]
    splits = math.comb(ka + kb, ka)
    total = 0.0
    for row in np.concatenate([A, B], axis=1):
        counts = Counter(row.tolist())
        if len(counts) == 2:
            (_, n1), (_, n2) = counts.items()
            ways = (n1 == ka and n2 == kb) + (n1 == kb and n2 == ka)
            total += ways / splits
    return total


# --- binary fast path: everything depends only on the count of True per condition --------------

def _binary_delta(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> np.ndarray:
    within_a = 2 * a * (ka - a) / (ka * (ka - 1))
    within_b = 2 * b * (kb - b) / (kb * (kb - 1))
    cross = (a * (kb - b) + (ka - a) * b) / (ka * kb)
    return cross - 0.5 * (within_a + within_b)


def _binary_stable(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> np.ndarray:
    return (((a == ka) & (b == 0)) | ((a == 0) & (b == kb))).sum(axis=-1)


def _binary_permutation(A: np.ndarray, B: np.ndarray, n_perm: int,
                        rng: np.random.Generator) -> tuple[float, float]:
    ka, kb = A.shape[1], B.shape[1]
    a, b = A.sum(axis=1), B.sum(axis=1)
    ones = a + b
    observed = _binary_delta(a, b, ka, kb).mean()
    observed_stable = _binary_stable(a, b, ka, kb)
    tol = 1e-12 * max(1.0, abs(observed))
    hits = hits_stable = 0
    chunk = max(1, int(5e6 // max(1, len(a))))
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        a_star = rng.hypergeometric(ones, ka + kb - ones, ka, size=(m, len(a)))
        b_star = ones - a_star
        hits += int((_binary_delta(a_star, b_star, ka, kb).mean(axis=1) >= observed - tol).sum())
        hits_stable += int((_binary_stable(a_star, b_star, ka, kb) >= observed_stable).sum())
    return (hits + 1) / (n_perm + 1), (hits_stable + 1) / (n_perm + 1)


def _general_permutation(A: np.ndarray, B: np.ndarray, kind: str, n_perm: int,
                         rng: np.random.Generator) -> tuple[float, float]:
    ka = A.shape[1]
    pooled = np.concatenate([A, B], axis=1)
    n, k = pooled.shape
    observed = item_deltas(A, B, kind)[0].mean()
    observed_stable = stable_count(A, B)
    tol = 1e-12 * max(1.0, abs(observed))
    hits = hits_stable = 0
    chunk = max(1, int(4e6 // max(1, n * k * k)))
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        order = np.argsort(rng.random((m, n, k)), axis=2)
        perm = np.take_along_axis(np.broadcast_to(pooled, (m, n, k)), order, axis=2)
        stat = item_deltas(perm[..., :ka], perm[..., ka:], kind)[0].mean(axis=-1)
        hits += int((stat >= observed - tol).sum())
        hits_stable += int((stable_count(perm[..., :ka], perm[..., ka:]) >= observed_stable).sum())
    return (hits + 1) / (n_perm + 1), (hits_stable + 1) / (n_perm + 1)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) of Student's t distribution."""
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def cluster_robust_test(values: np.ndarray, ways: list[np.ndarray]) -> dict[str, Any]:
    """One-sided test that the mean exceeds 0, with a variance robust to dependence within clusters.

    One way: the usual cluster-robust variance. Two ways: Cameron, Gelbach and Miller's estimator,
    V = V_1 + V_2 - V_12, falling back to the larger one-way variance if V is not positive. Each term
    has the finite-cluster factor G/(G-1); the reference is Student's t with min(G) - 1 df.
    """
    n = len(values)
    resid = values - values.mean()

    def variance(labels: np.ndarray) -> tuple[float, int]:
        _, inverse = np.unique(labels, return_inverse=True)
        g = int(inverse.max()) + 1
        sums = np.bincount(inverse, weights=resid)
        return (g / (g - 1) if g > 1 else 1.0) * float((sums ** 2).sum()) / n ** 2, g

    parts = [variance(w) for w in ways]
    if len(ways) == 1:
        var = parts[0][0]
    else:
        joint = np.array([f"{x}\x1f{y}" for x, y in zip(ways[0], ways[1])])
        var = parts[0][0] + parts[1][0] - variance(joint)[0]
        if var <= 0:
            var = max(parts[0][0], parts[1][0])
    n_clusters = [g for _, g in parts]
    df = min(n_clusters) - 1
    mean = float(values.mean())
    if df < 1:
        p = 1.0
    elif var <= 0:
        p = 0.0 if mean > 0 else 1.0
    else:
        p = t_sf(mean / math.sqrt(var), df)
    return {"p": p, "se": math.sqrt(max(var, 0.0)), "df": df, "n_clusters": n_clusters}


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05,
                 clusters: np.ndarray | None = None,
                 crossed: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, float]:
    """Percentile interval for the mean. With `clusters`, whole clusters are resampled (items that
    share records or calls), and the mean is the ratio of resampled sums to resampled counts. With
    `crossed` (two crossed groupings), the two groupings are resampled independently and each item
    is weighted by the product of its groups' multiplicities (Owen's pigeonhole bootstrap)."""
    if crossed is not None:
        inv1 = np.unique(crossed[0], return_inverse=True)[1]
        inv2 = np.unique(crossed[1], return_inverse=True)[1]
        g1, g2 = int(inv1.max()) + 1, int(inv2.max()) + 1
        means = np.empty(n_boot)
        chunk = max(1, int(2e6 // max(1, len(values))))
        for start in range(0, n_boot, chunk):
            m = min(chunk, n_boot - start)
            w = (rng.multinomial(g1, np.full(g1, 1.0 / g1), size=m)[:, inv1]
                 * rng.multinomial(g2, np.full(g2, 1.0 / g2), size=m)[:, inv2]).astype(float)
            total = w.sum(axis=1)
            total[total == 0] = np.nan
            means[start:start + m] = (w @ values) / total
        means = means[~np.isnan(means)]
    elif clusters is None:
        n = len(values)
        means = np.empty(n_boot)
        chunk = max(1, int(5e6 // max(1, n)))
        for start in range(0, n_boot, chunk):
            m = min(chunk, n_boot - start)
            means[start:start + m] = values[rng.integers(0, n, size=(m, n))].mean(axis=1)
    else:
        _, inverse = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse).astype(float)
        g = len(sums)
        means = np.empty(n_boot)
        chunk = max(1, int(5e6 // max(1, g)))
        for start in range(0, n_boot, chunk):
            m = min(chunk, n_boot - start)
            idx = rng.integers(0, g, size=(m, g))
            means[start:start + m] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def encode(draws: dict[str, tuple[list[Any], list[Any]]], kind: str, invalid_as_category: bool = False):
    """Matrices of draws. By default complete-case: items with any invalid or missing draw are
    excluded and counted. With `invalid_as_category` (categorical kinds only; the pre-registered
    sensitivity analysis), INVALID is kept as an output category of its own.

    Returns (items, A, B, n_excluded, binary). Boolean outputs are coded 1 = True, 0 = False so the
    binary fast path applies; other categorical outputs get arbitrary integer codes.
    """
    keep_invalid = invalid_as_category and kind != "numeric"
    shapes = Counter((len(a), len(b)) for a, b in draws.values())
    if not shapes:
        return [], None, None, 0, False
    ka, kb = shapes.most_common(1)[0][0]
    items, rows_a, rows_b, excluded = [], [], [], 0
    for item in sorted(draws):
        a, b = draws[item]
        if len(a) != ka or len(b) != kb or (not keep_invalid and any(v == INVALID or v is None for v in (*a, *b))):
            excluded += 1
            continue
        items.append(item)
        rows_a.append(a)
        rows_b.append(b)
    if not items:
        return [], None, None, excluded, False
    if kind == "numeric":
        return items, np.array(rows_a, dtype=float), np.array(rows_b, dtype=float), excluded, False
    if all(isinstance(v, bool) for row in rows_a + rows_b for v in row):
        return (items, np.array(rows_a, dtype=np.int64), np.array(rows_b, dtype=np.int64),
                excluded, True)
    vocab: dict[str, int] = {}

    def code(v: Any) -> int:
        return vocab.setdefault(json.dumps(v, sort_keys=True), len(vocab))

    A = np.array([[code(v) for v in r] for r in rows_a], dtype=np.int64)
    B = np.array([[code(v) for v in r] for r in rows_b], dtype=np.int64)
    return items, A, B, excluded, False


def compare(draws: dict[str, tuple[list[Any], list[Any]]], kind: str, n_perm: int = 5000,
            n_boot: int = 5000, seed: int = 0, clusters: dict[str, Any] | None = None,
            invalid_as_category: bool = False, n_perm_binary: int | None = None,
            call_clusters: dict[str, tuple[Any, ...]] | None = None) -> dict[str, Any]:
    """Excess divergence of condition B over condition A, with its permutation test.

    - p_perm is the exact (Monte Carlo) permutation p-value; p_perm_mc_se is its Monte Carlo error.
    - call_clusters maps each item to the one or two groups of calls its draws come from (the batch
      prompts its row appears in). With it, p_cluster_robust is computed and the reported p is
      max(p_perm, p_cluster_robust); without it, p = p_perm. Inference rests on p.
    - delta_ci is an approximate bootstrap interval truncated at zero, because the true excess cannot
      be negative; the raw interval is kept as delta_ci_raw. It resamples the call groups when given
      (both groupings, crossed, when there are two), else `clusters` (items that share records),
      else items.
    """
    items, A, B, excluded, binary = encode(draws, kind, invalid_as_category)
    result: dict[str, Any] = {"n_items": len(items), "n_excluded": excluded}
    if len(items) < 2 or A.shape[1] < 2 or B.shape[1] < 2:
        result["status"] = "insufficient data"
        return result
    boot_rng, perm_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    deltas, cross, within = item_deltas(A, B, kind)
    ways = crossed = None
    if call_clusters:
        n_ways = {len(call_clusters[i]) for i in items}
        if len(n_ways) != 1 or n_ways.pop() not in (1, 2):
            raise ValueError("call_clusters must give every item the same number (1 or 2) of groups")
        ways = [np.array([str(call_clusters[i][w]) for i in items]) for w in range(len(call_clusters[items[0]]))]
        groups = ways[0] if len(ways) == 1 else None
        crossed = (ways[0], ways[1]) if len(ways) == 2 else None
    else:
        groups = np.array([str(clusters.get(i, i)) for i in items]) if clusters else None
    lo, hi = bootstrap_ci(deltas, n_boot, boot_rng, clusters=groups, crossed=crossed)
    perms = n_perm_binary if (binary and n_perm_binary) else n_perm
    if binary:
        p_perm, p_stable = _binary_permutation(A, B, perms, perm_rng)
    else:
        p_perm, p_stable = _general_permutation(A, B, kind, perms, perm_rng)
    stable = int(stable_count(A, B))
    delta = float(deltas.mean())
    robust = cluster_robust_test(deltas, ways) if ways else None
    result.update(
        status="ok",
        noise_within=float(within.mean()),
        cross=float(cross.mean()),
        delta=delta,
        sqrt_delta=math.sqrt(max(delta, 0.0)),
        delta_ci=[max(lo, 0.0), max(hi, 0.0)],
        delta_ci_raw=[lo, hi],
        ci_resampling=("crossed call groups" if crossed is not None else "call groups" if ways
                       else "clusters" if clusters else "items"),
        n_clusters=(robust["n_clusters"] if robust else int(len(np.unique(groups))) if clusters else len(items)),
        p=max(p_perm, robust["p"]) if robust else p_perm,
        p_rule="max(permutation, cluster-robust)" if robust else "permutation",
        p_cluster_robust=robust["p"] if robust else None,
        p_perm=p_perm,
        p_perm_mc_se=math.sqrt(p_perm * (1 - p_perm) / perms),
        n_perm=perms,
        stable_violations=stable,
        stable_expected=expected_stable(A, B),
        stable_rate_ci=list(wilson(stable, len(items))),
        p_stable=p_stable,
    )
    if kind == "numeric":
        diffs = A.mean(axis=1) - B.mean(axis=1)
        result["mean_diff"] = float(diffs.mean())
        result["mean_diff_ci"] = list(bootstrap_ci(diffs, n_boot, boot_rng, clusters=groups, crossed=crossed))
    if binary:
        # Result-set size: how many items pass in each repeat under each condition. Comparing the gap
        # between conditions with the gap between repeats gives the COUNT-level excess.
        sa, sb = A.sum(axis=0), B.sum(axis=0)
        gaps_within = [abs(int(x) - int(y)) for s in (sa, sb) for i, x in enumerate(s) for y in s[i + 1:]]
        result.update(
            positives_a=sa.tolist(),
            positives_b=sb.tolist(),
            count_gap_between=float(np.mean([abs(int(x) - int(y)) for x in sa for y in sb])),
            count_gap_within=float(np.mean(gaps_within)) if gaps_within else float("nan"),
        )
    return result
