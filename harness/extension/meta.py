"""Random-effects meta-analysis of excess divergence across predicate pairs.

semaudit's confirmatory test pools items across runs, which answers "is the identity broken for
this model?". It does not say how much the effect varies between predicate pairs, or what to expect
for a new one. Here each run (a dataset with one predicate pair) is a study:

- the per-run estimate is the run's mean excess divergence Δ̄ (semaudit.stats.item_deltas over the
  run's complete-case items), and its standard error is the dependence-robust one semaudit uses
  for the comparison (cluster-robust over matching blocks for H1, over batch calls for H4, S2 and
  S4, over items otherwise; semaudit.stats.cluster_robust_test). A run whose items are all
  deterministic estimates SE = 0; it gets a floor instead (`se_floor`: the SE of the same design with
  every item deterministic except one, which has one flipped draw in each condition), and the output
  flags it. The floor depends on the design only, so runs with real variation keep their own SE;
- between-pair heterogeneity τ² is estimated by REML (the global maximum of the restricted
  likelihood: a log grid refined by golden-section search, because with near-zero standard errors
  the likelihood can have a second, local maximum near zero), with DerSimonian-Laird alongside;
- the pooled mean has a Wald interval and a Hartung-Knapp-Sidik-Jonkman interval (t with k - 1
  degrees of freedom, better calibrated for few studies);
- the 95% prediction interval for a new predicate pair is μ̂ ± t(k-2) sqrt(τ² + SE(μ̂)²)
  (Higgins, Thompson and Spiegelhalter 2009); it needs k >= 3 studies.

Δ is non-negative by construction but its estimates are not, so intervals are reported as computed
and truncated at 0 only on the √Δ scale used for interpretation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from semaudit.identities import comparisons
from semaudit.stats import cluster_robust_test, encode, item_deltas, t_sf

from .logs import collect_ext, load_design, read_compact, run_logs

Z975 = NormalDist().inv_cdf(0.975)


# --- distributions -------------------------------------------------------------------------------

def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t (inverse of semaudit.stats.t_sf), by bisection."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    target = 1 - p  # upper-tail probability
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _gamma_series(a: float, x: float) -> float:
    total = term = 1.0 / a
    n = a
    for _ in range(10000):
        n += 1
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(q: float, df: float) -> float:
    """Upper tail of the chi-square distribution (regularized upper incomplete gamma)."""
    if df <= 0:
        return float("nan")
    if q <= 0:
        return 1.0
    a, x = df / 2, q / 2
    if x < a + 1:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


# --- the meta-analysis ---------------------------------------------------------------------------

def restricted_loglik(tau2: float, y: np.ndarray, v: np.ndarray) -> float:
    """REML log-likelihood of the between-study variance (up to a constant)."""
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    return float(-0.5 * (np.log(v + tau2).sum() + math.log(w.sum()) + (w * (y - mu) ** 2).sum()))


def _reml_tau2(y: np.ndarray, v: np.ndarray) -> tuple[float, int]:
    """The global maximum of the restricted likelihood over tau2 >= 0.

    The usual fixed-point iteration can stop at a local maximum near zero when some studies have
    tiny standard errors (here: runs whose items are almost all deterministic), so the likelihood is
    first scanned on a log grid and the best grid cell is refined by golden-section search."""
    upper = max(10 * float(np.var(y)) + float(v.max()), 1e-12)
    grid = np.concatenate([[0.0], np.geomspace(upper * 1e-12, upper, 600)])
    values = [restricted_loglik(t, y, v) for t in grid]
    i = int(np.argmax(values))
    if i == 0:
        lo, hi = 0.0, grid[1]
    else:
        lo, hi = grid[i - 1], grid[min(i + 1, len(grid) - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = restricted_loglik(c, y, v), restricted_loglik(d, y, v)
    iterations = 0
    for iterations in range(1, 200):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = restricted_loglik(c, y, v)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = restricted_loglik(d, y, v)
        if b - a < 1e-12 * max(1.0, b):
            break
    best = (a + b) / 2
    candidates = [(restricted_loglik(t, y, v), t) for t in (best, grid[i], 0.0)]
    return max(candidates)[1], iterations


def random_effects(y: list[float], se: list[float], method: str = "REML") -> dict[str, Any]:
    """Random-effects pooling of estimates y with standard errors se."""
    y_arr, v = np.asarray(y, float), np.asarray(se, float) ** 2
    k = len(y_arr)
    if k == 0:
        return {"k": 0, "status": "no studies"}
    if np.any(v <= 0):
        raise ValueError("every study needs a positive standard error; floor zero ones with se_floor() first")
    w = 1 / v
    mu_fe = float((w * y_arr).sum() / w.sum())
    q = float((w * (y_arr - mu_fe) ** 2).sum())
    df = k - 1
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2_dl = max(0.0, (q - df) / c) if k > 1 and c > 0 else 0.0
    tau2 = tau2_dl
    iterations = 0
    if method == "REML" and k > 1:
        tau2, iterations = _reml_tau2(y_arr, v)
    wi = 1 / (v + tau2)
    mu = float((wi * y_arr).sum() / wi.sum())
    se_mu = float(math.sqrt(1 / wi.sum()))
    out: dict[str, Any] = {
        "k": k, "method": method, "mu": mu, "se": se_mu, "ci_wald": [mu - Z975 * se_mu, mu + Z975 * se_mu],
        "tau2": tau2, "tau": math.sqrt(tau2), "tau2_dl": tau2_dl, "mu_fixed": mu_fe, "Q": q, "df": df,
        "p_Q": chi2_sf(q, df) if df > 0 else None, "I2": max(0.0, (q - df) / q) if q > 0 and df > 0 else 0.0,
        "iterations": iterations,
    }
    if method == "REML":
        wdl = 1 / (v + tau2_dl)
        out["mu_dl"] = float((wdl * y_arr).sum() / wdl.sum())
    if k >= 2:
        q_hk = float((wi * (y_arr - mu) ** 2).sum() / (k - 1))
        se_hk = math.sqrt(q_hk / wi.sum())
        t = t_quantile(0.975, k - 1)
        out["ci_hksj"] = [mu - t * se_hk, mu + t * se_hk]
        out["se_hksj"] = se_hk
    if k >= 3:
        t = t_quantile(0.975, k - 2)
        half = t * math.sqrt(tau2 + se_mu ** 2)
        out["prediction_interval"] = [mu - half, mu + half]
    return out


def _se(deltas: np.ndarray, items: list[str], comp: Any) -> float:
    if comp.call_clusters:
        n_ways = len(comp.call_clusters[items[0]])
        ways = [np.array([str(comp.call_clusters[i][w]) for i in items]) for w in range(n_ways)]
    elif comp.clusters:
        ways = [np.array([str(comp.clusters.get(i, i)) for i in items])]
    else:
        ways = [np.array(items, dtype=str)]
    return float(cluster_robust_test(deltas, ways)["se"])


def min_item_delta(k_a: int, k_b: int) -> float:
    """|Delta_i| of an item whose draws all agree except one flipped draw in each condition: 2/(k_a k_b).
    (One flipped draw in one condition only leaves Delta_i at 0.) For counts, a one-unit change."""
    return 2.0 / (k_a * k_b)


def se_floor(items: list[str], comp: Any, k_a: int, k_b: int) -> float:
    """A floor for a run's standard error: the SE of a run with these items and clusters in which every
    item is deterministic except one, which has one flipped draw in each condition (|Delta_i| =
    2/(k_a k_b); d/n for n unclustered items). It depends on the design only, so a run with real
    variation keeps its own SE. A run whose items are all deterministic estimates SE = 0, which would
    give it infinite weight; its sampling error is not 0, only below what k draws can resolve."""
    resolution = np.zeros(len(items))
    resolution[0] = min_item_delta(k_a, k_b)
    return _se(resolution, items, comp)


def run_estimates(run_dir: Path, backends: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """{backend: {comparison: {hypothesis, kind, n_items, delta, se}}} for one run."""
    design = load_design(run_dir)
    result: dict[str, dict[str, Any]] = {}
    for backend, path in run_logs(run_dir, backends):
        out = collect_ext(read_compact(path))
        rows = {}
        for comp in comparisons(out, design):
            items, a, b, excluded, _ = encode(comp.draws, comp.kind)
            if len(items) < 2 or a.shape[1] < 2 or b.shape[1] < 2:
                rows[comp.name] = {"hypothesis": comp.hypothesis, "status": "insufficient data"}
                continue
            deltas = item_deltas(a, b, comp.kind)[0]
            se_raw = _se(deltas, items, comp)
            floor = se_floor(items, comp, a.shape[1], b.shape[1])
            rows[comp.name] = {"hypothesis": comp.hypothesis, "kind": comp.kind, "primary": comp.primary,
                               "n_items": len(items), "n_excluded": excluded, "delta": float(deltas.mean()),
                               "se": max(se_raw, floor), "se_raw": se_raw, "se_floor": floor,
                               "se_floored": se_raw < floor}
        result[backend] = rows
    return result


def meta_report(run_dirs: list[Path], backends: list[str] | None = None,
                estimates: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Per backend and comparison: the per-run estimates and their random-effects summary."""
    per_run = estimates or {run_dir.name: run_estimates(run_dir, backends) for run_dir in run_dirs}
    models = sorted({b for runs in per_run.values() for b in runs})
    report: dict[str, Any] = {"runs": list(per_run), "backends": {}}
    for backend in models:
        names: dict[str, dict[str, Any]] = {}
        for run, by_backend in per_run.items():
            for name, row in by_backend.get(backend, {}).items():
                if row.get("status"):
                    continue
                entry = names.setdefault(name, {"hypothesis": row["hypothesis"], "kind": row["kind"],
                                                "primary": row.get("primary"), "studies": []})
                entry["studies"].append({"run": run, **{k: row.get(k) for k in (
                    "n_items", "n_excluded", "delta", "se", "se_raw", "se_floored")}})
        rows = {}
        for name, entry in names.items():
            y = [s["delta"] for s in entry["studies"]]
            se = [s["se"] for s in entry["studies"]]
            reml = random_effects(y, se, "REML")
            rows[name] = {**entry, "reml": reml, "dl": random_effects(y, se, "DL")}
        report["backends"][backend] = rows
    return report


def _root(x: float) -> float:
    return math.sqrt(max(x, 0.0))


def render(report: dict[str, Any], title: str) -> str:
    lines = [f"# {title}", "",
             f"Studies (runs): {', '.join(report['runs'])}.", "",
             "Each run is one predicate pair. Δ is the excess divergence (rows² for H7 COUNT); √Δ is the RMS shift in "
             "P(True) (rows per batch for H7). τ² is the between-pair variance of Δ (REML); the prediction interval is "
             "for the Δ of a new predicate pair and needs at least 3 runs. HKSJ: Hartung-Knapp-Sidik-Jonkman interval. "
             "An SE marked * is a floor: the run's items were all (nearly) deterministic, so its estimated SE was "
             "below the SE of the same design with every item deterministic except one, which has one flipped draw "
             "in each condition (|Δ_i| = 2/(k_A k_B); for n unclustered items, 2/(k_A k_B n)), and the floor is used "
             "instead.",
             ""]
    order = {"H": 0, "S": 1, "X": 2}
    for backend, rows in report["backends"].items():
        lines += [f"## {backend}", "",
                  "| Hyp. | Comparison | Runs | Per-run Δ (SE) | Pooled Δ (REML) | 95% CI (HKSJ) | τ² (τ) | I² | Q p | "
                  "95% prediction interval | √Δ pooled [PI upper] | DL: Δ, τ² |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, row in sorted(rows.items(), key=lambda kv: (order.get(kv[1]["hypothesis"][0], 9), kv[1]["hypothesis"])):
            r, dl = row["reml"], row["dl"]
            per = "; ".join(f"{s['run']}: {s['delta']:.4f} ({s['se']:.4f}{'*' if s.get('se_floored') else ''})"
                            for s in row["studies"])
            ci = r.get("ci_hksj")
            pi = r.get("prediction_interval")
            p_q = r.get("p_Q")
            lines.append(
                f"| {row['hypothesis']} | {name} | {r['k']} | {per} | {r['mu']:.4f} | "
                + (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "n/a (k = 1)") + " | "
                + f"{r['tau2']:.5f} ({r['tau']:.4f}) | {r['I2']:.0%} | " + ("" if p_q is None else f"{p_q:.3g}") + " | "
                + (f"[{pi[0]:.4f}, {pi[1]:.4f}]" if pi else "n/a (k < 3)") + " | "
                + f"{_root(r['mu']):.3f}" + (f" [{_root(pi[1]):.3f}]" if pi else "") + " | "
                + f"{dl['mu']:.4f}, {dl['tau2']:.5f} |")
        lines.append("")
    return "\n".join(lines)


def write(report: dict[str, Any], out_dir: Path, name: str, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(render(report, title), encoding="utf-8")
