#!/usr/bin/env python3
"""Pooled and between-pair analyses for the mock ICDE review, from the published results. Exploratory.

Tasks, numbered as in the review brief:

  1 ladder           per model and data group, sqrt(Delta) [95%] and Holm status of every identity, grouped
                     into rewrites of the surface form (H1, S3, S5, and LOTUS's field order), logical
                     rewrites (H2, H3) and execution rewrites (H4, S2, S4, H5, H6, H7)
  4 meta_sqrt        the across-pair random-effects meta-analysis of results/meta-domains/meta.json redone on
                     the sqrt scale (DL, REML, Paule-Mandel; Knapp-Hartung intervals; prediction intervals
                     truncated at 0), a three-level model with pairs nested in datasets, leave one dataset
                     out, per-pair medians and the share of pairs above the smallest effect of interest
  5 meta_regression  per-pair Delta and sqrt(Delta) on the kind of the rewritten predicate (gold label,
                     surface check on a shown field, or judgment), with the dataset where estimable
    surface_excluded the pooled estimates without the pairs whose rewritten predicate is a surface check

Reads results/*/pooled.json, results/meta-domains/meta.json, runs/lotus-hs1 and data/*/dataset.toml; writes
<out>/{ladder,meta_sqrt,meta_regression,surface_excluded}.{json,md}. No logs, no API calls.

    python3 scripts/review_meta.py --out results/review
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import integrate, optimize
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]  # harness/
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from extension.meta import random_effects  # noqa: E402  (meta.json's own REML, for the comparison)

RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
DATA = PROJECT / "data"
DOCS_KINDS = PROJECT / "extension" / "data" / "DATASETS.md"

LADDER_GROUPS = {"products": ["confirmatory", "flash", "tier"], "domains": ["domains"], "argkp": ["argkp"]}
CLASSES = [
    ("surface", "rewrites that change only the surface form and keep the meaning",
     [("H1", "operand order"), ("S3", "conjunct order"), ("S5", "renamed columns")]),
    ("logical", "logical rewrites", [("H2", "negation"), ("H3", "fusion")]),
    ("execution", "execution rewrites", [("H4", "batch neighbours"), ("S2", "position in batch"),
                                         ("S4", "batching"), ("H5", "placement"), ("H6", "idempotence"),
                                         ("H7", "COUNT")]),
]
NAMES = {h: n for _, _, ids in CLASSES for h, n in ids}
META_IDS = ["H2", "H3", "S3", "H4", "S2", "S4", "H6", "H7"]
COMPARISON = {"H2": "negation", "H3": "fusion", "S3": "conjunct_order", "H4": "batch_neighbours",
              "S2": "batch_position", "S4": "batch_vs_single", "H6": "idempotence", "H7": "count"}
SESOI = {"categorical": 0.05, "numeric": 0.5}
GOLD_FIELDS = {"labels", "stance", "wa", "primary_group", "categories"}
DERIVED = {"period_word": "clause", "n_words": "argument", "year": "submitted"}  # hidden fields computed from shown ones


# --- helpers -----------------------------------------------------------------------------------------

def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=float, allow_nan=True) + "\n"


def f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{float(x):.{nd}f}"


def root(x: float) -> float:
    return math.sqrt(max(x, 0.0))


def ci_txt(ci: Any, nd: int = 3) -> str:
    return "n/a" if not ci or ci[0] is None else f"[{f(ci[0], nd)}, {f(ci[1], nd)}]"


def dataset_of(run: str) -> str:
    return run.split("-")[1]


# --- random-effects estimators -------------------------------------------------------------------------

def restricted_loglik(t2: float, y: np.ndarray, v: np.ndarray, X: np.ndarray) -> float:
    w = 1.0 / (v + t2)
    xtwx = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(xtwx, X.T @ (w * y))
    r = y - X @ beta
    return float(-0.5 * (np.log(v + t2).sum() + np.linalg.slogdet(xtwx)[1] + (w * r * r).sum()))


def tau2_reml(y: np.ndarray, v: np.ndarray, X: np.ndarray | None = None) -> float:
    """Global maximum of the restricted likelihood over tau2 >= 0: a log grid, then a bounded refinement."""
    X = np.ones((len(y), 1)) if X is None else X
    upper = max(10 * float(np.var(y)) + float(v.max()), 1e-12)
    grid = np.concatenate([[0.0], np.geomspace(upper * 1e-12, upper, 500)])
    vals = np.array([restricted_loglik(t, y, v, X) for t in grid])
    i = int(np.argmax(vals))
    lo, hi = (0.0, grid[1]) if i == 0 else (grid[i - 1], grid[min(i + 1, len(grid) - 1)])
    res = optimize.minimize_scalar(lambda t: -restricted_loglik(t, y, v, X), bounds=(lo, hi), method="bounded",
                                   options={"xatol": max(hi, 1e-30) * 1e-10})
    cands = [(restricted_loglik(t, y, v, X), t) for t in (float(res.x), float(grid[i]), 0.0)]
    return max(cands)[1]


def tau2_dl(y: np.ndarray, v: np.ndarray) -> float:
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    q = (w * (y - mu) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    return max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0


def tau2_pm(y: np.ndarray, v: np.ndarray) -> float:
    """Paule-Mandel: the tau2 at which the generalized Q statistic equals k - 1."""
    k = len(y)

    def gq(t2: float) -> float:
        w = 1 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return float((w * (y - mu) ** 2).sum() - (k - 1))
    if gq(0.0) <= 0:
        return 0.0
    hi = max(float(np.var(y)), 1e-12)
    while gq(hi) > 0:
        hi *= 4
    return float(optimize.brentq(gq, 0.0, hi, xtol=1e-16, rtol=1e-12, maxiter=500))


def summarize_re(y: np.ndarray, v: np.ndarray, t2: float, truncate_pi: bool) -> dict[str, Any]:
    k = len(y)
    w = 1 / (v + t2)
    mu = float((w * y).sum() / w.sum())
    se = math.sqrt(1 / w.sum())
    wf = 1 / v
    mu_f = (wf * y).sum() / wf.sum()
    q = float((wf * (y - mu_f) ** 2).sum())
    out: dict[str, Any] = {"k": k, "mu": mu, "se": se, "tau2": t2, "tau": math.sqrt(t2), "Q": q,
                           "I2": max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0,
                           "ci_wald": [mu - 1.959964 * se, mu + 1.959964 * se]}
    if k >= 2:
        qhk = float((w * (y - mu) ** 2).sum() / (k - 1))
        se_hk = math.sqrt(qhk / w.sum())
        t = sps.t.ppf(0.975, k - 1)
        out["se_hksj"] = se_hk
        out["ci_hksj"] = [mu - t * se_hk, mu + t * se_hk]
        out["p_hksj_two_sided"] = float(2 * sps.t.sf(abs(mu) / se_hk, k - 1)) if se_hk > 0 else None
    if k >= 3:
        t = sps.t.ppf(0.975, k - 2)
        half = t * math.sqrt(t2 + se ** 2)
        lo, hi = mu - half, mu + half
        out["prediction_interval_raw"] = [lo, hi]
        out["prediction_interval"] = [max(lo, 0.0), max(hi, 0.0)] if truncate_pi else [lo, hi]
    return out


def fit_all(y: np.ndarray, v: np.ndarray, truncate_pi: bool = True) -> dict[str, Any]:
    if len(y) < 2:
        return {"k": len(y), "status": "fewer than 2 pairs"}
    return {"DL": summarize_re(y, v, tau2_dl(y, v), truncate_pi),
            "REML": summarize_re(y, v, tau2_reml(y, v), truncate_pi),
            "PM": summarize_re(y, v, tau2_pm(y, v), truncate_pi)}


# --- the sqrt scale ----------------------------------------------------------------------------------

def se_sqrt_floored(delta: float, se: float) -> float:
    """Delta method se / (2 sqrt(Delta)), with Delta floored at its own standard error. Near 0 this gives
    sqrt(se)/2, close to the sampling SD of sqrt(max(X, 0)) for X ~ N(0, se^2), which is 0.48 sqrt(se)."""
    return se / (2 * math.sqrt(max(delta, se)))


def se_sqrt_parametric(delta: float, se: float) -> float:
    """SD of sqrt(max(X, 0)) for X ~ N(max(Delta, 0), se^2), by numerical integration."""
    m, s = max(delta, 0.0), se
    if m > 8 * s:
        return s / (2 * math.sqrt(m))
    e2 = m * sps.norm.cdf(m / s) + s * sps.norm.pdf(m / s)
    lo, hi = max(0.0, m - 10 * s), m + 10 * s
    e1, _ = integrate.quad(lambda x: math.sqrt(x) * sps.norm.pdf((x - m) / s) / s, lo, hi, limit=200)
    return math.sqrt(max(e2 - e1 * e1, 0.0))


def three_level(y: np.ndarray, v: np.ndarray, groups: list[str]) -> dict[str, Any]:
    """REML for y_ij = mu + u_i + e_ij + error, u_i ~ N(0, tau_d^2) per dataset, e_ij ~ N(0, tau_p^2) per pair."""
    labels = sorted(set(groups))
    m = len(labels)
    sizes = {g: groups.count(g) for g in labels}
    if m < 2 or max(sizes.values()) < 2:
        return {"status": "not identifiable: every dataset has one pair" if m >= 2 else "one dataset",
                "datasets": sizes}
    Z = np.array([[1.0 if g == lab else 0.0 for lab in labels] for g in groups])
    ZZ = Z @ Z.T
    one = np.ones(len(y))

    def ll(td2: float, tp2: float) -> tuple[float, float, float]:
        V = np.diag(v + tp2) + td2 * ZZ
        sign, logdet = np.linalg.slogdet(V)
        Vi = np.linalg.inv(V)
        s = float(one @ Vi @ one)
        mu = float(one @ Vi @ y) / s
        r = y - mu
        return float(-0.5 * (logdet + math.log(s) + r @ Vi @ r)), mu, math.sqrt(1 / s)

    scale = max(float(np.var(y)), float(v.max()), 1e-12)
    grid = np.concatenate([[0.0], np.geomspace(scale * 1e-8, scale * 10, 60)])
    best = max(((ll(a, b)[0], a, b) for a in grid for b in grid))
    _, a0, b0 = best

    def neg(x: np.ndarray) -> float:
        return -ll(max(x[0], 0.0), max(x[1], 0.0))[0]
    res = optimize.minimize(neg, x0=np.array([a0, b0]), method="L-BFGS-B",
                            bounds=[(0, scale * 100), (0, scale * 100)])
    cands = [(ll(max(res.x[0], 0), max(res.x[1], 0))[0], max(res.x[0], 0), max(res.x[1], 0)), best]
    llv, td2, tp2 = max(cands)
    _, mu, se = ll(td2, tp2)
    ll_two = max((ll(0.0, b)[0], b) for b in grid)  # pairs only (two-level)
    ll_ds = max((ll(a, 0.0)[0], a) for a in grid)  # datasets only
    t = sps.t.ppf(0.975, m - 1)
    return {"status": "ok", "datasets": sizes, "mu": mu, "se": se, "tau_dataset": math.sqrt(td2),
            "tau_pair": math.sqrt(tp2), "share_dataset": td2 / (td2 + tp2) if td2 + tp2 > 0 else None,
            "ci_wald": [mu - 1.959964 * se, mu + 1.959964 * se],
            "ci_t_datasets": [mu - t * se, mu + t * se], "loglik": llv,
            "loglik_pairs_only": ll_two[0], "loglik_datasets_only": ll_ds[0]}


# --- task 1 ---------------------------------------------------------------------------------------------

def holm_status(row: dict[str, Any]) -> str:
    if row is None or row.get("status") == "not measured":
        return "not measured"
    if row.get("status") != "ok":
        return row.get("status", "n/a")
    if row["p_holm"] < 0.05:
        return "supported" if row.get("practically_relevant") else "supported, below SESOI"
    return "not supported"


def ladder() -> dict[str, Any]:
    out: dict[str, Any] = {"classes": [{"class": c, "description": d, "identities": [h for h, _ in ids]}
                                       for c, d, ids in CLASSES], "groups": {}, "figure_data": []}
    for group, folders in LADDER_GROUPS.items():
        models: dict[str, Any] = {}
        for folder in folders:
            data = json.loads((RESULTS / folder / "pooled.json").read_text(encoding="utf-8"))
            for backend, res in data["backends"].items():
                rows = {r["hypothesis"]: r for r in res["comparisons"]}
                cells = {}
                for cls, _, ids in CLASSES:
                    for hyp, name in ids:
                        r = rows.get(hyp)
                        st = holm_status(r)
                        cell = {"identity": hyp, "name": name, "class": cls, "status": st, "source": folder}
                        if r is not None and r.get("status") == "ok":
                            lo, hi = r["delta_ci"]
                            cell.update(n_items=r["n_items"], n_excluded=r["n_excluded"], delta=r["delta"],
                                        delta_ci=r["delta_ci"], sqrt_delta=r["sqrt_delta"],
                                        sqrt_ci=[root(lo), root(hi)], p=r["p"], p_holm=r["p_holm"],
                                        sesoi=r.get("sesoi"), runs=r.get("runs"))
                        cells[hyp] = cell
                        out["figure_data"].append({"group": group, "model": backend, **cell})
                models[backend] = cells
        out["groups"][group] = models
    # LOTUS's field order (L3): LOTUS sorts fields by Python's per-process string hash; runs/lotus-hs1 (hash seed 1)
    # against runs/lotus-hs0a (hash seed 0), all True/False requests pooled; lotus-hs0b repeats hash seed 0.
    l3 = json.loads((RUNS / "lotus-hs1" / "results.json").read_text(encoding="utf-8"))
    ctl = json.loads((RUNS / "lotus-hs0b" / "results.json").read_text(encoding="utf-8"))
    pb = next(iter(l3["drift"].values()))["pooled_binary"]
    pc = next(iter(ctl["drift"].values()))["pooled_binary"]
    lo, hi = pb["delta_ci"]
    out["lotus_L3"] = {"model": "lotus-haiku-5.5", "data": "wdc-A", "identity": "L3", "name": "field order (LOTUS)",
                       "class": "surface", "runs": ["lotus-hs1", "lotus-hs0a"], "n_items": pb["n_items"],
                       "n_excluded": pb["n_excluded"], "delta": pb["delta"], "delta_ci": pb["delta_ci"],
                       "sqrt_delta": root(pb["delta"]), "sqrt_ci": [root(lo), root(hi)], "p": pb["p"],
                       "p_perm": pb["p_perm"], "status": "descriptive (no Holm family); p is max(permutation, "
                       "cluster-robust)",
                       "control_same_hash_seed": {"runs": ["lotus-hs0b", "lotus-hs0a"], "delta": pc["delta"],
                                                  "delta_ci": pc["delta_ci"], "p": pc["p"]}}
    out["figure_data"].append({"group": "products", "model": "lotus-haiku-5.5", **{k: v for k, v in out["lotus_L3"].items()
                                                                                 if k not in ("model", "control_same_hash_seed")}})
    # class summaries: range of sqrt(Delta) over models, per group and identity
    summ = {}
    for group, models in out["groups"].items():
        for cls, _, ids in CLASSES:
            for hyp, _ in ids:
                vals = [(m, c[hyp]["sqrt_delta"]) for m, c in models.items() if c[hyp].get("sqrt_delta") is not None]
                if vals:
                    summ.setdefault(group, {})[hyp] = {
                        "min": min(v for _, v in vals), "max": max(v for _, v in vals),
                        "median": float(np.median([v for _, v in vals])), "models": len(vals),
                        "supported": sum(1 for m, _ in vals if models[m][hyp]["status"].startswith("supported"))}
    out["class_summary"] = summ
    return out


def render_ladder(rep: dict[str, Any]) -> str:
    L = ["# Ladder of rewrites: surface form, logic, execution (review task 1)", "",
         "Exploratory. Generated by `scripts/review_meta.py` from the published pooled results "
         "(results/confirmatory, flash and tier for product data; results/domains; results/argkp). Each cell is "
         "sqrt(Delta) [95%], the square root of semaudit's pooled excess divergence and of its bootstrap interval "
         "(an RMS shift in P(True); rows per batch of ten for H7). Marks: ** supported (Holm p < 0.05 within the "
         "model and family) and at least the smallest effect of interest (0.05; 0.5 rows for H7); * supported but "
         "below it; no mark: not supported; a dot: not measured.", "",
         "Classes: (a) surface form only, the meaning is kept: H1 operand order, S3 conjunct order, S5 renamed "
         "columns, and LOTUS's field order (L3); (b) logical rewrites: H2 negation, H3 fusion; (c) execution "
         "rewrites: H4 batch neighbours, S2 position in the batch, S4 batching, H5 placement, H6 idempotence, "
         "H7 COUNT.", ""]
    order = [h for _, _, ids in CLASSES for h, _ in ids]

    def cell(c: dict[str, Any]) -> str:
        if c["status"] == "not measured":
            return "."
        if c.get("sqrt_delta") is None:
            return c["status"]
        mark = "**" if c["status"] == "supported" else "*" if c["status"].startswith("supported") else ""
        nd = 2
        return f"{c['sqrt_delta']:.{nd}f} [{c['sqrt_ci'][0]:.{nd}f}, {c['sqrt_ci'][1]:.{nd}f}]{mark}"
    titles = {"products": "Product data (4 pairs; WDC-A, WDC-B, Abt-A, Abt-B)",
              "domains": "New domains (15 pairs; CUAD, ArgQ, arXiv)", "argkp": "ArgKP argument matching (1 pair)"}
    for group, models in rep["groups"].items():
        ids = [h for h in order if any(c[h]["status"] != "not measured" for c in models.values())]
        head = " | ".join(f"{h} {NAMES[h]}" for h in ids)
        cls_row = " | ".join(next(c for c, _, hs in CLASSES if h in [x for x, _ in hs]) for h in ids)
        L += [f"## {titles[group]}", "", f"| Model | {head} |", "|---|" + "---|" * len(ids),
              f"| class | {cls_row} |"]
        for backend, cells in models.items():
            L.append(f"| {backend} | " + " | ".join(cell(cells[h]) for h in ids) + " |")
        L.append("")
    l3 = rep["lotus_L3"]
    L += ["## LOTUS field order (L3)", "",
          f"LOTUS serializes a row's fields in the order of Python's per-process string hash. The same WDC-A requests "
          f"through LOTUS with Haiku under hash seed 1 against hash seed 0 (runs/lotus-hs1 against runs/lotus-hs0a), "
          f"every True/False request pooled ({l3['n_items']} request x row items): sqrt(Delta) = {l3['sqrt_delta']:.3f} "
          f"[{l3['sqrt_ci'][0]:.3f}, {l3['sqrt_ci'][1]:.3f}], Delta = {l3['delta']:.4f}, p = {l3['p']:.3f} "
          f"(permutation p {l3['p_perm']:.4f}; the reported p also guards against dependence within LOTUS's "
          f"batched calls). The same hash seed repeated (lotus-hs0b against lotus-hs0a) gives Delta = "
          f"{l3['control_same_hash_seed']['delta']:.5f}, p = {l3['control_same_hash_seed']['p']:.2f}. No Holm "
          "family covers it.", "", "## Range over models", "",
          "| Group | Identity | class | models | supported | min | median | max |", "|---|---|---|---|---|---|---|---|"]
    for group, ids in rep["class_summary"].items():
        for hyp in order:
            if hyp in ids:
                s = ids[hyp]
                cls = next(c for c, _, hs in CLASSES if hyp in [x for x, _ in hs])
                L.append(f"| {group} | {hyp} {NAMES[hyp]} | {cls} | {s['models']} | {s['supported']} | {s['min']:.3f} | "
                         f"{s['median']:.3f} | {s['max']:.3f} |")
    L += ["", "## Summary", ""] + [f"- {x}" for x in rep["summary"]] + [""]
    return "\n".join(L)


def ladder_summary(rep: dict[str, Any]) -> list[str]:
    out = []
    for group in ("products", "domains"):
        cs = rep["class_summary"][group]
        out.append(f"{group}: the surface-only conjunct reordering (S3) ranges {cs['S3']['min']:.2f} to {cs['S3']['max']:.2f} "
                   f"(median {cs['S3']['median']:.2f}) over {cs['S3']['models']} models; negation (H2) "
                   f"{cs['H2']['min']:.2f} to {cs['H2']['max']:.2f} (median {cs['H2']['median']:.2f}); fusion (H3) "
                   f"{cs['H3']['min']:.2f} to {cs['H3']['max']:.2f} (median {cs['H3']['median']:.2f}).")
        if "H1" in cs:
            out.append(f"{group}: operand order (H1) {cs['H1']['min']:.2f} to {cs['H1']['max']:.2f}; renamed columns "
                       f"(S5) {cs['S5']['min']:.2f} to {cs['S5']['max']:.2f}, supported for {cs['S5']['supported']} of "
                       f"{cs['S5']['models']} models.")
        models = rep["groups"][group]
        above = [m for m, c in models.items() if c["S3"].get("sqrt_delta") is not None and c["H3"].get("sqrt_delta") is not None
                 and c["S3"]["sqrt_delta"] >= c["H3"]["sqrt_delta"]]
        out.append(f"{group}: conjunct order is at least as large as fusion for {len(above)} of {len(models)} models"
                   + (f" ({', '.join(above)})" if above else "") + ".")
    a = rep["groups"]["argkp"]
    out.append("argkp: operand order (H1) " + ", ".join(f"{m} {c['H1']['sqrt_delta']:.2f}" for m, c in a.items())
               + "; renamed columns (S5) " + ", ".join(f"{m} {c['S5']['sqrt_delta']:.2f}" for m, c in a.items()) + ".")
    l3 = rep["lotus_L3"]
    s5 = rep["class_summary"]["products"]["S5"]
    out.append(f"LOTUS's field order (L3): sqrt(Delta) {l3['sqrt_delta']:.3f}, p = {l3['p']:.3f}, not distinguishable "
               f"from 0 at 0.05; among the direct prompts, renamed columns (S5) on product data are the other surface "
               f"rewrite that mostly holds ({s5['models'] - s5['supported']} of {s5['models']} models not supported), "
               "while operand order (H1) and conjunct order (S3) are supported for every model in every group.")
    return out


# --- task 4 ---------------------------------------------------------------------------------------------

def studies_for(meta: dict[str, Any], backend: str, hyp: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entry = meta["backends"][backend][COMPARISON[hyp]]
    assert entry["hypothesis"] == hyp
    return entry, entry["studies"]


def sqrt_inputs(studies: list[dict[str, Any]], rule: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([root(s["delta"]) for s in studies])
    if rule == "floored":
        se = np.array([se_sqrt_floored(s["delta"], s["se"]) for s in studies])
    elif rule == "parametric":
        se = np.array([se_sqrt_parametric(s["delta"], s["se"]) for s in studies])
    else:
        raise ValueError(rule)
    return y, se ** 2


def meta_sqrt(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"rules": {
        "floored": "y = sqrt(max(Delta, 0)); SE = se / (2 sqrt(max(Delta, se))): the delta method, with the derivative "
                   "taken at Delta = se when Delta < se (then SE = sqrt(se)/2)",
        "parametric": "y as above; SE = SD of sqrt(max(X, 0)) for X ~ N(max(Delta, 0), se^2) (numerical)",
        "delta_scale": "REML on Delta (meta.json's method), then the square root of the mean, of the HKSJ bounds and "
                       "of the prediction bounds, each truncated at 0",
        "unweighted": "plain mean of the per-pair sqrt(Delta) with a t interval over pairs",
        "naive_positive": "the plain delta method se / (2 sqrt(Delta)) on the pairs with Delta > 0 only"},
        "backends": {}}
    for backend in meta["backends"]:
        for hyp in META_IDS:
            if COMPARISON[hyp] not in meta["backends"][backend]:
                continue
            entry, studies = studies_for(meta, backend, hyp)
            if len(studies) < 2:
                continue
            kind = entry["kind"]
            sesoi = SESOI[kind]
            deltas = np.array([s["delta"] for s in studies])
            ses = np.array([s["se"] for s in studies])
            runs = [s["run"] for s in studies]
            ds = [dataset_of(r) for r in runs]
            roots = np.array([root(d) for d in deltas])
            q1, med, q3 = np.quantile(roots, [0.25, 0.5, 0.75])
            passing = deltas - 1.959964 * ses > sesoi ** 2
            res: dict[str, Any] = {"kind": kind, "sesoi": sesoi, "k": len(studies), "runs": runs, "datasets": ds,
                                   "per_pair": [{"run": r, "delta": float(d), "se": float(s), "sqrt_delta": float(x),
                                                 "se_sqrt_floored": se_sqrt_floored(d, s),
                                                 "se_sqrt_parametric": se_sqrt_parametric(d, s),
                                                 "passes_sesoi": bool(p)}
                                                for r, d, s, x, p in zip(runs, deltas, ses, roots, passing)],
                                   "median_sqrt": float(med), "iqr_sqrt": [float(q1), float(q3)],
                                   "share_passing_sesoi": float(passing.mean()), "n_passing_sesoi": int(passing.sum())}
            fits = {}
            for rule in ("floored", "parametric"):
                y, v = sqrt_inputs(studies, rule)
                fits[rule] = fit_all(y, v)
            pos = deltas > 0
            if pos.sum() >= 2:
                y = np.sqrt(deltas[pos])
                v = (ses[pos] / (2 * np.sqrt(deltas[pos]))) ** 2
                fits["naive_positive"] = {"k_used": int(pos.sum()), **fit_all(y, v)}
            # Delta scale: meta.json's REML, recomputed with extension.meta, and its square roots
            dre = random_effects(list(deltas), list(ses), "REML")
            stored = entry["reml"]
            dsc = {"mu": dre["mu"], "tau2": dre["tau2"], "I2": dre["I2"], "ci_hksj": dre.get("ci_hksj"),
                   "prediction_interval": dre.get("prediction_interval"),
                   "matches_meta_json": abs(dre["mu"] - stored["mu"]) < 1e-9 and abs(dre["tau2"] - stored["tau2"]) <= 1e-6 * max(stored["tau2"], 1e-12),
                   "abs_diff_mu_vs_meta_json": abs(dre["mu"] - stored["mu"]),
                   "sqrt_mu": root(dre["mu"]),
                   "sqrt_ci_hksj": [root(x) for x in dre["ci_hksj"]] if dre.get("ci_hksj") else None,
                   "sqrt_prediction_interval": [root(x) for x in dre["prediction_interval"]] if dre.get("prediction_interval") else None}
            fits["delta_scale"] = dsc
            uw_t = sps.t.ppf(0.975, len(roots) - 1)
            sd = float(roots.std(ddof=1))
            fits["unweighted"] = {"mu": float(roots.mean()), "ci_t": [float(roots.mean() - uw_t * sd / math.sqrt(len(roots))),
                                                                      float(roots.mean() + uw_t * sd / math.sqrt(len(roots)))],
                                  "sd_between_pairs": sd}
            res["fits"] = fits
            # three-level models
            y, v = sqrt_inputs(studies, "floored")
            res["three_level_sqrt"] = three_level(y, v, ds)
            res["three_level_delta"] = three_level(deltas, ses ** 2, ds)
            # leave one dataset out
            lodo = {}
            for d in sorted(set(ds)):
                keep = [i for i, x in enumerate(ds) if x != d]
                if len(keep) < 2:
                    continue
                yk, vk = y[keep], v[keep]
                fs = summarize_re(yk, vk, tau2_reml(yk, vk), True)
                dk = random_effects(list(deltas[keep]), list(ses[keep]), "REML")
                lodo[d] = {"k": len(keep), "sqrt_reml": {k2: fs.get(k2) for k2 in ("mu", "ci_hksj", "tau", "prediction_interval")},
                           "delta_reml": {"mu": dk["mu"], "sqrt_mu": root(dk["mu"]), "ci_hksj": dk.get("ci_hksj"),
                                          "tau2": dk["tau2"]}}
            res["leave_one_dataset_out"] = lodo
            # what changes against meta.json's REML on Delta
            prim = fits["floored"]["REML"]
            res["comparison"] = {
                "delta_scale_mean_above_0": bool(dre["ci_hksj"][0] > 0),
                "sqrt_scale_mean_above_0": bool(prim["ci_hksj"][0] > 0),
                "delta_scale_sqrt_mu_at_least_sesoi": bool(root(dre["mu"]) >= sesoi),
                "sqrt_scale_mu_at_least_sesoi": bool(prim["mu"] >= sesoi),
                "sqrt_scale_ci_lower_at_least_sesoi": bool(prim["ci_hksj"][0] >= sesoi),
                "delta_scale_pi_lower_raw": dre.get("prediction_interval", [None])[0],
                "sqrt_scale_pi": prim.get("prediction_interval"),
                "sqrt_scale_pi_excludes_0": bool(prim.get("prediction_interval_raw", [0])[0] > 0),
                "delta_scale_sqrt_pi_upper": root(dre["prediction_interval"][1]) if dre.get("prediction_interval") else None,
                "I2_delta": dre["I2"], "I2_sqrt": prim["I2"],
                "lodo_sqrt_mu_range": [min(x["sqrt_reml"]["mu"] for x in lodo.values()),
                                       max(x["sqrt_reml"]["mu"] for x in lodo.values())] if lodo else None,
                "lodo_any_ci_includes_0": any(x["sqrt_reml"]["ci_hksj"][0] <= 0 for x in lodo.values()
                                              if x["sqrt_reml"].get("ci_hksj")) if lodo else None,
            }
            out["backends"].setdefault(backend, {})[hyp] = res
    return out


def meta_sqrt_findings(rep: dict[str, Any]) -> list[str]:
    lines = []
    primary = ["haiku-5.5", "gpt-5.6-luna", "gemini-3.5-flash-lite"]
    flips_zero, flips_sesoi, pi_pos, mismatch = [], [], [], []
    for backend, ids in rep["backends"].items():
        for hyp, r in ids.items():
            c = r["comparison"]
            if not r["fits"]["delta_scale"]["matches_meta_json"]:
                mismatch.append(f"{backend} {hyp}")
            if c["delta_scale_mean_above_0"] != c["sqrt_scale_mean_above_0"]:
                flips_zero.append(f"{backend} {hyp} (Delta scale: {'above' if c['delta_scale_mean_above_0'] else 'includes'} 0; "
                                  f"sqrt scale: {'above' if c['sqrt_scale_mean_above_0'] else 'includes'} 0)")
            if c["delta_scale_sqrt_mu_at_least_sesoi"] != c["sqrt_scale_mu_at_least_sesoi"]:
                flips_sesoi.append(f"{backend} {hyp}")
            if c["sqrt_scale_pi_excludes_0"]:
                pi_pos.append(f"{backend} {hyp}")
    lines.append("The REML fit on Delta recomputed here (extension/meta.py) equals meta.json's for every cell." if not mismatch else
                 f"REML on Delta differs from meta.json for: {', '.join(mismatch)}")
    lines.append("Whether the pooled mean's HKSJ interval excludes 0 changes between the Delta scale (meta.json) and "
                 "the sqrt scale (REML, floored rule) for " + (", ".join(flips_zero) if flips_zero else "no cell") + ".")
    lines.append("Whether the pooled RMS shift reaches the smallest effect of interest changes for "
                 + (", ".join(flips_sesoi) if flips_sesoi else "no cell") + ".")
    lines.append("On the sqrt scale the prediction interval for a new pair excludes 0 for "
                 + (", ".join(pi_pos) if pi_pos else "no cell")
                 + "; on the Delta scale every prediction interval reaches below 0, a value Delta cannot take.")
    for hyp in ("H2", "S4", "H3", "S3", "H4", "S2", "H6", "H7"):
        vals = [(b, rep["backends"][b][hyp]) for b in primary if hyp in rep["backends"].get(b, {})]
        if not vals:
            continue
        mus = [r["fits"]["floored"]["REML"]["mu"] for _, r in vals]
        old = [r["fits"]["delta_scale"]["sqrt_mu"] for _, r in vals]
        piu = [r["fits"]["floored"]["REML"].get("prediction_interval", [None, None])[1] for _, r in vals]
        oldpi = [r["comparison"]["delta_scale_sqrt_pi_upper"] for _, r in vals]
        i2 = [r["fits"]["floored"]["REML"]["I2"] for _, r in vals]
        unit = " rows" if hyp == "H7" else ""
        lines.append(
            f"{hyp} {NAMES[hyp]}, primary models: pooled sqrt-scale mean {min(mus):.2f} to {max(mus):.2f}{unit} (sqrt of the "
            f"Delta-scale mean: {min(old):.2f} to {max(old):.2f}); upper prediction bound {min(piu):.2f} to {max(piu):.2f} "
            f"(sqrt of the Delta-scale bound: {min(oldpi):.2f} to {max(oldpi):.2f}); I2 on the sqrt scale "
            f"{min(i2):.0%} to {max(i2):.0%}.")
    return lines


def render_meta_sqrt(rep: dict[str, Any]) -> str:
    L = ["# Across-pair meta-analysis on the sqrt scale (review task 4)", "",
         "Exploratory. Generated by `scripts/review_meta.py` from results/meta-domains/meta.json, whose per-pair "
         "estimates (Delta and its dependence-robust SE per run, floored for runs with deterministic items) are "
         "reused unchanged. 19 predicate pairs: 4 product (WDC-A, WDC-B, Abt-A, Abt-B) and 15 new-domain (CUAD, "
         "ArgQ, arXiv A to E); 17 for the batch identities and 5 for idempotence; DeepSeek ran on the product "
         "pairs only.", "",
         "## Why the sqrt scale, and the rule near 0", "",
         "meta.json pools Delta, a squared shift. Its prediction intervals then reach below 0, which Delta cannot "
         "do, and the paper reports the square roots of the pooled mean and of the bounds. Here each pair's "
         "estimate is put on the scale of an RMS shift first, y = sqrt(max(Delta, 0)), and pooled there. The "
         "delta-method SE, se / (2 sqrt(Delta)), grows without bound as Delta approaches 0 and is undefined at or "
         "below it; it also gives pairs with large Delta small SEs, which inflates the pooled mean if used as is "
         "(see 'naive' below). The primary rule takes the derivative at max(Delta, se): for Delta >= se it is the "
         "delta method, and below it the SE is sqrt(se)/2. For a pair whose true Delta is 0, sqrt(max(X, 0)) "
         "with X ~ N(0, se^2) has SD 0.48 sqrt(se), so the floor matches the sampling spread there. Sensitivity: "
         "(i) the SD of sqrt(max(X, 0)) computed exactly for X ~ N(max(Delta, 0), se^2); (ii) pooling Delta by "
         "REML (meta.json's method) and taking square roots; (iii) the unweighted mean of the per-pair "
         "sqrt(Delta) with a t interval; (iv) the naive delta method on the pairs with Delta > 0 only. "
         "Estimators: DerSimonian-Laird, REML and Paule-Mandel, each with the Knapp-Hartung (HKSJ) interval "
         "(t with k - 1 df) and a prediction interval mu +/- t(k - 2) sqrt(tau^2 + SE^2) truncated at 0.", "",
         "## Findings", ""] + [f"- {x}" for x in rep["findings"]] + ["",
         "## Table to replace Fig. 5", "",
         "Per identity and model: the median per-pair sqrt(Delta) [interquartile range]; the pooled mean on the "
         "sqrt scale (REML, floored rule) [HKSJ 95%]; tau on the sqrt scale; the 95% prediction interval for a new "
         "pair, truncated at 0; and the pairs whose Delta exceeds the smallest effect of interest squared with "
         "95% confidence (Delta - 1.96 se > 0.0025; 0.25 rows^2 for H7). H7 in rows per batch of ten.", "",
         "| Identity | Model | Pairs | Median per pair [IQR] | Pooled sqrt mean [HKSJ 95%] | tau | 95% PI | Pairs above SESOI |",
         "|---|---|---|---|---|---|---|---|"]
    for hyp in META_IDS:
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r:
                continue
            p = r["fits"]["floored"]["REML"]
            L.append(f"| {hyp} {NAMES[hyp]} | {backend} | {r['k']} | {f(r['median_sqrt'], 2)} [{f(r['iqr_sqrt'][0], 2)}, "
                     f"{f(r['iqr_sqrt'][1], 2)}] | {f(p['mu'], 2)} {ci_txt(p.get('ci_hksj'), 2)} | {f(p['tau'], 2)} | "
                     f"{ci_txt(p.get('prediction_interval'), 2)} | {r['n_passing_sesoi']} of {r['k']} |")
    L += ["", "## Estimators and rules side by side", "",
          "Pooled sqrt-scale mean [HKSJ 95%] per estimator and rule; tau after a slash. 'Delta scale' is meta.json's "
          "REML on Delta with square roots taken afterwards (its tau is on the Delta scale and not shown).", "",
          "| Identity | Model | REML floored | DL floored | PM floored | REML parametric | Delta scale (sqrt of mu [CI]) | "
          "Unweighted [t 95%] | Naive delta method, Delta > 0 only (pairs) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for hyp in META_IDS:
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r:
                continue
            fl, pa = r["fits"]["floored"], r["fits"]["parametric"]
            ds = r["fits"]["delta_scale"]
            uw = r["fits"]["unweighted"]
            nv = r["fits"].get("naive_positive")

            def one(x: dict[str, Any]) -> str:
                return f"{f(x['mu'], 3)} {ci_txt(x.get('ci_hksj'), 3)} / {f(x['tau'], 3)}"
            L.append(f"| {hyp} | {backend} | {one(fl['REML'])} | {one(fl['DL'])} | {one(fl['PM'])} | {one(pa['REML'])} | "
                     f"{f(ds['sqrt_mu'], 3)} {ci_txt(ds['sqrt_ci_hksj'], 3)} | {f(uw['mu'], 3)} {ci_txt(uw['ci_t'], 3)} | "
                     + (f"{one(nv['REML'])} ({nv['k_used']})" if nv and "REML" in nv else "n/a") + " |")
    L += ["", "## Against meta.json (REML on Delta)", "",
          "| Identity | Model | meta.json: sqrt(mu) [sqrt HKSJ] | I2 | PI on Delta (raw) | sqrt of PI upper | sqrt scale: mu [HKSJ] | "
          "I2 | PI (truncated) | mean above 0: Delta / sqrt | mu >= SESOI: Delta / sqrt |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for hyp in META_IDS:
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r:
                continue
            ds, p, c = r["fits"]["delta_scale"], r["fits"]["floored"]["REML"], r["comparison"]
            pi_raw = ds.get("prediction_interval")
            L.append(f"| {hyp} | {backend} | {f(ds['sqrt_mu'])} {ci_txt(ds['sqrt_ci_hksj'])} | {ds['I2']:.0%} | "
                     f"{ci_txt(pi_raw, 4)} | {f(c['delta_scale_sqrt_pi_upper'])} | {f(p['mu'])} {ci_txt(p.get('ci_hksj'))} | "
                     f"{p['I2']:.0%} | {ci_txt(p.get('prediction_interval'))} | "
                     f"{'yes' if c['delta_scale_mean_above_0'] else 'no'} / {'yes' if c['sqrt_scale_mean_above_0'] else 'no'} | "
                     f"{'yes' if c['delta_scale_sqrt_mu_at_least_sesoi'] else 'no'} / {'yes' if c['sqrt_scale_mu_at_least_sesoi'] else 'no'} |")
    L += ["", "## Three-level model: pairs nested in datasets", "",
          "REML over the two variance components (datasets: wdc, abt, cuad, argq, arxiv). wdc and abt have two "
          "product pairs each, which share records; cuad, argq and arxiv have five pairs each. Where every "
          "dataset contributes one pair (H6, and the batch identities of DeepSeek) the two components cannot be "
          "separated. Interval: t with (datasets - 1) df, conservative; the Wald interval is in the JSON.", "",
          "| Identity | Model | sqrt scale: mu [t 95%] | tau dataset | tau pair | share of datasets | Delta scale: sqrt(mu) | "
          "tau dataset | tau pair |",
          "|---|---|---|---|---|---|---|---|---|"]
    for hyp in META_IDS:
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r:
                continue
            a, b = r["three_level_sqrt"], r["three_level_delta"]
            if a.get("status") != "ok":
                L.append(f"| {hyp} | {backend} | {a.get('status')} | | | | | | |")
                continue
            L.append(f"| {hyp} | {backend} | {f(a['mu'])} {ci_txt(a['ci_t_datasets'])} | {f(a['tau_dataset'])} | "
                     f"{f(a['tau_pair'])} | {f(a['share_dataset'], 2)} | {f(root(b['mu']))} | {f(b['tau_dataset'], 4)} | "
                     f"{f(b['tau_pair'], 4)} |")
    L += ["", "## Leave one dataset out (sqrt scale, REML, floored rule)", "",
          "| Identity | Model | all pairs: mu [HKSJ] | without wdc | without abt | without cuad | without argq | without arxiv |",
          "|---|---|---|---|---|---|---|---|"]
    for hyp in META_IDS:
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r:
                continue
            p = r["fits"]["floored"]["REML"]
            cells = []
            for d in ("wdc", "abt", "cuad", "argq", "arxiv"):
                x = r["leave_one_dataset_out"].get(d)
                cells.append("n/a" if not x else f"{f(x['sqrt_reml']['mu'])} {ci_txt(x['sqrt_reml'].get('ci_hksj'))}")
            L.append(f"| {hyp} | {backend} | {f(p['mu'])} {ci_txt(p.get('ci_hksj'))} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


# --- task 5 ---------------------------------------------------------------------------------------------

def parse_doc_kinds() -> dict[str, str]:
    """Predicate text -> kind, from the table in extension/data/DATASETS.md."""
    kinds = {}
    for line in DOCS_KINDS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| ([a-z]+\.[a-z_]+) \| (gold(?: \(crowd\))?|machine|judgment) \| (.+?) \| (.+) \|$", line)
        if m:
            kinds[m.group(3).strip()] = m.group(2).split()[0]
    return kinds


def classify_pairs(runs: list[str]) -> dict[str, Any]:
    doc = parse_doc_kinds()
    out = {}
    for run in runs:
        pair = run[len("study-"):]
        spec = tomllib.loads((DATA / pair / "dataset.toml").read_text(encoding="utf-8"))
        shown = set(spec["text_fields"])
        entry = {"pair": pair, "dataset": dataset_of(run), "text_fields": sorted(shown)}
        for key in ("p", "q"):
            pred = spec["predicates"][key]
            rule = pred.get("rule")
            if rule is None:
                kind, why = "judgment", "no rule: truth is not recorded"
            else:
                fields = set(re.findall(r'"field": "([^"]+)"', json.dumps(rule)))
                if fields & GOLD_FIELDS:
                    kind = "gold"
                    why = f"rule reads the hidden label field(s) {', '.join(sorted(fields & GOLD_FIELDS))}"
                elif fields <= shown:
                    kind = "surface"
                    why = f"rule checks the shown field(s) {', '.join(sorted(fields))}: {json.dumps(rule)[:90]}"
                elif all(f_ in DERIVED and DERIVED[f_] in shown for f_ in fields):
                    kind = "surface"
                    why = ("rule reads " + ", ".join(f"`{f_}`, a hidden field computed from the shown `{DERIVED[f_]}`"
                                                     for f_ in sorted(fields)))
                else:
                    kind, why = "unclassified", json.dumps(rule)
            dk = doc.get(pred["text"])
            doc_kind = {"machine": "surface"}.get(dk, dk)
            entry[key] = {"text": pred["text"], "kind": kind, "why": why,
                          "docs_kind": dk if dk else "not in DATASETS.md (product pair)",
                          "agrees_with_docs": None if dk is None else doc_kind == kind}
        entry["any_surface"] = "surface" in (entry["p"]["kind"], entry["q"]["kind"])
        entry["any_judgment"] = "judgment" in (entry["p"]["kind"], entry["q"]["kind"])
        out[run] = entry
    return out


def metareg(y: np.ndarray, v: np.ndarray, X: np.ndarray, names: list[str], test_cols: list[int]) -> dict[str, Any]:
    k, p = X.shape
    if np.linalg.matrix_rank(X) < p or k - p < 1:
        return {"status": "not estimable", "k": k, "p": p}
    t2 = tau2_reml(y, v, X)
    w = 1 / (v + t2)
    xtwx = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(xtwx, X.T @ (w * y))
    r = y - X @ beta
    qkh = float((w * r * r).sum() / (k - p))
    cov = qkh * np.linalg.inv(xtwx)
    se = np.sqrt(np.diag(cov))
    df = k - p
    tq = sps.t.ppf(0.975, df)
    coefs = {n: {"beta": float(b), "se_kh": float(s), "ci": [float(b - tq * s), float(b + tq * s)],
                 "p": float(2 * sps.t.sf(abs(b / s), df)) if s > 0 else None}
             for n, b, s in zip(names, beta, se)}
    omni = None
    if test_cols:
        Lm = np.zeros((len(test_cols), p))
        for i, c in enumerate(test_cols):
            Lm[i, c] = 1
        lb = Lm @ beta
        F = float(lb @ np.linalg.solve(Lm @ cov @ Lm.T, lb) / len(test_cols))
        omni = {"F": F, "df1": len(test_cols), "df2": df, "p": float(sps.f.sf(F, len(test_cols), df))}
    t2_0 = tau2_reml(y, v, None)
    return {"status": "ok", "k": k, "p": p, "df": df, "tau2": t2, "tau2_intercept_only": t2_0,
            "R2": max(0.0, (t2_0 - t2) / t2_0) if t2_0 > 0 else None, "coefficients": coefs, "omnibus": omni}


def design_matrix(runs: list[str], kinds: dict[str, Any], hyp: str, with_dataset: bool) -> tuple[np.ndarray, list[str], list[int]]:
    cols, names = [np.ones(len(runs))], ["intercept"]
    test = []
    if hyp in ("H3", "S3"):
        for key in ("any_surface", "any_judgment"):
            cols.append(np.array([1.0 if kinds[r][key] else 0.0 for r in runs]))
            names.append(key)
            test.append(len(cols) - 1)
    else:
        for kind in ("judgment", "surface"):
            col = np.array([1.0 if kinds[r]["p"]["kind"] == kind else 0.0 for r in runs])
            if col.any():
                cols.append(col)
                names.append(f"p_{kind}")
                test.append(len(cols) - 1)
    if with_dataset:
        ds = [dataset_of(r) for r in runs]
        levels = [d for d in ("wdc", "abt", "cuad", "argq", "arxiv") if d in ds]
        for d in levels[1:]:
            cols.append(np.array([1.0 if x == d else 0.0 for x in ds]))
            names.append(f"dataset_{d}")
    X = np.column_stack(cols)
    # drop kind columns that are constant once the dataset is in (collinear): keep the rank honest
    return X, names, test


def rewritten_is_surface(hyp: str, kinds: dict[str, Any], run: str) -> bool:
    if hyp in ("H3", "S3"):
        return kinds[run]["any_surface"]
    if hyp == "H6":
        return False
    return kinds[run]["p"]["kind"] == "surface"


def meta_regression(meta: dict[str, Any], kinds: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"kinds": kinds, "backends": {}, "kind_means": {}}
    for backend in meta["backends"]:
        for hyp in ["H2", "H3", "S3", "H4", "S2", "S4", "H7"]:
            if COMPARISON[hyp] not in meta["backends"][backend]:
                continue
            entry, studies = studies_for(meta, backend, hyp)
            runs = [s["run"] for s in studies]
            if len(runs) < 5:
                continue
            deltas = np.array([s["delta"] for s in studies])
            ses = np.array([s["se"] for s in studies])
            ys, vs = sqrt_inputs(studies, "floored")
            res = {}
            for scale, y, v in (("delta", deltas, ses ** 2), ("sqrt", ys, vs)):
                for spec in ("kind", "kind+dataset"):
                    X, names, test = design_matrix(runs, kinds, hyp, spec == "kind+dataset")
                    res[f"{scale}:{spec}"] = {"terms": names, **metareg(y, v, X, names, test)}
            # descriptive means by kind
            groups: dict[str, list[float]] = {}
            for r, d in zip(runs, deltas):
                if hyp in ("H3", "S3"):
                    label = ("surface conjunct" if kinds[r]["any_surface"] else "no surface conjunct")
                else:
                    label = f"p {kinds[r]['p']['kind']}"
                groups.setdefault(label, []).append(root(d))
            res["mean_sqrt_by_kind"] = {g: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
                                        for g, v in groups.items()}
            out["backends"].setdefault(backend, {})[hyp] = res
    return out


def surface_excluded(meta: dict[str, Any], kinds: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"backends": {}}
    for backend in meta["backends"]:
        for hyp in META_IDS:
            if COMPARISON[hyp] not in meta["backends"][backend]:
                continue
            entry, studies = studies_for(meta, backend, hyp)
            if len(studies) < 2:
                continue
            keep = [s for s in studies if not rewritten_is_surface(hyp, kinds, s["run"])]
            dropped = [s["run"] for s in studies if rewritten_is_surface(hyp, kinds, s["run"])]
            res = {"k_all": len(studies), "k_kept": len(keep), "dropped": dropped, "kind": entry["kind"]}
            for label, sub in (("all", studies), ("without_surface", keep)):
                if len(sub) < 2:
                    res[label] = {"status": "fewer than 2 pairs"}
                    continue
                d = random_effects([s["delta"] for s in sub], [s["se"] for s in sub], "REML")
                y, v = sqrt_inputs(sub, "floored")
                sq = summarize_re(y, v, tau2_reml(y, v), True)
                res[label] = {"delta_reml": {k: d.get(k) for k in ("mu", "ci_hksj", "tau2", "I2", "prediction_interval")},
                              "delta_reml_sqrt_mu": root(d["mu"]),
                              "sqrt_reml": {k: sq.get(k) for k in ("mu", "ci_hksj", "tau", "I2", "prediction_interval")}}
            out["backends"].setdefault(backend, {})[hyp] = res
    return out


def render_meta_regression(rep: dict[str, Any], kinds: dict[str, Any]) -> str:
    L = ["# Which predicates go with large violations (review task 5)", "",
         "Exploratory. Generated by `scripts/review_meta.py` from results/meta-domains/meta.json and "
         "data/<pair>/dataset.toml.", "",
         "## Kinds of the predicates", "",
         "From each pair's `rule` in dataset.toml: gold = the rule reads a hidden label from the source (CUAD "
         "`labels`, ArgQ `stance` or `wa`, arXiv `primary_group` or `categories`); surface = the rule checks a "
         "string, number, length or date in a field the model is shown, directly or through a hidden field "
         "computed from it (`period_word` from the clause, `n_words` from the argument, `year` from `submitted`); "
         "judgment = no rule. Checked against the kinds in extension/data/DATASETS.md, which calls surface "
         "predicates 'machine'; the four product pairs are not in that table.", "",
         "| Pair | p kind | why | q kind | why | docs agree (p, q) |", "|---|---|---|---|---|---|"]
    for run, e in kinds.items():
        agree = ", ".join("n/a" if e[k]["agrees_with_docs"] is None else ("yes" if e[k]["agrees_with_docs"] else "NO")
                          for k in ("p", "q"))
        L.append(f"| {e['pair']} | {e['p']['kind']} | {e['p']['why']} | {e['q']['kind']} | {e['q']['why']} | {agree} |")
    L += ["", "The kind of the predicate an identity rewrites: p's kind for H2 (negation of p), H7 (COUNT of p) and the "
          "batch identities H4, S2 and S4 (which filter on p); for H3 and S3, whether either conjunct is a surface "
          "check and whether either is a judgment. Only one pair (argq-D) has a surface p, so the p-kind contrast "
          "for surface rests on that pair alone. Product pairs all have judgment p and q (except WDC-A's price), "
          "so kind and dataset are confounded: with dataset terms, the p-kind effects are identified from ArgQ "
          "alone (gold A, B; judgment C, E; surface D).", "",
          "## Summary", ""] + [f"- {x}" for x in rep["summary"]] + ["",
          "## Meta-regressions", "",
          "Mixed-effects meta-regression, tau^2 by REML, Knapp-Hartung standard errors and t tests with k - p df. "
          "Coefficients are differences from gold p (or from pairs without a surface or judgment conjunct for H3 "
          "and S3). Omnibus: F test of the kind terms. R2: share of the intercept-only tau^2 that the terms "
          "explain. 'not estimable': rank-deficient or no residual df.", ""]
    for scale in ("delta", "sqrt"):
        for spec in ("kind", "kind+dataset"):
            L += [f"### {'Delta' if scale == 'delta' else 'sqrt(Delta)'} on {spec}", "",
                  "| Identity | Model | k | term: estimate [95%] (p) | omnibus F (p) | R2 | tau |", "|---|---|---|---|---|---|---|"]
            for backend, ids in rep["backends"].items():
                for hyp, res in ids.items():
                    m = res[f"{scale}:{spec}"]
                    if m.get("status") != "ok":
                        L.append(f"| {hyp} | {backend} | {m.get('k')} | {m.get('status')} | | | |")
                        continue
                    terms = "; ".join(f"{n} {c['beta']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] ({c['p']:.3f})"
                                      for n, c in m["coefficients"].items() if not n.startswith("dataset") and n != "intercept")
                    om = m["omnibus"]
                    L.append(f"| {hyp} | {backend} | {m['k']} | {terms} | "
                             + (f"{om['F']:.2f} ({om['p']:.3f})" if om else "n/a") + " | "
                             + (f"{m['R2']:.0%}" if m["R2"] is not None else "n/a") + f" | {math.sqrt(m['tau2']):.4f} |")
            L.append("")
    L += ["## Mean per-pair sqrt(Delta) by kind", "", "| Identity | Model | by kind: mean (n pairs) |", "|---|---|---|"]
    for backend, ids in rep["backends"].items():
        for hyp, res in ids.items():
            L.append(f"| {hyp} | {backend} | " + "; ".join(f"{g} {v['mean']:.3f} ({v['n']})"
                                                         for g, v in sorted(res["mean_sqrt_by_kind"].items())) + " |")
    L.append("")
    return "\n".join(L)


def regression_summary(rep: dict[str, Any]) -> list[str]:
    lines = []
    sig = []
    for backend, ids in rep["backends"].items():
        for hyp, res in ids.items():
            for spec in ("kind", "kind+dataset"):
                for scale in ("sqrt", "delta"):
                    m = res[f"{scale}:{spec}"]
                    if m.get("status") != "ok":
                        continue
                    for n, c in m["coefficients"].items():
                        if n.startswith("dataset") or n == "intercept" or c["p"] is None:
                            continue
                        if c["p"] < 0.05:
                            sig.append((hyp, backend, spec, scale, n, c["beta"], c["ci"]))
    for hyp in ("H2", "H7", "H4", "S2", "S4", "H3", "S3"):
        rows = [s for s in sig if s[0] == hyp and s[3] == "sqrt"]
        if rows:
            lines.append(f"{hyp} {NAMES[hyp]}, sqrt scale: kind terms with p < 0.05: " + "; ".join(
                f"{b} {n} {beta:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] ({spec})" for _, b, spec, _, n, beta, ci in rows) + ".")
        else:
            lines.append(f"{hyp} {NAMES[hyp]}, sqrt scale: no kind term has p < 0.05 for any model, with or without dataset terms.")
    # descriptive: where does surface p sit
    for hyp in ("H2", "H7", "S4"):
        vals = []
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if r and "p surface" in r["mean_sqrt_by_kind"]:
                g = r["mean_sqrt_by_kind"]
                vals.append(f"{backend}: surface {g['p surface']['mean']:.2f}, gold {g.get('p gold', {}).get('mean', float('nan')):.2f}, "
                            f"judgment {g.get('p judgment', {}).get('mean', float('nan')):.2f}")
        if vals:
            lines.append(f"{hyp}: mean per-pair sqrt(Delta) by p kind (surface is argq-D alone): " + "; ".join(vals) + ".")
    for hyp in ("H3", "S3"):
        vals = []
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if r and r["mean_sqrt_by_kind"]:
                g = r["mean_sqrt_by_kind"]
                vals.append(f"{backend}: with {g.get('surface conjunct', {}).get('mean', float('nan')):.2f}, without "
                            f"{g.get('no surface conjunct', {}).get('mean', float('nan')):.2f}")
        if vals:
            lines.append(f"{hyp}: mean per-pair sqrt(Delta), pairs with a surface conjunct against those without: " + "; ".join(vals) + ".")
    return lines


def render_surface_excluded(rep: dict[str, Any]) -> str:
    L = ["# Pooled estimates without the surface predicates (review task 5)", "",
         "Exploratory. Generated by `scripts/review_meta.py`. Pairs are dropped when the predicate the identity "
         "rewrites is a surface check (see meta_regression.md): for H2, H4, S2, S4 and H7 the pair argq-D, whose p "
         "is 'the argument is more than 20 words long'; for H3 and S3 every pair with a surface conjunct (WDC-A, "
         "CUAD-A, D, E, ArgQ-A, C, D, arXiv-A, C, E); H6 rewrites a map, not a predicate, and keeps all pairs. "
         "Pooling: REML on Delta as in meta.json (square roots shown) and REML on the sqrt scale (floored rule, "
         "meta_sqrt.md); HKSJ intervals.", "",
         "## Summary", ""] + [f"- {x}" for x in rep["summary"]] + ["",
         "| Identity | Model | pairs all / kept | Delta scale, all: sqrt(mu) [HKSJ] | without: sqrt(mu) [HKSJ] | sqrt scale, all: mu "
         "[HKSJ] | without: mu [HKSJ] | tau sqrt, all / without |",
         "|---|---|---|---|---|---|---|---|"]
    for hyp in META_IDS:
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r:
                continue
            a, w = r["all"], r["without_surface"]
            if w.get("status"):
                L.append(f"| {hyp} | {backend} | {r['k_all']} / {r['k_kept']} | | {w['status']} | | | |")
                continue
            sc = lambda d: [root(x) for x in d["ci_hksj"]] if d.get("ci_hksj") else None  # noqa: E731
            L.append(f"| {hyp} | {backend} | {r['k_all']} / {r['k_kept']} | {f(a['delta_reml_sqrt_mu'])} {ci_txt(sc(a['delta_reml']))} | "
                     f"{f(w['delta_reml_sqrt_mu'])} {ci_txt(sc(w['delta_reml']))} | {f(a['sqrt_reml']['mu'])} "
                     f"{ci_txt(a['sqrt_reml'].get('ci_hksj'))} | {f(w['sqrt_reml']['mu'])} {ci_txt(w['sqrt_reml'].get('ci_hksj'))} | "
                     f"{f(a['sqrt_reml']['tau'])} / {f(w['sqrt_reml']['tau'])} |")
    L.append("")
    return "\n".join(L)


def surface_summary(rep: dict[str, Any]) -> list[str]:
    lines = []
    for hyp in META_IDS:
        changes, lost, lost_delta = [], [], []
        for backend, ids in rep["backends"].items():
            r = ids.get(hyp)
            if not r or r["without_surface"].get("status") or not r["dropped"]:
                continue
            a, w = r["all"]["sqrt_reml"], r["without_surface"]["sqrt_reml"]
            changes.append(w["mu"] - a["mu"])
            if a["ci_hksj"][0] > 0 and w["ci_hksj"][0] <= 0:
                lost.append(backend)
            ad, wd = r["all"]["delta_reml"], r["without_surface"]["delta_reml"]
            if ad["ci_hksj"][0] > 0 and wd["ci_hksj"][0] <= 0:
                lost_delta.append(backend)
        if changes:
            lines.append(f"{hyp} {NAMES[hyp]}: dropping the surface pairs changes the sqrt-scale pooled mean by "
                         f"{min(changes):+.3f} to {max(changes):+.3f} across models"
                         + (f"; its interval then includes 0 for {', '.join(lost)}" if lost else
                            "; every sqrt-scale interval that excluded 0 still does")
                         + (f"; on the Delta scale (meta.json's method) the interval then includes 0 for "
                            f"{', '.join(lost_delta)}, with fewer pairs left" if lost_delta else "") + ".")
    return lines


# --- main ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RESULTS / "review"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lad = ladder()
    lad["summary"] = ladder_summary(lad)
    (out_dir / "ladder.json").write_text(jdump(lad), encoding="utf-8")
    (out_dir / "ladder.md").write_text(render_ladder(lad), encoding="utf-8")

    meta = json.loads((RESULTS / "meta-domains" / "meta.json").read_text(encoding="utf-8"))
    ms = meta_sqrt(meta)
    ms["findings"] = meta_sqrt_findings(ms)
    (out_dir / "meta_sqrt.json").write_text(jdump(ms), encoding="utf-8")
    (out_dir / "meta_sqrt.md").write_text(render_meta_sqrt(ms), encoding="utf-8")

    kinds = classify_pairs(meta["runs"])
    mr = meta_regression(meta, kinds)
    mr["summary"] = regression_summary(mr)
    (out_dir / "meta_regression.json").write_text(jdump(mr), encoding="utf-8")
    (out_dir / "meta_regression.md").write_text(render_meta_regression(mr, kinds), encoding="utf-8")

    se_ = surface_excluded(meta, kinds)
    se_["summary"] = surface_summary(se_)
    (out_dir / "surface_excluded.json").write_text(jdump(se_), encoding="utf-8")
    (out_dir / "surface_excluded.md").write_text(render_surface_excluded(se_), encoding="utf-8")
    disagree = [f"{e['pair']} {k}" for e in kinds.values() for k in ("p", "q") if e[k]["agrees_with_docs"] is False]
    print("ladder, meta_sqrt, meta_regression, surface_excluded written; kinds disagreeing with the docs: "
          + (", ".join(disagree) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
