"""Simulation-based calibration and power for the binary excess-divergence test.

Item model: most items are easy (P(True) of 0.01 or 0.99); a share are ambiguous with P(True)
uniform on [0.25, 0.75]. Under the alternative, a share of items has its P(True) under condition B
moved toward the opposite answer by `shift`. With affected = 0 the rejection rate is the test's
false-positive rate, which must not exceed alpha.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import stable_seed
from .stats import _binary_permutation


def simulate(n_items: int, k: int, affected: float, shift: float, n_sims: int, n_perm: int,
             alphas: tuple[float, ...], seed: int, hard_frac: float = 0.15) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rejections = np.zeros(len(alphas))
    true_delta = []
    for _ in range(n_sims):
        hard = rng.random(n_items) < hard_frac
        easy = np.where(rng.random(n_items) < 0.5, 0.01, 0.99)
        pi_a = np.where(hard, rng.uniform(0.25, 0.75, n_items), easy)
        moved = rng.random(n_items) < affected
        direction = np.where(pi_a >= 0.5, -1.0, 1.0)
        pi_b = np.clip(pi_a + moved * direction * shift, 0.005, 0.995)
        true_delta.append(float(((pi_a - pi_b) ** 2).mean()))
        A = (rng.random((n_items, k)) < pi_a[:, None]).astype(np.int64)
        B = (rng.random((n_items, k)) < pi_b[:, None]).astype(np.int64)
        p, _ = _binary_permutation(A, B, n_perm, rng)
        rejections += np.array([p <= a for a in alphas])
    return {
        "n_items": n_items, "repeats": k, "affected": affected, "shift": shift,
        "true_delta": float(np.mean(true_delta)),
        **{f"power@{a:.4g}": float(r / n_sims) for a, r in zip(alphas, rejections)},
    }


def power_grid(n_items_list=(100, 200, 300, 600), k: int = 3, affected_list=(0.0, 0.02, 0.05, 0.10),
               shifts=(0.3, 0.6), n_sims: int = 300, n_perm: int = 2000,
               alphas=(0.05, 0.05 / 7), seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    for n in n_items_list:
        for affected in affected_list:
            for shift in (shifts[:1] if affected == 0 else shifts):
                rows.append(simulate(n, k, affected, 0.0 if affected == 0 else shift, n_sims, n_perm,
                                     tuple(alphas), seed=stable_seed(n, affected, shift, seed) % 2**32))
    return rows


def pipeline_calibration(config: dict[str, Any], seeds: range, n_perm: int = 1000,
                         alpha: float = 0.05, sim_params: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    """False-positive rate of every comparison through the full pipeline, using the null simulator.

    Each seed draws new analysis units and new simulated answers; no effect is injected, so every
    rejection is a false positive.
    """
    import copy
    import json
    import shutil
    import tempfile
    from collections import defaultdict
    from pathlib import Path

    from .analyze import analyze_records
    from .offline import SimulatedBackend
    from .runner import find_log, read_log, run_experiment

    p_values: dict[str, list[float]] = defaultdict(list)
    p_perm_only: dict[str, list[float]] = defaultdict(list)
    deltas: dict[str, list[float]] = defaultdict(list)
    ci_above_zero: dict[str, list[bool]] = defaultdict(list)
    from .diagnostics import independent_subset_tests
    from .identities import collect
    with tempfile.TemporaryDirectory() as tmp:
        for s in seeds:
            cfg = copy.deepcopy(config)
            cfg["run"]["seed"] = s
            run_dir = run_experiment(cfg, f"cal{s}", [SimulatedBackend("sim", seed=s, **(sim_params or {}))],
                                     Path(tmp), progress_every=None)
            design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
            records = read_log(find_log(run_dir, "sim"))
            res = analyze_records(records, design, n_perm, 200, s, "sim")
            for name, row in independent_subset_tests(collect(records), design, n_perm=n_perm).get("tests", {}).items():
                if row.get("status") == "ok":
                    p_values[f"{name} (independent column)"].append(row["p"])
                    p_perm_only[f"{name} (independent column)"].append(row["p_perm"])
                    deltas[f"{name} (independent column)"].append(row["delta"])
            shutil.rmtree(run_dir)
            for row in res["comparisons"]:
                if row.get("status") == "ok":
                    p_values[row["comparison"]].append(row["p"])
                    p_perm_only[row["comparison"]].append(row["p_perm"])
                    deltas[row["comparison"]].append(row["delta"])
                    ci_above_zero[row["comparison"]].append(row["delta_ci"][0] > 0)
    return {name: {"runs": len(ps), "false_positive_rate": float(np.mean(np.array(ps) <= alpha)),
                   "false_positive_rate_strict": float(np.mean(np.array(ps) <= alpha / 7)),
                   "false_positive_rate_permutation_only": float(np.mean(np.array(p_perm_only[name]) <= alpha)),
                   "mean_delta": float(np.mean(deltas[name])),
                   "ci_lower_above_zero": float(np.mean(ci_above_zero[name])) if ci_above_zero.get(name) else None}
            for name, ps in p_values.items()}


def render(rows: list[dict[str, Any]]) -> str:
    keys = [k for k in rows[0] if k.startswith("power@")]
    lines = ["| Items | Repeats | Share affected | Shift | True Δ | " + " | ".join(keys) + " |",
             "|---" * (5 + len(keys)) + "|"]
    for r in rows:
        lines.append(f"| {r['n_items']} | {r['repeats']} | {r['affected']:.2f} | {r['shift']:.1f} | "
                     f"{r['true_delta']:.4f} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    return "\n".join(lines)
