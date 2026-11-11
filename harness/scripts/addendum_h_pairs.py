"""Addendum H's pair-level analysis: do logical rewrites change answers more than rewordings that keep the meaning,
with the predicate pair as the unit?

For each primary arm and each of Addendum G's contrasts (extension/placebo.py: C1, C2 and the four secondary ones),
the per-pair differences in mean excess divergence and their standard errors (from the per-pair t intervals of the
pooled placebo outputs, df = items - 1) are pooled by REML, with the Knapp-Hartung (HKSJ) interval and a 95%
prediction interval (extension.meta.random_effects, pre-registered in Addendum C). The HKSJ p value is two-sided, t
with k - 1 df.
- Confirmatory: C1 and C2 over Addendum H's 15 pairs, none of them chosen by its results. Per arm, Holm over the two
  HKSJ p values. A contrast exceeds across pairs for an arm if its Holm p < 0.05 and its mean is positive; it does so
  in general if it does for at least two of the three primary arms.
- Secondary: the four secondary contrasts over the 15 pairs (Holm over the four, per arm); and every contrast over
  all 19 pairs (Addendum G's four and H's fifteen), reported without a decision, since G's four were chosen for
  large violations and their values were known when this analysis was fixed.
A pair whose interval has zero width (every item's contrast is equal) gets the smallest positive standard error of
the same arm and contrast.

    python3 scripts/addendum_h_pairs.py results/placebo/placebo.json results/placebo-h/placebo.json \\
        --out results/placebo-h                     # from harness/; writes pairs.json and pairs.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from extension.meta import random_effects, t_quantile  # noqa: E402
from semaudit.stats import holm, t_sf  # noqa: E402

PRIMARY = ["haiku-5.5", "gpt-5.6-luna", "gemini-3.5-flash-lite"]
CONFIRMATORY = ["C1", "C2"]
SECONDARY = ["H2_vs_tmpl", "H2_vs_order", "S3_vs_pqpara", "H3_vs_S3"]


def per_pair(placebo: dict, arm: str, contrast: str) -> list[dict]:
    """The contrast's per-pair estimates: run, difference, standard error (None for a zero-width interval)."""
    rows = []
    for c in placebo["backends"][arm]["contrasts"]:
        if c["contrast"] != contrast:
            continue
        for r in c["by_run"]:
            if r.get("status") != "ok":
                continue
            lo, hi = r["difference_ci"]
            se = (hi - lo) / (2 * t_quantile(0.975, r["n_items"] - 1))
            rows.append({"run": r["run"], "difference": r["difference"], "se": se if se > 0 else None,
                         "n_items": r["n_items"]})
    return rows


def pool(rows: list[dict]) -> dict:
    positive = [r["se"] for r in rows if r["se"]]
    if len(rows) < 2 or not positive:
        return {"k": len(rows), "status": "insufficient data"}
    floored = [r["run"] for r in rows if not r["se"]]
    ses = [r["se"] or min(positive) for r in rows]
    fit = random_effects([r["difference"] for r in rows], ses, method="REML")
    fit["p_hksj"] = 2 * t_sf(abs(fit["mu"]) / fit["se_hksj"], fit["k"] - 1) if fit["se_hksj"] > 0 else 1.0
    fit["floored"] = floored
    fit["pairs"] = [{"run": r["run"], "difference": r["difference"], "se": s} for r, s in zip(rows, ses)]
    return fit


def analyze(g: dict, h: dict) -> dict:
    out: dict = {"arms": {}, "decision": {}}
    for arm in PRIMARY:
        res: dict = {"h15": {}, "all19": {}}
        for contrast in CONFIRMATORY + SECONDARY:
            hrows = per_pair(h, arm, contrast)
            res["h15"][contrast] = pool(hrows)
            res["all19"][contrast] = pool(per_pair(g, arm, contrast) + hrows)
        for family in (CONFIRMATORY, SECONDARY):
            fits = [res["h15"][c] for c in family]
            ps = [f.get("p_hksj", 1.0) for f in fits]
            for f, adj in zip(fits, holm(ps)):
                f["p_holm"] = adj
                f["exceeds"] = bool(f.get("status") != "insufficient data" and adj < 0.05 and f["mu"] > 0)
        out["arms"][arm] = res
    for contrast in CONFIRMATORY:
        n = sum(out["arms"][a]["h15"][contrast]["exceeds"] for a in PRIMARY)
        out["decision"][contrast] = {"arms_exceeding": n, "in_general": n >= 2}
    return out


def render(res: dict) -> str:
    def row(arm: str, contrast: str, f: dict, decided: bool) -> str:
        if f.get("status") == "insufficient data":
            return f"| {arm} | {contrast} | {f['k']} | insufficient data | | | | |"
        lo, hi = f["ci_hksj"]
        pi = f.get("prediction_interval", [math.nan, math.nan])
        verdict = ("exceeds" if f["exceeds"] else "no") if decided else ""
        holm_p = f"{f['p_holm']:.3g}" if decided else "n/a"
        return (f"| {arm} | {contrast} | {f['k']} | {f['mu']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {f['tau']:.4f} | "
                f"[{pi[0]:+.4f}, {pi[1]:+.4f}] | {holm_p} | {verdict} |")

    head = ["| Arm | Contrast | Pairs | REML mean [HKSJ 95%] | τ | 95% prediction | Holm p | Verdict |",
            "|---|---|---|---|---|---|---|---|"]
    lines = ["# Addendum H: logical rewrites against rewordings, with the pair as the unit", "",
             "## Confirmatory and secondary: Addendum H's 15 pairs", "", *head]
    for arm in PRIMARY:
        for c in CONFIRMATORY + SECONDARY:
            lines.append(row(arm, c, res["arms"][arm]["h15"][c], True))
    lines += ["", "Decision (confirmatory): " + "; ".join(
        f"{c} exceeds across pairs for {d['arms_exceeding']} of 3 primary arms"
        f"{', in general' if d['in_general'] else ''}" for c, d in res["decision"].items()) + ".", "",
        "## Secondary, without a decision: all 19 pairs (Addendum G's four, chosen for large violations, and H's 15)",
        "", *head]
    for arm in PRIMARY:
        for c in CONFIRMATORY + SECONDARY:
            lines.append(row(arm, c, res["arms"][arm]["all19"][c], False))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("g_pooled", type=Path, help="Addendum G's pooled placebo.json")
    ap.add_argument("h_pooled", type=Path, help="Addendum H's pooled placebo.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = analyze(json.loads(args.g_pooled.read_text(encoding="utf-8")),
                  json.loads(args.h_pooled.read_text(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    text = render(res)
    (args.out / "pairs.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
