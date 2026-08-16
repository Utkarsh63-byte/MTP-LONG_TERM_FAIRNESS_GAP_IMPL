"""Weight sweeps and Pareto fronts, with BOTH gaps active in every configuration.

Hard constraint enforced here, not assumed: every swept configuration keeps the
Gap 1b term (rate fairness, within + between) and the Gap 2 term (utilisation)
switched on. `_assert_both_gaps_active` fails loudly otherwise. Turning a term off
is a legitimate *ablation* -- it lives in Table 2 as a diagnostic -- but it must
never silently become the method being tuned.

Search strategy
---------------
A full grid over three lambdas would be ~170 configurations at roughly 1.5 min
each, so this runs a *coordinate search* instead: sweep one weight at a time,
holding the others at the best value found so far, then refine. That is stated
plainly because it does not certify a global optimum -- it finds a good operating
point and, more importantly, traces the trade-off curves a reviewer will ask for.

An exploratory pass showed parity sits outside the obvious range and that the two
gap weights interact rather than acting independently:
    lambda_between 0.1 -> 1.0   moved PT/FT ratio 0.562 -> 0.677  (towards parity)
    lambda_util    4.0 -> 0.25  moved PT/FT ratio 0.581 -> 0.749  (towards parity)
So both a *higher* between-group weight and a *lower* utilisation weight push
towards group parity, and the ranges below are extended accordingly.

Stages
  A  lambda_between, extended upwards
  B  lambda_util at the best lambda_between
  C  lambda_within at the best (lambda_between, lambda_util)
  D  local refinement around the best point
  Pareto fronts are then extracted over every configuration evaluated.

Run:  .venv/bin/python -m scripts.run_sweeps
      .venv/bin/python -m scripts.run_sweeps --quick
"""
from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import pandas as pd

from ltf.config import Config
from ltf.experiments import prepare
from ltf.metrics import compute
from ltf.methods.momaql import MOMAQL
from ltf.utils import LOG, output_path

# Gap 2's term must stay strictly positive; this floor is what stops a sweep from
# quietly optimising Gap 1b alone.
MIN_LAMBDA_UTIL = 0.05
MIN_LAMBDA_RATE = 0.05


def _assert_both_gaps_active(lam_within: float, lam_between: float,
                             lam_util: float, use_util: bool) -> None:
    if not use_util or lam_util < MIN_LAMBDA_UTIL:
        raise ValueError(
            f"Gap 2 term inactive (use_utilisation={use_util}, "
            f"lambda_util={lam_util}). Both gaps must stay active in a sweep.")
    if max(lam_within, lam_between) < MIN_LAMBDA_RATE:
        raise ValueError(
            f"Gap 1b term inactive (lambda_within={lam_within}, "
            f"lambda_between={lam_between}). Both gaps must stay active.")


def run_config(bench, lam_within: float, lam_between: float, lam_util: float,
               episodes: int, label: str) -> dict:
    """Train and score one weight configuration. Both gap terms stay on."""
    _assert_both_gaps_active(lam_within, lam_between, lam_util, True)
    cfg = copy.deepcopy(bench.cfg)
    cfg.fairness.target = "rate_grouped"
    cfg.fairness.lambda_within = lam_within
    cfg.fairness.lambda_between = lam_between
    cfg.fairness.lambda_util = lam_util
    # keep the calibrated weights fixed so lambda is the only thing varying
    cfg.fairness.omega = bench.omegas["omega_total"]
    cfg.fairness.omega_rate = bench.omegas["omega_rate"]
    cfg.fairness.omega_util = bench.omegas["omega_util"]

    agent = MOMAQL(cfg, bench.n_nodes, name=label, use_lookahead=True,
                   fairness_target="rate_grouped", use_utilisation=True)
    assert agent.lam_util >= MIN_LAMBDA_UTIL, "Gap 2 weight lost in MOMAQL init"
    agent.train([bench.real_sc, bench.synth_sc], episodes=episodes)
    res = bench.env.run(agent, method=label)
    met = compute(res, bench.drivers, cfg)
    d = met.to_dict()
    d.update({"lambda_within": lam_within, "lambda_between": lam_between,
              "lambda_util": lam_util,
              "parity_error": abs(met.rate_group_ratio - 1.0)})
    return d


def pareto_front(df: pd.DataFrame, x: str, y: str,
                 maximise_x: bool = True, minimise_y: bool = True) -> pd.DataFrame:
    """Non-dominated rows on (x, y)."""
    pts = df[[x, y]].to_numpy(float)
    keep = np.ones(len(df), bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            bx = pts[j, 0] >= pts[i, 0] if maximise_x else pts[j, 0] <= pts[i, 0]
            by = pts[j, 1] <= pts[i, 1] if minimise_y else pts[j, 1] >= pts[i, 1]
            strict = (pts[j, 0] != pts[i, 0]) or (pts[j, 1] != pts[i, 1])
            if bx and by and strict:
                keep[i] = False
                break
    return df[keep].sort_values(x)


def show(df: pd.DataFrame, cols: list[str], title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df[cols].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    episodes = 8 if args.quick else 30
    bench = prepare(cfg)

    KEY = ["lambda_within", "lambda_between", "lambda_util", "total_utility",
           "fairness_total_var", "rate_var", "rate_var_within",
           "rate_var_between", "rate_full_time", "rate_part_time",
           "rate_group_ratio", "parity_error", "util_mean", "util_var",
           "util_adj_var", "n_idle_drivers", "service_rate"]

    all_rows: list[dict] = []
    seen: set[tuple] = set()

    def evaluate(w: float, b: float, u: float, tag: str) -> dict:
        key = (round(w, 4), round(b, 4), round(u, 4))
        for r in all_rows:
            if (round(r["lambda_within"], 4), round(r["lambda_between"], 4),
                    round(r["lambda_util"], 4)) == key:
                return r
        r = run_config(bench, w, b, u, episodes, tag)
        all_rows.append(r)
        return r

    def best_parity(rows: list[dict]) -> dict:
        return min(rows, key=lambda r: r["parity_error"])

    # ---------------- Stage A: lambda_between, extended upwards ----------
    betas = [0.5, 1.0, 4.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    rows_a = [evaluate(1.0, b, 0.25, f"A: lam_between={b}") for b in betas]
    A = pd.DataFrame(rows_a)
    show(A, KEY, "STAGE A  lambda_between (lambda_within=1.0, lambda_util=0.25; "
                 "both gaps ON)")
    best_b = float(best_parity(rows_a)["lambda_between"])
    print(f"\n  best parity so far: lambda_between={best_b}, "
          f"ratio {best_parity(rows_a)['rate_group_ratio']:.3f}, "
          f"error {best_parity(rows_a)['parity_error']:.3f}")

    # ---------------- Stage B: lambda_util ------------------------------
    utils = [0.05, 0.25, 1.0] if args.quick else [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    rows_b = [evaluate(1.0, best_b, u, f"B: lam_util={u}") for u in utils]
    B = pd.DataFrame(rows_b)
    show(B, KEY, f"STAGE B  lambda_util at lambda_between={best_b} "
                 f"(Gap 1b stays ON)")
    best_u = float(best_parity(rows_b)["lambda_util"])
    print(f"\n  best parity so far: lambda_util={best_u}, "
          f"ratio {best_parity(rows_b)['rate_group_ratio']:.3f}, "
          f"error {best_parity(rows_b)['parity_error']:.3f}")

    # ---------------- Stage C: lambda_within ----------------------------
    withins = [0.5, 1.0] if args.quick else [0.1, 0.25, 0.5, 1.0, 2.0]
    rows_c = [evaluate(w, best_b, best_u, f"C: lam_within={w}") for w in withins]
    C_ = pd.DataFrame(rows_c)
    show(C_, KEY, f"STAGE C  lambda_within at lambda_between={best_b}, "
                  f"lambda_util={best_u}")
    best_w = float(best_parity(rows_c)["lambda_within"])
    print(f"\n  best parity so far: lambda_within={best_w}, "
          f"ratio {best_parity(rows_c)['rate_group_ratio']:.3f}, "
          f"error {best_parity(rows_c)['parity_error']:.3f}")

    # ---------------- Stage D: local refinement -------------------------
    if not args.quick:
        for b in [best_b * 0.5, best_b * 2.0]:
            for u in [max(MIN_LAMBDA_UTIL, best_u * 0.5), best_u * 2.0]:
                evaluate(best_w, b, u, f"D: w={best_w},b={b},u={u}")
    C = pd.DataFrame(all_rows).drop_duplicates(
        subset=["lambda_within", "lambda_between", "lambda_util"])
    show(C.sort_values("parity_error"), KEY,
         "ALL CONFIGURATIONS, sorted by distance from group parity "
         "(both gaps ON in every row)")

    bp = C.loc[C.parity_error.idxmin()]
    print(f"\n  PARITY-BEST: lambda_within={bp.lambda_within}, "
          f"lambda_between={bp.lambda_between}, lambda_util={bp.lambda_util}")
    print(f"    PT/FT hourly-rate ratio {bp.rate_group_ratio:.3f} "
          f"(parity = 1.000, error {bp.parity_error:.3f})")
    print(f"    utility {bp.total_utility:,.1f}, rate_var {bp.rate_var:.3f}, "
          f"between {bp.rate_var_between:.3f}, util_adj_var {bp.util_adj_var:.3f}, "
          f"idle {int(bp.n_idle_drivers)}")

    # ---------------- Pareto fronts ----------------
    for y, name in [("rate_var", "Gap 1b: Var(hourly rate)"),
                    ("rate_var_between", "Gap 1b: between-group Var"),
                    ("util_adj_var", "Gap 2: opportunity-normalised Var(util)"),
                    ("parity_error", "Gap 1b: distance from group parity")]:
        f = pareto_front(C, "total_utility", y)
        show(f, ["lambda_within", "lambda_between", "lambda_util",
                 "total_utility", y],
             f"PARETO FRONT  total utility vs {name}")

    A.to_csv(output_path("sweep_lambda_between.csv"), index=False)
    B.to_csv(output_path("sweep_lambda_util.csv"), index=False)
    C.to_csv(output_path("sweep_joint_grid.csv"), index=False)
    print("\nwrote outputs/sweep_lambda_between.csv, sweep_lambda_util.csv, "
          "sweep_joint_grid.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
