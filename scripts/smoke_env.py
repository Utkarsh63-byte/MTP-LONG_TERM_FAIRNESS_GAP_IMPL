"""End-to-end smoke test: scenario -> environment -> policy -> metrics.

Runs the efficiency-only and paper-Greedy allocations and prints both the
paper's metric columns and the gap metrics, so the first real numbers can be
sanity-checked before the RL machinery lands.

Run:  .venv/bin/python -m scripts.smoke_env
"""
from __future__ import annotations

import sys

import numpy as np

from ltf.config import Config
from ltf.metrics import compute, check_decomposition
from ltf.methods.greedy import make_efficiency_only, make_paper_greedy
from ltf.sim.env import Environment
from ltf.sim.scenario import build_scenario
from ltf.utils import output_path

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    cfg = Config()
    sc = build_scenario(cfg)
    env = Environment(sc)

    print("\n" + "=" * 78)
    print("SMOKE: environment + greedy policies")
    print("=" * 78)

    runs = []
    for maker in (make_efficiency_only, make_paper_greedy):
        pol = maker(cfg)
        res = env.run(pol, method=pol.name)
        met = compute(res, sc.drivers, cfg)
        runs.append((res, met))

    print("\n-- paper's Table 1 columns --")
    print(runs[0][1].paper_header())
    for _, m in runs:
        print(m.paper_row())

    print("\n-- gap metrics --")
    hdr = (f"{'Method':<26} {'rate_var':>12} {'within':>12} {'between':>12} "
           f"{'FT rate':>9} {'PT rate':>9} {'util':>7} {'util_var':>9} "
           f"{'util_adj_var':>13} {'idle':>5}")
    print(hdr)
    for _, m in runs:
        print(f"{m.method:<26} {m.rate_var:>12,.2f} {m.rate_var_within:>12,.2f} "
              f"{m.rate_var_between:>12,.2f} {m.rate_full_time:>9.2f} "
              f"{m.rate_part_time:>9.2f} {m.util_mean:>7.3f} {m.util_var:>9.4f} "
              f"{m.util_adj_var:>13.4f} {m.n_idle_drivers:>5}")

    print("\n-- invariants --")
    res0, met0 = runs[0]
    res1, met1 = runs[1]
    check("assignments recorded consistently",
          res0.n_assigned == len(res0.assign_drv) == len(res0.assign_utility))
    check("per-driver utility reconciles with the assignment log",
          np.allclose(np.bincount(res0.assign_drv, weights=res0.assign_utility,
                                  minlength=sc.drivers.n), res0.cum_utility),
          "sum of logged utilities == cum_utility")
    check("busy slots reconcile with the log",
          np.array_equal(np.bincount(res0.assign_drv, weights=res0.assign_occupancy,
                                     minlength=sc.drivers.n).astype(int),
                         res0.busy_slots))
    check("no driver busier than they were online",
          bool(np.all(res0.busy_slots <= res0.online_slots)),
          f"max utilisation {met0.util_max:.3f}")
    check("no request served twice",
          len(np.unique(res0.assign_req)) == res0.n_assigned)
    check("service rate in (0,1]", 0 < res0.service_rate <= 1.0,
          f"{100*res0.service_rate:.1f}%")
    check("variance decomposition is exact (rate)",
          check_decomposition(res0.cum_utility / np.maximum(res0.online_hours, 1e-9),
                              res0.group),
          "within + between == total")
    check("efficiency-only beats fairness-weighted on total utility",
          met0.total_utility > met1.total_utility,
          f"{met0.total_utility:,.0f} vs {met1.total_utility:,.0f}")
    check("fairness-weighted beats efficiency-only on the paper's fairness",
          met1.fairness_total_var < met0.fairness_total_var,
          f"{met1.fairness_total_var:,.0f} vs {met0.fairness_total_var:,.0f}")

    print("\n-- versus the paper's Greedy row (Table 1: total utility -1,514,736) --")
    print(f"  Greedy total utility here: {met1.total_utility:,.1f}, "
          f"service rate {100*met1.service_rate:.1f}% "
          f"(efficiency-only: {100*met0.service_rate:.1f}%)")
    print("  NOT reproduced, and deliberately so: the paper's Greedy drives total")
    print("  utility negative, which means it keeps assigning trips whose utility")
    print("  is negative in order to level earnings. Our greedy stops when the")
    print("  marginal scalarised gain turns negative, using the no-action option")
    print("  the paper's own MDP defines (Sec. 4.3). Forcing assignments would")
    print("  reproduce the negative row but would be a straw-man baseline.")

    print("\n-- what the fairness objective did to the GROUP rate gap (Gap 1b) --")
    print(f"  efficiency-only: FT {met0.rate_full_time:.2f}/h vs "
          f"PT {met0.rate_part_time:.2f}/h  (ratio {met0.rate_group_ratio:.2f}x), "
          f"between-group Var {met0.rate_var_between:.3f}")
    print(f"  paper Greedy   : FT {met1.rate_full_time:.2f}/h vs "
          f"PT {met1.rate_part_time:.2f}/h  (ratio {met1.rate_group_ratio:.2f}x), "
          f"between-group Var {met1.rate_var_between:.3f}")
    print("  Optimising Var(totals) -- the paper's Eq. 2 -- pushed the part-time")
    print("  hourly rate UP and the full-time rate DOWN, widening the between-group")
    print("  gap it cannot measure. Levelling totals across unequal hours")
    print("  necessarily transfers rate from long-hours to short-hours drivers.")
    check("paper's fairness objective widens the between-group rate gap",
          met1.rate_var_between > met0.rate_var_between,
          f"between-group Var {met0.rate_var_between:.3f} -> "
          f"{met1.rate_var_between:.3f}")

    import pandas as pd
    pd.DataFrame([m.to_dict() for _, m in runs]).to_csv(
        output_path("smoke_greedy_metrics.csv"), index=False)
    print("\n  wrote outputs/smoke_greedy_metrics.csv")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("SMOKE TEST PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
