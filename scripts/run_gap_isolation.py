"""One-gap-at-a-time isolation, so each gap can be compared to the paper alone.

The main experiment table (run_experiments.py) evaluates every method inside the
same time-aware environment, which is the right control for comparing *fairness
objectives* but does not isolate the gaps individually. This script does:

  Run 0  Paper baseline        paper utility  + fairness on totals
  Run 1  Paper + Gap 1a only   time-aware utility + fairness on totals
  Run 2  Paper + Gap 1b only   paper utility  + rate fairness with group split
  Run 3  Paper + Gap 1 whole   time-aware utility + rate fairness with group split
                               (both halves of Gap 1 together, no Gap 2)
  Run 4  Paper + Gap 2 only    paper utility  + fairness on totals + utilisation term
  Run 5  Paper + all gaps      time-aware utility + rate fairness + utilisation term

Run 3 matters because Gap 1 was originally stated as a single gap with two parts
(static utility, and group-blind fairness). Runs 1 and 2 split it for diagnosis;
run 3 is Gap 1 as it was actually proposed.

The request sampling rate is PINNED across all five runs, so every configuration
sees the identical 6,604 requests and the identical fleet. The only thing that
changes is the utility definition and the fairness objective.

Output: outputs/gap_isolation.csv

Run:  .venv/bin/python -m scripts.run_gap_isolation
      .venv/bin/python -m scripts.run_gap_isolation --quick
"""
from __future__ import annotations

import argparse
import copy
import sys

import pandas as pd

from ltf.config import Config
from ltf.experiments import prepare
from ltf.metrics import compute
from ltf.methods.momaql import MOMAQL
from ltf.utils import LOG, output_path

# every run sees exactly this request stream (the calibrated rate from the
# main experiment, so results stay comparable with the published tables)
PINNED_RATE = 0.00296


def build_bench(utility_mode: str, base: Config):
    cfg = copy.deepcopy(base)
    cfg.utility.mode = utility_mode
    cfg.protocol.sample_rate = PINNED_RATE
    return prepare(cfg)


def run(bench, label: str, gap1a: bool, gap1b: bool, gap2: bool,
        episodes: int) -> dict:
    cfg = copy.deepcopy(bench.cfg)
    target = "rate_grouped" if gap1b else "total"
    cfg.fairness.target = target
    if not gap1b:
        # paper's fairness on totals, at the paper's own preference weight
        cfg.fairness.lambda_fair = 1.0
    agent = MOMAQL(cfg, bench.n_nodes, name=label, use_lookahead=True,
                   fairness_target=target, use_utilisation=gap2)
    agent.train([bench.real_sc, bench.synth_sc], episodes=episodes)
    res = bench.env.run(agent, method=label)
    met = compute(res, bench.drivers, cfg)
    d = met.to_dict()
    d.update({"gap1a": gap1a, "gap1b": gap1b, "gap2": gap2,
              "utility_mode": cfg.utility.mode, "fairness_target": target,
              "n_requests": bench.requests_n if hasattr(bench, "requests_n")
              else bench.test_sc.requests.n})
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    episodes = 8 if args.quick else 30

    base = Config()
    LOG.info("building the two environments (paper utility, time-aware utility)")
    b_paper = build_bench("paper", base)
    b_time = build_bench("time_aware", base)
    LOG.info("paper env requests   : %d", b_paper.test_sc.requests.n)
    LOG.info("time-aware requests  : %d", b_time.test_sc.requests.n)

    rows = []
    rows.append(run(b_paper, "0. Paper baseline", False, False, False, episodes))
    rows.append(run(b_time, "1. Paper + Gap 1a only", True, False, False, episodes))
    rows.append(run(b_paper, "2. Paper + Gap 1b only", False, True, False, episodes))
    rows.append(run(b_time, "3. Paper + Gap 1 whole (1a+1b)", True, True, False, episodes))
    rows.append(run(b_paper, "4. Paper + Gap 2 only", False, False, True, episodes))
    rows.append(run(b_time, "5. Paper + all gaps", True, True, True, episodes))

    df = pd.DataFrame(rows)
    cols = ["method", "gap1a", "gap1b", "gap2", "total_utility",
            "fairness_total_var", "rate_var", "rate_var_within",
            "rate_var_between", "rate_full_time", "rate_part_time",
            "rate_group_ratio", "util_mean", "util_var", "util_adj_var",
            "n_idle_drivers", "service_rate", "gini_rate", "min_utility"]
    print("\n" + "=" * 118)
    print("ONE GAP AT A TIME  (identical fleet and identical request stream in every row)")
    print("=" * 118)
    show = ["method", "total_utility", "rate_var", "rate_var_between",
            "rate_group_ratio", "util_var", "util_adj_var", "n_idle_drivers"]
    print(df[show].to_string(index=False))

    base_row = df.iloc[0]
    print("\n" + "=" * 118)
    print("EACH GAP'S OWN CONTRIBUTION, measured against the paper baseline")
    print("=" * 118)
    for i in range(1, len(df)):
        r = df.iloc[i]
        print(f"\n{r.method}")
        for lab, key, better in [
            ("total utility", "total_utility", "up"),
            ("Var(hourly rate)", "rate_var", "down"),
            ("between-group gap", "rate_var_between", "down"),
            ("group ratio (1.0 ideal)", "rate_group_ratio", "one"),
            ("Var(utilisation)", "util_var", "down"),
            ("Var(util) adjusted", "util_adj_var", "down"),
            ("idle drivers", "n_idle_drivers", "down"),
        ]:
            a, b = float(base_row[key]), float(r[key])
            if better == "one":
                verdict = f"|1-x| {abs(1-a):.3f} -> {abs(1-b):.3f}"
            else:
                d = 100 * (b - a) / abs(a) if a else float("nan")
                verdict = f"{d:+.1f}%"
            print(f"   {lab:<26} {a:>12.4f} -> {b:>12.4f}   {verdict}")

    df[cols].to_csv(output_path("gap_isolation.csv"), index=False)
    print(f"\nwrote outputs/gap_isolation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
