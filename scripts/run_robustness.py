"""Robustness checks. Both gaps stay active throughout.

1. Money-denominated utility. The main results are in distance units, which keeps
   them comparable to the paper's Table 1. But "hourly rate" is only a meaningful
   fairness quantity if it survives translation into money, so the whole pipeline
   is re-run with utility priced at the calibrated fare per km ($3.95, measured
   from the data). If the group gap is an artefact of distance units it will
   vanish here; if it is real it will persist.

2. The paper's own protocol. `Config.paper_faithful()` switches to the peak 2-hour
   window at a 1-hour decision step with the paper's sampling rate, static utility
   and fairness on totals. This is the closest we can get to Table 1's setting.
   Note the known limitation, stated rather than hidden: a 2 h/day window caps a
   driver at 14 h/week, so the 40 h vs 20 h contrast Gap 1b needs cannot exist
   there. The run is reported for comparability of the *paper's* metrics, and its
   group numbers are explicitly not interpretable.

3. Seed sensitivity. The driver fleet, request sample and RL exploration are all
   stochastic, so the headline comparison is repeated across seeds to show the
   gap improvements are not one lucky draw.

Run:  .venv/bin/python -m scripts.run_robustness
      .venv/bin/python -m scripts.run_robustness --quick
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
from ltf.sim.drivers import FULL_TIME, PART_TIME
from ltf.utils import LOG, output_path

MIN_LAMBDA_UTIL = 0.05


def _pair(bench, episodes: int, label_suffix: str = "") -> list[dict]:
    """Train Kang-style (totals) and ours (both gaps) on the same bench."""
    out = []
    specs = [
        ("Kang et al. (totals)", "total", False),
        ("Ours (Gap 1+2)", "rate_grouped", True),
    ]
    for name, target, use_util in specs:
        cfg = copy.deepcopy(bench.cfg)
        cfg.fairness.target = target
        if use_util:
            assert cfg.fairness.lambda_util >= MIN_LAMBDA_UTIL, \
                "Gap 2 weight must stay active for our method"
        agent = MOMAQL(cfg, bench.n_nodes, name=name + label_suffix,
                       use_lookahead=True, fairness_target=target,
                       use_utilisation=use_util)
        if use_util:
            assert agent.lam_util >= MIN_LAMBDA_UTIL, "Gap 2 term lost"
        agent.train([bench.real_sc, bench.synth_sc], episodes=episodes)
        res = bench.env.run(agent, method=name + label_suffix)
        met = compute(res, bench.drivers, cfg)
        d = met.to_dict()
        d["variant"] = label_suffix.strip(" []") or "main"
        out.append(d)
    return out


COLS = ["method", "variant", "total_utility", "fairness_total_var", "rate_var",
        "rate_var_between", "rate_full_time", "rate_part_time",
        "rate_group_ratio", "util_var", "util_adj_var", "n_idle_drivers",
        "service_rate"]


def show(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df[[c for c in COLS if c in df.columns]].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    episodes = 8 if args.quick else 30
    rows: list[dict] = []

    # ---------------- 1. distance units (reference) ----------------
    cfg = Config()
    b_dist = prepare(cfg)
    rows += _pair(b_dist, episodes, " [distance]")
    show(pd.DataFrame(rows), "1. DISTANCE-DENOMINATED UTILITY (reference)")

    # ---------------- 2. money units ----------------
    cfg_m = Config()
    cfg_m.utility.unit = "money"
    b_money = prepare(cfg_m)
    money_rows = _pair(b_money, episodes, " [money]")
    rows += money_rows
    show(pd.DataFrame(money_rows), "2. MONEY-DENOMINATED UTILITY "
         f"(fare per km = ${b_money.test_sc.utility.fare_per_km:.2f})")
    k, o = money_rows[0], money_rows[1]
    print(f"\n  group ratio: Kang {k['rate_group_ratio']:.3f} -> "
          f"ours {o['rate_group_ratio']:.3f}")
    print(f"  between-group Var: {k['rate_var_between']:.4f} -> "
          f"{o['rate_var_between']:.4f}")
    print(f"  hourly rate in $/h: Kang FT ${k['rate_full_time']:.2f} vs PT "
          f"${k['rate_part_time']:.2f}; ours FT ${o['rate_full_time']:.2f} vs PT "
          f"${o['rate_part_time']:.2f}")
    if k["rate_group_ratio"] > 1.2:
        print("  The group gap survives the change of units, so it is a property")
        print("  of the allocation, not of distance-denominated utility.")
    else:
        print("  NOTE: the group gap did NOT reproduce in money units. Reported")
        print("  as-is; it would mean the effect is unit-dependent.")

    # ---------------- 3. paper-faithful protocol ----------------
    print("\n" + "=" * 100)
    print("3. PAPER-FAITHFUL PROTOCOL (peak 2h/day, 1-hour steps, rate 0.05, "
          "static utility)")
    print("=" * 100)
    cfg_p = Config().paper_faithful()
    try:
        b_paper = prepare(cfg_p)
        pr = []
        for name, target, use_util in [("Kang et al. (paper protocol)", "total", False),
                                       ("Ours (paper protocol)", "rate_grouped", True)]:
            c = copy.deepcopy(b_paper.cfg)
            c.fairness.target = target
            ag = MOMAQL(c, b_paper.n_nodes, name=name, use_lookahead=True,
                        fairness_target=target, use_utilisation=use_util)
            ag.train([b_paper.real_sc, b_paper.synth_sc], episodes=episodes)
            res = b_paper.env.run(ag, method=name)
            m = compute(res, b_paper.drivers, c)
            d = m.to_dict(); d["variant"] = "paper_protocol"
            pr.append(d); rows.append(d)
        show(pd.DataFrame(pr), "paper-protocol results")
        oh = b_paper.drivers.online_hours
        print(f"\n  driver hours under this protocol: max {oh.max():.1f} h/week, "
              f"mean {oh.mean():.1f} h")
        print(f"  full-time threshold is {cfg_p.driver.full_time_threshold_h:.0f} h, "
              f"so {(oh >= cfg_p.driver.full_time_threshold_h).sum()} of "
              f"{len(oh)} drivers can reach it.")
        print("  This is the structural limitation of the paper's protocol: the")
        print("  window is too short for a full-time/part-time contrast to exist,")
        print("  which is exactly why the main experiments use a full-day timeline.")
        print("  The group columns above are therefore NOT interpretable.")
    except Exception as e:  # noqa: BLE001
        LOG.warning("paper-faithful protocol run failed: %s", e)
        print(f"  run failed: {e}")

    # ---------------- 4. seed sensitivity ----------------
    print("\n" + "=" * 100)
    print("4. SEED SENSITIVITY (fleet, request sample and RL exploration all "
          "re-drawn)")
    print("=" * 100)
    seed_rows = []
    seeds = [1] if args.quick else [1, 2, 3]
    for s in seeds:
        c = Config()
        c.driver.seed = 20260812 + s * 101
        c.rl.seed = 20260812 + s * 101
        c.predict.seed = 20260812 + s * 101
        b = prepare(c, with_forecaster=True)
        for r in _pair(b, episodes, f" [seed{s}]"):
            r["seed"] = s
            seed_rows.append(r); rows.append(r)
    S = pd.DataFrame(seed_rows)
    show(S, "per-seed results")
    if len(S):
        S["family"] = np.where(S.method.str.startswith("Ours"), "Ours", "Kang")
        agg = S.groupby("family").agg(
            utility=("total_utility", "mean"),
            rate_var=("rate_var", "mean"), rate_var_sd=("rate_var", "std"),
            between=("rate_var_between", "mean"),
            between_sd=("rate_var_between", "std"),
            ratio=("rate_group_ratio", "mean"), ratio_sd=("rate_group_ratio", "std"),
            util_adj=("util_adj_var", "mean"), idle=("n_idle_drivers", "mean"))
        print("\n  across seeds (mean, sd):")
        print(agg.to_string())

    pd.DataFrame(rows).to_csv(output_path("robustness.csv"), index=False)
    print("\nwrote outputs/robustness.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
