"""Main experiment: all methods, both metric families, ablations, and the
horizon-stability curves.

Structure mirrors the paper so results are comparable, then adds the gap
columns:
  Table 1  method comparison on the paper's metrics (Eq. 1, Eq. 2, Eq. 8)
  Table 1b the same runs scored on the Gap 1b and Gap 2 metrics
  Table 2  ablations: prediction on/off, fairness on/off, congestion on/off,
           fairness target totals vs rate, utilisation term on/off
  Fig. 4   fairness as the horizon grows from 1 to 7 days

Run:  .venv/bin/python -m scripts.run_experiments
      .venv/bin/python -m scripts.run_experiments --quick
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np
import pandas as pd

from ltf.config import Config
from ltf.data import demand as demand_mod
from ltf.data import graph as graph_mod
from ltf.data import traveltime as tt_mod
from ltf.data.prepare import load_trips
from ltf.metrics import compute
from ltf.methods.classical import LAFPolicy, ReassignPolicy
from ltf.methods.greedy import ScalarisedGreedy, make_efficiency_only, make_paper_greedy
from ltf.methods.momaql import MOMAQL
from ltf.predict import Forecaster
from ltf.sim.env import Environment
from ltf.sim.requests import synthetic_requests_from_prediction
from ltf.sim.scenario import Scenario, build_scenario
from ltf.utils import LOG, output_path, timed


# --------------------------------------------------------------------------
def build_training_scenarios(cfg: Config, test_sc: Scenario, forecaster,
                             tt_train, dem, graph):
    """Real training week + a synthetic week drawn from predicted demand."""
    train_cfg = copy.deepcopy(cfg)
    train_cfg.protocol.test_start = cfg.protocol.sim_train_start
    train_cfg.protocol.test_end = cfg.protocol.sim_train_end
    train_cfg.protocol.sample_rate = test_sc.requests.sample_rate
    real = build_scenario(train_cfg)

    # synthetic future: same timeline as the test week, requests from forecast
    import pandas as pd
    from ltf.sim.timeline import EPOCH
    hours = np.unique(test_sc.timeline.hour_index)
    dem_grid, con_grid = forecaster.predict_grids(cfg, dem, graph, tt_train, hours)
    synth_rq = synthetic_requests_from_prediction(
        cfg, test_sc.timeline, dem_grid, hours,
        test_sc.requests.sample_rate, seed=cfg.rl.seed)
    synth = Scenario(cfg=cfg, timeline=test_sc.timeline, drivers=test_sc.drivers,
                     requests=synth_rq, graph=test_sc.graph, tt=test_sc.tt,
                     demand=test_sc.demand, utility=test_sc.utility,
                     n_nodes=test_sc.n_nodes, offered_load=float("nan"))
    return real, synth, dem_grid, con_grid


def train_agent(agent: MOMAQL, scenarios, episodes: int):
    info = agent.train(scenarios, episodes=episodes)
    LOG.info("%s trained: %d/%d states visited", agent.name,
             info["states_visited"], info["states_total"])
    return info


def score(env: Environment, policy, drivers, cfg: Config, name: str):
    res = env.run(policy, method=name)
    return res, compute(res, drivers, cfg)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer RL episodes and skip the horizon sweep")
    args = ap.parse_args()

    cfg = Config()
    episodes = 8 if args.quick else cfg.rl.episodes
    graph = graph_mod.load_or_build(cfg)
    dem = demand_mod.load_or_build(cfg, n_nodes=graph.n_nodes)
    tt_full = tt_mod.load_or_build(cfg)
    tt_train = tt_mod.load_or_build(cfg, name="traveltime_train.npz")
    trips = load_trips(columns=["o", "d", "dur_min", "km", "is_weekend", "hour",
                                "date"])
    tt_test = tt_mod.load_or_build(cfg, name="traveltime_test.npz")

    test_sc = build_scenario(cfg)
    K = test_sc.n_nodes
    env = Environment(test_sc)

    # Calibrate the scalarisation weights from a measured efficiency-only run
    # rather than inheriting the paper's hard-coded omega = 0.6, which leaves the
    # utilisation penalty ~3 orders of magnitude below a trip's utility.
    from ltf.methods.base import calibrate_omegas
    warm = env.run(make_efficiency_only(cfg), method="warmup (efficiency only)")
    om = calibrate_omegas(warm, test_sc.drivers, cfg)
    cfg.fairness.omega = om["omega_total"]
    cfg.fairness.omega_rate = om["omega_rate"]
    cfg.fairness.omega_util = om["omega_util"]
    print(f"\ncalibrated scalarisation weights: omega_total={om['omega_total']:.3g}, "
          f"omega_rate={om['omega_rate']:.3g}, omega_util={om['omega_util']:.3g}")
    print(f"  (paper hard-codes omega = 0.6 for the totals variant)")

    with timed("forecaster"):
        fc = Forecaster(cfg)
        fres = fc.fit(cfg, dem, graph, tt_train, tt_test)

    real_sc, synth_sc, dem_grid, con_grid = build_training_scenarios(
        cfg, test_sc, fc, tt_train, dem, graph)

    rows, gap_rows, results = [], [], {}

    # ---------------- baselines ----------------
    print("\n" + "=" * 78)
    print("RUNNING METHODS")
    print("=" * 78)

    for name, pol in [
        ("Greedy", make_paper_greedy(cfg)),
        ("REASSIGN", ReassignPolicy(cfg)),
        ("LAF", LAFPolicy(cfg)),
    ]:
        res, met = score(env, pol, test_sc.drivers, cfg, name)
        rows.append(met)
        results[name] = (res, met)

    # ---------------- RL methods ----------------
    # Balance Ride-Pooling: fairness on totals, trained WITHOUT predicted future
    brp = MOMAQL(cfg, K, name="Balance Ride-Pooling", use_lookahead=False,
                 fairness_target="total")
    train_agent(brp, [real_sc], episodes)
    res, met = score(env, brp, test_sc.drivers, cfg, brp.name)
    rows.append(met); results[brp.name] = (res, met)

    # Kang et al. reproduced: fairness on totals, trained WITH predicted future
    kang = MOMAQL(cfg, K, name="Kang et al. (reproduced)", use_lookahead=True,
                  fairness_target="total")
    train_agent(kang, [real_sc, synth_sc], episodes)
    res, met = score(env, kang, test_sc.drivers, cfg, kang.name)
    rows.append(met); results[kang.name] = (res, met)

    # Ours: rate fairness with group split + utilisation term
    ours = MOMAQL(cfg, K, name="Ours (Gap 1+2)", use_lookahead=True,
                  fairness_target="rate_grouped", use_utilisation=True)
    train_agent(ours, [real_sc, synth_sc], episodes)
    res, met = score(env, ours, test_sc.drivers, cfg, ours.name)
    rows.append(met); results[ours.name] = (res, met)

    # ---------------- Table 1 ----------------
    print("\n" + "=" * 78)
    print("TABLE 1  paper's metrics (utility in km-equivalent units)")
    print("=" * 78)
    print(rows[0].paper_header())
    for m in rows:
        print(m.paper_row())

    print("\n" + "=" * 78)
    print("TABLE 1b  the same runs on the gap metrics")
    print("=" * 78)
    hdr = (f"{'Method':<26} {'rate_var':>9} {'within':>9} {'between':>9} "
           f"{'FT/hr':>7} {'PT/hr':>7} {'ratio':>6} {'util':>6} "
           f"{'util_var':>9} {'utiladj_var':>11} {'idle':>5} {'serv%':>6}")
    print(hdr)
    for m in rows:
        print(f"{m.method:<26} {m.rate_var:>9.3f} {m.rate_var_within:>9.3f} "
              f"{m.rate_var_between:>9.3f} {m.rate_full_time:>7.2f} "
              f"{m.rate_part_time:>7.2f} {m.rate_group_ratio:>6.2f} "
              f"{m.util_mean:>6.3f} {m.util_var:>9.4f} {m.util_adj_var:>11.4f} "
              f"{m.n_idle_drivers:>5} {100*m.service_rate:>6.1f}")

    # ---------------- Table 2: ablations ----------------
    print("\n" + "=" * 78)
    print("TABLE 2  ablations")
    print("=" * 78)
    abl = []

    # (a) utility: paper's static utility vs time-aware  [Gap 1a]
    cfg_static = copy.deepcopy(cfg)
    cfg_static.utility.mode = "paper"
    sc_static = build_scenario(cfg_static)
    env_static = Environment(sc_static)
    ag = MOMAQL(cfg_static, K, name="Ours, static utility (c=1)",
                use_lookahead=True, fairness_target="rate_grouped",
                use_utilisation=True)
    train_agent(ag, [real_sc], max(2, episodes // 2))
    r, m = score(env_static, ag, sc_static.drivers, cfg_static, ag.name)
    abl.append(m)

    # (b) fairness off
    ag = MOMAQL(cfg, K, name="Ours, w/o fairness", use_lookahead=True,
                fairness_target="rate_grouped", use_utilisation=False)
    ag.lam = 0.0; ag.lam_within = 0.0; ag.lam_between = 0.0
    train_agent(ag, [real_sc], max(2, episodes // 2))
    r, m = score(env, ag, test_sc.drivers, cfg, ag.name)
    abl.append(m)

    # (c) prediction off
    ag = MOMAQL(cfg, K, name="Ours, w/o prediction", use_lookahead=False,
                fairness_target="rate_grouped", use_utilisation=True)
    train_agent(ag, [real_sc], episodes)
    r, m = score(env, ag, test_sc.drivers, cfg, ag.name)
    abl.append(m)

    # (d) utilisation term off  [Gap 2]
    ag = MOMAQL(cfg, K, name="Ours, w/o utilisation term", use_lookahead=True,
                fairness_target="rate_grouped", use_utilisation=False)
    train_agent(ag, [real_sc, synth_sc], episodes)
    r, m = score(env, ag, test_sc.drivers, cfg, ag.name)
    abl.append(m)

    # (e) fairness on totals instead of rate  [Gap 1b]
    abl.append(results["Kang et al. (reproduced)"][1])

    print(f"{'Ablation':<32} {'Utility':>12} {'Var(total)':>12} "
          f"{'rate_var':>9} {'between':>9} {'util_var':>9} {'idle':>5}")
    for m in abl + [results["Ours (Gap 1+2)"][1]]:
        print(f"{m.method:<32} {m.total_utility:>12,.1f} "
              f"{m.fairness_total_var:>12,.1f} {m.rate_var:>9.3f} "
              f"{m.rate_var_between:>9.3f} {m.util_var:>9.4f} {m.n_idle_drivers:>5}")

    # ---------------- Fig. 4: horizon stability ----------------
    curves = {}
    if not args.quick:
        print("\n" + "=" * 78)
        print("FIG. 4  fairness vs horizon length (1..7 days)")
        print("=" * 78)
        for label, agent in [("Kang et al. (reproduced)", kang),
                             ("Ours (Gap 1+2)", ours)]:
            row = []
            for nd in range(1, cfg.protocol.n_history_days
                            + cfg.protocol.n_current_days
                            + cfg.protocol.n_future_days + 1):
                sc_h = build_scenario(cfg, n_days=nd)
                r_h = Environment(sc_h).run(agent, method=f"{label} d{nd}")
                m_h = compute(r_h, sc_h.drivers, cfg)
                row.append({"days": nd, "method": label,
                            "fairness_total_var": m_h.fairness_total_var,
                            "fairness_normalised": m_h.fairness_normalised,
                            "rate_var": m_h.rate_var,
                            "rate_var_between": m_h.rate_var_between,
                            "util_var": m_h.util_var,
                            "total_utility": m_h.total_utility})
            curves[label] = row
            print(f"\n  {label}")
            print(f"    {'days':>5} {'Var(total)':>13} {'norm':>8} "
                  f"{'rate_var':>9} {'between':>9} {'util_var':>9}")
            for e in row:
                print(f"    {e['days']:>5} {e['fairness_total_var']:>13,.1f} "
                      f"{e['fairness_normalised']:>8.4f} {e['rate_var']:>9.3f} "
                      f"{e['rate_var_between']:>9.3f} {e['util_var']:>9.4f}")

    # ---------------- save ----------------
    pd.DataFrame([m.to_dict() for m in rows]).to_csv(
        output_path("table1_methods.csv"), index=False)
    pd.DataFrame([m.to_dict() for m in abl
                  + [results["Ours (Gap 1+2)"][1]]]).to_csv(
        output_path("table2_ablations.csv"), index=False)
    if curves:
        pd.DataFrame([e for v in curves.values() for e in v]).to_csv(
            output_path("fig4_horizon.csv"), index=False)
    with open(output_path("forecaster_results.json"), "w") as f:
        json.dump(fres.__dict__, f, indent=2)

    print("\nwrote outputs/table1_methods.csv, table2_ablations.csv"
          + (", fig4_horizon.csv" if curves else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
