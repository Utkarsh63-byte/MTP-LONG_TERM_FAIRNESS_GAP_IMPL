"""Shared experiment setup, so every script runs on an identical pipeline.

Builds, once:
  * the test scenario (7-day horizon) and its environment
  * calibrated scalarisation weights, measured from an efficiency-only warmup
    rather than inherited from the paper's hard-coded omega = 0.6
  * the forecaster, plus the two training scenarios: the real training week and
    a synthetic week drawn from predicted demand (the mechanism that puts
    forecast requests in the MDP's action space, paper Sec. 4.3)

Sharing this means a sweep, a robustness run and the main table are all
comparing like with like: same fleet, same request stream, same weights.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..data import demand as demand_mod
from ..data import graph as graph_mod
from ..data import traveltime as tt_mod
from ..data.prepare import load_trips
from ..methods.base import calibrate_omegas
from ..methods.greedy import make_efficiency_only
from ..predict import Forecaster
from ..sim.env import Environment
from ..sim.requests import synthetic_requests_from_prediction
from ..sim.scenario import Scenario, build_scenario
from ..utils import LOG, timed


@dataclass
class Bench:
    cfg: Config
    test_sc: Scenario
    env: Environment
    real_sc: Scenario
    synth_sc: Scenario
    omegas: dict
    forecast: object
    n_nodes: int

    @property
    def drivers(self):
        return self.test_sc.drivers


def prepare(cfg: Config | None = None, with_forecaster: bool = True) -> Bench:
    cfg = cfg or Config()
    graph = graph_mod.load_or_build(cfg)
    dem = demand_mod.load_or_build(cfg, n_nodes=graph.n_nodes)
    tt_train = tt_mod.load_or_build(cfg, name="traveltime_train.npz")
    tt_test = tt_mod.load_or_build(cfg, name="traveltime_test.npz")

    test_sc = build_scenario(cfg)
    env = Environment(test_sc)
    K = test_sc.n_nodes

    warm = env.run(make_efficiency_only(cfg), method="warmup (efficiency only)")
    om = calibrate_omegas(warm, test_sc.drivers, cfg)
    cfg.fairness.omega = om["omega_total"]
    cfg.fairness.omega_rate = om["omega_rate"]
    cfg.fairness.omega_util = om["omega_util"]
    LOG.info("calibrated omegas: total=%.4g rate=%.4g util=%.4g",
             om["omega_total"], om["omega_rate"], om["omega_util"])

    # real training week
    tcfg = copy.deepcopy(cfg)
    tcfg.protocol.test_start = cfg.protocol.sim_train_start
    tcfg.protocol.test_end = cfg.protocol.sim_train_end
    tcfg.protocol.sample_rate = test_sc.requests.sample_rate
    real_sc = build_scenario(tcfg)

    fres, synth_sc = None, real_sc
    if with_forecaster:
        with timed("forecaster"):
            fc = Forecaster(cfg)
            fres = fc.fit(cfg, dem, graph, tt_train, tt_test)
        hours = np.unique(test_sc.timeline.hour_index)
        dem_grid, _ = fc.predict_grids(cfg, dem, graph, tt_train, hours)
        synth_rq = synthetic_requests_from_prediction(
            cfg, test_sc.timeline, dem_grid, hours,
            test_sc.requests.sample_rate, seed=cfg.rl.seed)
        synth_sc = Scenario(cfg=cfg, timeline=test_sc.timeline,
                            drivers=test_sc.drivers, requests=synth_rq,
                            graph=test_sc.graph, tt=test_sc.tt, demand=test_sc.demand,
                            utility=test_sc.utility, n_nodes=K,
                            offered_load=float("nan"))

    return Bench(cfg=cfg, test_sc=test_sc, env=env, real_sc=real_sc,
                 synth_sc=synth_sc, omegas=om, forecast=fres, n_nodes=K)
