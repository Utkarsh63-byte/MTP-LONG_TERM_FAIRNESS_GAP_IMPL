"""A Scenario bundles everything an allocation method needs to be run or scored.

Constructing one is the single entry point for every experiment: it wires the
Phase-1 tensors to the Phase-2 timeline, driver fleet and request stream, and
performs the demand calibration so that supply and demand are in a regime where
fairness is actually contested (if demand were far below capacity every driver
would be served and all fairness metrics would collapse to zero).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data import graph as graph_mod
from ..data import demand as demand_mod
from ..data import traveltime as tt_mod
from ..data.prepare import load_nodes, load_trips
from ..utility import UtilityModel, calibrate_fare_per_km
from ..utils import LOG
from .drivers import DriverPopulation, build_drivers
from .requests import (RequestStream, build_requests, calibrate_sample_rate,
                       estimate_occupancy_slots, refine_sample_rate)
from .timeline import Timeline, build_timeline


@dataclass
class Scenario:
    cfg: Config
    timeline: Timeline
    drivers: DriverPopulation
    requests: RequestStream
    graph: "graph_mod.RoadGraph"
    tt: "tt_mod.TravelTime"
    demand: "demand_mod.Demand"
    utility: UtilityModel
    n_nodes: int
    offered_load: float = float("nan")

    def describe(self) -> str:
        c = self.drivers.concurrent_online()
        return "\n".join([
            self.timeline.describe(),
            f"drivers: {self.drivers.n}, online slots "
            f"{int(self.drivers.online_slots.sum()):,}, "
            f"concurrent min/med/max {c.min()}/{int(np.median(c))}/{c.max()}",
            f"requests: {self.requests.n:,} "
            f"(rate {self.requests.sample_rate:.5f}), "
            f"{self.requests.n / self.timeline.n_slots:.1f} per slot, "
            f"offered load {self.offered_load:.3f}",
            f"nodes: {self.n_nodes}, utility mode "
            f"'{self.cfg.utility.mode}' in {self.cfg.utility.unit} units",
        ])


_TRIPS_CACHE: pd.DataFrame | None = None


def _trips() -> pd.DataFrame:
    global _TRIPS_CACHE
    if _TRIPS_CACHE is None:
        _TRIPS_CACHE = load_trips(columns=["o", "d", "km", "dur_min", "fare",
                                           "pickup_ts", "hour", "date"])
    return _TRIPS_CACHE


def build_scenario(cfg: Config | None = None, n_days: int | None = None,
                   start: str | None = None, end: str | None = None,
                   seed: int | None = None) -> Scenario:
    cfg = cfg or Config()
    nodes = load_nodes()
    K = nodes.n_nodes
    graph = graph_mod.load_or_build(cfg)
    tt = tt_mod.load_or_build(cfg)
    dem = demand_mod.load_or_build(cfg, n_nodes=K)
    trips = _trips()

    timeline = build_timeline(cfg, start=start, end=end, n_days=n_days)
    drivers = build_drivers(cfg, timeline, demand=dem, n_nodes=K, seed=seed)

    fpk = calibrate_fare_per_km(trips.sample(min(500_000, len(trips)),
                                             random_state=0))
    um = UtilityModel(cfg, graph, tt, fare_per_km=fpk)

    occ = estimate_occupancy_slots(cfg, timeline, trips, tt, graph)
    rate0 = calibrate_sample_rate(cfg, timeline, trips,
                                  online_slots=int(drivers.online_slots.sum()),
                                  occupancy_slots=occ)
    rate, requests, load = refine_sample_rate(cfg, timeline, trips, drivers, um,
                                              rate0, seed)

    sc = Scenario(cfg=cfg, timeline=timeline, drivers=drivers, requests=requests,
                  graph=graph, tt=tt, demand=dem, utility=um, n_nodes=K,
                  offered_load=load)
    LOG.info("scenario ready\n%s", sc.describe())
    return sc
