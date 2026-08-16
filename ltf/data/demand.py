"""Demand tensors for the request forecaster (paper Sec. 4.2).

The paper predicts, per OD pair and per 1-hour step, how many requests will be
raised, using a 3-layer MLP over lagged counts, trained on one month and
forecasting the next 7 days (reported MSE 94.69). Predicted requests then enter
the *action space* of the MDP, which is the paper's actual novelty.

We build:
  od[o, d, h]   requests per OD pair per hour-of-month   (87 x 87 x 744)
  org[o, h]     requests raised at each node per hour    (the aggregate the
                paper's Fig. 2 pipeline actually needs)
  cong[o, d, p] congestion targets, supplied by traveltime.py

`h` is hours since 2016-03-01 00:00 local, so h = day*24 + hour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import LOG, cache_path, timed
from .prepare import load_trips

DEMAND_NPZ = "demand.npz"
EPOCH = pd.Timestamp("2016-03-01")


@dataclass
class Demand:
    od: np.ndarray        # (K,K,H) int32
    org: np.ndarray       # (K,H)   int32
    hours: np.ndarray     # (H,) int32 hour-of-month index
    hour_of_day: np.ndarray
    is_weekend: np.ndarray

    @property
    def n_hours(self) -> int:
        return self.od.shape[2]

    def save(self) -> None:
        np.savez_compressed(cache_path(DEMAND_NPZ), od=self.od, org=self.org,
                            hours=self.hours, hour_of_day=self.hour_of_day,
                            is_weekend=self.is_weekend)
        LOG.info("wrote %s", DEMAND_NPZ)

    @staticmethod
    def load() -> "Demand":
        z = np.load(cache_path(DEMAND_NPZ))
        return Demand(**{k: z[k] for k in
                         ["od", "org", "hours", "hour_of_day", "is_weekend"]})


def build(cfg: Config | None = None, trips: pd.DataFrame | None = None,
          n_nodes: int | None = None) -> Demand:
    cfg = cfg or Config()
    if trips is None:
        trips = load_trips(columns=["o", "d", "pickup_ts"])
    if n_nodes is None:
        n_nodes = int(max(trips.o.max(), trips.d.max())) + 1
    K = n_nodes

    with timed("demand tensors"):
        hidx = ((trips.pickup_ts.to_numpy().astype("datetime64[h]")
                 - np.datetime64(EPOCH, "h")).astype(np.int64))
        H = int(hidx.max()) + 1
        od = np.zeros((K, K, H), np.int32)
        np.add.at(od, (trips.o.to_numpy(), trips.d.to_numpy(), hidx), 1)
        org = od.sum(axis=1).astype(np.int32)
        hours = np.arange(H, dtype=np.int32)
        hod = (hours % 24).astype(np.int8)
        dow = ((EPOCH.dayofweek + hours // 24) % 7).astype(np.int8)
        wknd = (dow >= 5).astype(np.int8)

    d = Demand(od=od, org=org, hours=hours, hour_of_day=hod, is_weekend=wknd)
    LOG.info("demand: %d nodes x %d hours, %d requests total, "
             "busiest node-hour %d", K, H, int(od.sum()), int(org.max()))
    LOG.info("nonzero OD-hour cells: %.2f%%", 100 * (od > 0).mean())
    d.save()
    return d


def load_or_build(cfg: Config | None = None, force: bool = False,
                  n_nodes: int | None = None) -> Demand:
    if not force and cache_path(DEMAND_NPZ).exists():
        return Demand.load()
    return build(cfg, n_nodes=n_nodes)


if __name__ == "__main__":
    build(Config())
