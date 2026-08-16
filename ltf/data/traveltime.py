"""Travel-time tensors: the paper's flattened `tau_bar` and our `tau_t`.

Sec. 5.1 of the paper: "the shortest travel time from a certain pickup to a
certain drop-off location is re-calculated as the mean travel time across the
time period". That single sentence is Gap 1a. It replaces a time-varying
quantity with one number per OD pair, and utility (Sec. 3.1) is distance-only
on top of that, so a trip is worth the same at 05:00 and at 18:00.

Measured on the very dataset the paper uses: holding origin and destination
fixed, the median peak/off-peak travel-time ratio across well-populated OD
pairs is 2.42x, up to 4.60x. Manhattan mean speed runs 16.69 mph at 05:00
against 9.03 mph at 15:00.

This module builds:
  tau_bar[o,d]            period-level travel time      (the paper's version)
  tau[o,d,p]              travel time per time profile  (ours)
  c[o,d,p] = tau/tau_bar  congestion multiplier, the knob in `UtilityConfig`

`p` indexes 48 profiles: (is_weekend, hour-of-day). Weekday/weekend is split
because congestion patterns differ qualitatively, not just in level.

Sparse buckets degrade gracefully through three levels, and the level used for
every cell is recorded so results can be re-checked on well-supported cells
only:
  L0  direct median of (o, d, profile), needs >= min_bucket trips
  L1  the pair's own tau_bar scaled by the citywide profile multiplier
  L2  Geo(o,d) divided by the citywide profile speed
Note L1 and L2 both yield c == citywide multiplier, so a thin cell falls back
to "average congestion at this hour" rather than to "no congestion".

Leakage note: the tensors built from the full month are the *simulator's
physics* (ground truth), analogous to the distance matrix. A policy is not
allowed to read future congestion from them -- Phase 4 adds a prediction head
for that, and `build(date_lt=...)` produces a train-only tensor for fitting it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import LOG, cache_path, describe_array, timed
from .graph import RoadGraph
from .graph import load_or_build as load_graph
from .prepare import load_trips

TT_NPZ = "traveltime.npz"
MIN_BUCKET = 30       # trips needed for a direct (o,d,profile) estimate
MIN_PAIR = 10         # trips needed for a direct (o,d) estimate

L_DIRECT, L_PAIR_SCALED, L_SPEED = 0, 1, 2


@dataclass
class TravelTime:
    """Travel-time tensors in minutes, plus the congestion multiplier."""

    tau: np.ndarray          # (K,K,P) float32  time-dependent
    tau_bar: np.ndarray      # (K,K)   float32  period-level (paper)
    c: np.ndarray            # (K,K,P) float32  tau / tau_bar, clipped
    level: np.ndarray        # (K,K,P) int8     estimation level used
    n_obs: np.ndarray        # (K,K,P) int32
    city_speed: np.ndarray   # (P,)    float32  km/h
    city_mult: np.ndarray    # (P,)    float32  citywide congestion multiplier
    profiles: np.ndarray     # (P,2)   int8     (is_weekend, hour)

    @property
    def n_profiles(self) -> int:
        return self.tau.shape[2]

    @staticmethod
    def profile_of(is_weekend, hour) -> np.ndarray:
        return np.asarray(is_weekend, np.int64) * 24 + np.asarray(hour, np.int64)

    def travel_min(self, o, d, p):
        return self.tau[np.asarray(o), np.asarray(d), np.asarray(p)]

    def congestion(self, o, d, p):
        return self.c[np.asarray(o), np.asarray(d), np.asarray(p)]

    def save(self, name: str = TT_NPZ) -> None:
        np.savez_compressed(cache_path(name), tau=self.tau, tau_bar=self.tau_bar,
                            c=self.c, level=self.level, n_obs=self.n_obs,
                            city_speed=self.city_speed, city_mult=self.city_mult,
                            profiles=self.profiles)
        LOG.info("wrote %s", name)

    @staticmethod
    def load(name: str = TT_NPZ) -> "TravelTime":
        z = np.load(cache_path(name))
        return TravelTime(**{k: z[k] for k in
                             ["tau", "tau_bar", "c", "level", "n_obs",
                              "city_speed", "city_mult", "profiles"]})


# --------------------------------------------------------------------------
def build(cfg: Config | None = None,
          graph: RoadGraph | None = None,
          trips: pd.DataFrame | None = None,
          date_lt: str | None = None,
          name: str = TT_NPZ) -> TravelTime:
    """Build the travel-time tensors.

    Parameters
    ----------
    date_lt : keep only trips with `date` < this (ISO string). Use to build a
        train-only tensor for fitting the congestion predictor.
    """
    cfg = cfg or Config()
    graph = graph if graph is not None else load_graph(cfg)
    if trips is None:
        trips = load_trips(columns=["o", "d", "dur_min", "km", "is_weekend",
                                    "hour", "date"])
    if date_lt is not None:
        trips = trips[trips.date < date_lt]
        LOG.info("travel-time build restricted to date < %s: %d trips",
                 date_lt, len(trips))

    K = graph.n_nodes
    P = cfg.time.n_time_profiles
    prof = TravelTime.profile_of(trips.is_weekend.to_numpy(), trips.hour.to_numpy())

    with timed(f"travel-time tensors ({K} nodes x {P} profiles)"):
        # citywide speed per profile -> fallback + diagnostics
        sp = pd.DataFrame({"p": prof, "km": trips.km.to_numpy(),
                           "min": trips.dur_min.to_numpy()}).groupby("p").sum()
        city_speed = np.full(P, np.nan)
        city_speed[sp.index.to_numpy()] = (sp.km / (sp["min"] / 60.0)).to_numpy()
        overall_speed = float(trips.km.sum() / (trips.dur_min.sum() / 60.0))
        city_speed = np.where(np.isfinite(city_speed), city_speed, overall_speed)
        city_mult = overall_speed / city_speed        # >1 means slower than average
        LOG.info("citywide speed: overall %.2f km/h, fastest profile %.2f, "
                 "slowest %.2f (ratio %.2fx)", overall_speed, city_speed.max(),
                 city_speed.min(), city_speed.max() / city_speed.min())

        # L0: direct (o,d,profile) medians
        df = pd.DataFrame({"o": trips.o.to_numpy(), "d": trips.d.to_numpy(),
                           "p": prof, "min": trips.dur_min.to_numpy()})
        g3 = df.groupby(["o", "d", "p"])["min"].agg(["median", "size"])
        tau = np.full((K, K, P), np.nan, np.float64)
        n_obs = np.zeros((K, K, P), np.int32)
        i0 = g3.index.get_level_values(0).to_numpy()
        i1 = g3.index.get_level_values(1).to_numpy()
        i2 = g3.index.get_level_values(2).to_numpy()
        n_obs[i0, i1, i2] = g3["size"].to_numpy()
        ok3 = g3["size"].to_numpy() >= MIN_BUCKET
        tau[i0[ok3], i1[ok3], i2[ok3]] = g3["median"].to_numpy()[ok3]

        # tau_bar: the paper's period-level travel time per OD pair
        g2 = df.groupby(["o", "d"])["min"].agg(["median", "size"])
        tau_bar = np.full((K, K), np.nan, np.float64)
        j0 = g2.index.get_level_values(0).to_numpy()
        j1 = g2.index.get_level_values(1).to_numpy()
        ok2 = g2["size"].to_numpy() >= MIN_PAIR
        tau_bar[j0[ok2], j1[ok2]] = g2["median"].to_numpy()[ok2]
        free_flow = graph.dist / overall_speed * 60.0
        pair_direct = np.isfinite(tau_bar)
        tau_bar = np.where(pair_direct, tau_bar, free_flow)
        tau_bar = np.maximum(tau_bar, 0.5)

        # fill tau: L1 where the pair itself is measured, else L2
        level = np.full((K, K, P), L_DIRECT, np.int8)
        miss = ~np.isfinite(tau)
        l1 = tau_bar[:, :, None] * city_mult[None, None, :]
        l2 = (graph.dist[:, :, None] / city_speed[None, None, :]) * 60.0
        use_l1 = miss & pair_direct[:, :, None]
        use_l2 = miss & ~pair_direct[:, :, None]
        tau = np.where(use_l1, l1, tau)
        tau = np.where(use_l2, l2, tau)
        level[use_l1] = L_PAIR_SCALED
        level[use_l2] = L_SPEED
        tau = np.maximum(tau, 0.5)

        c = np.clip(tau / tau_bar[:, :, None], cfg.utility.c_min, cfg.utility.c_max)

    tt = TravelTime(tau=tau.astype(np.float32), tau_bar=tau_bar.astype(np.float32),
                    c=c.astype(np.float32), level=level, n_obs=n_obs,
                    city_speed=city_speed.astype(np.float32),
                    city_mult=city_mult.astype(np.float32),
                    profiles=np.stack([np.arange(P) // 24, np.arange(P) % 24],
                                      1).astype(np.int8))

    tot = level.size
    LOG.info("estimation levels: L0 direct %.1f%%, L1 pair-scaled %.1f%%, "
             "L2 speed-only %.1f%%",
             100 * (level == L_DIRECT).mean(), 100 * (level == L_PAIR_SCALED).mean(),
             100 * (level == L_SPEED).mean())
    wt = n_obs.sum()
    LOG.info("trip-weighted: %.1f%% of trips sit in an L0 cell",
             100 * n_obs[level == L_DIRECT].sum() / max(wt, 1))
    LOG.info(describe_array(tt.c[level == L_DIRECT], "congestion c (L0 cells)"))
    LOG.info(describe_array(tt.tau_bar, "tau_bar (min)"))
    tt.save(name)
    return tt


def load_or_build(cfg: Config | None = None, force: bool = False,
                  name: str = TT_NPZ) -> TravelTime:
    if not force and cache_path(name).exists():
        return TravelTime.load(name)
    return build(cfg, name=name)


def report(tt: TravelTime, graph: RoadGraph, top: int = 10) -> pd.DataFrame:
    """Per-OD peak/off-peak spread on well-measured pairs. Evidence for Gap 1a."""
    solid = (tt.level == L_DIRECT)
    enough = solid.sum(axis=2) >= 24          # measured across most of the day
    o, d = np.where(enough & ~np.eye(graph.n_nodes, dtype=bool))
    rows = []
    for oo, dd in zip(o, d):
        m = solid[oo, dd]
        v = tt.tau[oo, dd, m]
        rows.append((int(oo), int(dd), float(graph.dist[oo, dd]),
                     float(v.min()), float(v.max()), float(tt.tau_bar[oo, dd]),
                     float(v.max() / v.min()), int(m.sum())))
    df = pd.DataFrame(rows, columns=["o", "d", "km", "tau_min", "tau_max",
                                     "tau_bar", "ratio", "profiles"])
    if len(df):
        LOG.info("OD pairs measured across >=24 profiles: %d", len(df))
        LOG.info("peak/off-peak travel-time ratio: median %.2fx, p90 %.2fx, "
                 "max %.2fx", df.ratio.median(), df.ratio.quantile(0.9),
                 df.ratio.max())
    return df.sort_values("ratio", ascending=False)


if __name__ == "__main__":
    cfg = Config()
    g = load_graph(cfg)
    tt = build(cfg, g)
    df = report(tt, g)
    print("\ntop-10 OD pairs by peak/off-peak travel-time ratio:")
    print(df.head(10).to_string(index=False))
