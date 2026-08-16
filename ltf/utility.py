"""Utility definitions: the paper's (Sec. 3.1) and the time-aware one (Gap 1a).

Paper, Sec. 3.1
---------------
    U(r, v) = Geo(d_r, s_r) - Geo(s_r, g_v)

profit (trip distance) minus cost (deadhead distance), evaluated the moment a
request is assigned, on a graph whose travel times were flattened to a period
mean (Sec. 5.1). The paper is explicit that this makes utility independent of
when the trip happens.

Gap 1a
------
    c_t(a,b) = tau_t(a,b) / tau_bar(a,b)                  (congestion multiplier)
    U_t(r, v) = Geo(d_r,s_r) / c_t(s_r,d_r)
                - w * Geo(s_r,g_v) * c_t(g_v,s_r)

Reading: congested loaded distance delivers less value per unit of the driver's
clock, and congested deadhead costs more than free-flowing deadhead. Both terms
move in the direction that hurts, which is the asymmetry the paper's formula
cannot express.

The design constraint that makes the ablations clean: at c == 1 everywhere,
`time_aware` is *identically* `paper`. `assert_reduction()` checks this to
floating-point tolerance, and test_utility_reduction() in scripts/ runs it on
the real tensors.

Occupancy
---------
`occupancy_slots` converts an assignment into the number of decision epochs the
driver is actually busy, tau_t(g,s) + tau_t(s,d). The paper has no equivalent:
its drivers may accept unboundedly many concurrent requests and time never
enters, which is precisely why "was this driver busy in this slot?" -- the
question Gap 2 asks -- has no answer there.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data.graph import RoadGraph
from .data.traveltime import TravelTime


class UtilityModel:
    """Computes request utility and occupancy for a batch of (request, driver) pairs.

    All methods are vectorised: pass arrays of equal length (or broadcastable
    scalars) for origin `s`, destination `d`, driver location `g`, and time
    profile `p`.
    """

    def __init__(self, cfg: Config, graph: RoadGraph, tt: TravelTime,
                 fare_per_km: float | None = None):
        self.cfg = cfg
        self.graph = graph
        self.tt = tt
        self.mode = cfg.utility.mode
        self.unit = cfg.utility.unit
        self.w = cfg.utility.deadhead_weight
        self.slot_min = cfg.time.slot_minutes
        # money mode needs a price for deadhead km; calibrated from the data
        self.fare_per_km = fare_per_km if fare_per_km is not None else 3.05

    # -- congestion -------------------------------------------------------
    def c(self, a, d, p) -> np.ndarray:
        """Congestion multiplier; identically 1.0 in `paper` mode."""
        if self.mode == "paper":
            return np.ones(np.broadcast(np.asarray(a), np.asarray(d),
                                        np.asarray(p)).shape, np.float32)
        return self.tt.c[np.asarray(a), np.asarray(d), np.asarray(p)]

    # -- utility ----------------------------------------------------------
    def trip_value(self, s, d, p) -> np.ndarray:
        """The profit term: what the loaded leg is worth."""
        if self.unit == "money":
            base = self.graph.dist[np.asarray(s), np.asarray(d)] * self.fare_per_km
        else:
            base = self.graph.dist[np.asarray(s), np.asarray(d)]
        return base / self.c(s, d, p)

    def deadhead_cost(self, g, s, p) -> np.ndarray:
        """The cost term: getting to the pickup."""
        base = self.graph.dist[np.asarray(g), np.asarray(s)]
        if self.unit == "money":
            base = base * self.fare_per_km
        return self.w * base * self.c(g, s, p)

    def utility(self, g, s, d, p) -> np.ndarray:
        """U(request, driver). `paper` mode reproduces Sec. 3.1 exactly."""
        return self.trip_value(s, d, p) - self.deadhead_cost(g, s, p)

    # -- time -------------------------------------------------------------
    def travel_min(self, a, b, p) -> np.ndarray:
        """Clock time for a leg. `paper` mode uses the flattened tau_bar."""
        if self.mode == "paper":
            return self.tt.tau_bar[np.asarray(a), np.asarray(b)]
        return self.tt.tau[np.asarray(a), np.asarray(b), np.asarray(p)]

    def occupancy_min(self, g, s, d, p) -> np.ndarray:
        return self.travel_min(g, s, p) + self.travel_min(s, d, p)

    def occupancy_slots(self, g, s, d, p) -> np.ndarray:
        """Decision epochs the driver is unavailable for. At least 1."""
        m = self.occupancy_min(g, s, d, p)
        return np.maximum(1, np.ceil(m / self.slot_min)).astype(np.int32)

    # -- feasibility ------------------------------------------------------
    def pickup_km(self, g, s) -> np.ndarray:
        return self.graph.dist[np.asarray(g), np.asarray(s)]

    def feasible(self, g, s) -> np.ndarray:
        return self.pickup_km(g, s) <= self.cfg.utility.max_pickup_km

    # -- self-check -------------------------------------------------------
    def assert_reduction(self, n: int = 200_000, seed: int = 0) -> dict:
        """Verify time_aware == paper when c is forced to 1.

        Every Phase-4 ablation claims the paper is a special case of our
        utility. This checks it numerically rather than by assertion in prose.
        """
        import copy

        rng = np.random.default_rng(seed)
        K = self.graph.n_nodes
        P = self.tt.n_profiles
        g = rng.integers(0, K, n)
        s = rng.integers(0, K, n)
        d = rng.integers(0, K, n)
        p = rng.integers(0, P, n)

        cfg_p = copy.deepcopy(self.cfg)
        cfg_p.utility.mode = "paper"
        um_paper = UtilityModel(cfg_p, self.graph, self.tt, self.fare_per_km)

        tt_flat = copy.copy(self.tt)
        tt_flat.c = np.ones_like(self.tt.c)
        tt_flat.tau = np.repeat(self.tt.tau_bar[:, :, None], P, axis=2)
        cfg_t = copy.deepcopy(self.cfg)
        cfg_t.utility.mode = "time_aware"
        um_flat = UtilityModel(cfg_t, self.graph, tt_flat, self.fare_per_km)

        u_paper = um_paper.utility(g, s, d, p)
        u_flat = um_flat.utility(g, s, d, p)
        t_paper = um_paper.occupancy_slots(g, s, d, p)
        t_flat = um_flat.occupancy_slots(g, s, d, p)
        max_u = float(np.max(np.abs(u_paper - u_flat)))
        max_t = int(np.max(np.abs(t_paper - t_flat)))

        # and confirm the real (unflattened) tensors genuinely differ,
        # otherwise the reduction test is vacuous
        u_real = self.utility(g, s, d, p) if self.mode == "time_aware" else None
        spread = (float(np.mean(np.abs(u_real - u_paper))) if u_real is not None
                  else float("nan"))
        return {"max_abs_utility_diff": max_u, "max_abs_slot_diff": max_t,
                "mean_abs_diff_vs_real_congestion": spread,
                "reduction_holds": max_u < 1e-4 and max_t == 0}


def calibrate_fare_per_km(trips) -> float:
    """Median fare+tip per km, for the money-denominated utility variant."""
    v = (trips.fare / np.maximum(trips.km, 0.1)).to_numpy()
    v = v[np.isfinite(v) & (v > 0) & (v < 50)]
    return float(np.median(v))
