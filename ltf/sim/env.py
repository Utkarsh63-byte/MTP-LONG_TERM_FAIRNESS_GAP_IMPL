"""The assignment environment: slot-by-slot simulation of the market.

This is the piece that turns an allocation policy into measurable outcomes, and
it is where one of the paper's modelling shortcuts had to be repaired.

Kang et al. (Sec. 4.3) let "an agent (driver) accept multiple requests
concurrently" and their utility carries no time dimension at all, so a driver
can absorb unbounded work and "was this driver busy at time t?" has no answer.
Gap 2 asks exactly that question, so here a driver assigned at slot t is
occupied for the next `occupancy_slots` epochs -- deadhead to the pickup plus
the loaded leg, both time-dependent under the Gap 1a utility -- and is not
available again until the trip completes. Capacity is `cfg.driver.capacity`
(default 1), so the environment is a ride-hailing (not ride-pooling) market,
matching the paper's driver tuple where `c_v` exists but is never varied.

Bookkeeping per driver, all of which the metrics layer consumes:
  cum_utility   o_v, the paper's Eq. 1/2 quantity
  busy_slots    occupied decision epochs           -> Gap 2 numerator
  online_slots  epochs the driver was available    -> Gap 2 denominator, and
                                                      H_v for Gap 1b's rate
  n_trips       assignments received
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..utils import LOG
from .drivers import DriverPopulation
from .requests import RequestStream
from .scenario import Scenario
from .timeline import Timeline

UNASSIGNED = -1


@dataclass
class SlotContext:
    """Everything a policy may look at when deciding one slot's assignments.

    A policy is handed only present-tense facts plus whatever forecast the
    scenario supplies. It never sees future ground truth -- that separation is
    what keeps the Gap 1a congestion term honest (Phase 4 predicts it).
    """

    t: int                      # slot index
    profile: int                # travel-time profile of this slot
    req_idx: np.ndarray         # (m,) global request indices pending now
    req_o: np.ndarray           # (m,) pickup nodes
    req_d: np.ndarray           # (m,) dropoff nodes
    drv_idx: np.ndarray         # (k,) available driver ids
    drv_node: np.ndarray        # (k,) current node of each available driver
    utility: np.ndarray         # (m,k) utility of each pairing
    occupancy: np.ndarray       # (m,k) slots each pairing would consume
    feasible: np.ndarray        # (m,k) bool, pickup within max_pickup_km
    # running per-driver state, indexed by *global* driver id
    cum_utility: np.ndarray
    busy_slots: np.ndarray
    online_so_far: np.ndarray
    n_trips: np.ndarray
    group: np.ndarray
    online_total: np.ndarray    # full-horizon online slots (known from shifts)

    @property
    def m(self) -> int:
        return len(self.req_idx)

    @property
    def k(self) -> int:
        return len(self.drv_idx)


@dataclass
class EpisodeResult:
    """Outcome of one full pass over the timeline."""

    cum_utility: np.ndarray       # (n,) o_v
    busy_slots: np.ndarray        # (n,)
    online_slots: np.ndarray      # (n,)
    n_trips: np.ndarray           # (n,)
    group: np.ndarray             # (n,)
    online_hours: np.ndarray      # (n,) H_v
    assign_req: np.ndarray        # (a,) request index of each assignment
    assign_drv: np.ndarray        # (a,) driver index
    assign_slot: np.ndarray       # (a,) slot index
    assign_utility: np.ndarray    # (a,) utility realised
    assign_occupancy: np.ndarray  # (a,) slots consumed
    n_requests: int
    n_servable: int               # requests arriving when >=1 driver was online
    busy_by_slot: np.ndarray      # (T,) busy drivers per slot
    online_by_slot: np.ndarray    # (T,) online drivers per slot
    method: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_assigned(self) -> int:
        return len(self.assign_req)

    @property
    def total_utility(self) -> float:
        """Eq. 1: total utility across all drivers."""
        return float(self.cum_utility.sum())

    @property
    def service_rate(self) -> float:
        return self.n_assigned / max(self.n_servable, 1)

    def summary(self) -> str:
        return (f"{self.method}: total utility {self.total_utility:,.1f}, "
                f"{self.n_assigned:,}/{self.n_servable:,} requests served "
                f"({100*self.service_rate:.1f}%), "
                f"mean utilisation {self.busy_slots.sum()/max(self.online_slots.sum(),1):.3f}")


class Environment:
    """Runs a policy over a scenario and records everything the metrics need."""

    def __init__(self, scenario: Scenario):
        self.sc = scenario
        self.cfg: Config = scenario.cfg
        self.tl: Timeline = scenario.timeline
        self.dr: DriverPopulation = scenario.drivers
        self.rq: RequestStream = scenario.requests
        self.um = scenario.utility

    # ------------------------------------------------------------------
    def run(self, policy, method: str = "", reset_policy: bool = True,
            log_every: int = 0) -> EpisodeResult:
        n, T = self.dr.n, self.tl.n_slots
        cum_u = np.zeros(n)
        busy = np.zeros(n, np.int32)
        online_so_far = np.zeros(n, np.int32)
        n_trips = np.zeros(n, np.int32)
        node = self.dr.home_node.astype(np.int32).copy()
        busy_until = np.zeros(n, np.int32)          # slot when driver frees up
        online_total = self.dr.online_slots

        a_req, a_drv, a_slot, a_u, a_occ = [], [], [], [], []
        busy_by_slot = np.zeros(T, np.int32)
        online_by_slot = np.zeros(T, np.int32)
        n_servable = 0

        if reset_policy and hasattr(policy, "reset"):
            policy.reset(self.sc)

        for t in range(T):
            online = self.dr.online[:, t]
            online_by_slot[t] = online.sum()
            online_so_far[online] += 1
            # a driver returning for a new shift starts from their home node
            fresh = self.dr.shift_start[:, t]
            if fresh.any():
                node[fresh] = self.dr.home_node[fresh]
                busy_until[fresh] = t
            occupied = online & (busy_until > t)
            busy_by_slot[t] = occupied.sum()

            sl = self.rq.of_slot(t)
            m = sl.stop - sl.start
            if m == 0:
                continue
            if online_by_slot[t] > 0:
                n_servable += m

            avail = np.where(online & (busy_until <= t))[0]
            if len(avail) == 0:
                continue

            p = int(self.tl.profile[t])
            ro = self.rq.o[sl].astype(np.int32)
            rd = self.rq.d[sl].astype(np.int32)
            gn = node[avail]
            # (m,k) pairing tensors
            G = np.broadcast_to(gn[None, :], (m, len(avail)))
            O = np.broadcast_to(ro[:, None], (m, len(avail)))
            D = np.broadcast_to(rd[:, None], (m, len(avail)))
            U = self.um.utility(G, O, D, p)
            OC = self.um.occupancy_slots(G, O, D, p)
            FS = self.um.feasible(G, O)

            ctx = SlotContext(
                t=t, profile=p,
                req_idx=np.arange(sl.start, sl.stop), req_o=ro, req_d=rd,
                drv_idx=avail, drv_node=gn, utility=U, occupancy=OC, feasible=FS,
                cum_utility=cum_u, busy_slots=busy, online_so_far=online_so_far,
                n_trips=n_trips, group=self.dr.group, online_total=online_total)

            pairs = policy.assign(ctx)
            if pairs is None or len(pairs) == 0:
                continue
            pairs = np.asarray(pairs, np.int64).reshape(-1, 2)

            # apply, enforcing capacity: one open trip per driver per slot
            taken = np.zeros(len(avail), bool)
            for ri, dj in pairs:
                if taken[dj] or not FS[ri, dj]:
                    continue
                taken[dj] = True
                v = int(avail[dj])
                u = float(U[ri, dj])
                oc = int(OC[ri, dj])
                cum_u[v] += u
                n_trips[v] += 1
                busy[v] += oc
                busy_until[v] = t + oc
                node[v] = int(rd[ri])
                a_req.append(int(sl.start + ri))
                a_drv.append(v)
                a_slot.append(t)
                a_u.append(u)
                a_occ.append(oc)

            if hasattr(policy, "observe"):
                policy.observe(ctx, pairs)
            if log_every and t % log_every == 0:
                LOG.info("  slot %d/%d: %d assigned so far, utility %.0f",
                         t, T, len(a_req), cum_u.sum())

        res = EpisodeResult(
            cum_utility=cum_u, busy_slots=busy,
            online_slots=online_total.copy(), n_trips=n_trips,
            group=self.dr.group.copy(),
            online_hours=online_total * self.tl.slot_hours,
            assign_req=np.asarray(a_req, np.int32),
            assign_drv=np.asarray(a_drv, np.int32),
            assign_slot=np.asarray(a_slot, np.int32),
            assign_utility=np.asarray(a_u, float),
            assign_occupancy=np.asarray(a_occ, np.int32),
            n_requests=self.rq.n, n_servable=n_servable,
            busy_by_slot=busy_by_slot, online_by_slot=online_by_slot,
            method=method or type(policy).__name__)
        LOG.info(res.summary())
        return res
