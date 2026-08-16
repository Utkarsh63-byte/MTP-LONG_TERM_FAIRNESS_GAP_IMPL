"""Driver population: shifts, hours worked, and activity groups.

Why this module exists
----------------------
The NYC yellow-taxi feed carries no driver identifier, so the driver side is
simulated in every paper in this line of work. Kang et al. never report the
driver count, the shift model, or any notion of hours worked -- and that
omission *is* Gap 1b. Their fairness measure, Var over weekly total utility
(Eq. 2), is invariant to how long a driver was online: a 60 h/week driver and a
15 h/week driver on equal totals score a perfect 0, hiding a 4x gap in hourly
rate.

Making online time explicit is therefore the precondition for both gaps:
  * Gap 1b needs H_v (hours online) to form the rate rho_v = o_v / H_v, and a
    group label to decompose Var(rho) into within- and between-group parts.
  * Gap 2 needs the *set* of online slots, not just their count, to ask whether
    an available driver was actually given work.

Shift model
-----------
Drivers are not online at uniformly random 5-minute instants -- real drivers
work contiguous shifts on a personal schedule. Each driver draws:
  * a group (full-time / part-time) and a weekly hour target,
  * a start-time archetype (morning / day / evening / night) held fixed all
    week, so schedules are consistent rather than reshuffled daily,
  * a number of working days and a shift length consistent with the target,
  * a home node, mostly demand-weighted, partly uniform.

Holding the archetype fixed matters for the research question: it produces
drivers who are systematically online in thin hours. That is exactly the
population on which raw utilisation variance would blame the platform for a
driver's own schedule, which is why Phase 6 also reports an
opportunity-normalised utilisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import LOG, rng
from .timeline import Timeline

FULL_TIME, PART_TIME = 0, 1
GROUP_NAMES = {FULL_TIME: "full-time", PART_TIME: "part-time"}

# start-hour window per archetype; "night" wraps past midnight
ARCHETYPES = ["morning", "day", "evening", "night"]
ARCH_START = {"morning": (5.0, 9.0), "day": (9.0, 13.0),
              "evening": (15.0, 19.0), "night": (20.0, 25.0)}
ARCH_WEIGHTS = np.array([0.25, 0.30, 0.30, 0.15])


@dataclass
class DriverPopulation:
    """A simulated driver fleet over a fixed timeline."""

    group: np.ndarray          # (n,) int8
    archetype: np.ndarray      # (n,) int8
    target_hours: np.ndarray   # (n,) float32 weekly target
    home_node: np.ndarray      # (n,) int16
    online: np.ndarray         # (n,T) bool
    shift_start: np.ndarray    # (n,T) bool, first slot of each shift
    n_shifts: np.ndarray       # (n,) int32
    slot_hours: float

    @property
    def n(self) -> int:
        return self.online.shape[0]

    @property
    def n_slots(self) -> int:
        return self.online.shape[1]

    @property
    def online_slots(self) -> np.ndarray:
        return self.online.sum(axis=1).astype(np.int32)

    @property
    def online_hours(self) -> np.ndarray:
        """H_v: the denominator of the hourly rate in Gap 1b."""
        return self.online_slots * self.slot_hours

    def group_mask(self, g: int) -> np.ndarray:
        return self.group == g

    def realised_group(self, cfg: Config) -> np.ndarray:
        """Group label implied by *realised* hours, per the config thresholds."""
        h = self.online_hours
        out = np.full(self.n, -1, np.int8)
        out[h >= cfg.driver.full_time_threshold_h] = FULL_TIME
        out[h <= cfg.driver.part_time_threshold_h] = PART_TIME
        return out

    def concurrent_online(self) -> np.ndarray:
        """(T,) drivers online per slot -- the supply curve."""
        return self.online.sum(axis=0).astype(np.int32)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "driver": np.arange(self.n),
            "group": [GROUP_NAMES[g] for g in self.group],
            "archetype": [ARCHETYPES[a] for a in self.archetype],
            "target_hours": self.target_hours,
            "online_hours": self.online_hours,
            "n_shifts": self.n_shifts,
            "home_node": self.home_node,
        })


def _demand_weights(demand, hour_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Pickup volume per node over the horizon, as a placement distribution."""
    h = np.unique(np.clip(hour_index, 0, demand.org.shape[1] - 1))
    w = demand.org[:, h].sum(axis=1).astype(float)
    if len(w) < n_nodes:
        w = np.pad(w, (0, n_nodes - len(w)))
    w = w[:n_nodes]
    return w / w.sum() if w.sum() > 0 else np.full(n_nodes, 1.0 / n_nodes)


def build_drivers(cfg: Config, timeline: Timeline, demand=None,
                  n_nodes: int = 87, seed: int | None = None) -> DriverPopulation:
    d = cfg.driver
    r = rng(d.seed if seed is None else seed)
    n, T, D = d.n_drivers, timeline.n_slots, timeline.n_days
    slot_h = timeline.slot_hours
    week_scale = D / 7.0          # weekly targets scaled to the horizon length

    group = np.where(r.random(n) < d.full_time_share, FULL_TIME,
                     PART_TIME).astype(np.int8)
    arch = r.choice(len(ARCHETYPES), size=n, p=ARCH_WEIGHTS).astype(np.int8)

    target = np.empty(n, np.float32)
    ft, pt = group == FULL_TIME, group == PART_TIME
    target[ft] = r.integers(d.full_time_hours[0], d.full_time_hours[1] + 1, ft.sum())
    target[pt] = r.integers(d.part_time_hours[0], d.part_time_hours[1] + 1, pt.sum())

    # home node
    if d.start_from_demand and demand is not None:
        w = _demand_weights(demand, timeline.hour_index, n_nodes)
    else:
        w = np.full(n_nodes, 1.0 / n_nodes)
    home = np.where(r.random(n) < d.start_uniform_mix,
                    r.integers(0, n_nodes, n),
                    r.choice(n_nodes, size=n, p=w)).astype(np.int16)

    # midnight of each day on the absolute axis, plus weekend flags
    day_of_slot = timeline.day
    day_midnight = np.empty(D, dtype="datetime64[m]")
    is_weekend_day = np.zeros(D, bool)
    for di in range(D):
        s = np.where(day_of_slot == di)[0]
        if len(s):
            t0 = pd.Timestamp(timeline.ts[s[0]])
            day_midnight[di] = np.datetime64(t0.normalize(), "m")
            is_weekend_day[di] = bool(t0.dayofweek >= 5)

    online = np.zeros((n, T), bool)
    shift_start = np.zeros((n, T), bool)
    n_shifts = np.zeros(n, np.int32)

    for v in range(n):
        h_target = float(target[v]) * week_scale
        if group[v] == FULL_TIME:
            preferred = r.uniform(7.0, d.max_shift_hours)
            day_w = np.where(is_weekend_day, 0.7, 1.0)
        else:
            preferred = r.uniform(d.min_shift_hours, 6.0)
            day_w = np.where(is_weekend_day, 1.6, 1.0)   # part-timers skew to weekends
        n_days = int(np.clip(round(h_target / preferred), 1, D))
        shift_len = float(np.clip(h_target / n_days, d.min_shift_hours,
                                  d.max_shift_hours))

        day_w = day_w / day_w.sum()
        days = r.choice(D, size=n_days, replace=False, p=day_w)

        lo, hi = ARCH_START[ARCHETYPES[arch[v]]]
        base = r.uniform(lo, hi)
        for di in days:
            # Paint on the absolute time axis, not within the day. A night
            # shift starting 23:00 must run forward into the next day's early
            # slots; selecting by hour-of-day within one day would instead put
            # the wrapped hours *before* the shift start and split it in two.
            start_ts = day_midnight[di] + np.timedelta64(
                int(round((base + r.normal(0.0, 0.75)) * 60)), "m")
            end_ts = start_ts + np.timedelta64(int(round(shift_len * 60)), "m")
            a, b = np.searchsorted(timeline.ts, [start_ts, end_ts])
            if b <= a:
                continue
            online[v, a:b] = True
            n_shifts[v] += 1
        # mark the first slot of every contiguous online run
        row = online[v]
        starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
        shift_start[v, starts] = True

    pop = DriverPopulation(group=group, archetype=arch, target_hours=target,
                           home_node=home, online=online,
                           shift_start=shift_start, n_shifts=n_shifts,
                           slot_hours=slot_h)

    oh = pop.online_hours
    LOG.info("drivers: %d (%d full-time, %d part-time) over %d days",
             n, int(ft.sum()), int(pt.sum()), D)
    LOG.info("  online hours: full-time mean %.1f (min %.1f max %.1f), "
             "part-time mean %.1f (min %.1f max %.1f)",
             oh[ft].mean(), oh[ft].min(), oh[ft].max(),
             oh[pt].mean(), oh[pt].min(), oh[pt].max())
    LOG.info("  hours ratio between groups: %.2fx; total online slots %d",
             oh[ft].mean() / max(oh[pt].mean(), 1e-9), int(pop.online_slots.sum()))
    conc = pop.concurrent_online()
    LOG.info("  concurrent online drivers: min %d, median %d, max %d",
             conc.min(), int(np.median(conc)), conc.max())
    return pop
