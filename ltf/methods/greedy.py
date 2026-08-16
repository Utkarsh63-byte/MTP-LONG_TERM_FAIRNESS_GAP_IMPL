"""Greedy allocation, and the objective every other method is compared against.

`ScalarisedGreedy` is one class serving several roles, selected by config:

  paper Greedy baseline   fairness on totals, lambda = 1, omega = 0.6
                          (Sec. 5.2: "Greedy with the objective of balancing
                          efficiency and fairness according to Eq. 3")
  efficiency-only         lambda = 0; the "w/o fairness" ablation of Table 2
  gap-aware greedy        fairness on rate (within/between) and utilisation

Within a slot it repeatedly takes the pair with the highest marginal scalarised
gain

    gain(i, j) = U[i,j] - lambda * omega * dFairness(j | U[i,j])

recomputing the fairness marginal after every pick, since the penalty for giving
another trip to an already-rich driver depends on what has been handed out so
far. That is a maximum-weight greedy matching, not first-come-first-served, so
the baseline is a fair fight rather than a straw man.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..sim.env import SlotContext
from .base import VarianceTracker

EPS = 1e-12


class ScalarisedGreedy:
    """Greedy maximum-gain matching under the scalarised objective."""

    def __init__(self, cfg: Config, name: str = "Greedy",
                 fairness_target: str | None = None,
                 lambda_fair: float | None = None,
                 omega: float | None = None,
                 use_utilisation: bool = False):
        self.cfg = cfg
        self.name = name
        f = cfg.fairness
        self.target = fairness_target or f.target
        self.lam = f.lambda_fair if lambda_fair is None else lambda_fair
        self.omega = f.omega if omega is None else omega
        self.lam_within = f.lambda_within
        self.lam_between = f.lambda_between
        self.lam_util = f.lambda_util if use_utilisation else 0.0
        self.omega_rate = f.omega_rate
        self.omega_util = f.omega_util
        self._tr_fair: VarianceTracker | None = None
        self._tr_util: VarianceTracker | None = None
        self._hours: np.ndarray | None = None
        self._online: np.ndarray | None = None

    # ------------------------------------------------------------------
    def reset(self, scenario) -> None:
        dr = scenario.drivers
        n = dr.n
        self._hours = np.maximum(dr.online_hours, EPS)
        self._online = np.maximum(dr.online_slots, 1)
        # fairness quantity starts at zero for everyone
        self._tr_fair = VarianceTracker(np.zeros(n), dr.group)
        self._tr_util = VarianceTracker(np.zeros(n), dr.group)
        # omega defaults: the paper's 0.6 for totals; for the rate and
        # utilisation terms the natural scale differs by orders of magnitude
        # (rate is utility/hour, utilisation is a fraction), so unset weights
        # are scaled by the quantity's own denominator rather than left at 0.6.
        if self.omega_rate is None:
            self.omega_rate = float(np.mean(self._hours))
        if self.omega_util is None:
            self.omega_util = float(np.mean(self._online))

    # ------------------------------------------------------------------
    def _fair_increment(self, drv_global: np.ndarray, u: np.ndarray,
                        occ: np.ndarray) -> np.ndarray:
        """dx applied to the fairness quantity by giving driver j utility u."""
        if self.target == "total":
            return u
        return u / self._hours[drv_global]          # rate targets

    def _penalty(self, drv_global: np.ndarray, u: np.ndarray,
                 occ: np.ndarray) -> np.ndarray:
        """lambda * omega * dFairness for each candidate pairing."""
        if self.lam == 0.0 and self.lam_util == 0.0:
            return np.zeros(len(u))
        pen = np.zeros(len(u))
        dx = self._fair_increment(drv_global, u, occ)
        if self.target == "total":
            pen += self.lam * self.omega * self._tr_fair.delta_vector_total(
                drv_global, dx)
        elif self.target == "rate":
            pen += (self.lam * self.omega_rate
                    * self._tr_fair.delta_vector_total(drv_global, dx))
        else:  # rate_grouped
            w = self._tr_fair.delta_vector_within(drv_global, dx)
            t = self._tr_fair.delta_vector_total(drv_global, dx)
            b = t - w
            pen += self.omega_rate * (self.lam_within * w + self.lam_between * b)
        if self.lam_util > 0.0:
            du = occ / self._online[drv_global]
            pen += (self.lam_util * self.omega_util
                    * self._tr_util.delta_vector_total(drv_global, du))
        return pen

    # ------------------------------------------------------------------
    def assign(self, ctx: SlotContext) -> np.ndarray:
        m, k = ctx.m, ctx.k
        if m == 0 or k == 0:
            return np.empty((0, 2), np.int64)

        free_req = np.ones(m, bool)
        free_drv = np.ones(k, bool)
        pairs: list[tuple[int, int]] = []
        n_max = min(m, k)

        for _ in range(n_max):
            ri, ci = np.where(free_req[:, None] & free_drv[None, :] & ctx.feasible)
            if len(ri) == 0:
                break
            u = ctx.utility[ri, ci]
            occ = ctx.occupancy[ri, ci].astype(float)
            gj = ctx.drv_idx[ci]
            gain = u - self._penalty(gj, u, occ)
            best = int(np.argmax(gain))
            # a strictly negative gain means this assignment hurts the
            # objective; the paper's MDP has an explicit no-action for exactly
            # this case (Sec. 4.3), so stop rather than force a match
            if gain[best] <= 0.0:
                break
            bi, bj = int(ri[best]), int(ci[best])
            pairs.append((bi, bj))
            free_req[bi] = False
            free_drv[bj] = False
            v = int(ctx.drv_idx[bj])
            uu = float(ctx.utility[bi, bj])
            self._tr_fair.apply(v, float(self._fair_increment(
                np.array([v]), np.array([uu]), np.array([occ[best]]))[0]))
            self._tr_util.apply(v, float(ctx.occupancy[bi, bj]) / self._online[v])

        return np.asarray(pairs, np.int64).reshape(-1, 2)


def make_paper_greedy(cfg: Config) -> ScalarisedGreedy:
    """Greedy as the paper specifies it: Eq. 3 on totals, lambda=1, omega=0.6."""
    return ScalarisedGreedy(cfg, name="Greedy", fairness_target="total",
                            lambda_fair=cfg.fairness.lambda_fair,
                            omega=cfg.fairness.omega)


def make_efficiency_only(cfg: Config) -> ScalarisedGreedy:
    """Pure efficiency: the 'w/o fairness' ablation of Table 2."""
    return ScalarisedGreedy(cfg, name="Greedy (efficiency only)",
                            fairness_target="total", lambda_fair=0.0)
