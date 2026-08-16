"""Optimisation-based baselines: REASSIGN and LAF.

REASSIGN (Lesmana et al. [7], the paper's baseline of the same name)
    Two-stage by construction, which is what distinguishes it from a
    fairness-penalised matching: first solve for the efficient assignment, then
    *reassign* to lift the worst-off driver while giving up no more than a bounded
    fraction of total utility. Its native objective is min-max (maximise the
    minimum utility); Sec. 5.2 of the paper says the fairness definition was
    swapped to Eq. 2, so the reassignment step here targets Var(totals) while
    keeping the bounded-efficiency-loss mechanism.

LAF (Shi et al. [17])
    An MDP acts as a *re-weighting module* on the edges, then the Hungarian
    algorithm optimises total re-weighted utility. So unlike greedy, LAF solves
    the slot's matching optimally; the fairness pressure enters through the
    weights, not the matching. The re-weighting term here is the marginal
    fairness effect plus a learned per-node value, matching that structure.

Both use `scipy.optimize.linear_sum_assignment`, so within a slot the matching is
optimal rather than sequential -- the greedy baseline is the weaker matcher, and
keeping that difference real is the point of having both.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import Config
from ..sim.env import SlotContext
from .base import VarianceTracker

EPS = 1e-12
BIG = -1e9


def _hungarian(weight: np.ndarray, feasible: np.ndarray,
               require_positive: bool = True) -> list[tuple[int, int]]:
    """Max-weight matching on a (m,k) weight matrix, honouring feasibility."""
    m, k = weight.shape
    w = np.where(feasible, weight, BIG)
    if not np.isfinite(w).any():
        return []
    ri, ci = linear_sum_assignment(-w)
    out = []
    for i, j in zip(ri, ci):
        if not feasible[i, j]:
            continue
        if require_positive and weight[i, j] <= 0:
            continue
        out.append((int(i), int(j)))
    return out


class ReassignPolicy:
    """Efficiency-first matching, then bounded-loss reassignment for fairness."""

    def __init__(self, cfg: Config, name: str = "REASSIGN",
                 efficiency_loss_budget: float = 0.15, max_passes: int = 3):
        self.cfg = cfg
        self.name = name
        self.budget = efficiency_loss_budget
        self.max_passes = max_passes
        self._tr: VarianceTracker | None = None

    def reset(self, scenario) -> None:
        self._tr = VarianceTracker(np.zeros(scenario.drivers.n),
                                   scenario.drivers.group)

    def assign(self, ctx: SlotContext) -> np.ndarray:
        if ctx.m == 0 or ctx.k == 0:
            return np.empty((0, 2), np.int64)

        # stage 1: efficient assignment
        base = _hungarian(ctx.utility, ctx.feasible)
        if not base:
            return np.empty((0, 2), np.int64)
        eff_total = sum(ctx.utility[i, j] for i, j in base)
        budget = self.budget * max(eff_total, 0.0)

        # stage 2: reassign requests to worse-off drivers while the utility
        # given up stays inside the budget and Var(totals) strictly improves
        assign = dict(base)                     # request -> driver column
        used = set(assign.values())
        spent = 0.0
        for _ in range(self.max_passes):
            improved = False
            for i, j in list(assign.items()):
                cur_u = ctx.utility[i, j]
                gj = int(ctx.drv_idx[j])
                # candidates: unused, feasible, and currently poorer
                for jj in range(ctx.k):
                    if jj in used or not ctx.feasible[i, jj]:
                        continue
                    alt_u = ctx.utility[i, jj]
                    if alt_u <= 0:
                        continue
                    loss = cur_u - alt_u
                    if loss < 0:
                        loss = 0.0
                    if spent + loss > budget:
                        continue
                    gjj = int(ctx.drv_idx[jj])
                    d_now = self._tr.delta_total(gj, float(cur_u))
                    d_alt = self._tr.delta_total(gjj, float(alt_u))
                    if d_alt < d_now - 1e-12:
                        used.discard(j)
                        used.add(jj)
                        assign[i] = jj
                        spent += loss
                        improved = True
                        break
            if not improved:
                break

        pairs = [(i, j) for i, j in assign.items()]
        for i, j in pairs:
            self._tr.apply(int(ctx.drv_idx[j]), float(ctx.utility[i, j]))
        return np.asarray(pairs, np.int64).reshape(-1, 2)


class LAFPolicy:
    """Edge re-weighting (MDP term + fairness marginal) then optimal matching."""

    def __init__(self, cfg: Config, name: str = "LAF",
                 value_table: np.ndarray | None = None,
                 fairness_target: str | None = None):
        self.cfg = cfg
        self.name = name
        self.V = value_table            # optional (n_profiles, n_nodes) node values
        self.target = fairness_target or "total"
        f = cfg.fairness
        self.lam = f.lambda_fair
        self.omega = f.omega
        self.omega_rate = f.omega_rate
        self._tr: VarianceTracker | None = None
        self._hours: np.ndarray | None = None

    def reset(self, scenario) -> None:
        self._tr = VarianceTracker(np.zeros(scenario.drivers.n),
                                   scenario.drivers.group)
        self._hours = np.maximum(scenario.drivers.online_hours, EPS)
        if self.omega_rate is None:
            self.omega_rate = float(np.mean(self._hours))

    def _increment(self, gj: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.target == "total":
            return u
        return u / self._hours[gj]

    def assign(self, ctx: SlotContext) -> np.ndarray:
        if ctx.m == 0 or ctx.k == 0:
            return np.empty((0, 2), np.int64)
        m, k = ctx.m, ctx.k
        gj = ctx.drv_idx
        w = ctx.utility.copy()

        # fairness re-weighting: penalise edges that worsen the variance
        dx = self._increment(np.tile(gj, (m, 1)).ravel(), ctx.utility.ravel())
        pen = self._tr.delta_vector_total(np.tile(gj, (m, 1)).ravel(), dx)
        scale = self.omega if self.target == "total" else self.omega_rate
        w = w - self.lam * scale * pen.reshape(m, k)

        # MDP re-weighting term: value of the node the driver ends up at
        if self.V is not None:
            w = w + self.V[ctx.profile, ctx.req_d][:, None]

        pairs = _hungarian(w, ctx.feasible)
        out = []
        for i, j in pairs:
            if ctx.utility[i, j] <= 0:
                continue
            out.append((i, j))
            self._tr.apply(int(gj[j]), float(self._increment(
                np.array([gj[j]]), np.array([ctx.utility[i, j]]))[0]))
        return np.asarray(out, np.int64).reshape(-1, 2)
