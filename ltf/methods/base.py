"""Policy interface and the incremental variance machinery every method shares.

A policy sees a `SlotContext` and returns (request, driver) index pairs. Because
all methods here optimise some variant of

    max  sum_v u_v  -  lambda * omega * Fairness

the expensive part is the *marginal* effect of one assignment on a variance, so
`VarianceTracker` maintains sufficient statistics and answers "what would
Var change by if driver j gained dx?" in O(#groups) rather than O(n).

It supports the three fairness quantities used across the phases:
  totals  x_v = o_v                       (paper, Eq. 2)
  rate    x_v = o_v / H_v                 (Gap 1b)
  util    x_v = busy_v / online_v         (Gap 2)
and, for the grouped variants, the within/between split from the law of total
variance so the two components can carry separate weights.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..sim.env import SlotContext

EPS = 1e-12


class Policy(Protocol):
    """Anything that can allocate one slot's requests."""

    def assign(self, ctx: SlotContext) -> np.ndarray:
        """Return an (a,2) array of (request row, driver column) index pairs."""
        ...


class VarianceTracker:
    """Incremental Var / within-Var / between-Var over a per-driver quantity."""

    def __init__(self, x: np.ndarray, group: np.ndarray):
        self.x = np.asarray(x, float).copy()
        self.group = np.asarray(group)
        self.groups = np.unique(self.group)
        self.n = len(self.x)
        self._idx = {int(g): np.where(self.group == g)[0] for g in self.groups}
        self._ng = {int(g): len(v) for g, v in self._idx.items()}
        self._sg = {int(g): float(self.x[v].sum()) for g, v in self._idx.items()}
        self._qg = {int(g): float((self.x[v] ** 2).sum()) for g, v in self._idx.items()}

    # -- current values -------------------------------------------------
    @property
    def S(self) -> float:
        return sum(self._sg.values())

    @property
    def Q(self) -> float:
        return sum(self._qg.values())

    def total(self) -> float:
        return self.Q / self.n - (self.S / self.n) ** 2

    def within(self) -> float:
        out = 0.0
        for g in self.groups:
            g = int(g)
            ng = self._ng[g]
            if ng == 0:
                continue
            out += (ng / self.n) * (self._qg[g] / ng - (self._sg[g] / ng) ** 2)
        return out

    def between(self) -> float:
        mu = self.S / self.n
        out = 0.0
        for g in self.groups:
            g = int(g)
            ng = self._ng[g]
            if ng == 0:
                continue
            out += (ng / self.n) * (self._sg[g] / ng - mu) ** 2
        return out

    # -- marginals ------------------------------------------------------
    def delta_total(self, j: int, dx: float) -> float:
        """Change in Var(x) if x[j] += dx. Closed form, O(1)."""
        n, S = self.n, self.S
        xj = self.x[j]
        dQ = 2.0 * xj * dx + dx * dx
        dS = dx
        return dQ / n - (2.0 * S * dS + dS * dS) / (n * n)

    def delta_within(self, j: int, dx: float) -> float:
        g = int(self.group[j])
        ng = self._ng[g]
        if ng == 0:
            return 0.0
        xj = self.x[j]
        dQ = 2.0 * xj * dx + dx * dx
        old = self._qg[g] / ng - (self._sg[g] / ng) ** 2
        new = (self._qg[g] + dQ) / ng - ((self._sg[g] + dx) / ng) ** 2
        return (ng / self.n) * (new - old)

    def delta_between(self, j: int, dx: float) -> float:
        return self.delta_total(j, dx) - self.delta_within(j, dx)

    def delta_vector_total(self, js: np.ndarray, dxs: np.ndarray) -> np.ndarray:
        """Vectorised `delta_total` over candidate (driver, increment) pairs."""
        n, S = self.n, self.S
        xj = self.x[np.asarray(js)]
        dx = np.asarray(dxs, float)
        dQ = 2.0 * xj * dx + dx * dx
        return dQ / n - (2.0 * S * dx + dx * dx) / (n * n)

    def delta_vector_within(self, js: np.ndarray, dxs: np.ndarray) -> np.ndarray:
        js = np.asarray(js)
        dx = np.asarray(dxs, float)
        out = np.zeros(len(js))
        gs = self.group[js]
        for g in self.groups:
            g = int(g)
            m = gs == g
            if not m.any():
                continue
            ng = self._ng[g]
            if ng == 0:
                continue
            xj = self.x[js[m]]
            d = dx[m]
            dQ = 2.0 * xj * d + d * d
            old = self._qg[g] / ng - (self._sg[g] / ng) ** 2
            new = (self._qg[g] + dQ) / ng - ((self._sg[g] + d) / ng) ** 2
            out[m] = (ng / self.n) * (new - old)
        return out

    # -- mutation -------------------------------------------------------
    def apply(self, j: int, dx: float) -> None:
        g = int(self.group[j])
        self._qg[g] += 2.0 * self.x[j] * dx + dx * dx
        self._sg[g] += dx
        self.x[j] += dx


def auto_omega(utility_scale: float, fairness_scale: float) -> float:
    """The paper's omega, chosen rather than guessed.

    Sec. 4.4 introduces omega purely "to scale utility and fairness into the same
    range", and fixes it at 0.6 without saying what range that produced. A
    variance term has squared units, so its magnitude relative to utility depends
    on the fleet size and the horizon -- hard-coding one number does not
    transfer. This returns the omega that puts the fairness penalty on the same
    scale as a typical utility, so lambda alone controls the trade-off.
    """
    if fairness_scale <= EPS:
        return 1.0
    return float(utility_scale / fairness_scale)


def calibrate_omegas(res, drivers, cfg) -> dict:
    """Measure the omega that puts each fairness penalty on the utility scale.

    The paper fixes omega = 0.6 "to scale utility and fairness into the same
    range" (Sec. 4.4) without reporting what range that was. It does not
    transfer: a variance carries squared units, so the ratio between a fairness
    marginal and a single trip's utility depends on fleet size, horizon, and --
    critically -- on *which* quantity the variance is taken over.

    Left uncalibrated the effect is not subtle. With omega_util defaulted to the
    mean online-slot count, the utilisation penalty came out around 0.009 against
    a typical trip utility of 2.4, i.e. roughly 270 times too small, so
    the Gap 2 term was inert and its ablation showed no effect.

    Here each omega is set so that a typical assignment's fairness marginal is
    comparable to a typical assignment's utility, leaving lambda as the only
    trade-off knob. Inputs are measured from an efficiency-only episode.
    """
    u = np.asarray(res.assign_utility, float)
    if len(u) == 0:
        return {"omega_total": 1.0, "omega_rate": 1.0, "omega_util": 1.0}
    mean_u = float(np.mean(np.abs(u)))
    n = len(res.cum_utility)
    o = np.asarray(res.cum_utility, float)
    h = np.maximum(np.asarray(res.online_hours, float), EPS)
    online = np.maximum(np.asarray(res.online_slots, float), 1.0)
    occ = np.asarray(res.assign_occupancy, float)

    def omega_for(x: np.ndarray, dx: float) -> float:
        # d(Var)/d(one assignment) ~ 2 * mean(x) * dx / n
        scale = 2.0 * float(np.mean(np.abs(x))) * dx / max(n, 1)
        return float(mean_u / scale) if scale > EPS else 1.0

    out = {
        "omega_total": omega_for(o, mean_u),
        "omega_rate": omega_for(o / h, mean_u / float(np.mean(h))),
        "omega_util": omega_for(res.busy_slots / online,
                                float(np.mean(occ)) / float(np.mean(online))),
    }
    return out
