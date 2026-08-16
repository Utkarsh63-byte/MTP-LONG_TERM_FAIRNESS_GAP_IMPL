"""Multi-objective multi-agent Q-learning (paper Sec. 4.3-4.4) and its variants.

State abstraction
-----------------
The paper's state is "the various locations where drivers are initially
positioned and where ride requests originate and conclude" (Sec. 4.3) -- i.e.
location, with time implicit. A tabular value over (time profile, node) is the
standard formulation in this literature and is what makes the method non-myopic:
the value of finishing a trip at node d at time t captures what that position is
worth later.

We augment it (Phase 7) with a discretised *rate deficit* and *utilisation
deficit* bucket per driver. This is not cosmetic. A policy cannot optimise a
fairness term it cannot observe, so leaving the state as location-only would let
us measure the Gap 1b / Gap 2 objectives but never improve them. That extra
state dimension is the mechanism by which the gap objectives become optimisable
rather than merely reportable.

    V[profile, node, deficit_bucket]     48 x 87 x 5 = 20,880 entries

Scalarisation (Eq. 7)
---------------------
    score(i,j) = r_scalarised(i,j) + gamma^k * V[s'] - V[s]
    r_scalarised = u - lambda * omega * dFairness

with k the occupancy in slots, so a long trip is discounted for the time it
locks the driver up -- something the paper's timeless utility cannot express.

Prediction in the action space
------------------------------
The paper's stated novelty is that forecast requests enter the MDP's action
space so the learned policy anticipates demand. `train()` therefore runs
episodes over the training week *and* over a synthetic week sampled from the
forecaster's predicted demand. Disabling that synthetic portion yields exactly
the "w/o Prediction" ablation of Table 2, and is also how Balance Ride-Pooling
(Raman et al. [14]) is positioned by the paper: an RL method with fairness but
no view of future request patterns.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..sim.env import Environment, SlotContext
from ..utils import LOG, rng, timed
from .base import VarianceTracker

EPS = 1e-12


class MOMAQL:
    """Tabular value-based allocation with a scalarised fairness objective."""

    def __init__(self, cfg: Config, n_nodes: int, n_profiles: int = 48,
                 name: str = "MOMAQL", use_lookahead: bool = True,
                 fairness_target: str | None = None,
                 use_utilisation: bool = False):
        self.cfg = cfg
        self.name = name
        self.n_nodes = n_nodes
        self.n_profiles = n_profiles
        self.use_lookahead = use_lookahead
        f = cfg.fairness
        self.target = fairness_target or f.target
        self.lam = f.lambda_fair
        self.omega = f.omega
        self.lam_within = f.lambda_within
        self.lam_between = f.lambda_between
        self.lam_util = f.lambda_util if use_utilisation else 0.0
        self.omega_rate = f.omega_rate
        self.omega_util = f.omega_util
        r = cfg.rl
        self.gamma = r.gamma
        self.alpha = r.alpha
        self.n_rate_buckets = r.n_rate_buckets if r.use_rate_state else 1
        self.n_util_buckets = r.n_util_buckets if r.use_util_state else 1
        self.n_buckets = self.n_rate_buckets * self.n_util_buckets
        self.V = np.zeros((n_profiles, n_nodes, self.n_buckets), np.float64)
        self.epsilon = 0.0
        self._tr_fair: VarianceTracker | None = None
        self._tr_util: VarianceTracker | None = None
        self._hours = None
        self._online = None
        self._rng = rng(r.seed)
        self.trained = False

    # ------------------------------------------------------------------
    def reset(self, scenario) -> None:
        dr = scenario.drivers
        self._hours = np.maximum(dr.online_hours, EPS)
        self._online = np.maximum(dr.online_slots, 1)
        self._tr_fair = VarianceTracker(np.zeros(dr.n), dr.group)
        self._tr_util = VarianceTracker(np.zeros(dr.n), dr.group)
        if self.omega_rate is None:
            self.omega_rate = float(np.mean(self._hours))
        if self.omega_util is None:
            self.omega_util = float(np.mean(self._online))

    # -- state ----------------------------------------------------------
    def _bucket(self, drv_global: np.ndarray) -> np.ndarray:
        """Deficit bucket: how far behind peers this driver is, discretised.

        Deficit is measured on the *fairness quantity* the policy optimises, so
        the same code serves the totals, rate and utilisation objectives.
        """
        b = np.zeros(len(drv_global), np.int64)
        if self.n_rate_buckets > 1:
            x = self._tr_fair.x
            mu, sd = x.mean(), x.std() + EPS
            z = (x[drv_global] - mu) / sd
            rb = np.clip(np.digitize(z, [-1.0, -0.33, 0.33, 1.0]), 0,
                         self.n_rate_buckets - 1)
            b = b * self.n_rate_buckets + rb
        if self.n_util_buckets > 1:
            y = self._tr_util.x
            mu, sd = y.mean(), y.std() + EPS
            z = (y[drv_global] - mu) / sd
            ub = np.clip(np.digitize(z, [-1.0, -0.33, 0.33, 1.0]), 0,
                         self.n_util_buckets - 1)
            b = b * self.n_util_buckets + ub
        return np.clip(b, 0, self.n_buckets - 1)

    # -- objective ------------------------------------------------------
    def _increment(self, drv_global: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.target == "total":
            return u
        return u / self._hours[drv_global]

    def _penalty(self, drv_global: np.ndarray, u: np.ndarray,
                 occ: np.ndarray) -> np.ndarray:
        pen = np.zeros(len(u))
        dx = self._increment(drv_global, u)
        if self.target == "total":
            pen += self.lam * self.omega * self._tr_fair.delta_vector_total(
                drv_global, dx)
        elif self.target == "rate":
            pen += (self.lam * self.omega_rate
                    * self._tr_fair.delta_vector_total(drv_global, dx))
        else:
            w = self._tr_fair.delta_vector_within(drv_global, dx)
            t = self._tr_fair.delta_vector_total(drv_global, dx)
            pen += self.omega_rate * (self.lam_within * w
                                      + self.lam_between * (t - w))
        if self.lam_util > 0.0:
            du = occ / self._online[drv_global]
            pen += (self.lam_util * self.omega_util
                    * self._tr_util.delta_vector_total(drv_global, du))
        return pen

    def _score(self, ctx: SlotContext, ri: np.ndarray, ci: np.ndarray):
        gj = ctx.drv_idx[ci]
        u = ctx.utility[ri, ci]
        occ = ctx.occupancy[ri, ci].astype(float)
        r = u - self._penalty(gj, u, occ)
        b = self._bucket(gj)
        v_now = self.V[ctx.profile, ctx.drv_node[ci], b]
        # profile of the slot the driver frees up in, approximated by advancing
        # hours within the same day-type; exact wall-clock is unnecessary at
        # this granularity and keeps the table small
        adv = np.maximum(1, (occ * self.cfg.time.slot_minutes / 60.0).astype(int))
        prof_next = (ctx.profile // 24) * 24 + (ctx.profile + adv) % 24
        v_next = self.V[prof_next, ctx.req_d[ri], b]
        disc = self.gamma ** np.maximum(occ, 1.0)
        return r + disc * v_next - v_now, r, occ, gj, b, prof_next

    # -- acting ---------------------------------------------------------
    def assign(self, ctx: SlotContext, learn: bool = False) -> np.ndarray:
        if ctx.m == 0 or ctx.k == 0:
            return np.empty((0, 2), np.int64)
        free_r = np.ones(ctx.m, bool)
        free_d = np.ones(ctx.k, bool)
        pairs: list[tuple[int, int]] = []

        for _ in range(min(ctx.m, ctx.k)):
            ri, ci = np.where(free_r[:, None] & free_d[None, :] & ctx.feasible)
            if len(ri) == 0:
                break
            score, r, occ, gj, b, prof_next = self._score(ctx, ri, ci)
            if learn and self.epsilon > 0 and self._rng.random() < self.epsilon:
                pick = int(self._rng.integers(len(ri)))
            else:
                pick = int(np.argmax(score))
            if score[pick] <= 0.0 and not (learn and self.epsilon > 0):
                break
            bi, bj = int(ri[pick]), int(ci[pick])
            v = int(ctx.drv_idx[bj])

            if learn:
                s_prof, s_node, s_b = ctx.profile, int(ctx.drv_node[bj]), int(b[pick])
                target = r[pick] + (self.gamma ** max(occ[pick], 1.0)) * self.V[
                    int(prof_next[pick]), int(ctx.req_d[bi]), s_b]
                self.V[s_prof, s_node, s_b] += self.alpha * (
                    target - self.V[s_prof, s_node, s_b])

            pairs.append((bi, bj))
            free_r[bi] = False
            free_d[bj] = False
            self._tr_fair.apply(v, float(self._increment(
                np.array([v]), np.array([ctx.utility[bi, bj]]))[0]))
            self._tr_util.apply(v, float(ctx.occupancy[bi, bj]) / self._online[v])

        return np.asarray(pairs, np.int64).reshape(-1, 2)

    # -- training -------------------------------------------------------
    def train(self, scenarios: list, episodes: int | None = None) -> dict:
        """TD-learn the value table over one or more scenarios.

        Passing the real training-week scenario plus a synthetic scenario built
        from predicted demand is what puts forecast requests in the action space.
        """
        r = self.cfg.rl
        episodes = episodes or r.episodes
        hist = []
        with timed(f"training {self.name} ({episodes} episodes over "
                   f"{len(scenarios)} scenario(s))"):
            for ep in range(episodes):
                frac = min(1.0, ep / max(r.epsilon_decay_episodes, 1))
                self.epsilon = (r.epsilon_start
                                + frac * (r.epsilon_end - r.epsilon_start))
                tot = 0.0
                for sc in scenarios:
                    env = Environment(sc)
                    res = env.run(_LearningWrapper(self), method=self.name)
                    tot += res.total_utility
                hist.append(tot)
                if (ep + 1) % 10 == 0:
                    LOG.info("  episode %d/%d: eps=%.2f, train utility %.1f, "
                             "|V|>0 on %d/%d states", ep + 1, episodes,
                             self.epsilon, tot, int((self.V != 0).sum()), self.V.size)
        self.epsilon = 0.0
        self.trained = True
        return {"utility_history": hist,
                "states_visited": int((self.V != 0).sum()),
                "states_total": int(self.V.size)}


class _LearningWrapper:
    """Adapts MOMAQL to the Policy interface with learning switched on."""

    def __init__(self, agent: MOMAQL):
        self.agent = agent
        self.name = agent.name

    def reset(self, scenario) -> None:
        self.agent.reset(scenario)

    def assign(self, ctx: SlotContext) -> np.ndarray:
        return self.agent.assign(ctx, learn=True)


def make_balance_ride_pooling(cfg: Config, n_nodes: int) -> MOMAQL:
    """Raman et al. [14]: RL with fairness on totals, no future-demand module.

    The paper's own framing of this baseline (Sec. 5.3): "the method does not
    consider the patterns of requests in the future".
    """
    return MOMAQL(cfg, n_nodes, name="Balance Ride-Pooling",
                  use_lookahead=False, fairness_target="total")


def make_paper_method(cfg: Config, n_nodes: int) -> MOMAQL:
    """Kang et al. as published: fairness on totals + prediction in action space."""
    return MOMAQL(cfg, n_nodes, name="Kang et al. (reproduced)",
                  use_lookahead=True, fairness_target="total")


def make_gap_method(cfg: Config, n_nodes: int) -> MOMAQL:
    """Ours: rate fairness with group decomposition + utilisation term."""
    return MOMAQL(cfg, n_nodes, name="Ours (Gap 1+2)", use_lookahead=True,
                  fairness_target="rate_grouped", use_utilisation=True)
