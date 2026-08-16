"""Fairness metrics: the paper's, and the two the gaps add.

Paper's measures
----------------
Eq. 1  total utility            sum_v o_v
Eq. 2  long-term fairness       Var(o_v)          over weekly *totals*
Eq. 8  normalised fairness      sigma(o_v)/mean(o_v)

Gap 1b: rate fairness
---------------------
The paper's Eq. 2 is invariant to hours worked, so equalising totals -- which is
exactly what minimising Var(o_v) drives towards -- can hide a multi-fold gap in
hourly rate. Measured on this fleet: equalised totals give Var(o_v) = 0
("perfectly fair") while hourly rates span 47.9 to 375.7, a 7.8x gap, with
part-timers on 3.59x the rate of full-timers.

So we measure the rate rho_v = o_v / H_v and decompose its variance with the law
of total variance:

    Var(rho) = E_g[Var(rho | g)]  +  Var_g(E[rho | g])
               \\___ within ___/      \\___ between ___/

which yields the within-group and between-group contrast Gap 1b asks for as an
identity, not an ad-hoc construction. `within + between == total` is asserted.

Gap 2: utilisation / access fairness
------------------------------------
The paper checks only final earnings, never whether an available driver was
given work. A driver online 8 h and dispatched twice can match the earnings of
one dispatched ten times, and Eq. 2 calls that fair while 6 h of forced idleness
go unrecorded.

    nu_v = busy slots / online slots

Reported raw *and* opportunity-normalised. The raw form conflates two very
different things: a platform that starves a driver, and a driver who chose to be
online at 04:00 when there is no demand. The normalised form divides each
driver's utilisation by what drivers online in the *same slots* actually
achieved, so it isolates the part the algorithm is responsible for. Reporting
only the raw form would overstate the unfairness; reporting only the normalised
form would hide genuine starvation. Both are given.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..config import Config
from ..sim.drivers import FULL_TIME, PART_TIME, GROUP_NAMES

EPS = 1e-12


# --------------------------------------------------------------------------
def gini(x: np.ndarray) -> float:
    """Gini coefficient; 0 = perfect equality. Shifted if negatives are present."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan")
    if x.min() < 0:
        x = x - x.min()
    s = x.sum()
    if s <= EPS:
        return 0.0
    xs = np.sort(x)
    n = len(xs)
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * s))


def cv(x: np.ndarray) -> float:
    """Coefficient of variation = the paper's normalised fairness (Eq. 8)."""
    x = np.asarray(x, float)
    mu = x.mean()
    return float(x.std() / mu) if abs(mu) > EPS else float("nan")


def decompose_variance(x: np.ndarray, group: np.ndarray) -> tuple[float, float, float]:
    """Law of total variance: returns (total, within, between).

    within  = sum_g w_g * Var(x | g)
    between = sum_g w_g * (mean(x | g) - mean(x))^2
    with w_g the group's population share. within + between == total exactly.
    """
    x = np.asarray(x, float)
    total = float(x.var())
    mu = x.mean()
    within = 0.0
    between = 0.0
    for g in np.unique(group):
        m = group == g
        w = m.mean()
        within += w * float(x[m].var())
        between += w * float((x[m].mean() - mu) ** 2)
    return total, within, between


# --------------------------------------------------------------------------
@dataclass
class Metrics:
    """Every number a method is scored on, paper's and ours side by side."""

    method: str
    # -- efficiency (paper Eq. 1)
    total_utility: float
    n_assigned: int
    service_rate: float
    mean_utility: float
    min_utility: float
    max_utility: float
    # -- paper's fairness (Eq. 2, Eq. 8)
    fairness_total_var: float
    fairness_normalised: float
    gini_total: float
    # -- Gap 1b: rate fairness
    rate_mean: float
    rate_min: float
    rate_max: float
    rate_var: float
    rate_var_within: float
    rate_var_between: float
    rate_normalised: float
    gini_rate: float
    rate_full_time: float
    rate_part_time: float
    rate_group_ratio: float
    # -- Gap 2: utilisation fairness
    util_mean: float
    util_min: float
    util_max: float
    util_var: float
    util_var_within: float
    util_var_between: float
    util_normalised: float
    gini_util: float
    util_full_time: float
    util_part_time: float
    util_adj_var: float
    util_adj_min: float
    util_adj_max: float
    # -- context
    n_drivers: int
    n_idle_drivers: int
    hours_total: float

    def to_dict(self) -> dict:
        return asdict(self)

    def paper_row(self) -> str:
        """Table-1-format row: the columns Kang et al. report."""
        return (f"{self.method:<26} {self.total_utility:>14,.2f} "
                f"{self.fairness_total_var:>16,.2f} "
                f"{self.fairness_normalised:>10.4f} "
                f"{self.min_utility:>12,.2f} {self.mean_utility:>12,.2f} "
                f"{self.max_utility:>12,.2f}")

    @staticmethod
    def paper_header() -> str:
        return (f"{'Method':<26} {'Total Utility':>14} {'Fairness':>16} "
                f"{'Norm.Fair':>10} {'Min':>12} {'Mean':>12} {'Max':>12}")


def opportunity_normalised_utilisation(res, drivers) -> np.ndarray:
    """Each driver's utilisation relative to peers online in the same slots.

    opportunity_v = mean over v's online slots of the system-wide busy fraction
                    in that slot
    nu_adj_v      = nu_v / opportunity_v

    1.0 means the driver did as well as a typical driver online at the same
    times; below 1.0 means the algorithm under-served them relative to what
    those hours actually offered.
    """
    online_by_slot = np.maximum(res.online_by_slot, 1)
    slot_busy_frac = res.busy_by_slot / online_by_slot          # (T,)
    on = drivers.online                                         # (n,T) bool
    denom = np.maximum(on.sum(axis=1), 1)
    opportunity = (on * slot_busy_frac[None, :]).sum(axis=1) / denom
    nu = res.busy_slots / np.maximum(res.online_slots, 1)
    return nu / np.maximum(opportunity, EPS)


def compute(res, drivers, cfg: Config | None = None) -> Metrics:
    """Score an EpisodeResult on the paper's metrics and the gap metrics."""
    cfg = cfg or Config()
    o = np.asarray(res.cum_utility, float)
    h = np.asarray(res.online_hours, float)
    grp = np.asarray(res.group)

    # Gap 1b: hourly rate. Drivers with too few hours are excluded from the
    # rate statistics -- one lucky trip in a 20-minute session is not a "rate".
    trusted = h >= cfg.fairness.min_hours_for_rate
    rho = np.where(trusted, o / np.maximum(h, EPS), np.nan)
    rho_ok = rho[trusted]
    grp_ok = grp[trusted]
    r_tot, r_win, r_btw = decompose_variance(rho_ok, grp_ok)

    ft = grp_ok == FULL_TIME
    pt = grp_ok == PART_TIME
    rate_ft = float(rho_ok[ft].mean()) if ft.any() else float("nan")
    rate_pt = float(rho_ok[pt].mean()) if pt.any() else float("nan")

    # Gap 2: utilisation
    nu = res.busy_slots / np.maximum(res.online_slots, 1)
    u_tot, u_win, u_btw = decompose_variance(nu, grp)
    nu_adj = opportunity_normalised_utilisation(res, drivers)
    nu_ft = float(nu[grp == FULL_TIME].mean()) if (grp == FULL_TIME).any() else float("nan")
    nu_pt = float(nu[grp == PART_TIME].mean()) if (grp == PART_TIME).any() else float("nan")

    return Metrics(
        method=res.method,
        total_utility=float(o.sum()),
        n_assigned=int(res.n_assigned),
        service_rate=float(res.service_rate),
        mean_utility=float(o.mean()), min_utility=float(o.min()),
        max_utility=float(o.max()),
        fairness_total_var=float(o.var()),
        fairness_normalised=cv(o),
        gini_total=gini(o),
        rate_mean=float(np.nanmean(rho)), rate_min=float(np.nanmin(rho)),
        rate_max=float(np.nanmax(rho)),
        rate_var=r_tot, rate_var_within=r_win, rate_var_between=r_btw,
        rate_normalised=cv(rho_ok), gini_rate=gini(rho_ok),
        rate_full_time=rate_ft, rate_part_time=rate_pt,
        rate_group_ratio=float(rate_pt / rate_ft) if rate_ft not in (0.0,) else float("nan"),
        util_mean=float(nu.mean()), util_min=float(nu.min()), util_max=float(nu.max()),
        util_var=u_tot, util_var_within=u_win, util_var_between=u_btw,
        util_normalised=cv(nu), gini_util=gini(nu),
        util_full_time=nu_ft, util_part_time=nu_pt,
        util_adj_var=float(np.var(nu_adj)), util_adj_min=float(nu_adj.min()),
        util_adj_max=float(nu_adj.max()),
        n_drivers=int(len(o)), n_idle_drivers=int((res.n_trips == 0).sum()),
        hours_total=float(h.sum()),
    )


def check_decomposition(x: np.ndarray, group: np.ndarray, tol: float = 1e-8) -> bool:
    t, w, b = decompose_variance(x, group)
    return abs(t - (w + b)) <= tol * max(1.0, abs(t))
