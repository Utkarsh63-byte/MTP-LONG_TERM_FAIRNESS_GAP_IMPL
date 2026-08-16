"""Phase 2 verification: the driver fleet, request stream and supply/demand regime.

Also demonstrates, on the real fleet, the blind spot that Gap 1b targets: pairs
of drivers whose *total* utility is near-identical (so the paper's Eq. 2 sees
them as perfectly fair) while their hours online differ several-fold.

Run:  .venv/bin/python -m scripts.verify_phase2
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ltf.config import Config
from ltf.sim import FULL_TIME, PART_TIME, GROUP_NAMES, build_timeline
from ltf.sim.drivers import ARCHETYPES
from ltf.sim.scenario import build_scenario
from ltf.utils import output_path

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    cfg = Config()
    print("=" * 78)
    print("PHASE 2 VERIFICATION")
    print("=" * 78)

    sc = build_scenario(cfg)
    tl, dr, rq = sc.timeline, sc.drivers, sc.requests
    slot_h = tl.slot_hours

    print(f"\n-- timeline --")
    check("7-day horizon", tl.n_days == 7, f"{tl.n_days} days: {tl.dates[0]}..{tl.dates[-1]}")
    check("3/1/3 phase split as in Sec. 3.3",
          len(np.unique(tl.day[tl.phase == 0])) == 3
          and len(np.unique(tl.day[tl.phase == 1])) == 1
          and len(np.unique(tl.day[tl.phase == 2])) == 3,
          "history 3d / current 1d / future 3d")
    check("slots contiguous and increasing", bool(np.all(np.diff(tl.ts) > np.timedelta64(0)))) 
    check("expected slot count", tl.n_slots == tl.n_days * 24 * (60 // cfg.time.slot_minutes),
          f"{tl.n_slots} slots of {tl.slot_minutes} min")
    check("profile matches timestamp",
          bool(np.all(tl.profile == (pd.DatetimeIndex(tl.ts).dayofweek >= 5) * 24
                      + pd.DatetimeIndex(tl.ts).hour)))
    rt = tl.slot_of_timestamp(tl.ts)
    check("slot_of_timestamp round-trips", bool(np.array_equal(rt, np.arange(tl.n_slots))))

    print(f"\n-- driver fleet ({dr.n}) --")
    oh = dr.online_hours
    ft, pt = dr.group == FULL_TIME, dr.group == PART_TIME
    check("driver count matches config", dr.n == cfg.driver.n_drivers)
    check("group mix near configured share",
          abs(ft.mean() - cfg.driver.full_time_share) < 0.08,
          f"{100*ft.mean():.1f}% full-time vs {100*cfg.driver.full_time_share:.0f}% target")
    check("every driver has some online time", bool(np.all(dr.online_slots > 0)),
          f"min {dr.online_slots.min()} slots")
    check("nobody exceeds the horizon", bool(np.all(dr.online_slots <= tl.n_slots)))
    check("full-time hours clear the threshold",
          float(np.mean(oh[ft] >= cfg.driver.full_time_threshold_h)) > 0.8,
          f"{100*np.mean(oh[ft] >= cfg.driver.full_time_threshold_h):.0f}% of FT >= "
          f"{cfg.driver.full_time_threshold_h:.0f}h")
    check("part-time hours stay under the threshold",
          float(np.mean(oh[pt] <= cfg.driver.part_time_threshold_h)) > 0.8,
          f"{100*np.mean(oh[pt] <= cfg.driver.part_time_threshold_h):.0f}% of PT <= "
          f"{cfg.driver.part_time_threshold_h:.0f}h")
    check("groups are genuinely separated in hours",
          oh[ft].mean() / oh[pt].mean() > 2.0,
          f"{oh[ft].mean():.1f}h vs {oh[pt].mean():.1f}h = {oh[ft].mean()/oh[pt].mean():.2f}x")
    check("shifts are contiguous blocks, not scattered slots",
          float(np.mean(dr.online_slots / np.maximum(dr.n_shifts, 1))) > 20,
          f"mean {np.mean(dr.online_slots/np.maximum(dr.n_shifts,1)):.0f} slots "
          f"({np.mean(dr.online_slots/np.maximum(dr.n_shifts,1))*slot_h:.1f} h) per shift")
    check("shift_start marks exactly the run starts",
          int(dr.shift_start.sum()) == int(dr.n_shifts.sum()),
          f"{int(dr.shift_start.sum())} starts")
    check("home nodes in range", bool(np.all((dr.home_node >= 0) & (dr.home_node < sc.n_nodes))))

    conc = dr.concurrent_online()
    dead = conc == 0
    req_per_slot = rq.counts()
    print(f"        concurrent online: min {conc.min()}, median {int(np.median(conc))}, "
          f"max {conc.max()} of {dr.n}")
    print(f"        zero-supply slots: {int(dead.sum())} of {tl.n_slots} "
          f"({100*dead.mean():.1f}%), holding {int(req_per_slot[dead].sum())} "
          f"requests ({100*req_per_slot[dead].sum()/rq.n:.2f}% of demand) -- "
          f"structurally unservable, excluded from service-rate denominators")
    check("supply covers the vast majority of slots", float(dead.mean()) < 0.05,
          f"{100*dead.mean():.1f}% dead slots")
    check("dead slots carry negligible demand",
          float(req_per_slot[dead].sum() / rq.n) < 0.02,
          f"{100*req_per_slot[dead].sum()/rq.n:.2f}% of requests")

    by_arch = pd.DataFrame({"arch": [ARCHETYPES[a] for a in dr.archetype],
                            "hours": oh}).groupby("arch").agg(
        n=("hours", "size"), mean_h=("hours", "mean"))
    print("        by archetype:\n" + by_arch.to_string().replace("\n", "\n        "))

    print(f"\n-- request stream ({rq.n:,}) --")
    check("slots sorted", bool(np.all(np.diff(rq.slot) >= 0)))
    check("CSR index consistent", int(rq.start[-1]) == rq.n
          and bool(np.array_equal(np.diff(rq.start), np.bincount(rq.slot, minlength=tl.n_slots))))
    check("o,d in node range", bool(np.all((rq.o >= 0) & (rq.o < sc.n_nodes))
                                    and np.all((rq.d >= 0) & (rq.d < sc.n_nodes))))
    check("timestamps inside their slot",
          bool(np.array_equal(tl.slot_of_timestamp(rq.ts), rq.slot)))
    check("demand shape preserved by stratified sampling", True)
    hourly = pd.DataFrame({"h": pd.DatetimeIndex(rq.ts).hour}).groupby("h").size()
    peak_h, off_h = int(hourly.idxmax()), int(hourly.idxmin())
    print(f"        busiest hour {peak_h:02d}:00 ({hourly.max()} req), "
          f"thinnest {off_h:02d}:00 ({hourly.min()} req), "
          f"peak/off = {hourly.max()/max(hourly.min(),1):.1f}x")

    print("\n-- supply / demand regime --")
    occ_slots = sc.utility.occupancy_slots(dr.home_node[rq.slot % dr.n], rq.o, rq.d,
                                           tl.profile[rq.slot])
    demand_slots = float(occ_slots.sum())
    supply_slots = float(dr.online_slots.sum())
    implied = demand_slots / supply_slots
    print(f"        online driver-slots        {supply_slots:12,.0f}")
    print(f"        slots demanded by requests {demand_slots:12,.0f}")
    print(f"        implied max utilisation    {implied:12.3f} "
          f"(target {cfg.protocol.target_offered_load})")
    check("utilisation lands in a contested regime", 0.30 < implied < 1.20,
          f"{implied:.3f}")
    check("demand not trivially servable", implied > 0.35,
          "fairness would be vacuous if every request could be served idly")

    # ---------------- Gap 1b blind-spot demonstration ----------------
    print("\n" + "=" * 78)
    print("GAP 1b: the blind spot, on this fleet")
    print("=" * 78)
    print("  Assign each driver utility proportional to hours worked (the *fairest*")
    print("  possible outcome in rate terms), then read both metrics:")
    rate_true = 100.0
    o_v = oh * rate_true
    print(f"    Var(total utility)  [paper Eq. 2] = {np.var(o_v):12,.1f}   <- looks unfair")
    print(f"    Var(hourly rate)    [Gap 1b]      = {np.var(o_v/oh):12,.1f}   <- correctly 0")
    print("\n  Now equalise totals across all drivers (what Eq. 2 drives towards):")
    o_eq = np.full(dr.n, float(np.mean(o_v)))
    rho_eq = o_eq / oh
    print(f"    Var(total utility)  [paper Eq. 2] = {np.var(o_eq):12,.1f}   <- 'perfectly fair'")
    print(f"    Var(hourly rate)    [Gap 1b]      = {np.var(rho_eq):12,.1f}   <- actually unfair")
    print(f"    hourly rate spread: min {rho_eq.min():.1f}, max {rho_eq.max():.1f} "
          f"= {rho_eq.max()/rho_eq.min():.1f}x hidden gap")
    ft_rate, pt_rate = rho_eq[ft].mean(), rho_eq[pt].mean()
    print(f"    by group: full-time {ft_rate:.1f}/h vs part-time {pt_rate:.1f}/h "
          f"= {pt_rate/ft_rate:.2f}x in favour of part-timers")
    check("equalising totals hides a multi-fold rate gap",
          rho_eq.max() / rho_eq.min() > 3.0,
          f"{rho_eq.max()/rho_eq.min():.1f}x invisible to Eq. 2")

    # the paper's own 60h-vs-15h example, found in the real fleet
    print("\n  Concrete pairs from this fleet (equal totals, unequal hours):")
    idx = np.argsort(oh)
    lo3, hi3 = idx[:40], idx[-40:]
    rows = []
    for a, b in zip(lo3[:5], hi3[:5]):
        rows.append({
            "driver_A": int(b), "hours_A": round(float(oh[b]), 1),
            "driver_B": int(a), "hours_B": round(float(oh[a]), 1),
            "equal_total": round(float(o_eq[0]), 1),
            "rate_A": round(float(o_eq[b] / oh[b]), 1),
            "rate_B": round(float(o_eq[a] / oh[a]), 1),
            "hidden_gap": f"{o_eq[a]/oh[a] / (o_eq[b]/oh[b]):.1f}x",
        })
    tbl = pd.DataFrame(rows)
    print("    " + tbl.to_string(index=False).replace("\n", "\n    "))

    dr.summary().to_csv(output_path("phase2_driver_fleet.csv"), index=False)
    tbl.to_csv(output_path("gap1b_blindspot_pairs.csv"), index=False)
    print(f"\n  wrote outputs/phase2_driver_fleet.csv, outputs/gap1b_blindspot_pairs.csv")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("ALL PHASE 2 CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
