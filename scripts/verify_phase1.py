"""Phase 1 verification: assert the data pipeline's invariants, then report the
measured evidence for Gap 1a.

Run:  .venv/bin/python -m scripts.verify_phase1
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ltf.config import Config
from ltf.data import graph as graph_mod
from ltf.data import demand as demand_mod
from ltf.data import traveltime as tt_mod
from ltf.data.prepare import load_nodes, load_trips
from ltf.utility import UtilityModel, calibrate_fare_per_km
from ltf.utils import LOG, output_path

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    cfg = Config()
    print("=" * 78)
    print("PHASE 1 VERIFICATION")
    print("=" * 78)

    nodes = load_nodes()
    trips = load_trips()
    g = graph_mod.load_or_build(cfg)
    tt = tt_mod.load_or_build(cfg)
    dem = demand_mod.load_or_build(cfg, n_nodes=nodes.n_nodes)
    K = nodes.n_nodes

    print(f"\n-- nodes ({K}) --")
    check("node ids are 0..K-1", list(nodes.node_id) == list(range(K)))
    check("centroids inside bbox",
          bool(np.all((nodes.lat > cfg.data.lat_min) & (nodes.lat < cfg.data.lat_max)
                      & (nodes.lon > cfg.data.lon_min) & (nodes.lon < cfg.data.lon_max))))
    check("every node carries volume", bool(np.all(nodes.trips >= cfg.data.min_cell_trips)),
          f"min {nodes.trips.min():,}")

    print(f"\n-- trips ({len(trips):,}) --")
    check("no nulls", not bool(trips.isna().any().any()))
    check("o,d within range", bool(trips.o.between(0, K - 1).all()
                                   and trips.d.between(0, K - 1).all()))
    check("all in March 2016",
          bool((trips.pickup_ts >= "2016-03-01").all() and (trips.pickup_ts < "2016-04-01").all()))
    check("dropoff after pickup", bool((trips.dropoff_ts > trips.pickup_ts).all()))
    check("duration within filter",
          bool(trips.dur_min.between(cfg.data.min_duration_min,
                                     cfg.data.max_duration_min).all()))
    check("sorted by pickup", bool(trips.pickup_ts.is_monotonic_increasing))
    print(f"        date span {trips.date.min()} .. {trips.date.max()}, "
          f"{trips.date.nunique()} days")

    print("\n-- distance matrix --")
    check("shape (K,K)", g.dist.shape == (K, K))
    check("all finite", bool(np.all(np.isfinite(g.dist))))
    check("strictly positive", bool(np.all(g.dist > 0)))
    check("diagonal = intra-node distance", bool(np.allclose(np.diag(g.dist), g.intra)))
    asym = np.abs(g.dist - g.dist.T)
    check("digraph really is asymmetric", float(asym.max()) > 0.5,
          f"max |d(a,b)-d(b,a)| = {asym.max():.2f} km, "
          f"median {np.median(asym[~np.eye(K, dtype=bool)]):.3f} km")
    check("observed pairs dominate", float((g.source == 0).mean()) > 0.75,
          f"{100*(g.source==0).mean():.1f}% observed")

    print("\n-- travel time --")
    P = tt.n_profiles
    check("shape (K,K,P)", tt.tau.shape == (K, K, P), f"P={P}")
    check("tau finite and positive", bool(np.all(np.isfinite(tt.tau)) and np.all(tt.tau > 0)))
    check("tau_bar finite and positive",
          bool(np.all(np.isfinite(tt.tau_bar)) and np.all(tt.tau_bar > 0)))
    check("c within configured clip",
          bool(np.all(tt.c >= cfg.utility.c_min - 1e-6)
               and np.all(tt.c <= cfg.utility.c_max + 1e-6)))
    check("c centred near 1 on measured cells",
          0.9 < float(tt.c[tt.level == 0].mean()) < 1.1,
          f"mean {tt.c[tt.level==0].mean():.3f}")
    check("implied speeds are plausible",
          bool(np.all(tt.city_speed > 5) and np.all(tt.city_speed < 60)),
          f"{tt.city_speed.min():.1f}-{tt.city_speed.max():.1f} km/h")

    print("\n-- demand --")
    check("od total == trip count", int(dem.od.sum()) == len(trips),
          f"{dem.od.sum():,} vs {len(trips):,}")
    check("org == od collapsed over d", bool(np.array_equal(dem.org, dem.od.sum(axis=1))))
    check("744 hours in March", dem.n_hours == 744, f"{dem.n_hours}")

    print("\n-- utility: does time_aware reduce to the paper at c=1? --")
    fpk = calibrate_fare_per_km(trips.sample(500_000, random_state=0))
    um = UtilityModel(cfg, g, tt, fare_per_km=fpk)
    res = um.assert_reduction()
    check("time_aware == paper when c==1", bool(res["reduction_holds"]),
          f"max |dU| = {res['max_abs_utility_diff']:.2e}, "
          f"max slot diff = {res['max_abs_slot_diff']}")
    check("real congestion actually changes utility",
          res["mean_abs_diff_vs_real_congestion"] > 0.1,
          f"mean |dU| = {res['mean_abs_diff_vs_real_congestion']:.3f} km-equivalent")
    print(f"        calibrated fare per km = ${fpk:.2f}")

    # ---------------- Gap 1a evidence ----------------
    print("\n" + "=" * 78)
    print("GAP 1a EVIDENCE (measured on the paper's own dataset and granularity)")
    print("=" * 78)
    fastest, slowest = int(np.argmax(tt.city_speed)), int(np.argmin(tt.city_speed))
    lbl = lambda p: f"{'weekend' if p >= 24 else 'weekday'} {p % 24:02d}:00"
    print(f"  citywide speed: {tt.city_speed[fastest]:.1f} km/h ({lbl(fastest)}) "
          f"vs {tt.city_speed[slowest]:.1f} km/h ({lbl(slowest)}) "
          f"= {tt.city_speed[fastest]/tt.city_speed[slowest]:.2f}x")
    rep = tt_mod.report(tt, g)
    rep.to_csv(output_path("gap1a_od_congestion_spread.csv"), index=False)
    print(f"  OD pairs measured across >=24 of 48 profiles: {len(rep)}")
    print(f"  peak/off-peak travel-time ratio: median {rep.ratio.median():.2f}x, "
          f"p90 {rep.ratio.quantile(0.9):.2f}x, max {rep.ratio.max():.2f}x")

    # what the paper's flattening costs, in its own utility units
    rng = np.random.default_rng(0)
    n = 300_000
    gg, ss, dd = (rng.integers(0, K, n) for _ in range(3))
    pp = rng.integers(0, P, n)
    import copy
    cfg_p = copy.deepcopy(cfg); cfg_p.utility.mode = "paper"
    u_paper = UtilityModel(cfg_p, g, tt, fpk).utility(gg, ss, dd, pp)
    u_time = um.utility(gg, ss, dd, pp)
    solid = tt.level[gg, ss, pp] == 0
    diff = (u_time - u_paper)[solid]
    print(f"  on {solid.sum():,} well-measured random (driver, request, time) triples:")
    print(f"    paper utility     mean {u_paper[solid].mean():7.3f} km")
    print(f"    time-aware utility mean {u_time[solid].mean():7.3f} km")
    print(f"    mean signed shift {diff.mean():+.3f} km, "
          f"mean |shift| {np.abs(diff).mean():.3f} km, "
          f"p95 |shift| {np.percentile(np.abs(diff),95):.3f} km")
    same = (np.sign(u_paper[solid]) != np.sign(u_time[solid])).mean()
    print(f"    sign of utility flips on {100*same:.1f}% of them "
          f"(profitable <-> unprofitable)")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("ALL PHASE 1 CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
