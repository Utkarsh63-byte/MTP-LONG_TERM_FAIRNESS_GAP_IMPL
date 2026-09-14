"""Full dataset audit: per-filter attrition, artefact statistics, usage map.

Measures things not previously quantified, in particular how many rows each
individual filter is responsible for removing, and the anomalies present in the
raw file. Results are cached to outputs/dataset_audit.json so the DOCX builder
does not need to re-scan the 1.9 GB CSV.

Run:  .venv/bin/python -m scripts.audit_dataset
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ltf.config import RAW_CSV, Config
from ltf.data import graph as graph_mod
from ltf.data import traveltime as tt_mod
from ltf.data import demand as demand_mod
from ltf.data.prepare import CSV_COLS, load_nodes, load_trips
from ltf.utils import LOG, output_path, timed

CHUNK = 1_500_000


def main() -> int:
    cfg = Config()
    d = cfg.data
    A: dict = {}

    # ---------------- raw file structure ----------------
    head = pd.read_csv(RAW_CSV, nrows=5)
    A["raw_columns"] = list(head.columns)
    A["raw_n_columns"] = len(head.columns)
    A["used_columns"] = CSV_COLS
    A["ignored_columns"] = [c for c in head.columns if c not in CSV_COLS]

    # ---------------- one pass: per-filter attrition ----------------
    keys = ["total", "pickup_bbox", "dropoff_bbox", "trip_distance", "fare",
            "date_window", "duration", "speed"]
    excl = {k: 0 for k in keys}          # rows failing ONLY this test
    fail_any = 0
    kept = 0
    anomalies = {"dist_gt_1000mi": 0, "dist_zero": 0, "fare_negative": 0,
                 "dropoff_before_pickup": 0, "dropoff_next_month": 0,
                 "outside_march": 0, "zero_coords": 0}
    per_date = {}
    hour_speed_sum = np.zeros(24)
    hour_speed_n = np.zeros(24)
    hour_trips = np.zeros(24, dtype=np.int64)
    dur_sum = np.zeros(24)
    fare_total = 0.0
    km_total = 0.0

    reader = pd.read_csv(RAW_CSV, usecols=CSV_COLS, chunksize=CHUNK,
                         parse_dates=["tpep_pickup_datetime",
                                      "tpep_dropoff_datetime"])
    with timed("single pass over the raw CSV (per-filter attrition)"):
        for ci, ch in enumerate(reader):
            n = len(ch)
            excl["total"] += n

            dur = (ch.tpep_dropoff_datetime
                   - ch.tpep_pickup_datetime).dt.total_seconds() / 60.0
            with np.errstate(divide="ignore", invalid="ignore"):
                spd = ch.trip_distance.to_numpy() / (dur.to_numpy() / 60.0)

            tests = {
                "pickup_bbox": (ch.pickup_longitude.between(d.lon_min, d.lon_max)
                                & ch.pickup_latitude.between(d.lat_min, d.lat_max)),
                "dropoff_bbox": (ch.dropoff_longitude.between(d.lon_min, d.lon_max)
                                 & ch.dropoff_latitude.between(d.lat_min, d.lat_max)),
                "trip_distance": ch.trip_distance.between(d.min_trip_miles,
                                                          d.max_trip_miles),
                "fare": ch.fare_amount.between(d.min_fare, d.max_fare),
                "date_window": ((ch.tpep_pickup_datetime >= "2016-03-01")
                                & (ch.tpep_pickup_datetime < "2016-04-01")),
                "duration": pd.Series((dur > d.min_duration_min)
                                      & (dur < d.max_duration_min),
                                      index=ch.index),
                "speed": pd.Series(np.nan_to_num(spd) > d.min_speed_mph,
                                   index=ch.index)
                         & pd.Series(np.nan_to_num(spd) < d.max_speed_mph,
                                     index=ch.index),
            }
            passes = pd.DataFrame(tests)
            n_fail = (~passes).sum(axis=1)
            for kk in tests:
                # rows removed solely because of this test
                excl[kk] += int(((~passes[kk]) & (n_fail == 1)).sum())
            good = passes.all(axis=1)
            fail_any += int((~good).sum())
            kept += int(good.sum())

            # anomaly counters on the raw chunk
            anomalies["dist_gt_1000mi"] += int((ch.trip_distance > 1000).sum())
            anomalies["dist_zero"] += int((ch.trip_distance <= 0).sum())
            anomalies["fare_negative"] += int((ch.fare_amount < 0).sum())
            anomalies["dropoff_before_pickup"] += int((dur <= 0).sum())
            anomalies["dropoff_next_month"] += int(
                (ch.tpep_dropoff_datetime >= "2016-04-01").sum())
            anomalies["outside_march"] += int(
                ((ch.tpep_pickup_datetime < "2016-03-01")
                 | (ch.tpep_pickup_datetime >= "2016-04-01")).sum())
            anomalies["zero_coords"] += int(
                ((ch.pickup_longitude == 0) | (ch.pickup_latitude == 0)).sum())

            g = ch[good]
            if len(g):
                gd = dur[good].to_numpy()
                gs = spd[good]
                hod = g.tpep_pickup_datetime.dt.hour.to_numpy()
                np.add.at(hour_trips, hod, 1)
                np.add.at(dur_sum, hod, gd)
                ok = np.isfinite(gs)
                np.add.at(hour_speed_sum, hod[ok], gs[ok])
                np.add.at(hour_speed_n, hod[ok], 1)
                fare_total += float(g.fare_amount.sum() + g.tip_amount.sum())
                km_total += float(g.trip_distance.sum() * 1.60934)
                vc = g.tpep_pickup_datetime.dt.strftime("%Y-%m-%d").value_counts()
                for k2, v2 in vc.items():
                    per_date[k2] = per_date.get(k2, 0) + int(v2)
            LOG.info("  chunk %d: scanned %d, kept %d", ci, excl["total"], kept)

    A["raw_rows"] = excl.pop("total")
    A["kept_rows"] = kept
    A["removed_rows"] = fail_any
    A["kept_pct"] = 100.0 * kept / A["raw_rows"]
    A["sole_cause_removals"] = excl
    A["multi_cause_removals"] = fail_any - sum(excl.values())
    A["anomalies"] = anomalies
    A["per_date"] = dict(sorted(per_date.items()))
    A["fare_plus_tip_total"] = fare_total
    A["km_total"] = km_total
    A["hour_speed_mph"] = (hour_speed_sum / np.maximum(hour_speed_n, 1)).round(2).tolist()
    A["hour_trips"] = hour_trips.tolist()
    A["hour_mean_dur_min"] = (dur_sum / np.maximum(hour_trips, 1)).round(2).tolist()

    # ---------------- artefacts ----------------
    nodes = load_nodes()
    g = graph_mod.load_or_build(cfg)
    tt = tt_mod.load_or_build(cfg)
    dem = demand_mod.load_or_build(cfg, n_nodes=nodes.n_nodes)
    trips = load_trips(columns=["o", "d", "km", "dur_min", "fare", "date", "hour"])

    A["n_nodes"] = int(nodes.n_nodes)
    A["node_trips_min"] = int(nodes.trips.min())
    A["node_trips_max"] = int(nodes.trips.max())
    A["node_trips_median"] = int(np.median(nodes.trips))
    A["grid_deg"] = d.grid_deg
    A["grid_km"] = round(d.grid_deg * 111.0, 2)
    A["min_cell_trips"] = d.min_cell_trips

    K = nodes.n_nodes
    off = ~np.eye(K, dtype=bool)
    A["od_pairs_total"] = int(K * K)
    A["od_observed_pct"] = float(100 * (g.source == 0).mean())
    A["od_trip_coverage_pct"] = float(
        100 * g.n_obs[g.source == 0].sum() / g.n_obs.sum())
    A["dist_median_km"] = float(np.median(g.dist[off]))
    A["dist_max_km"] = float(g.dist[off].max())
    A["asym_max_km"] = float(np.abs(g.dist - g.dist.T).max())
    A["asym_median_km"] = float(np.median(np.abs(g.dist - g.dist.T)[off]))
    A["intra_median_km"] = float(np.median(g.intra))

    A["tt_shape"] = list(tt.tau.shape)
    A["tt_L0_cells_pct"] = float(100 * (tt.level == 0).mean())
    A["tt_L1_cells_pct"] = float(100 * (tt.level == 1).mean())
    A["tt_L2_cells_pct"] = float(100 * (tt.level == 2).mean())
    A["tt_L0_trip_pct"] = float(
        100 * tt.n_obs[tt.level == 0].sum() / max(tt.n_obs.sum(), 1))
    A["city_speed_max"] = float(tt.city_speed.max())
    A["city_speed_min"] = float(tt.city_speed.min())
    A["city_speed_ratio"] = float(tt.city_speed.max() / tt.city_speed.min())
    A["c_mean_L0"] = float(tt.c[tt.level == 0].mean())
    A["c_clip_pct"] = float(100 * (((tt.c <= cfg.utility.c_min + 1e-9)
                                    | (tt.c >= cfg.utility.c_max - 1e-9)).mean()))
    rep = tt_mod.report(tt, g)
    A["od_pairs_wellmeasured"] = int(len(rep))
    A["ratio_median"] = float(rep.ratio.median())
    A["ratio_p90"] = float(rep.ratio.quantile(0.9))
    A["ratio_max"] = float(rep.ratio.max())

    A["demand_shape"] = list(dem.od.shape)
    A["demand_total"] = int(dem.od.sum())
    A["demand_reconciles"] = bool(int(dem.od.sum()) == len(trips))
    A["demand_nonzero_pct"] = float(100 * (dem.od > 0).mean())
    A["busiest_node_hour"] = int(dem.org.max())

    A["parquet_rows"] = int(len(trips))
    A["parquet_mean_trip_km"] = float(trips.km.mean())
    A["parquet_mean_dur_min"] = float(trips.dur_min.mean())
    A["fare_per_km_median"] = float(np.median(
        (trips.fare / np.maximum(trips.km, 0.1)).clip(0, 50)))

    with open(output_path("dataset_audit.json"), "w") as f:
        json.dump(A, f, indent=2)

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print("DATASET AUDIT")
    print("=" * 78)
    print(f"raw rows                {A['raw_rows']:>12,}")
    print(f"kept                    {A['kept_rows']:>12,}  ({A['kept_pct']:.2f}%)")
    print(f"removed                 {A['removed_rows']:>12,}")
    print("\nrows removed by exactly ONE filter (sole cause):")
    for kk, vv in sorted(A["sole_cause_removals"].items(), key=lambda x: -x[1]):
        print(f"  {kk:<16} {vv:>12,}  ({100*vv/A['raw_rows']:.3f}%)")
    print(f"  {'multiple causes':<16} {A['multi_cause_removals']:>12,}")
    print("\nraw-file anomalies:")
    for kk, vv in A["anomalies"].items():
        print(f"  {kk:<24} {vv:>12,}")
    print(f"\nnodes {A['n_nodes']}, OD observed {A['od_observed_pct']:.1f}% "
          f"({A['od_trip_coverage_pct']:.2f}% of trips)")
    print(f"travel-time levels L0/L1/L2 = {A['tt_L0_cells_pct']:.1f}/"
          f"{A['tt_L1_cells_pct']:.1f}/{A['tt_L2_cells_pct']:.1f}% of cells; "
          f"L0 covers {A['tt_L0_trip_pct']:.1f}% of trips")
    print(f"demand tensor reconciles with parquet: {A['demand_reconciles']}")
    print(f"\nwrote outputs/dataset_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
