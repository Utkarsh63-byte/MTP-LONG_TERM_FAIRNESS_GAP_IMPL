"""Pick the grid size that yields ~90 *usable* nodes.

A cell is only usable as a graph node if it carries enough trips to estimate
a per-(OD, time-slot) travel time. Sweep grid size against a volume threshold
and report both the node count and the resulting travel-time density.
"""
import numpy as np
import pandas as pd

CSV = "yellow_tripdata_2016-03.csv"
COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
        "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "fare_amount"]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.02, -73.93, 40.70, 40.88

df = pd.read_csv(CSV, usecols=COLS, nrows=3_000_000,
                 parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
m = (df.pickup_longitude.between(LON_MIN, LON_MAX)
     & df.pickup_latitude.between(LAT_MIN, LAT_MAX)
     & df.dropoff_longitude.between(LON_MIN, LON_MAX)
     & df.dropoff_latitude.between(LAT_MIN, LAT_MAX)
     & df.trip_distance.between(0.1, 40) & df.fare_amount.between(2.5, 300))
df = df[m]
dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
df = df[(dur > 1) & (dur < 120)]
SCALE = 10_300_738 / len(df)
print(f"sample {len(df):,} trips, scale x{SCALE:.2f} to full month\n")

hod = df.tpep_pickup_datetime.dt.hour.to_numpy()
wknd = (df.tpep_pickup_datetime.dt.dayofweek >= 5).astype(int).to_numpy()
MIN_CELL = 2000     # endpoints per cell over the month
MIN_BUCKET = 30     # trips per (OD, daytype, hour) bucket

for deg in [0.0075, 0.009, 0.0105, 0.0115, 0.0135, 0.018]:
    nr = int(np.ceil((LAT_MAX - LAT_MIN) / deg)) + 2
    nc = int(np.ceil((LON_MAX - LON_MIN) / deg)) + 2
    pr = np.floor((df.pickup_latitude.to_numpy() - LAT_MIN) / deg).astype(int)
    pc = np.floor((df.pickup_longitude.to_numpy() - LON_MIN) / deg).astype(int)
    dr = np.floor((df.dropoff_latitude.to_numpy() - LAT_MIN) / deg).astype(int)
    dc = np.floor((df.dropoff_longitude.to_numpy() - LON_MIN) / deg).astype(int)
    pf, dfl = pr * nc + pc, dr * nc + dc
    cnt = np.bincount(np.concatenate([pf, dfl]), minlength=nr * nc) * SCALE
    usable = cnt >= MIN_CELL
    cov = cnt[usable].sum() / cnt.sum()

    keep = np.where(usable)[0]
    lut = -np.ones(nr * nc, int)
    lut[keep] = np.arange(len(keep))
    om, dm = lut[pf], lut[dfl]
    ok = (om >= 0) & (dm >= 0)
    s = pd.DataFrame({"o": om[ok], "d": dm[ok], "w": wknd[ok], "h": hod[ok]})
    b = s.groupby(["o", "d", "w", "h"]).size() * SCALE
    trip_w = b[b >= MIN_BUCKET].sum() / b.sum()   # trip-weighted, not bucket-weighted
    print(f"grid {deg:.4f} deg (~{deg*111:.2f} km): cells={int((cnt>0).sum()):4d} "
          f"usable_nodes={usable.sum():4d}  endpoint_coverage={100*cov:6.2f}%")
    print(f"{'':28s} (OD,daytype,hour) buckets={len(b):8,}  "
          f"n>=30: {100*(b>=MIN_BUCKET).mean():5.1f}% of buckets, "
          f"{100*trip_w:5.1f}% of trips")
