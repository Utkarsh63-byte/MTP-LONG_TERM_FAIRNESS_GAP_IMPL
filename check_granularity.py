"""Decide node granularity: can we estimate a per-(OD, time-slot) travel-time
table densely enough? Tests several grid sizes on a 3M-row sample."""
import numpy as np
import pandas as pd

CSV = "yellow_tripdata_2016-03.csv"
COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "fare_amount"]
LON_MIN, LON_MAX = -74.02, -73.93
LAT_MIN, LAT_MAX = 40.70, 40.88

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
print(f"sample trips used: {len(df):,}  "
      f"(scales to ~{len(df)*10_303_643/2_550_000:,.0f} for the full month)\n")
SCALE = 10_303_643 / len(df)

hod = df.tpep_pickup_datetime.dt.hour.to_numpy()
wk = (df.tpep_pickup_datetime.dt.dayofweek >= 5).astype(int).to_numpy()

for deg, label in [(0.0045, "~0.5 km"), (0.009, "~1.0 km"),
                   (0.0135, "~1.5 km"), (0.018, "~2.0 km")]:
    oi = (np.floor((df.pickup_latitude.to_numpy() - LAT_MIN) / deg).astype(int) * 100
          + np.floor((df.pickup_longitude.to_numpy() - LON_MIN) / deg).astype(int))
    di = (np.floor((df.dropoff_latitude.to_numpy() - LAT_MIN) / deg).astype(int) * 100
          + np.floor((df.dropoff_longitude.to_numpy() - LON_MIN) / deg).astype(int))
    nodes = len(np.union1d(np.unique(oi), np.unique(di)))
    s = pd.DataFrame({"o": oi, "d": di, "h": hod, "w": wk})
    od = s.groupby(["o", "d"]).size()
    od_big = (od * SCALE >= 50).sum()
    b24 = s.groupby(["o", "d", "h"]).size()
    b_ok = (b24 * SCALE >= 30).sum()
    b2 = s.groupby(["o", "d", "w", "h"]).size()
    b2_ok = (b2 * SCALE >= 30).sum()
    print(f"grid {label:8s} nodes={nodes:4d}  OD pairs seen={len(od):7,}  "
          f"OD w/ >=50 trips/month={od_big:6,} ({100*od_big/len(od):4.1f}%)")
    print(f"{'':14s} (OD,hour) buckets={len(b24):8,}  with >=30 trips={b_ok:7,} "
          f"({100*b_ok/len(b24):4.1f}%)")
    print(f"{'':14s} (OD,weekend,hour) buckets={len(b2):8,}  with >=30 trips="
          f"{b2_ok:7,} ({100*b2_ok/len(b2):4.1f}%)\n")
