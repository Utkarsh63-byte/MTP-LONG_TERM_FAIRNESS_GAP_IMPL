"""One-pass profile of yellow_tripdata_2016-03.csv for the MTP gap implementation.

Checks:
  1. How much data survives Manhattan + sanity filtering.
  2. Whether travel TIME varies by hour-of-day for the SAME OD pair
     (this is the empirical justification for Gap 1: time-aware utility).
  3. Trip volume per date / per hour, to size the simulation.
  4. Node granularity options (grid cell counts).
"""
import numpy as np
import pandas as pd
from collections import Counter

CSV = "yellow_tripdata_2016-03.csv"
COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "fare_amount", "tip_amount", "total_amount"]

# Manhattan bounding box (approx, excludes most of Bronx/Queens/Brooklyn)
LON_MIN, LON_MAX = -74.02, -73.93
LAT_MIN, LAT_MAX = 40.70, 40.88

GRID = 0.0045  # ~ 500 m in latitude

raw_rows = 0
kept_rows = 0
date_counts = Counter()
hour_counts = Counter()
# speed accumulators per hour-of-day
spd_sum = np.zeros(24)
spd_n = np.zeros(24)
dur_sum = np.zeros(24)
# per-OD-per-hour travel time for a fixed set of busy OD pairs
od_hour_time = {}      # (o,d,hour) -> [sum, n]
cell_counter = Counter()
fare_sum = 0.0
dist_sum = 0.0

reader = pd.read_csv(CSV, usecols=COLS, chunksize=1_500_000,
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

for ci, ch in enumerate(reader):
    raw_rows += len(ch)
    m = (
        ch.pickup_longitude.between(LON_MIN, LON_MAX)
        & ch.pickup_latitude.between(LAT_MIN, LAT_MAX)
        & ch.dropoff_longitude.between(LON_MIN, LON_MAX)
        & ch.dropoff_latitude.between(LAT_MIN, LAT_MAX)
        & ch.trip_distance.between(0.1, 40)
        & ch.fare_amount.between(2.5, 300)
        & (ch.tpep_pickup_datetime >= "2016-03-01")
        & (ch.tpep_pickup_datetime < "2016-04-01")
    )
    ch = ch[m].copy()
    dur = (ch.tpep_dropoff_datetime - ch.tpep_pickup_datetime).dt.total_seconds() / 60.0
    ch = ch[(dur > 1) & (dur < 120)]
    dur = dur[(dur > 1) & (dur < 120)]
    kept_rows += len(ch)
    if len(ch) == 0:
        continue

    hod = ch.tpep_pickup_datetime.dt.hour.to_numpy()
    speed = (ch.trip_distance.to_numpy() / (dur.to_numpy() / 60.0))  # mph
    ok = (speed > 1) & (speed < 60)

    np.add.at(spd_sum, hod[ok], speed[ok])
    np.add.at(spd_n, hod[ok], 1)
    np.add.at(dur_sum, hod, dur.to_numpy())

    date_counts.update(ch.tpep_pickup_datetime.dt.strftime("%Y-%m-%d").tolist())
    hour_counts.update(hod.tolist())
    fare_sum += ch.fare_amount.sum() + ch.tip_amount.sum()
    dist_sum += ch.trip_distance.sum()

    oi = np.floor((ch.pickup_latitude.to_numpy() - LAT_MIN) / GRID).astype(int) * 1000 \
        + np.floor((ch.pickup_longitude.to_numpy() - LON_MIN) / GRID).astype(int)
    di = np.floor((ch.dropoff_latitude.to_numpy() - LAT_MIN) / GRID).astype(int) * 1000 \
        + np.floor((ch.dropoff_longitude.to_numpy() - LON_MIN) / GRID).astype(int)
    cell_counter.update(oi.tolist())

    sub = pd.DataFrame({"o": oi, "d": di, "h": hod, "t": dur.to_numpy(),
                        "km": ch.trip_distance.to_numpy() * 1.60934})
    g = sub.groupby(["o", "d", "h"]).agg(t=("t", "sum"), n=("t", "size"),
                                         km=("km", "sum"))
    for key, row in g.iterrows():
        a = od_hour_time.setdefault(key, [0.0, 0, 0.0])
        a[0] += row.t
        a[1] += row.n
        a[2] += row.km
    print(f"chunk {ci}: raw={raw_rows:,} kept={kept_rows:,}", flush=True)

print("\n===== SUMMARY =====")
print(f"raw rows          : {raw_rows:,}")
print(f"kept (Manhattan)  : {kept_rows:,}  ({100*kept_rows/raw_rows:.1f}%)")
print(f"distinct ~500m pickup cells: {len(cell_counter)}")
print(f"cells covering 90% of pickups: "
      f"{int(np.searchsorted(np.cumsum(sorted(cell_counter.values(), reverse=True)) / kept_rows, 0.90) + 1)}")
print(f"total fare+tip    : ${fare_sum:,.0f}")
print(f"total miles       : {dist_sum:,.0f}")

print("\n--- mean speed (mph) and mean duration (min) by hour of day ---")
for h in range(24):
    print(f"  h{h:02d}  speed={spd_sum[h]/max(spd_n[h],1):6.2f} mph   "
          f"trips={hour_counts[h]:>9,}   mean_dur={dur_sum[h]/max(hour_counts[h],1):5.1f} min")

print("\n--- trips per date ---")
for d in sorted(date_counts):
    print(f"  {d}  {date_counts[d]:,}")

# how much does travel time for the SAME OD pair vary across hours?
print("\n--- same-OD travel-time variation across hour-of-day ---")
df = pd.DataFrame(
    [(k[0], k[1], k[2], v[0] / v[1], v[1], v[2] / v[1]) for k, v in od_hour_time.items()],
    columns=["o", "d", "h", "mean_min", "n", "mean_km"])
df = df[df.n >= 30]
piv = df.groupby(["o", "d"]).filter(lambda g: len(g) >= 20)
stats = piv.groupby(["o", "d"]).agg(
    lo=("mean_min", "min"), hi=("mean_min", "max"),
    avg=("mean_min", "mean"), km=("mean_km", "mean"), hours=("h", "size"))
stats["peak_ratio"] = stats.hi / stats.lo
print(f"OD pairs with >=20 hourly buckets (n>=30 each): {len(stats)}")
print(stats.peak_ratio.describe().to_string())
print("\ntop 15 OD pairs by peak/offpeak travel-time ratio:")
print(stats.sort_values("peak_ratio", ascending=False).head(15).to_string())
df.to_csv("od_hour_traveltime.csv", index=False)
print("\nwrote od_hour_traveltime.csv  rows:", len(df))
