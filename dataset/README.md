# Dataset for MTP Gap Implementation

Complete dataset package for "Equal Earnings Are Not Equal Pay: Hours-Aware and Access-Aware Fairness for Driver Allocation in Ride-Hailing"

## Directory Structure

```
dataset/
├── raw/                          Original unprocessed data
│   ├── yellow_tripdata_2016-03.csv    NYC TLC Yellow Taxi (12,210,952 rows, 1.9 GB)
│   └── od_hour_traveltime.csv         (if present) hourly OD travel times
│
├── processed/                    Derived artifacts from the pipeline
│   ├── nodes.csv                 87 graph nodes with centroids
│   ├── trips.parquet             Cleaned trips (10,300,738 rows, 132 MB)
│   ├── distance.npz              87×87 Geo distance matrix
│   ├── traveltime.npz            87×87×48 travel-time tensor (full month)
│   ├── traveltime_train.npz      Travel-time tensor (pre-test-week, for forecaster)
│   ├── traveltime_test.npz       Travel-time tensor (test week only, held out)
│   └── demand.npz                87×87×744 demand tensor
│
├── documentation/                Complete documentation
│   ├── MTP_Dataset_Report.docx   15-section dataset documentation
│   ├── dataset_audit.json        Per-filter attrition statistics
│   └── data_pipeline.png         (if generated) pipeline diagram
│
└── README.md                     This file
```

## Raw Data

### yellow_tripdata_2016-03.csv
- **Source:** NYC Taxi and Limousine Commission
- **URL:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Direct link:** https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/k67s-dv2t
- **Size:** 1.9 GB
- **Rows:** 12,210,952 trip records
- **Columns:** 19 (we use 9, ignore 10)
- **Period:** March 2016 (1-31 March 2016)
- **Format:** CSV with header
- **⚠️  NOT IN GIT:** Too large for GitHub (1.9 GB exceeds 100 MB limit). Download from TLC link above.

**Columns used:**
- `tpep_pickup_datetime`, `tpep_dropoff_datetime` — timestamps
- `pickup_longitude`, `pickup_latitude` — origin coordinates
- `dropoff_longitude`, `dropoff_latitude` — destination coordinates
- `trip_distance` — driven distance in miles
- `fare_amount`, `tip_amount` — payment components
- `total_amount` — (loaded for profiling, not used downstream)

**Columns ignored:**
- `VendorID`, `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `payment_type`, `extra`, `mta_tax`, `tolls_amount`, `improvement_surcharge`

**Known anomalies in raw file:**
- 71,126 rows with zero distance
- 182,001 rows with (0,0) coordinates
- 21 rows with distance > 1,000 miles (max in file: 5,000,000 mi)
- 4,581 rows with negative fare
- 12,550 rows with dropoff at or before pickup
- 4,912 rows with dropoff timestamp in April or later

See `documentation/dataset_audit.json` for complete per-filter statistics.

## Processed Data

All files in `processed/` are **derived artifacts**. They can be regenerated from the raw CSV by running:

```bash
python -m ltf.data.prepare      # builds nodes.csv and trips.parquet
python -m ltf.data.graph        # builds distance.npz
python -m ltf.data.traveltime   # builds traveltime*.npz
python -m ltf.data.demand       # builds demand.npz
```

Total rebuild time: approximately 2 minutes on the reference machine.

### nodes.csv
- **Rows:** 87
- **Columns:** `node_id`, `lat`, `lon`, `trips`, `cell_i`, `cell_j`
- **Purpose:** Graph node definitions with demand-weighted centroids
- **Construction:**
  - 0.01° grid (≈1.11 km) overlaid on Manhattan bounding box
  - Keep cells with ≥2,000 trip endpoints
  - Centroid = empirical mean of endpoints in cell (not geometric centre)
- **Coverage:** 99.94% of trip endpoints fall in a kept cell
- **Trips per node:** min 2,602, max 1,390,863, median 53,413

### trips.parquet
- **Rows:** 10,300,738 (84.36% of raw file retained)
- **Size:** 132 MB
- **Columns:** `o`, `d`, `km`, `dur_min`, `fare`, `date`, `hour`, `daytype`, `profile`
- **Purpose:** Cleaned trip records snapped to graph nodes
- **⚠️  NOT IN GIT:** Too large for GitHub (132 MB exceeds 100 MB limit). Rebuild from raw CSV in 30 seconds (see below).
- **Filters applied:**
  1. Pickup inside Manhattan bbox: lon [-74.025, -73.9], lat [40.685, 40.88]
  2. Dropoff inside same bbox
  3. Trip distance [0.1, 40] miles
  4. Fare [$2.50, $500]
  5. Duration [1, 180] minutes
  6. Implied speed [1, 80] mph
  7. Pickup in March 2016

**Per-filter attrition (rows removed by exactly one filter):**
- Dropoff outside Manhattan: 811,053 (6.64%)
- Pickup outside Manhattan: 501,599 (4.11%)
- Duration out of range: 17,954 (0.15%)
- Speed out of range: 2,905 (0.02%)
- Distance out of range: 2,732 (0.02%)
- Fare out of range: 2,550 (0.02%)
- Multiple causes: 571,421 (4.68%)

**Key finding:** Filtering is almost entirely geographic (10.75% of rows), not quality-driven (0.21%).

### distance.npz
- **Shape:** 87 × 87
- **Purpose:** Geo(a,b) distance matrix for the utility function
- **Units:** kilometres
- **Estimation:**
  - Primary: median observed `trip_distance` per OD pair (80.8% of pairs observed)
  - Fallback: rotated L1 distance for unobserved pairs (scale fitted by weighted least squares)
  - Diagonal: median intra-node trip distance (median 0.97 km)
- **Validation:**
  - Circuity (road/straight): 1.320, matches independently reported NYC value ≈1.3
  - Median inter-node distance: 7.21 km
  - Maximum distance: 23.33 km (approximately Manhattan island length)
  - Asymmetry: median |d(a,b) - d(b,a)| = 0.21 km, max 7.90 km (one-way streets)
- **Arrays in file:**
  - `dist`: the distance matrix
  - `source`: provenance flag per cell (0=observed, 1=fallback, 2=diagonal)
  - `n_obs`: observation count per pair
  - `intra`: intra-node distances (diagonal entries)

### traveltime.npz
- **Shape:** 87 × 87 × 48
- **Purpose:** Time-dependent travel time for congestion-aware utility (Gap 1a)
- **Profiles:** 48 = 2 day types (weekday, weekend) × 24 hours
- **Units:** minutes
- **Estimation hierarchy (recorded per cell):**
  - **Level 0 (direct):** Median duration for this (pair, profile) if ≥30 observations  
    13.5% of cells, 90.7% of trips
  - **Level 1 (scaled):** Pair's own period mean × citywide multiplier for this profile  
    60.3% of cells
  - **Level 2 (fallback):** Distance ÷ citywide speed for this profile  
    26.2% of cells
- **Key quantities derived:**
  - `tau_bar[o,d]`: period-level travel time (the paper's version)
  - `tau[o,d,p]`: travel time per profile (ours)
  - `c[o,d,p] = tau / tau_bar`: congestion multiplier, clipped to [0.5, 3.0]
- **Citywide speed variation:** 12.8 to 28.5 km/h, ratio 2.22×
- **Well-measured pairs:** 969 pairs observed across ≥24 of 48 profiles
- **Peak/off-peak spread on those pairs:** median 2.11×, p90 2.93×, max 4.35×
- **Arrays in file:**
  - `tau`: 87×87×48 travel times
  - `tau_bar`: 87×87 period means
  - `c`: 87×87×48 congestion multipliers
  - `level`: 87×87×48 estimation level flags (0/1/2)
  - `n_obs`: 87×87×48 observation counts
  - `city_speed`: 48-element citywide speed per profile

### traveltime_train.npz & traveltime_test.npz
- **Same structure as traveltime.npz**
- **Purpose:** Prevent temporal leakage
  - `traveltime_train.npz`: built from trips before 25 March (test week excluded)
  - `traveltime_test.npz`: built from test week only (25-31 March)
- **Usage:**
  - The **forecaster** trains on `traveltime_train.npz` (no future information)
  - The **environment** uses `traveltime.npz` (full month, the true physics)
  - `traveltime_test.npz` held out entirely, used once to score predictions
- **Without this separation:** Gap 1a result would be a leakage artefact

### demand.npz
- **Shape:** 87 × 87 × 744
- **Dimensions:** origin × destination × hour of month (31 days × 24 hours)
- **Purpose:** Request counts for forecaster training and driver placement
- **Total requests:** 10,300,738 (reconciles exactly with trips.parquet)
- **Non-zero cells:** 25.4%
- **Busiest single (node, hour):** 2,184 requests
- **Arrays in file:**
  - `od`: 87×87×744 request counts per OD pair per hour
  - `org`: 87×744 marginal origin counts (sum over destinations)
  - `dest`: 87×744 marginal destination counts (sum over origins)

## Data Provenance and Reproducibility

### What can be rebuilt
Everything in `processed/` is derived. Given `raw/yellow_tripdata_2016-03.csv`:

```bash
# Full rebuild (≈2 minutes)
python -m ltf.data.prepare
python -m ltf.data.graph
python -m ltf.data.traveltime
python -m ltf.data.demand

# Verify integrity
python -m scripts.verify_phase1
```

The verification script runs 25 checks including:
- Node IDs are contiguous 0 to 86
- All distances are finite and positive
- Travel-time tensor has no non-finite values
- Congestion multiplier c lies in [0.5, 3.0]
- Demand tensor reconciles with trip count
- Time-aware utility reduces to prior work's utility at c=1

### What cannot be rebuilt
`raw/yellow_tripdata_2016-03.csv` is published data. If the TLC page moves or the file is updated, the exact row count and anomaly statistics may differ. The URL above was valid as of March 2024.

### Git status
- **Not committed:** The entire `dataset/` folder is excluded by `.gitignore`
- **Why:** `yellow_tripdata_2016-03.csv` is 1.9 GB and `trips.parquet` is 132 MB, both exceed GitHub's 100 MB file limit
- **Preservation strategy:** Keep the raw CSV locally, rebuild `processed/` on demand

### Checksums (for verification)
If you want to verify your raw CSV matches ours:

```bash
# MD5 (macOS/Linux)
md5sum raw/yellow_tripdata_2016-03.csv

# SHA256 (macOS/Linux)
shasum -a 256 raw/yellow_tripdata_2016-03.csv
```

Expected values (from our copy):
```
MD5:    [to be filled when verified]
SHA256: [to be filled when verified]
Rows:   12,210,952
Size:   1,988,474,675 bytes (1.9 GB)
```

## Documentation

### MTP_Dataset_Report.docx (15 sections)
Complete account of the data, every transformation applied, per-filter attrition measured on a full scan, every derived artefact, and a usage map showing where each artefact is consumed in the implementation.

Regenerate with:
```bash
python -m scripts.audit_dataset        # measures attrition → dataset_audit.json
python -m scripts.make_dataset_docx    # reads JSON → DOCX
```

### dataset_audit.json
Programmatically measured statistics:
- Per-filter row removal counts (sole cause vs multi-cause)
- Raw-file anomaly counts
- Node and graph properties
- Travel-time tensor estimation levels
- Demand tensor coverage
- All figures in the DOCX are read from this file

**Never edit this file by hand.** It is the single source of truth and regenerates from a full CSV scan.

## Frequently Asked Questions

**Q: Why March 2016?**  
A: It is the dataset used by Kang et al. (ECML PKDD 2024), the paper we extend. Using the same month makes our critique directly comparable.

**Q: Why not use a more recent month?**  
A: TLC replaced raw coordinates with pre-binned taxi zones in mid-2016. March 2016 is the last full month with exact lat/lon, which lets us choose spatial granularity by measurement rather than accepting a fixed zone set.

**Q: Why 87 nodes?**  
A: Measured, not assumed. At 1.11 km grid spacing with a 2,000-trip threshold, 87 nodes remain and about 90% of OD-profile cells are directly estimable. Finer grids are sparser; coarser grids wash out geography.

**Q: Why are traveltime_train and traveltime_test separate?**  
A: Temporal leakage prevention. The forecaster cannot be allowed to read future congestion. Without the split, Gap 1a's result would be an artefact of inadvertent future-peeking.

**Q: Can I use a different month?**  
A: Yes, but:
- Post-2016 data has taxi zones instead of coordinates, so nodes.csv would need a zone-to-centroid mapping
- The granularity selection (Section 4.2 of the paper) would need re-measurement
- Total rebuild time is ≈2 minutes, so experimentation is cheap

**Q: The paper says 10.3M trips but I only see 10,300,738 in trips.parquet. Is something missing?**  
A: 10,300,738 **is** 10.3M. They're the same number. The demand tensor reconciles exactly: `demand.od.sum() == len(trips)` is verified programmatically.

## Usage in the Implementation

| Artefact | Consumed by | For what |
|---|---|---|
| `nodes.csv` | `graph.py`, `traveltime.py`, `drivers.py` | Node coordinates for distance fallback; node count everywhere |
| `trips.parquet` | `graph.py`, `traveltime.py`, `demand.py`, `requests.py` | The single source for all estimation and for the request stream |
| `distance.npz` | `utility.py`, `env.py`, all methods | Every Geo(a,b) evaluation: trip value, deadhead cost, feasibility |
| `traveltime.npz` | `utility.py`, `env.py` | Congestion multiplier c in time-aware utility; trip occupancy in env |
| `traveltime_train.npz` | `forecaster.py` | Congestion features and targets, guaranteed free of test-week data |
| `traveltime_test.npz` | `run_forecaster.py` | Held-out ground truth, used once for scoring |
| `demand.npz` | `forecaster.py`, `drivers.py` | Forecaster training data; demand-weighted driver placement |

## License and Citation

**Data license:** The raw NYC TLC data is public domain.

**If you use this dataset package, cite:**

```bibtex
@misc{mtp2024dataset,
  title={Dataset Package for Hours-Aware and Access-Aware Fairness in Ride-Hailing},
  author={[Your Name]},
  year={2024},
  howpublished={GitHub repository},
  url={https://github.com/Utkarsh63-byte/MTP-LONG_TERM_FAIRNESS_GAP_IMPL}
}
```

And the original TLC source:

```bibtex
@misc{nyctlc2016,
  title={TLC Trip Record Data},
  author={{New York City Taxi and Limousine Commission}},
  year={2016},
  url={https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page}
}
```

## Contact

For questions about the dataset or the pipeline:
- Open an issue at: https://github.com/Utkarsh63-byte/MTP-LONG_TERM_FAIRNESS_GAP_IMPL/issues
- See `outputs/MTP_Dataset_Report.docx` for the complete 15-section documentation

---

Last updated: 2024  
Dataset version: 1.0 (March 2016 TLC data, 87-node graph at 1.11 km)
