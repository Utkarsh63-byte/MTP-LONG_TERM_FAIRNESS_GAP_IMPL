# How to Get the Missing Large Files

Two files are excluded from git because they exceed GitHub's 100 MB file size limit:
1. `raw/yellow_tripdata_2016-03.csv` (1.9 GB)
2. `processed/trips.parquet` (132 MB)

---

## Option 1: Download Raw CSV and Rebuild (Recommended)

### Step 1: Download the raw CSV

**Direct download link:**
```
https://data.cityofnewyork.us/api/views/uacg-pexx/rows.csv?accessType=DOWNLOAD
```

Or visit the TLC data page and download March 2016:
```
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
```

Save as: `dataset/raw/yellow_tripdata_2016-03.csv`

**Verify the download:**
```bash
# Expected size
ls -lh dataset/raw/yellow_tripdata_2016-03.csv
# Should show approximately 1.9 GB

# Expected row count
wc -l dataset/raw/yellow_tripdata_2016-03.csv
# Should show 12,210,953 (including header)

# Expected checksums
md5 dataset/raw/yellow_tripdata_2016-03.csv
# Expected: fbc95c81a17ef549e64881a456662460

shasum -a 256 dataset/raw/yellow_tripdata_2016-03.csv
# Expected: 12d11b731b522001142af5def35e77429cb5f3d520d1b7ef1d46b810789f8a0b
```

### Step 2: Rebuild trips.parquet

From the repository root:
```bash
cd /path/to/Research_MTP
python -m ltf.data.prepare
```

This will:
- Build `dataset/processed/nodes.csv` (87 nodes)
- Build `dataset/processed/trips.parquet` (10,300,738 cleaned trips)

**Time:** About 30-60 seconds on a modern machine.

**Verify:**
```bash
ls -lh dataset/processed/trips.parquet
# Should show approximately 132 MB

python -c "import pandas as pd; print(len(pd.read_parquet('dataset/processed/trips.parquet')))"
# Should print: 10300738
```

---

## Option 2: Download Pre-built trips.parquet (If Available)

If someone has shared the pre-built file:

```bash
# Copy trips.parquet to the correct location
cp /path/to/trips.parquet dataset/processed/

# Verify
python -m scripts.verify_phase1
```

---

## Option 3: Use a Different Month

You can use any NYC TLC yellow taxi CSV from the historical data page.

**Requirements:**
- Must have raw lat/lon coordinates (pre-2016 data has this; post-mid-2016 uses taxi zones)
- Minimum ~10M trips recommended for statistical stability

**Steps:**
1. Download your chosen month's CSV
2. Place in `dataset/raw/` with filename `yellow_tripdata_YYYY-MM.csv`
3. Update `ltf/config.py` with the new filename
4. Run the rebuild pipeline (see "Complete Rebuild" below)

**Note:** Results will differ from the paper's numbers because the data is different.

---

## Complete Rebuild Pipeline

If you have the raw CSV and want to rebuild **everything**:

```bash
cd /path/to/Research_MTP

# Rebuild all processed files (takes about 2 minutes)
python -m ltf.data.prepare       # → nodes.csv, trips.parquet
python -m ltf.data.graph         # → distance.npz
python -m ltf.data.traveltime    # → traveltime*.npz (3 files)
python -m ltf.data.demand        # → demand.npz

# Verify integrity (25 checks)
python -m scripts.verify_phase1

# Should print:
# ✓ 25/25 checks passed
```

---

## What If I Can't Download the Raw CSV?

The other 6 processed files (all except `trips.parquet`) **are** in git and total only ~8 MB:
- `nodes.csv`
- `distance.npz`
- `traveltime.npz`, `traveltime_train.npz`, `traveltime_test.npz`
- `demand.npz`

You can:
1. **Read the paper results** without needing the data
2. **Inspect the processed files** to understand the structure
3. **Read the documentation** (`MTP_Dataset_Report.docx`, `dataset_audit.json`)

But you **cannot run experiments** without either:
- The raw CSV (to rebuild `trips.parquet`), or
- A pre-built `trips.parquet` file

---

## Troubleshooting

### "The CSV I downloaded doesn't match the checksums"

If the TLC updates their data export, checksums may differ. As long as:
- Row count is approximately 12.2M
- The file spans March 1-31, 2016
- Columns match the expected schema

...the pipeline should work, but your exact numbers may differ slightly from the paper.

### "python -m ltf.data.prepare fails"

Check:
1. You have the virtual environment activated: `source .venv/bin/activate`
2. Dependencies installed: `pip install -r requirements.txt`
3. The CSV is in the correct location: `dataset/raw/yellow_tripdata_2016-03.csv`
4. The CSV has the expected columns (see `RAW_COLUMNS` in `ltf/data/prepare.py`)

### "trips.parquet is the wrong size"

The exact size depends on Parquet compression. Anywhere from 120-140 MB is normal. Check the **row count** instead:
```bash
python -c "import pandas as pd; print(len(pd.read_parquet('dataset/processed/trips.parquet')))"
```
Should print exactly: `10300738`

---

## Alternative Data Sources

If the TLC link is broken, try:

**AWS Open Data Registry:**
```
https://registry.opendata.aws/nyc-tlc-trip-records-pds/
```

**Kaggle:**
```
https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data
```

**Google BigQuery:**
```sql
SELECT * FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2016`
WHERE DATE(pickup_datetime) BETWEEN '2016-03-01' AND '2016-03-31'
```

---

## Questions?

Open an issue at:
```
https://github.com/Utkarsh63-byte/MTP-LONG_TERM_FAIRNESS_GAP_IMPL/issues
```

Include:
- Which step failed
- Error message
- File sizes you see
- Output of `python --version` and `pip list`
