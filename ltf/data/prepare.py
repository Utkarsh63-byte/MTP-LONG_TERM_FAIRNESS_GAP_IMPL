"""Pass 1/2 of the data pipeline: filter raw trips and define graph nodes.

Two streaming passes over the 1.9 GB CSV (chunked, only the needed columns):

  Pass 1  accumulate per-grid-cell trip counts and coordinate sums, then keep
          cells that clear `min_cell_trips`. Node centroids are the *empirical*
          mean pickup/dropoff coordinate inside the cell, not the geometric
          cell centre, so a node sits where the demand actually is.
  Pass 2  re-read, apply the same filters, snap every trip endpoint to a node
          id, and write a compact Parquet table.

The paper says only that "multiple locations are merged together as a node"
(Sec. 5.1) without giving a count, so the granularity is ours to choose. We
use ~1.5 km cells, which measurement showed is the coarsest setting that still
looks like geography while leaving the per-(OD, hour) travel-time buckets
populated enough to estimate (46.3% clear n>=30, versus 15.6% at 500 m).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import RAW_CSV, Config
from ..utils import LOG, cache_path, haversine_km, timed

CSV_COLS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "fare_amount", "tip_amount", "total_amount",
]
CHUNK = 1_500_000

TRIPS_PARQUET = "trips.parquet"
NODES_CSV = "nodes.csv"


@dataclass
class NodeTable:
    """Graph nodes: id, centroid, volume, and the grid cell they came from."""

    node_id: np.ndarray      # (K,) int32
    lat: np.ndarray          # (K,) float64 centroid
    lon: np.ndarray          # (K,) float64 centroid
    trips: np.ndarray        # (K,) int64 pickups+dropoffs in the month
    cell_row: np.ndarray
    cell_col: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.node_id)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "node_id": self.node_id, "lat": self.lat, "lon": self.lon,
            "trips": self.trips, "cell_row": self.cell_row,
            "cell_col": self.cell_col,
        })

    @staticmethod
    def from_frame(df: pd.DataFrame) -> "NodeTable":
        return NodeTable(
            node_id=df.node_id.to_numpy(np.int32),
            lat=df.lat.to_numpy(float), lon=df.lon.to_numpy(float),
            trips=df.trips.to_numpy(np.int64),
            cell_row=df.cell_row.to_numpy(np.int32),
            cell_col=df.cell_col.to_numpy(np.int32),
        )


# --------------------------------------------------------------------------
def _filter_chunk(ch: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply bbox + sanity filters. Returns (rows, duration_minutes)."""
    d = cfg.data
    m = (
        ch.pickup_longitude.between(d.lon_min, d.lon_max)
        & ch.pickup_latitude.between(d.lat_min, d.lat_max)
        & ch.dropoff_longitude.between(d.lon_min, d.lon_max)
        & ch.dropoff_latitude.between(d.lat_min, d.lat_max)
        & ch.trip_distance.between(d.min_trip_miles, d.max_trip_miles)
        & ch.fare_amount.between(d.min_fare, d.max_fare)
        & (ch.tpep_pickup_datetime >= "2016-03-01")
        & (ch.tpep_pickup_datetime < "2016-04-01")
    )
    ch = ch[m]
    if len(ch) == 0:
        return ch, np.empty(0)
    dur = (ch.tpep_dropoff_datetime - ch.tpep_pickup_datetime).dt.total_seconds() / 60.0
    ok = (dur > d.min_duration_min) & (dur < d.max_duration_min)
    ch, dur = ch[ok], dur[ok]
    if len(ch) == 0:
        return ch, np.empty(0)
    speed = ch.trip_distance.to_numpy() / (dur.to_numpy() / 60.0)
    ok2 = (speed > d.min_speed_mph) & (speed < d.max_speed_mph)
    return ch[ok2], dur.to_numpy()[ok2]


def _cell_index(lat, lon, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    d = cfg.data
    row = np.floor((np.asarray(lat) - d.lat_min) / d.grid_deg).astype(np.int32)
    col = np.floor((np.asarray(lon) - d.lon_min) / d.grid_deg).astype(np.int32)
    return row, col


def _reader():
    return pd.read_csv(RAW_CSV, usecols=CSV_COLS, chunksize=CHUNK,
                       parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])


# --------------------------------------------------------------------------
def build_nodes(cfg: Config) -> NodeTable:
    """Pass 1: define graph nodes from grid-cell trip volume."""
    n_col = int(np.ceil((cfg.data.lon_max - cfg.data.lon_min) / cfg.data.grid_deg)) + 2
    n_row = int(np.ceil((cfg.data.lat_max - cfg.data.lat_min) / cfg.data.grid_deg)) + 2
    size = n_row * n_col
    count = np.zeros(size, np.int64)
    lat_sum = np.zeros(size)
    lon_sum = np.zeros(size)
    kept = 0

    with timed(f"pass 1: grid-cell volumes ({n_row}x{n_col} cells)"):
        for ci, ch in enumerate(_reader()):
            ch, _ = _filter_chunk(ch, cfg)
            if len(ch) == 0:
                continue
            kept += len(ch)
            for la, lo in (("pickup_latitude", "pickup_longitude"),
                           ("dropoff_latitude", "dropoff_longitude")):
                r, c = _cell_index(ch[la].to_numpy(), ch[lo].to_numpy(), cfg)
                flat = np.clip(r, 0, n_row - 1) * n_col + np.clip(c, 0, n_col - 1)
                np.add.at(count, flat, 1)
                np.add.at(lat_sum, flat, ch[la].to_numpy())
                np.add.at(lon_sum, flat, ch[lo].to_numpy())
            LOG.info(f"  chunk {ci}: kept={kept:,}")

    keep = np.where(count >= cfg.data.min_cell_trips)[0]
    order = keep[np.argsort(-count[keep])]
    nodes = NodeTable(
        node_id=np.arange(len(order), dtype=np.int32),
        lat=lat_sum[order] / count[order],
        lon=lon_sum[order] / count[order],
        trips=count[order],
        cell_row=(order // n_col).astype(np.int32),
        cell_col=(order % n_col).astype(np.int32),
    )
    frac = count[order].sum() / count.sum()
    LOG.info("kept %d nodes (of %d non-empty cells); they hold %.2f%% of endpoints",
             nodes.n_nodes, int((count > 0).sum()), 100 * frac)
    nodes.to_frame().to_csv(cache_path(NODES_CSV), index=False)
    return nodes


def _snap_lookup(nodes: NodeTable, cfg: Config, n_row: int, n_col: int) -> np.ndarray:
    """Map every grid cell -> nearest node id (by centroid great-circle distance)."""
    d = cfg.data
    rr, cc = np.divmod(np.arange(n_row * n_col), n_col)
    cell_lat = d.lat_min + (rr + 0.5) * d.grid_deg
    cell_lon = d.lon_min + (cc + 0.5) * d.grid_deg
    dist = haversine_km(cell_lat[:, None], cell_lon[:, None],
                        nodes.lat[None, :], nodes.lon[None, :])
    lut = np.argmin(dist, axis=1).astype(np.int16)
    # exact cells keep their own node
    own = nodes.cell_row.astype(np.int64) * n_col + nodes.cell_col.astype(np.int64)
    lut[own] = nodes.node_id.astype(np.int16)
    return lut


def build_trips(cfg: Config, nodes: NodeTable) -> pd.DataFrame:
    """Pass 2: filter again, snap endpoints to nodes, write Parquet."""
    n_col = int(np.ceil((cfg.data.lon_max - cfg.data.lon_min) / cfg.data.grid_deg)) + 2
    n_row = int(np.ceil((cfg.data.lat_max - cfg.data.lat_min) / cfg.data.grid_deg)) + 2
    lut = _snap_lookup(nodes, cfg, n_row, n_col)

    parts: list[pd.DataFrame] = []
    with timed("pass 2: snap trips to nodes"):
        for ci, ch in enumerate(_reader()):
            ch, dur = _filter_chunk(ch, cfg)
            if len(ch) == 0:
                continue
            pr, pc = _cell_index(ch.pickup_latitude.to_numpy(),
                                 ch.pickup_longitude.to_numpy(), cfg)
            dr, dc = _cell_index(ch.dropoff_latitude.to_numpy(),
                                 ch.dropoff_longitude.to_numpy(), cfg)
            o = lut[np.clip(pr, 0, n_row - 1) * n_col + np.clip(pc, 0, n_col - 1)]
            dd = lut[np.clip(dr, 0, n_row - 1) * n_col + np.clip(dc, 0, n_col - 1)]
            parts.append(pd.DataFrame({
                "pickup_ts": ch.tpep_pickup_datetime.to_numpy(),
                "dropoff_ts": ch.tpep_dropoff_datetime.to_numpy(),
                "o": o.astype(np.int16),
                "d": dd.astype(np.int16),
                "km": (ch.trip_distance.to_numpy() * 1.60934).astype(np.float32),
                "dur_min": dur.astype(np.float32),
                "fare": (ch.fare_amount.to_numpy()
                         + ch.tip_amount.to_numpy()).astype(np.float32),
            }))
            LOG.info(f"  chunk {ci}: rows={sum(len(p) for p in parts):,}")

    trips = pd.concat(parts, ignore_index=True)
    trips.sort_values("pickup_ts", inplace=True, kind="mergesort")
    trips.reset_index(drop=True, inplace=True)
    trips["is_weekend"] = (trips.pickup_ts.dt.dayofweek >= 5).astype(np.int8)
    trips["hour"] = trips.pickup_ts.dt.hour.astype(np.int8)
    trips["date"] = trips.pickup_ts.dt.strftime("%Y-%m-%d")
    trips["speed_kmh"] = (trips.km / (trips.dur_min / 60.0)).astype(np.float32)
    trips.to_parquet(cache_path(TRIPS_PARQUET), index=False, compression="zstd")
    LOG.info(f"wrote {TRIPS_PARQUET}: {len(trips):,} rows")
    return trips


# --------------------------------------------------------------------------
def load_nodes() -> NodeTable:
    return NodeTable.from_frame(pd.read_csv(cache_path(NODES_CSV)))


def load_trips(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(cache_path(TRIPS_PARQUET), columns=columns)


def prepare(cfg: Config | None = None, force: bool = False):
    """Build (or load) nodes + trips."""
    cfg = cfg or Config()
    if not force and cache_path(NODES_CSV).exists() and cache_path(TRIPS_PARQUET).exists():
        LOG.info("cache hit: loading nodes + trips")
        return load_nodes(), load_trips()
    nodes = build_nodes(cfg)
    trips = build_trips(cfg, nodes)
    return nodes, trips


if __name__ == "__main__":
    cfg = Config()
    nodes, trips = prepare(cfg, force=True)
    print(f"\nnodes: {nodes.n_nodes}")
    print(f"trips: {len(trips):,}")
    print(trips.head().to_string())
