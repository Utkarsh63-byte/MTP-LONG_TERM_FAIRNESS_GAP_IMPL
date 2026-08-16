"""The `Geo` operator: an OD road-distance matrix over the graph nodes.

The paper defines edges as "the travel distance between locations" on a
*directed* graph, and utility as `Geo(d_r,s_r) - Geo(s_r,g_v)` where Geo is a
shortest geographical distance (Sec. 3.1). It never says how those distances
were obtained.

We estimate them from the data itself, which is stronger than a synthetic
metric: 10.3M observed trips give the actual driven distance between node
pairs, and the asymmetry the paper motivates its digraph with falls out for
free (one-way avenues really do make A->B differ from B->A).

  1. Direct estimate  = median observed `trip_distance` per (o, d), for pairs
     with at least `min_obs` trips.
  2. Fallback         = rotated-L1 distance. Manhattan's street grid runs
     ~29 deg east of north, so rotating the coordinate frame and taking an
     L1 distance approximates driving distance far better than a straight
     line. The scale is calibrated by least squares against (1).
  3. Metric closure   = min-plus (Floyd-Warshall) closure, OFF by default.
     It sounds right -- the paper says "shortest" distance -- but on a
     complete graph of noisy medians the min over ~87 candidate paths is
     systematically biased downward: enabling it shortens 74% of pairs by a
     median 1.11 km (~18% of the median trip). The direct estimates are
     already real driven distances from real routes, so closing them mostly
     destroys signal. Triangle-inequality violations are reported as a
     diagnostic instead. Set `enforce_shortest_path=True` to compare.

Intra-node distance (the diagonal) is kept as the empirical median of trips
that start and end inside one node: it is small but not zero, and treating it
as zero would make same-node deadhead free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import LOG, cache_path, describe_array, haversine_km, timed
from .prepare import NodeTable, load_nodes, load_trips

DIST_NPZ = "distance.npz"
MIN_OBS = 5


@dataclass
class RoadGraph:
    """OD distance matrix in km plus provenance bookkeeping."""

    dist: np.ndarray          # (K,K) float32, shortest road distance, km
    direct: np.ndarray        # (K,K) float32, pre-Floyd-Warshall estimate
    source: np.ndarray        # (K,K) int8: 0 = observed, 1 = rotated-L1 fallback
    n_obs: np.ndarray         # (K,K) int32 trips backing each direct estimate
    intra: np.ndarray         # (K,) float32 within-node median distance
    lat: np.ndarray
    lon: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.dist.shape[0]

    def geo(self, o, d):
        """Geo(o, d) in km. Vectorised over array-like inputs."""
        return self.dist[np.asarray(o), np.asarray(d)]

    def save(self) -> None:
        np.savez_compressed(
            cache_path(DIST_NPZ), dist=self.dist, direct=self.direct,
            source=self.source, n_obs=self.n_obs, intra=self.intra,
            lat=self.lat, lon=self.lon)
        LOG.info("wrote %s", DIST_NPZ)

    @staticmethod
    def load() -> "RoadGraph":
        z = np.load(cache_path(DIST_NPZ))
        return RoadGraph(**{k: z[k] for k in
                            ["dist", "direct", "source", "n_obs", "intra", "lat", "lon"]})


# --------------------------------------------------------------------------
def rotated_l1_km(lat_o, lon_o, lat_d, lon_d, rotation_deg: float) -> np.ndarray:
    """L1 distance in a frame aligned with the Manhattan street grid."""
    lat0 = float(np.mean(np.concatenate([np.ravel(lat_o), np.ravel(lat_d)])))
    ky = 111.32
    kx = 111.32 * np.cos(np.radians(lat0))
    xo, yo = np.asarray(lon_o) * kx, np.asarray(lat_o) * ky
    xd, yd = np.asarray(lon_d) * kx, np.asarray(lat_d) * ky
    th = np.radians(rotation_deg)
    c, s = np.cos(th), np.sin(th)
    dx, dy = xd - xo, yd - yo
    return np.abs(dx * c + dy * s) + np.abs(-dx * s + dy * c)


def build_graph(cfg: Config | None = None,
                nodes: NodeTable | None = None,
                trips: pd.DataFrame | None = None,
                enforce_shortest_path: bool = False) -> RoadGraph:
    cfg = cfg or Config()
    nodes = nodes or load_nodes()
    if trips is None:
        trips = load_trips(columns=["o", "d", "km"])
    K = nodes.n_nodes

    with timed(f"OD distance matrix ({K} nodes)"):
        g = trips.groupby(["o", "d"])["km"].agg(["median", "size"])
        direct = np.full((K, K), np.nan, np.float64)
        n_obs = np.zeros((K, K), np.int32)
        oi = g.index.get_level_values(0).to_numpy()
        di = g.index.get_level_values(1).to_numpy()
        n_obs[oi, di] = g["size"].to_numpy()
        enough = g["size"].to_numpy() >= MIN_OBS
        direct[oi[enough], di[enough]] = g["median"].to_numpy()[enough]
        observed = np.isfinite(direct)
        LOG.info("observed OD pairs with >=%d trips: %d / %d (%.1f%%), "
                 "covering %.2f%% of trips",
                 MIN_OBS, int(observed.sum()), K * K,
                 100 * observed.sum() / K / K,
                 100 * n_obs[observed].sum() / n_obs.sum())

        # rotated-L1 fallback, scale calibrated on the observed pairs
        l1 = rotated_l1_km(nodes.lat[:, None], nodes.lon[:, None],
                           nodes.lat[None, :], nodes.lon[None, :],
                           cfg.data.manhattan_rotation_deg)
        off = observed & ~np.eye(K, dtype=bool)
        w = n_obs[off].astype(float)
        num = float(np.sum(w * l1[off] * direct[off]))
        den = float(np.sum(w * l1[off] ** 2))
        scale = num / den if den > 0 else 1.0
        pred = scale * l1
        resid = direct[off] - pred[off]
        rel = np.abs(resid) / np.maximum(direct[off], 0.1)
        LOG.info("rotated-L1 calibration: scale=%.4f  weighted MAPE=%.1f%%  "
                 "median|err|=%.3f km", scale,
                 100 * float(np.sum(w * rel) / np.sum(w)),
                 float(np.median(np.abs(resid))))
        crow = haversine_km(nodes.lat[:, None], nodes.lon[:, None],
                           nodes.lat[None, :], nodes.lon[None, :])
        LOG.info("circuity (road / great-circle) on observed pairs: median %.3f",
                 float(np.median(direct[off] / np.maximum(crow[off], 1e-6))))

        source = np.where(observed, 0, 1).astype(np.int8)
        filled = np.where(observed, direct, pred)
        # a node's own diagonal: empirical intra-node trip distance
        intra = np.array([filled[i, i] if observed[i, i] else 0.35 * scale
                          for i in range(K)], np.float64)
        np.fill_diagonal(filled, 0.0)
        filled = np.maximum(filled, 0.0)

        # diagnostic: how far is the observed matrix from being a metric?
        closure = filled.copy()
        for k in range(K):
            np.minimum(closure, closure[:, k, None] + closure[None, k, :],
                       out=closure)
        viol = (closure < filled - 1e-9) & ~np.eye(K, dtype=bool)
        LOG.info("triangle-inequality violations: %d / %d pairs (%.1f%%), "
                 "median excess %.3f km -- closure NOT applied by default",
                 int(viol.sum()), K * K - K, 100 * viol.sum() / (K * K - K),
                 float(np.median((filled - closure)[viol])) if viol.any() else 0.0)
        if enforce_shortest_path:
            LOG.warning("applying min-plus closure (biases well-measured pairs down)")
            sp = closure
        else:
            sp = filled.copy()
        np.fill_diagonal(sp, intra)

    rg = RoadGraph(dist=sp.astype(np.float32), direct=filled.astype(np.float32),
                   source=source, n_obs=n_obs, intra=intra.astype(np.float32),
                   lat=nodes.lat, lon=nodes.lon)
    off_diag = sp[~np.eye(K, dtype=bool)]
    LOG.info(describe_array(off_diag, "Geo off-diagonal (km)"))
    LOG.info(describe_array(intra, "intra-node distance (km)"))
    rg.save()
    return rg


def load_or_build(cfg: Config | None = None, force: bool = False) -> RoadGraph:
    if not force and cache_path(DIST_NPZ).exists():
        return RoadGraph.load()
    return build_graph(cfg)


if __name__ == "__main__":
    build_graph(Config())
