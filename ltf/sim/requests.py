"""Request stream: the rider side of the market.

The paper thins the data with "a stratified sampling method with a sampling
rate of 0.05" (Sec. 5.2) without naming the strata. We stratify by
(date, hour) so the thinned stream keeps the real diurnal and day-of-week
demand shape -- which matters here, because the whole point of the forecasting
module is to anticipate that shape.

The sampling rate itself is not hard-coded. On a full-day timeline the paper's
0.05 would bury 200 drivers under demand they could never serve, and the
resulting utilisation would be pinned at 1.0, which would make Gap 2's metric
degenerate. `calibrate_sample_rate` instead solves for the rate that puts mean
system utilisation near a target (default 0.60).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import LOG, rng
from .timeline import Timeline


@dataclass
class RequestStream:
    """Requests bucketed into decision slots, sorted by slot."""

    slot: np.ndarray      # (N,) int32 decision slot
    o: np.ndarray         # (N,) int16 pickup node
    d: np.ndarray         # (N,) int16 dropoff node
    ts: np.ndarray        # (N,) datetime64[m] original request time
    km: np.ndarray        # (N,) float32 observed trip distance
    dur_min: np.ndarray   # (N,) float32 observed duration
    fare: np.ndarray      # (N,) float32 observed fare+tip
    start: np.ndarray     # (T+1,) int32 CSR-style index: requests of slot t are
                          #        [start[t]:start[t+1]]
    sample_rate: float

    @property
    def n(self) -> int:
        return len(self.slot)

    def of_slot(self, t: int) -> slice:
        return slice(int(self.start[t]), int(self.start[t + 1]))

    def counts(self) -> np.ndarray:
        return np.diff(self.start)


def _stratified_mask(trips: pd.DataFrame, rate: float, seed: int) -> np.ndarray:
    """Sample `rate` of rows within every (date, hour) stratum."""
    if rate >= 1.0:
        return np.ones(len(trips), bool)
    r = rng(seed)
    keep = np.zeros(len(trips), bool)
    codes, _ = pd.factorize(trips.date.to_numpy() + "T"
                            + trips.hour.to_numpy().astype(str))
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1],
                                  True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        k = int(round(rate * len(idx)))
        if k <= 0:
            # keep the stratum non-empty with probability = rate*len
            if r.random() < rate * len(idx):
                k = 1
            else:
                continue
        keep[r.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return keep


def build_requests(cfg: Config, timeline: Timeline, trips: pd.DataFrame,
                   sample_rate: float, seed: int | None = None) -> RequestStream:
    """Filter trips onto the timeline and thin them."""
    seed = cfg.driver.seed if seed is None else seed
    lo = pd.Timestamp(timeline.ts[0])
    hi = pd.Timestamp(timeline.ts[-1]) + pd.Timedelta(minutes=timeline.slot_minutes)
    sub = trips[(trips.pickup_ts >= lo) & (trips.pickup_ts < hi)]
    slot = timeline.slot_of_timestamp(sub.pickup_ts.to_numpy())
    sub = sub[slot >= 0]
    slot = slot[slot >= 0]

    keep = _stratified_mask(sub, sample_rate, seed)
    sub, slot = sub[keep], slot[keep]

    order = np.argsort(slot, kind="stable")
    sub, slot = sub.iloc[order], slot[order]
    start = np.searchsorted(slot, np.arange(timeline.n_slots + 1)).astype(np.int32)

    rs = RequestStream(
        slot=slot.astype(np.int32),
        o=sub.o.to_numpy(np.int16), d=sub.d.to_numpy(np.int16),
        ts=sub.pickup_ts.to_numpy().astype("datetime64[m]"),
        km=sub.km.to_numpy(np.float32), dur_min=sub.dur_min.to_numpy(np.float32),
        fare=sub.fare.to_numpy(np.float32), start=start, sample_rate=sample_rate)
    LOG.info("requests: %d over %d slots (rate %.4f), %.1f per slot, "
             "%.0f per day", rs.n, timeline.n_slots, sample_rate,
             rs.n / timeline.n_slots, rs.n / timeline.n_days)
    return rs


# --------------------------------------------------------------------------
def estimate_occupancy_slots(cfg: Config, timeline: Timeline,
                             trips: pd.DataFrame, tt, graph) -> float:
    """Mean decision slots an assignment consumes: deadhead + loaded leg.

    Deadhead is estimated as the distance to a nearby node (3rd-nearest, a
    stand-in for "some online driver is roughly there") at the prevailing
    citywide speed for the slots in the horizon.
    """
    lo = pd.Timestamp(timeline.ts[0])
    hi = pd.Timestamp(timeline.ts[-1])
    sub = trips[(trips.pickup_ts >= lo) & (trips.pickup_ts < hi)]
    loaded = float(sub.dur_min.mean()) if len(sub) else 12.0

    d = np.sort(graph.dist + np.eye(graph.n_nodes) * 1e6, axis=1)
    near_km = float(np.median(d[:, 2]))
    speed = float(np.mean(tt.city_speed[np.unique(timeline.profile)]))
    deadhead = near_km / speed * 60.0
    slots = max(1.0, np.ceil((loaded + deadhead) / cfg.time.slot_minutes))
    LOG.info("occupancy estimate: loaded %.1f min + deadhead %.1f min "
             "(%.2f km @ %.1f km/h) -> %.0f slots of %d min",
             loaded, deadhead, near_km, speed, slots, cfg.time.slot_minutes)
    return float(slots)


def calibrate_sample_rate(cfg: Config, timeline: Timeline, trips: pd.DataFrame,
                          online_slots: int, occupancy_slots: float,
                          available: int | None = None) -> float:
    """Solve for the sampling rate that hits `target_offered_load`.

    Balance: assignments * occupancy_slots  ~=  target * total online slots.
    """
    if cfg.protocol.sample_rate is not None:
        return float(cfg.protocol.sample_rate)
    if available is None:
        lo = pd.Timestamp(timeline.ts[0])
        hi = pd.Timestamp(timeline.ts[-1]) + pd.Timedelta(
            minutes=timeline.slot_minutes)
        m = (trips.pickup_ts >= lo) & (trips.pickup_ts < hi)
        if cfg.protocol.protocol == "paper_peak2h":
            m &= trips.hour.between(cfg.protocol.peak_hour_start,
                                    cfg.protocol.peak_hour_end - 1)
        available = int(m.sum())
    target = cfg.protocol.target_offered_load * online_slots / occupancy_slots
    rate = float(np.clip(target / max(available, 1), 1e-6, 1.0))
    LOG.info("calibrated sample rate %.5f: %d online slots x target %.2f "
             "/ %.0f slots per job = %.0f jobs, from %d available requests",
             rate, online_slots, cfg.protocol.target_offered_load,
             occupancy_slots, target, available)
    return rate


def measure_offered_load(cfg: Config, timeline: Timeline, rq: RequestStream,
                         drivers, utility, online_slots: int) -> float:
    """Offered load = driver-slots the requests would consume / driver-slots online.

    This is demand pressure, not realised utilisation: it assumes every request
    is served. Realised utilisation is offered load times the service rate, and
    is what Gap 2 actually measures. Keeping the distinction explicit avoids
    claiming a utilisation number the simulator has not produced yet.
    """
    r = rng(cfg.driver.seed + 7)
    g = drivers.home_node[r.integers(0, drivers.n, rq.n)]
    occ = utility.occupancy_slots(g, rq.o, rq.d, timeline.profile[rq.slot])
    return float(occ.sum()) / max(online_slots, 1)


def refine_sample_rate(cfg: Config, timeline: Timeline, trips: pd.DataFrame,
                       drivers, utility, rate: float, seed: int | None,
                       iters: int = 3) -> tuple[float, RequestStream, float]:
    """Rebuild the stream a couple of times so offered load hits its target.

    The analytic estimate uses an assumed deadhead and a mean trip duration; the
    real requests have their own distance and time-of-day mix, so one measured
    correction gets far closer than any closed form.
    """
    online_slots = int(drivers.online_slots.sum())
    target = cfg.protocol.target_offered_load
    rq = build_requests(cfg, timeline, trips, rate, seed=seed)
    load = measure_offered_load(cfg, timeline, rq, drivers, utility, online_slots)
    if cfg.protocol.sample_rate is not None:
        LOG.info("sample rate pinned at %.5f, measured offered load %.3f",
                 rate, load)
        return rate, rq, load
    for i in range(iters):
        if abs(load - target) <= 0.02 * target:
            break
        rate = float(np.clip(rate * target / max(load, 1e-6), 1e-7, 1.0))
        rq = build_requests(cfg, timeline, trips, rate, seed=seed)
        load = measure_offered_load(cfg, timeline, rq, drivers, utility,
                                    online_slots)
        LOG.info("  calibration step %d: rate %.6f -> offered load %.3f "
                 "(target %.2f)", i + 1, rate, load, target)
    return rate, rq, load


def synthetic_requests_from_prediction(cfg: Config, timeline: Timeline,
                                       dem_grid: np.ndarray,
                                       hours: np.ndarray,
                                       sample_rate: float,
                                       seed: int = 0) -> RequestStream:
    """Sample a request stream from *predicted* demand.

    This is the mechanism behind the paper's central claim: forecast requests
    enter the MDP's action space (Sec. 4.3) so the learned policy anticipates
    future demand rather than assuming the future repeats the past. Training on
    a synthetic week drawn from the forecaster -- rather than on ground truth --
    is also what keeps the look-ahead honest.

    Counts are thinned by the same `sample_rate` as the real stream and drawn
    Poisson, then scattered uniformly within their hour.
    """
    r = rng(seed)
    K = dem_grid.shape[0]
    hour_to_slots: dict[int, np.ndarray] = {}
    for t in range(timeline.n_slots):
        hour_to_slots.setdefault(int(timeline.hour_index[t]), []).append(t)
    hour_to_slots = {k: np.asarray(v) for k, v in hour_to_slots.items()}

    slots, os_, ds_ = [], [], []
    for hi, h in enumerate(hours):
        cand = hour_to_slots.get(int(h))
        if cand is None or len(cand) == 0:
            continue
        lam = dem_grid[:, :, hi] * sample_rate
        nz = np.argwhere(lam > 1e-6)
        if len(nz) == 0:
            continue
        draws = r.poisson(lam[nz[:, 0], nz[:, 1]])
        keep = draws > 0
        for (o, d), c in zip(nz[keep], draws[keep]):
            slots.append(r.choice(cand, size=int(c)))
            os_.append(np.full(int(c), o, np.int16))
            ds_.append(np.full(int(c), d, np.int16))

    if not slots:
        empty = np.zeros(0)
        return RequestStream(
            slot=empty.astype(np.int32), o=empty.astype(np.int16),
            d=empty.astype(np.int16), ts=timeline.ts[:0], km=empty.astype(np.float32),
            dur_min=empty.astype(np.float32), fare=empty.astype(np.float32),
            start=np.zeros(timeline.n_slots + 1, np.int32), sample_rate=sample_rate)

    slot = np.concatenate(slots).astype(np.int32)
    o = np.concatenate(os_)
    d = np.concatenate(ds_)
    order = np.argsort(slot, kind="stable")
    slot, o, d = slot[order], o[order], d[order]
    start = np.searchsorted(slot, np.arange(timeline.n_slots + 1)).astype(np.int32)
    n = len(slot)
    LOG.info("synthetic (predicted) requests: %d over %d slots", n, timeline.n_slots)
    return RequestStream(
        slot=slot, o=o, d=d, ts=timeline.ts[slot],
        km=np.zeros(n, np.float32), dur_min=np.zeros(n, np.float32),
        fare=np.zeros(n, np.float32), start=start, sample_rate=sample_rate)
