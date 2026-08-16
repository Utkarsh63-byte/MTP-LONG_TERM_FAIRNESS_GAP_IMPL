"""Decision-epoch timeline.

Holds the explicit list of slot start times, which keeps both protocols on one
code path:

  fullday      contiguous 5-minute slots across whole days
  paper_peak2h only the slots inside the paper's peak 2-hour window per day,
               at the paper's 1-hour step (Sec. 5.2)

Because the peak protocol skips most of the day, slot indices are not a simple
affine function of wall-clock time, so timestamps are materialised once and
everything else (time profile for the travel-time tensor, hour index for the
demand tensor, day index for the history/current/future split) is derived from
them.

The three-way split follows Sec. 3.3: within the one-week horizon the first
three days are history, the fourth is "current", the last three are future.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config

EPOCH = pd.Timestamp("2016-03-01")

HISTORY, CURRENT, FUTURE = 0, 1, 2
PHASE_NAMES = {HISTORY: "history", CURRENT: "current", FUTURE: "future"}


@dataclass
class Timeline:
    ts: np.ndarray            # (T,) datetime64[m], slot start times
    slot_minutes: int
    hour_index: np.ndarray    # (T,) int32, hours since 2016-03-01 00:00
    profile: np.ndarray       # (T,) int16, is_weekend*24 + hour
    day: np.ndarray           # (T,) int32, 0-based day within this timeline
    phase: np.ndarray         # (T,) int8, HISTORY / CURRENT / FUTURE
    dates: np.ndarray         # (D,) str, one per day index

    @property
    def n_slots(self) -> int:
        return len(self.ts)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0

    def slots_of_day(self, d: int) -> np.ndarray:
        return np.where(self.day == d)[0]

    def slots_of_phase(self, ph: int) -> np.ndarray:
        return np.where(self.phase == ph)[0]

    def slot_of_timestamp(self, when) -> np.ndarray:
        """Index of the slot containing each timestamp; -1 if outside."""
        w = np.asarray(when, dtype="datetime64[m]")
        idx = np.searchsorted(self.ts, w, side="right") - 1
        ok = (idx >= 0) & (idx < self.n_slots)
        span = np.timedelta64(self.slot_minutes, "m")
        within = np.zeros(len(np.atleast_1d(w)), bool)
        i = np.clip(idx, 0, self.n_slots - 1)
        within[ok] = (w[ok] - self.ts[i[ok]]) < span
        return np.where(ok & within, idx, -1)

    def describe(self) -> str:
        return (f"Timeline: {self.n_slots} slots of {self.slot_minutes} min "
                f"over {self.n_days} days ({self.dates[0]} .. {self.dates[-1]}), "
                f"history/current/future = "
                f"{int((self.phase==HISTORY).sum())}/"
                f"{int((self.phase==CURRENT).sum())}/"
                f"{int((self.phase==FUTURE).sum())} slots")


def build_timeline(cfg: Config, start: str | None = None,
                   end: str | None = None, n_days: int | None = None) -> Timeline:
    """Build the timeline for the test horizon (or any window).

    `n_days` truncates the horizon, which is how the Fig. 4 / Fig. 5 stability
    curves are produced (fairness as the horizon grows from 1 to 7 days).
    """
    p = cfg.protocol
    start = start or p.test_start
    end = end or p.test_end
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(t0, t1 - pd.Timedelta(days=1), freq="D")
    if n_days is not None:
        days = days[:n_days]

    step = pd.Timedelta(minutes=cfg.time.slot_minutes)
    ts_list, day_list = [], []
    for di, d0 in enumerate(days):
        if p.protocol == "paper_peak2h":
            lo = d0 + pd.Timedelta(hours=p.peak_hour_start)
            hi = d0 + pd.Timedelta(hours=p.peak_hour_end)
        else:
            lo, hi = d0, d0 + pd.Timedelta(days=1)
        cur = lo
        while cur < hi:
            ts_list.append(cur)
            day_list.append(di)
            cur = cur + step

    ts = np.array([t.to_datetime64() for t in ts_list], dtype="datetime64[m]")
    day = np.asarray(day_list, np.int32)
    tsp = pd.DatetimeIndex(ts)
    hour_index = ((ts.astype("datetime64[h]") - np.datetime64(EPOCH, "h"))
                  .astype(np.int32))
    profile = (np.asarray(tsp.dayofweek >= 5, np.int16) * 24
               + np.asarray(tsp.hour, np.int16))

    phase = np.full(len(ts), FUTURE, np.int8)
    n_hist, n_cur = p.n_history_days, p.n_current_days
    phase[day < n_hist] = HISTORY
    phase[(day >= n_hist) & (day < n_hist + n_cur)] = CURRENT

    return Timeline(ts=ts, slot_minutes=cfg.time.slot_minutes,
                    hour_index=hour_index, profile=profile, day=day,
                    phase=phase,
                    dates=np.array([d.strftime("%Y-%m-%d") for d in days]))
