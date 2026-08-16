"""Central configuration for the long-term-fairness (LTF) gap implementation.

Everything tunable lives here so experiments are reproducible from a single
object. Defaults reflect the decisions agreed with the user:
    * ~90 graph nodes (1.5 km grid over Manhattan)
    * n = 200 simulated drivers
    * distance-denominated utility (primary), fare-denominated (robustness)
    * full-time / part-time driver groups, 40 / 60 mix

Reference paper
---------------
Kang, Chan, Shao, Salim, Leckie.
"Long-term Fairness in Ride-Hailing Platform", ECML PKDD 2024
(arXiv:2407.17839). Section/equation numbers quoted throughout the codebase
refer to that paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import json

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "yellow_tripdata_2016-03.csv"
CACHE = ROOT / "cache"
OUTPUT = ROOT / "outputs"


# --------------------------------------------------------------------------
# Data / graph
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Filtering and graph-construction settings."""

    # Manhattan bounding box. 2016-03 predates the switch to zone-IDs, so
    # raw lon/lat are available and we can pick our own granularity.
    lon_min: float = -74.02
    lon_max: float = -73.93
    lat_min: float = 40.70
    lat_max: float = 40.88

    # sanity filters (measured: these keep 84.4% of the 12.21M raw trips)
    min_trip_miles: float = 0.1
    max_trip_miles: float = 40.0
    min_fare: float = 2.5
    max_fare: float = 300.0
    min_duration_min: float = 1.0
    max_duration_min: float = 120.0
    min_speed_mph: float = 1.0
    max_speed_mph: float = 60.0

    # Node granularity, chosen to land at ~90 *usable* nodes (a cell only
    # earns a node if it carries enough trips to estimate travel times).
    # Measured (check_nodecount.py), trip-weighted share of trips falling in
    # (OD, daytype, hour) buckets with n >= 30:
    #   0.0075 (~0.83 km): 143 nodes, 80.7% of trips
    #   0.0090 (~1.00 km): 108 nodes, 88.2%
    #   0.0100 (~1.11 km):  ~90 nodes            <- chosen
    #   0.0135 (~1.50 km):  57 nodes, 95.9%
    #   0.0180 (~2.00 km):  32 nodes, 98.3%
    # Finer than this and the travel-time table is too sparse to support the
    # time-aware utility of Gap 1a; coarser and the geography washes out.
    grid_deg: float = 0.0100
    # drop cells with fewer than this many pickups+dropoffs in the month;
    # their trips are snapped to the nearest surviving node centroid.
    min_cell_trips: int = 2000

    # street-grid rotation used by the fallback road-distance model.
    # Manhattan's avenue grid runs ~29 deg east of true north.
    manhattan_rotation_deg: float = 29.0


# --------------------------------------------------------------------------
# Time discretisation
# --------------------------------------------------------------------------
@dataclass
class TimeConfig:
    """Two distinct clocks.

    * `slot_minutes` is the simulator / MDP decision epoch. The paper uses a
      1-hour step, which is far coarser than the 12-min mean Manhattan trip
      and makes "was this driver busy in this slot?" ill-defined. Gap 2 needs
      that question answered, so we decide every 5 minutes.
    * `predict_hours` is the forecasting granularity, kept at the paper's
      1 hour so the prediction module stays comparable.
    """

    slot_minutes: int = 5
    predict_hours: int = 1

    # travel-time table keys: (is_weekend, hour) -> 48 profiles
    n_daytypes: int = 2
    n_hours: int = 24

    @property
    def slots_per_hour(self) -> int:
        return 60 // self.slot_minutes

    @property
    def slots_per_day(self) -> int:
        return 24 * self.slots_per_hour

    @property
    def n_time_profiles(self) -> int:
        return self.n_daytypes * self.n_hours


# --------------------------------------------------------------------------
# Experiment protocol (timeline splits)
# --------------------------------------------------------------------------
Protocol = Literal["fullday", "paper_peak2h"]


@dataclass
class ProtocolConfig:
    """Timeline splits.

    The paper (Sec. 5.2): train on data before 26/03, forecast 26/03-01/04,
    and extract a peak 2-hour slice from 19/03 onwards. Within the test week
    (Sec. 3.3) the first 3 days are history, day 4 is "current", the last
    3 days are the future to be allocated.

    `paper_peak2h` reproduces that exactly. `fullday` is the default for the
    gap experiments: a 2h/day window caps a driver at 14 h/week, which makes
    the 40 h vs 20 h full-time/part-time contrast of Gap 1b impossible to
    express.
    """

    protocol: Protocol = "fullday"

    # The paper tests on 26/03-01/04, but 01/04 lives in the April CSV which
    # we do not have; that window would give only 6 days and break the 3/1/3
    # split of Sec. 3.3. We shift one day earlier to keep a full 7-day horizon
    # inside March: 25-31/03, split 25,26,27 | 28 | 29,30,31.
    # Good Friday (25/03) and Easter Sunday (27/03, the month's lowest-volume
    # day at 278,805 trips) therefore land in the history segment, so the
    # forecaster is trained on a clean week and must generalise across a
    # holiday-contaminated history -- the "concept drift" the paper argues for.
    forecast_train_start: str = "2016-03-01"
    forecast_train_end: str = "2016-03-25"   # exclusive
    sim_train_start: str = "2016-03-18"
    sim_train_end: str = "2016-03-25"        # exclusive
    test_start: str = "2016-03-25"
    test_end: str = "2016-04-01"             # exclusive -> 25..31 Mar = 7 days

    # within the test week (Sec. 3.3 of the paper)
    n_history_days: int = 3   # 25, 26, 27 Mar
    n_current_days: int = 1   # 28 Mar
    n_future_days: int = 3    # 29, 30, 31 Mar

    # paper_peak2h only
    peak_hour_start: int = 18
    peak_hour_end: int = 20   # exclusive; measured busiest 2h in Manhattan

    # Demand thinning. The paper uses a stratified sample rate of 0.05 on a
    # 2-hour window. Over a full day that would swamp 200 drivers and pin
    # utilisation at 1.0, which would make Gap 2's metric degenerate, so
    # instead of hard-coding a rate we solve for the rate that puts *offered
    # load* at this target. Offered load = driver-slots the requests would
    # consume / driver-slots online, i.e. demand pressure assuming every
    # request is served. Realised utilisation is offered load times the
    # service rate and is what Gap 2 measures. Set `sample_rate` to override.
    target_offered_load: float = 0.60
    sample_rate: float | None = None
    paper_sample_rate: float = 0.05


# --------------------------------------------------------------------------
# Driver population
# --------------------------------------------------------------------------
@dataclass
class DriverConfig:
    """Driver population and shift model.

    The NYC yellow-taxi feed has no driver identifier, so the driver side is
    simulated in every paper in this line of work. The paper never reports
    n, the shift model, or any notion of hours worked. Making hours explicit
    is what turns Gap 1b (hourly rate) and Gap 2 (utilisation) into
    measurable quantities.
    """

    n_drivers: int = 200
    capacity: int = 1              # c_v in the paper's driver tuple
    seed: int = 20260812

    # group mix. Survey evidence is that part-timers are the majority of
    # rideshare drivers, so the population leans part-time.
    full_time_share: float = 0.40

    # weekly hours drawn per group (uniform integer hours within range)
    full_time_hours: tuple[int, int] = (40, 55)
    part_time_hours: tuple[int, int] = (8, 20)
    full_time_threshold_h: float = 40.0   # >= is full-time
    part_time_threshold_h: float = 20.0   # <= is part-time

    # shift structure: contiguous blocks so online time is realistic
    # (not scattered 5-minute fragments)
    max_shift_hours: float = 10.0
    min_shift_hours: float = 3.0

    # drivers cluster where demand is, but not perfectly
    start_from_demand: bool = True
    start_uniform_mix: float = 0.25   # fraction placed uniformly at random


# --------------------------------------------------------------------------
# Utility definition (Gap 1a)
# --------------------------------------------------------------------------
UtilityMode = Literal["paper", "time_aware"]
UtilityUnit = Literal["distance", "money"]


@dataclass
class UtilityConfig:
    """Utility definition.

    `paper` is Sec. 3.1 verbatim:
        U = Geo(d_r, s_r) - Geo(s_r, g_v)
    evaluated on a travel-distance graph whose travel times were flattened to
    a period mean (Sec. 5.1), i.e. explicitly time-invariant.

    `time_aware` is Gap 1a:
        U = Geo(d_r,s_r) / c_t(s_r,d_r)  -  Geo(s_r,g_v) * c_t(g_v,s_r)
        c_t(a,b) = tau_t(a,b) / tau_bar(a,b)
    Congested loaded miles deliver less value per unit clock time; congested
    deadhead costs more. At c == 1 this is *identically* the paper's utility,
    which is what makes the Phase-4 ablation clean.

    Measured motivation: holding origin and destination fixed, the median
    peak/off-peak travel-time ratio across well-populated OD pairs is 2.42x
    (max 4.60x). The paper's period-mean collapse discards that.
    """

    mode: UtilityMode = "time_aware"
    unit: UtilityUnit = "distance"

    # clamp the congestion multiplier so a thin travel-time bucket cannot
    # produce an absurd utility
    c_min: float = 0.5
    c_max: float = 3.0

    # deadhead weighting; 1.0 reproduces the paper's implicit 1:1 trade-off
    deadhead_weight: float = 1.0

    # a driver will not be sent more than this far to a pickup
    max_pickup_km: float = 5.0


# --------------------------------------------------------------------------
# Fairness definition (Gaps 1b + 2)
# --------------------------------------------------------------------------
FairnessTarget = Literal["total", "rate", "rate_grouped"]


@dataclass
class FairnessConfig:
    """Fairness definition.

    `total` is the paper's Eq. 2: Var over weekly *total* utility. Blind to
    hours worked -- a 60 h driver and a 15 h driver on equal totals score a
    perfect 0.

    `rate` replaces totals with the hourly rate rho_v = o_v / H_v.

    `rate_grouped` additionally splits Var(rho) by the law of total variance:
        Var(rho) = E_g[Var(rho|g)] + Var_g(E[rho|g])
                   \\_ within _/      \\_ between _/
    which is exactly the within/between contrast Gap 1b asks for, obtained
    from an identity rather than an ad-hoc construction.
    """

    target: FairnessTarget = "rate_grouped"

    # Scalarisation weights (generalising the paper's lambda and omega, Eq. 7).
    #
    # lambda_within / lambda_between / lambda_util below are the operating point
    # selected by the coordinate search in scripts/run_sweeps.py, which evaluated
    # 20 configurations with *both* gap terms active in every one. This point
    # beats the reproduced Kang et al. method on every metric at once -- total
    # utility, Var(hourly rate), between-group Var, Var(utilisation),
    # opportunity-normalised Var(utilisation) and idle drivers -- so it is a
    # dominating point rather than a trade-off pick. 12 of the 20 configurations
    # dominated on all six, so the result is not knife-edge.
    #
    # lambda_between = 16 looks large only because omega_rate is calibrated
    # against the *total* rate variance, of which the between-group component is
    # a small share; the weight has to be correspondingly larger to move it.
    lambda_fair: float = 1.0        # paper: lambda = 1
    omega: float = 0.6              # paper: omega = 0.6 (recalibrated at runtime)
    lambda_within: float = 1.0
    lambda_between: float = 16.0    # swept: 0.25 -> 32
    lambda_util: float = 0.5        # Gap 2 term; swept: 0.05 -> 2.0
    omega_rate: float | None = None   # auto-scaled if None
    omega_util: float | None = None   # auto-scaled if None

    # a driver needs at least this many online hours before their rate is
    # trusted, otherwise a 20-minute driver with one lucky trip dominates
    min_hours_for_rate: float = 2.0


# --------------------------------------------------------------------------
# Prediction module
# --------------------------------------------------------------------------
@dataclass
class PredictConfig:
    """Request-count forecaster (paper Sec. 4.2) plus a congestion head.

    The paper uses a 3-layer MLP over lagged per-OD-pair counts at 1-hour
    resolution, trained on one month, forecasting 7 days (reported MSE 94.69).
    We keep that and add a second head predicting the congestion multiplier
    c_t. Gap 1a needs c_t for *future* slots, and reading it from ground truth
    would leak. Sharing the trunk is also what the gap document asks for:
    reuse the existing demand-prediction module.
    """

    n_lags: int = 24
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 60
    patience: int = 8
    congestion_head: bool = True
    congestion_loss_weight: float = 1.0
    seed: int = 20260812
    # "cuda" / "mps" / "cpu"; None auto-detects so an unconfigured GPU today
    # and a configured one tomorrow both work with no code change.
    device: str | None = None


# --------------------------------------------------------------------------
# MOMAQL
# --------------------------------------------------------------------------
@dataclass
class RLConfig:
    """Multi-objective multi-agent Q-learning settings (paper Sec. 4.3)."""

    gamma: float = 0.9              # paper: gamma = 0.9
    alpha: float = 0.1
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 40
    episodes: int = 60
    seed: int = 20260812

    # State abstraction. The agent cannot optimise a fairness term it cannot
    # observe, so Phase 7 augments the paper's location-only state with
    # discretised rate-deficit and utilisation-deficit buckets.
    n_rate_buckets: int = 5
    n_util_buckets: int = 5
    use_rate_state: bool = True
    use_util_state: bool = True
    lookahead_slots: int = 12       # predicted-demand horizon in state


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    driver: DriverConfig = field(default_factory=DriverConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def paper_faithful(self) -> "Config":
        """A copy configured to reproduce Kang et al. as published."""
        import copy

        c = copy.deepcopy(self)
        c.protocol.protocol = "paper_peak2h"
        c.protocol.sample_rate = c.protocol.paper_sample_rate
        c.time.slot_minutes = 60
        c.utility.mode = "paper"
        c.fairness.target = "total"
        c.rl.use_rate_state = False
        c.rl.use_util_state = False
        c.predict.congestion_head = False
        return c

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))


def default_config() -> Config:
    return Config()
