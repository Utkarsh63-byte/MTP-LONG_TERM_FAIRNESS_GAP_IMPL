# Long-term Fairness in Ride-Hailing: Identifying and Closing Two Gaps

**MTP progress report**

**Base paper:** Y. Kang, J. Chan, W. Shao, F. D. Salim, C. Leckie, *"Long-term Fairness in Ride-Hailing Platform"*, ECML PKDD 2024 ([arXiv:2407.17839](https://arxiv.org/abs/2407.17839)), pp. 217–233.

**Dataset:** NYC TLC Yellow Taxi trip records, March 2016 (`yellow_tripdata_2016-03.csv`, 1.9 GB, 12,210,952 rows).

---

## 0. Executive summary

We identified two gaps in the base paper, quantified both on the paper's own dataset, implemented fixes, and validated them against five baselines.

| Finding | Number |
|---|---|
| The paper values a trip identically at 04:00 and 18:00, yet the same OD pair takes | **2.11× longer** at peak (median; max 4.35×) |
| Because of that, the paper's utility gets the **sign wrong** on | **10.2%** of (driver, request, time) cases |
| The paper's fairness metric `Var(total earnings)` hides an hourly-rate gap of | **7.8×** |
| Every one of the 5 prior methods pays part-timers more per hour than full-timers | **1.62–1.91×** |
| Switching the paper's fairness term ON widens the between-group gap by | **90×** |
| Our method brings the part-time/full-time hourly-rate ratio to | **0.97** (parity = 1.00) |
| Cost in total utility | **−0.7%** |
| Drivers who got no work all week: paper → ours | **1 → 0** |

**One sentence:** the paper equalises *total* earnings, which is the wrong target — it is blind to hours worked, it actively creates a full-time/part-time pay gap, and it never checks whether an available driver got any work. We re-target the objective at *hourly rate* (split within/between groups) and *utilisation*, achieving group parity for 0.7% of total utility.

---

## 1. The problem and what the base paper does

A ride-hailing platform must decide which incoming request goes to which driver. Two objectives conflict: **efficiency** (maximise total driver utility) and **fairness** (distribute earnings evenly). Kang et al. model this as a Markov Decision Process solved by multi-objective multi-agent Q-learning (MOMAQL), with a request forecaster feeding predicted demand into the action space.

### Their formulation

| Quantity | Definition | Ref |
|---|---|---|
| Request | `r = (t_r, s_r, d_r)` — raised at time `t_r`, from node `s_r` to `d_r` | Sec. 3.1 |
| Driver state | `v^t = (c_v, m_v^t, g_v^t, o_v^t)` — capacity, riders onboard, node, cumulative utility | Sec. 3.1 |
| **Utility** | `U(r,v) = Geo(d_r, s_r) − Geo(s_r, g_v^t)` | Sec. 3.1 |
| **Efficiency** | `π(M) = Σ_v o_v^{t_n}(M(v))` | Eq. 1 |
| **Long-term fairness** | `F(M) = Var(o_v^{t_n}(M(v)))` | Eq. 2 |
| Combined objective | `max_M π(M) − λ·F(M)` | Eq. 3 |
| Reward | `r^t_{s,A(v)} = Σ_{a∈A_v^t} Geo(d_a,s_a) − Geo(s_a,g_v^t)` | Eq. 6 |
| Scalarisation | `SR(M) = Σ_v r_{s,A(v)} − λω·Var(r_{s,A(v)})` | Eq. 7 |
| Normalised fairness | `F̂(M) = σ(U)/Ū` | Eq. 8 |

`Geo` = shortest road distance. Hyperparameters: `λ = 1, ω = 0.6, γ = 0.9`. Horizon = 1 week, split 3 days history / 1 day current / 3 days future (Sec. 3.3).

**In words:** utility = trip distance minus the distance the driver drives empty to reach the pickup. Fairness = how unequal the drivers' weekly totals are (variance; 0 means all equal).

---

## 2. The two gaps

### Gap 1a — Utility ignores time and traffic

The utility function contains only distance. Sec. 5.1 goes further and states that travel time per OD pair was *"re-calculated as the mean travel time across the time period"*.

**Example.** A 5 km trip at 04:00 and the same 5 km trip at 18:00 receive identical utility. In reality:

| | 04:00 | 18:00 |
|---|---|---|
| Same 5 km trip | ~12 min | ~26 min |

The driver's real cost — time — is invisible.

### Gap 1b — Fairness is blind to hours worked

`Var(o_v)` treats all drivers alike regardless of time online.

**Example.**

| Driver | Weekly earnings | Hours online | True hourly rate |
|---|---|---|---|
| A (full-time) | ₹6,000 | 60 h | **₹100/h** |
| B (part-time) | ₹6,000 | 15 h | **₹400/h** |

`Var = 0`, so the paper calls this *perfectly fair* while a 4× gap sits inside it.

**Fix.** Make utility time/traffic-aware (reusing the paper's own prediction module), split drivers into activity groups, and compare fairness *within* and *between* groups using hourly rate.

### Gap 2 — Utilisation / access fairness

The paper checks only final earnings, never whether an available driver was given work.

**Example.**

| Driver | Hours online | Trips assigned | Idle | Earnings |
|---|---|---|---|---|
| X | 8 h | 2 | **6 h (75%)** | ₹5,000 |
| Y | 8 h | 10 | 1 h | ₹5,000 |

Equal earnings, so Eq. 2 reports fairness — while X waited six hours with the app open and got nothing.

**Fix.** New metric `Utilisation ν_v = assigned slots / online slots`, and its variance across drivers and groups.

---

## 3. The dataset: everything we used and changed

### 3.1 What the raw file contains

`yellow_tripdata_2016-03.csv`, 1.9 GB, 12,210,952 rows, 19 columns. One row = one completed taxi trip.

| Column | Used? | Purpose |
|---|---|---|
| `tpep_pickup_datetime`, `tpep_dropoff_datetime` | **yes** | request time; duration = dropoff − pickup |
| `trip_distance` (miles) | **yes** | road distance → the `Geo` matrix |
| `pickup_longitude`, `pickup_latitude` | **yes** | origin node |
| `dropoff_longitude`, `dropoff_latitude` | **yes** | destination node |
| `fare_amount`, `tip_amount` | **yes** | money-denominated utility variant |
| `total_amount`, `passenger_count` | loaded | sanity profiling only |
| `VendorID`, `RatecodeID`, `store_and_fwd_flag`, `payment_type`, `extra`, `mta_tax`, `tolls_amount`, `improvement_surcharge` | no | irrelevant to allocation or fairness |

**Important property:** March 2016 predates the July 2016 format change, so raw latitude/longitude are present rather than pre-binned zone IDs. This is what lets us choose our own spatial granularity.

**Critical absence:** there is **no driver identifier**. The file records trips, not drivers. This single fact drives much of the methodology below — driver-side fairness cannot be measured without simulating drivers.

### 3.2 Change 1 — Cleaning and geographic filtering

Two streaming passes over the CSV in 1.5 M-row chunks (the file does not fit comfortably in memory).

| Filter | Threshold | Reason |
|---|---|---|
| Manhattan bounding box | lon [−74.02, −73.93], lat [40.70, 40.88] | The paper studies Manhattan (Sec. 5.1) |
| Trip distance | 0.1 – 40 miles | Removes 0 km trips and GPS errors (one row reads 5,000,000 miles) |
| Fare | \$2.50 – 300 | Removes negative fares (data-entry errors) |
| Duration | 1 – 120 min | Removes instant and multi-day records (some dropoffs are dated June for a March pickup) |
| Implied speed | 1 – 60 mph | Removes physically impossible trips |
| Date | within March 2016 | A few rows leak in from February |

**Result: 12,210,952 → 10,300,738 trips (84.4% retained).**

Output cached as `cache/trips.parquet` (139 MB) with a compact derived schema:

```
pickup_ts, dropoff_ts   timestamps
o, d                    int16   node IDs (see Change 2)
km                      float32 trip distance, converted miles → km
dur_min                 float32 observed duration
fare                    float32 fare_amount + tip_amount
is_weekend, hour, date  calendar keys
speed_kmh               float32 derived
```

### 3.3 Change 2 — Continuous coordinates → 87 graph nodes

**Problem.** Raw lat/lon gives ~10⁷ distinct locations. An MDP needs a finite state space, and the paper's `Geo(·,·)` operator needs a finite node set. The paper says only that *"multiple locations are merged together as a node"* — it never gives a count.

**Method.** Overlay a grid on Manhattan; each populated cell becomes a node. A node's coordinate is the **empirical mean** of the pickup/dropoff points inside it (not the geometric cell centre), so a node sits where the demand actually is. Cells with fewer than 2,000 endpoints in the month are dropped and their trips snapped to the nearest surviving node.

**Choosing the grid size.** Not a guess — measured. A cell only deserves to be a node if it carries enough trips to estimate a *time-dependent* travel time:

| Grid | Usable nodes | Share of trips in well-populated (OD, daytype, hour) cells |
|---|---|---|
| 0.83 km | 143 | 80.7% |
| 1.00 km | 108 | 88.2% |
| **1.11 km** | **87** | **≈90%** ← chosen |
| 1.50 km | 57 | 95.9% |
| 2.00 km | 32 | 98.3% |

Finer than ~1.1 km and the travel-time table becomes too sparse to estimate (Gap 1a would be unsupportable); coarser and the geography washes out. **87 nodes retain 99.94% of all trip endpoints.**

### 3.4 Change 3 — Building the `Geo` distance matrix (87 × 87)

The paper assumes a `Geo` operator but never says how it was obtained.

**Method.** Estimate it from the data itself:
1. **Direct:** median observed `trip_distance` for each ordered node pair. 80.8% of the 7,569 possible pairs are directly observed, covering **99.98% of all trips**.
2. **Fallback** for unobserved pairs: rotated-L1 distance. Manhattan's street grid runs ~29° east of true north, so rotating the coordinate frame and taking an L1 (city-block) distance approximates driving distance far better than a straight line. Scale calibrated by weighted least squares against the observed pairs — **MAPE 14%**.
3. Diagonal = empirical median distance of trips that start and end inside one node (~0.97 km), so same-node deadhead is not free.

**Validation.** Implied circuity (road ÷ great-circle distance) = **1.320**, matching the independently known NYC value of ≈1.3. Median OD distance 7.21 km. The matrix is genuinely asymmetric (max |d(a,b) − d(b,a)| = 7.90 km), which justifies the paper's directed-graph formulation.

**A design decision we made and rejected.** We implemented a Floyd–Warshall metric closure (to enforce "shortest" path consistency) but **do not apply it**: on a complete graph of noisy medians, the minimum over ~87 candidate paths is biased downward, shortening 74% of pairs by a median 1.11 km — destroying real measured distances. Triangle-inequality violations are reported as a diagnostic instead. Cached as `cache/distance.npz`.

### 3.5 Change 4 — Building the travel-time tensor (87 × 87 × 48) — **this is Gap 1a**

**What the paper does.** One travel-time number per OD pair (the period mean).

**What we do.** 48 numbers per OD pair: {weekday, weekend} × 24 hours of the day.

**Estimation** with three fallback levels, and the level used is recorded for every cell so results can be re-checked on well-supported cells only:

| Level | Rule | Requirement |
|---|---|---|
| L0 | direct median of (o, d, profile) | ≥ 30 trips |
| L1 | pair's own mean × citywide profile multiplier | pair observed |
| L2 | `Geo(o,d)` ÷ citywide speed for that profile | always |

L0 covers 13.5% of cells but **90.7% of actual trips** (dense routes are measured directly; rare pairs degrade to "average congestion at this hour", never to "no congestion").

Two derived objects:

```
τ̄[o,d]                  period-level travel time      ← the paper's version
τ[o,d,p]                 profile-specific travel time  ← ours
c[o,d,p] = τ / τ̄         congestion multiplier         ← the knob for Gap 1a
```

**What this revealed (all from the paper's own data):**

| Measurement | Value |
|---|---|
| Citywide speed, weekday 04:00 vs 12:00 | 28.5 vs 12.8 km/h = **2.22×** |
| Same OD pair, peak ÷ off-peak travel time (969 pairs measured across ≥24 profiles) | median **2.11×**, p90 2.93×, max **4.35×** |

The paper's period-mean collapse deletes a 2.11× effect.

**Leakage control.** Three versions are built: `traveltime.npz` (whole month — the simulator's physics), `traveltime_train.npz` (before the test week — features/targets for the congestion predictor), `traveltime_test.npz` (test week only — held-out ground truth for scoring). A policy never reads future congestion; it reads a prediction.

### 3.6 Change 5 — Demand tensor (87 × 87 × 744)

Request counts per OD pair per hour of the month (31 days × 24 h = 744 hours), used by the forecaster and to place drivers where demand is. Totals reconcile exactly with the trip count (10,300,738). Cached as `cache/demand.npz`.

### 3.7 Change 6 — Simulating 200 drivers (the dataset has none)

**Why this is unavoidable.** The feed has no driver ID, so driver-side fairness cannot be measured from it. Every paper in this line simulates drivers. Kang et al. never report the driver count, the shift model, or any notion of hours worked — **and that omission is precisely Gap 1b**: without hours `H_v` there is no hourly rate and no activity group.

**Our fleet.**

| Property | Value |
|---|---|
| Drivers | 200 |
| Full-time | 76 (38%), mean **46.4 h/week** |
| Part-time | 124 (62%), mean **14.0 h/week** |
| Group separation | **3.32×** in hours |
| Shift archetypes | morning / day / evening / night, held fixed per driver |
| Shifts | 825 total, mean 6.0 h, contiguous |
| Total online driver-slots | 63,120 |
| Starting node | 75% sampled from the empirical pickup distribution, 25% uniform |

Shifts are contiguous blocks on the absolute time axis, so a 23:00 shift correctly runs into the next morning. Part-timers are skewed toward weekends, full-timers toward weekdays, matching survey evidence that part-timers are the majority of rideshare drivers.

### 3.8 Change 7 — Decision epoch: 1 hour → 5 minutes

The paper's timestep is 1 hour. The mean Manhattan trip is 11.3 min, so at an hourly step the question *"was this driver busy during this slot?"* has no meaningful answer — and that is exactly the question Gap 2 asks. We decide every **5 minutes** (2,016 slots over the 7-day horizon), while keeping the forecaster at the paper's 1-hour granularity.

### 3.9 Change 8 — Timeline: peak 2-hour window → full day

The paper extracts a peak 2-hour window per day (Sec. 5.2). That caps a driver at **14 h/week**, making a 40 h full-time driver impossible — so Gap 1b could not be tested at all.

We use a full-day timeline. This is not an assumption; we ran the paper's protocol to confirm:

| Under the paper's protocol | Result |
|---|---|
| Max driver hours | 14.0 h/week |
| Drivers reaching the 40 h full-time threshold | **0 of 200** |
| Service rate | 2.6–3.0% |
| Drivers idle all week | 99–106 of 200 |
| Utilisation | pinned at 0.99 |

The paper's own setup structurally cannot measure either gap.

**Test week.** The paper tests 26/03–01/04, but 01/04 lives in the April file we do not have, and a 6-day window breaks the 3/1/3 split. We shift one day earlier: **25–31 March** (25, 26, 27 history | 28 current | 29, 30, 31 future). Good Friday (25/03) and Easter Sunday (27/03, the month's lowest-volume day at 278,805 trips) fall in the history segment, so the forecaster trains on a clean week and must generalise across a holiday-contaminated history — the "concept drift" the paper argues for.

### 3.10 Change 9 — Request sampling rate: 0.05 → calibrated 0.00296

The paper thins its data with a stratified 0.05 sample. On a full day that rate would swamp 200 drivers. We stratify by (date × hour) — preserving the real diurnal shape — and **solve** for the rate rather than fixing it:

| Quantity | Value |
|---|---|
| Online driver-slots | 63,120 |
| Mean slots consumed per trip (measured) | 5.74 |
| **Fleet capacity for the week** | **11,006 trips** |
| Requests offered | 6,604 |
| **Offered load = demand ÷ capacity** | **0.600** |
| Un-thinned test-week trips available | 2,231,191 |

Using all 2,231,191 would be **203× oversubscribed**: every driver busy every minute, utilisation pinned at 1.0, and `Var(utilisation)` — the entire Gap 2 metric — identically zero for every method. The rate is therefore solved so offered load = 0.60, the regime where fairness is genuinely contested.

### 3.11 Dataset usage summary

| Component | Trips used | Share of cleaned data |
|---|---|---|
| Node / graph construction | 10,300,738 | **100%** |
| `Geo` distance matrix | 10,300,738 | **100%** |
| Travel-time tensor (48 profiles) | 10,300,738 | **100%** |
| Demand tensor | 10,300,738 | **100%** |
| Forecaster training | 1,656,480 samples | 4,640 OD pairs × 357 hours |
| Allocation simulation | 6,604 requests | calibrated sample of the test week |

**100% of the cleaned data builds every model component.** Only the allocation *experiment* runs on a thinned stream — the same practice as the paper.

---

## 4. Methodology by phase

### Phase 1 — Data pipeline
Sections 3.2–3.6 above. Verified by 25 assertions in `scripts/verify_phase1.py` (all pass), including a numerical check that our new utility reduces **exactly** to the paper's when congestion is switched off.

### Phase 2 — Driver simulator
Section 3.7. Verified by `scripts/verify_phase2.py` (all pass), which also demonstrates the Gap 1b blind spot on the real fleet: equalising totals gives `Var(total) = 0` while hourly rates span **47.9 to 375.7 (7.8×)**, with part-timers on 3.59× the full-time rate.

### Phase 3 — Assignment environment
Slot-by-slot simulation. **One repair to the paper's model was necessary.** Sec. 4.3 allows a driver to *"accept multiple requests concurrently"* and its utility has no time dimension, so nothing bounds how much work one driver absorbs. Here an assigned driver is occupied for `τ_t(g_v,s_r) + τ_t(s_r,d_r)` slots and unavailable until the trip completes. Without this, "busy" is undefined and Gap 2 cannot exist.

Records per driver: cumulative utility `o_v`, busy slots, online slots, trip count — plus a full assignment log, reconciled by assertion.

### Phase 4 — Gap 1a: time-aware utility

```
c_t(a,b) = τ_t(a,b) / τ̄(a,b)                              (congestion multiplier)

U_t(r,v) = Geo(d_r,s_r) / c_t(s_r,d_r)  −  w · Geo(s_r,g_v) · c_t(g_v,s_r)
```

**Reading:** congested loaded distance delivers less value per unit of the driver's clock; congested deadhead costs more. Both terms move in the direction that hurts — an asymmetry the paper's formula cannot express.

**Design property:** at `c ≡ 1` this is *identically* the paper's Sec. 3.1 utility. Verified numerically: max |ΔU| = 0.00e+00, max slot difference = 0. This makes the ablation clean — the paper is a strict special case.

**Prediction, not leakage.** The forecaster gains a **second head predicting `c_t`**, sharing its trunk with the demand head (exactly what the gap document asked for: reuse the existing prediction module). Gap 1a needs congestion for *future* slots; the head is trained only on pre-test-week congestion.

### Phase 5 — Gap 1b: rate fairness with group decomposition

```
ρ_v = o_v / H_v                                            (hourly rate)

Var(ρ) = E_g[Var(ρ|g)]   +   Var_g(E[ρ|g])
         └─── within ───┘     └─── between ───┘            (law of total variance)
```

The within/between split follows from an **identity**, not an ad-hoc construction; `within + between = total` is asserted in code. Drivers with fewer than 2 online hours are excluded from rate statistics (one lucky trip in 20 minutes is not a "rate").

### Phase 6 — Gap 2: utilisation fairness

```
ν_v = busy slots / online slots
```

Reported **raw and opportunity-normalised**. The raw form conflates two different things: a platform that starves a driver, versus a driver who chose to be online at 04:00 when there is no demand. The normalised form divides each driver's utilisation by what drivers online in the *same slots* achieved:

```
opportunity_v = mean over v's online slots of (busy drivers / online drivers)
ν_adj,v       = ν_v / opportunity_v
```

`ν_adj = 1` means the driver did as well as a typical peer working the same hours; below 1 means the algorithm under-served them. Both are reported — raw alone overstates the unfairness, normalised alone would hide genuine starvation.

### Phase 7 — Unified objective

```
SR(M) = Σ_v u_v  −  ω_ρ[λ_w·Var_within(ρ) + λ_b·Var_between(ρ)]  −  ω_ν·λ_ν·Var(ν)
```

Two changes beyond metrics:

**1. MDP state augmentation.** A policy cannot optimise a fairness term it cannot observe, so the paper's location-only state is augmented with discretised rate-deficit and utilisation-deficit buckets: `V[profile, node, deficit]` = 48 × 87 × 5 = 20,880 entries. Scoring:

```
score(i,j) = r_scalarised(i,j) + γ^k · V[s'] − V[s]
```

with `k` the occupancy in slots, so a long trip is discounted for the time it locks the driver up.

**2. ω is calibrated, not hard-coded.** The paper fixes `ω = 0.6` "to scale utility and fairness into the same range" without reporting what range. It does not transfer: a variance carries squared units, so its magnitude relative to utility depends on fleet size, horizon, and *which* quantity the variance is over. Left uncalibrated, our utilisation penalty came out ≈0.009 against a typical trip utility of 2.4 — roughly **270 times too small**, leaving the Gap 2 term inert and its ablation showing no effect. Calibrating from a measured efficiency-only run gives `ω_total = 1.64, ω_ρ = 1.11e3, ω_ν = 5.96e4`. After this fix all ablations behave correctly.

### Phase 8 — Baselines, sweeps, experiments

Five baselines, each implemented to match how the paper describes it:

| Baseline | Mechanism |
|---|---|
| Greedy | max-gain matching on Eq. 3, recomputing the fairness marginal after every pick |
| REASSIGN (Lesmana et al.) | two-stage: efficient matching, then reassign to lift the worst-off driver within a bounded efficiency loss |
| LAF (Shi et al.) | MDP re-weights the edges, then Hungarian algorithm solves the matching optimally |
| Balance Ride-Pooling (Raman et al.) | RL with fairness on totals, **no** future-demand module |
| Kang et al. (reproduced) | RL with fairness on totals **plus** predicted requests in the action space |

Forecaster: 3-layer MLP, two heads, trained on MPS (Apple GPU); CUDA auto-detected when available. Leakage avoided by using only seasonal lags ≥ 168 h, so every test-week feature predates the forecast origin.

---

## 5. Results

Setup: 87 nodes, 200 drivers, 7-day horizon (25–31 March 2016), 6,604 requests, offered load 0.602, utility in km-equivalent units.

### Table 1 — the paper's own metrics

| Method | Total Utility | Var(total) ↓ | Norm. Fair ↓ | Min | Mean | Max |
|---|---|---|---|---|---|---|
| Greedy | 11,708 | 701 | 0.4522 | 0.00 | 58.54 | 118.87 |
| REASSIGN | 11,437 | 869 | 0.5156 | 0.00 | 57.19 | 126.13 |
| LAF | 11,704 | 635 | 0.4305 | 0.00 | 58.52 | 121.08 |
| Balance Ride-Pooling | 10,373 | 671 | 0.4996 | −1.11 | 51.86 | 107.94 |
| Kang et al. (reproduced) | 10,221 | 657 | 0.5016 | 0.00 | 51.11 | 108.96 |
| **Ours (Gap 1+2)** | 10,145 | 1,639 | 0.7982 | 0.37 | 50.72 | 172.10 |

Our `Var(total)` is deliberately worse. **Equal hourly rates across unequal hours mathematically implies unequal totals** — Eq. 2 and rate fairness cannot both be satisfied. This incompatibility is the central finding, not an error.

### Table 2 — the gap metrics, identical runs

| Method | Var(ρ) ↓ | within ↓ | between ↓ | FT /h | PT /h | ratio → 1.0 | Var(ν) ↓ | Var(ν) norm ↓ | idle ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | 1.809 | 1.279 | 0.530 | 1.78 | 3.28 | 1.84 | 0.0202 | 0.4262 | 1 |
| REASSIGN | 1.439 | 1.133 | 0.306 | 1.83 | 2.97 | 1.62 | 0.0214 | 0.4012 | 2 |
| LAF | 1.641 | 1.044 | 0.597 | 1.75 | 3.34 | 1.91 | 0.0216 | 0.4390 | 1 |
| Balance Ride-Pooling | 1.238 | 0.986 | 0.252 | 1.66 | 2.69 | 1.62 | 0.0140 | 0.2747 | 0 |
| Kang et al. (reproduced) | 1.251 | 1.004 | 0.247 | 1.64 | 2.67 | 1.62 | 0.0129 | 0.2527 | 1 |
| **Ours (Gap 1+2)** | **0.446** | **0.445** | **0.001** | 1.93 | 1.86 | **0.97** | **0.0081** | **0.1550** | **0** |

**Every prior method sits at 1.62–1.91**: part-timers earn 62–91% more per hour than full-timers. This is systematic across all five, because all five optimise `Var(totals)`.

### Table 3 — headline comparison

| Metric | Kang (repro) | Ours | Change |
|---|---|---|---|
| Total utility | 10,221 | 10,145 | **−0.7%** |
| Var(hourly rate) | 1.251 | 0.446 | **−64.4%** |
| within-group | 1.004 | 0.445 | −55.7% |
| between-group | 0.247 | 0.001 | **−99.6%** |
| Group ratio (parity = 1.0) | 1.62 | **0.97** | — |
| Full-time rate /h | 1.64 | 1.93 | +17.4% |
| Part-time rate /h | 2.67 | 1.86 | −30.2% |
| Var(utilisation) | 0.0129 | 0.0081 | −37.7% |
| Var(utilisation), normalised | 0.2527 | 0.1550 | −38.6% |
| Idle drivers (whole week) | 1 | **0** | −100% |
| Gini(hourly rate) | 0.258 | 0.198 | −23.4% |

### Table 4 — the paper's objective *causes* the harm its metric cannot see

| Configuration | FT rate /h | PT rate /h | between-group Var |
|---|---|---|---|
| No fairness term | 2.28 | 2.42 | 0.004 |
| Paper's fairness ON (Eq. 3) | **1.89** | **3.13** | **0.361 (90× worse)** |

Levelling totals across unequal hours can only be done by transferring rate from long-hours to short-hours drivers. This upgrades Gap 1b from "the metric is blind" to "the objective is harmful".

### Table 5 — ablations

| Configuration | Total Utility | Var(ρ) | between | Var(ν) | Var(ν) norm | idle |
|---|---|---|---|---|---|---|
| Ours, static utility (`c ≡ 1`) | 10,049 | 0.252 | 0.0023 | 0.0069 | 0.1332 | 2 |
| Ours, w/o fairness | 11,365 | 1.284 | 0.0062 | 0.0147 | 0.2484 | 4 |
| Ours, w/o prediction | 10,290 | 0.406 | 0.0012 | 0.0072 | 0.1246 | 0 |
| Ours, w/o utilisation term | 10,191 | 0.426 | 0.0005 | 0.0123 | 0.2224 | 0 |
| Kang et al. (fairness on totals) | 10,221 | 1.251 | 0.2473 | 0.0129 | 0.2527 | 1 |
| **Ours (full)** | 10,145 | 0.446 | 0.0011 | **0.0081** | **0.1550** | 0 |

The Gap 2 term does real work: removing it degrades `Var(ν)` from 0.0081 → 0.0123 (**+52%**) and the normalised form from 0.155 → 0.222.

### Table 6 — horizon stability (the paper's Fig. 4 view)

Between-group inequity as the horizon grows:

| Days | Kang (repro) | Ours |
|---|---|---|
| 1 | 0.0150 | 0.0004 |
| 3 | 0.1575 | 0.0011 |
| 5 | 0.1564 | 0.0021 |
| 7 | **0.2473** | **0.0011** |

The paper's group inequity **grows 16.5×** over the week; ours stays flat, two orders of magnitude lower. The paper claims *long-term fairness* — on the group metric it is long-term *unfairness* that accumulates.

### Table 7 — weight sweep and Pareto analysis

Coordinate search over 20 configurations, with **both gap terms active in every one** (enforced by assertion):

| λ_between | PT/FT ratio | between-group Var | Total Utility |
|---|---|---|---|
| 0.25 | 0.614 | 0.20057 | 11,011 |
| 1.00 | 0.761 | 0.06853 | 10,954 |
| 4.00 | 0.900 | 0.01071 | 10,901 |
| 16.00 | **0.955** | **0.00207** | 10,883 |
| *(paper)* | *1.624* | *0.24733* | *10,221* |

**12 of the 20 configurations beat the paper method on all six metrics simultaneously** (utility, Var(ρ), between-group, Var(ν), normalised Var(ν), idle drivers). The improvement is therefore not bought by sacrificing efficiency.

### Table 8 — robustness

**Money-denominated utility** (fare = \$3.95/km, calibrated from the data):

| | Kang (repro) | Ours |
|---|---|---|
| Full-time | \$6.62/h | **\$8.20/h** |
| Part-time | \$12.35/h | **\$7.93/h** |
| Ratio | 1.865 | **0.968** |
| between-group Var | 7.727 | **0.016** |

The gap survives the change of units, so it is a property of the allocation, not of distance-denominated utility.

**Seed sensitivity** (3 independent seeds; fleet, requests and RL exploration all re-drawn):

| | Total Utility | Var(ρ) | between | Group ratio | Var(ν) norm |
|---|---|---|---|---|---|
| Kang (repro) | 10,850 | 1.814 | 0.4602 | 1.838 ± 0.015 | 0.306 |
| **Ours** | 10,726 | **0.404** | **0.0012** | **0.965 ± 0.010** | **0.137** |

The paper's group ratio is 1.83–1.86 on *every* seed — systematic, not noise.

**Forecaster:**

| | Value |
|---|---|
| Demand MSE (counts) | 6.455 |
| Seasonal-naive baseline | 8.856 |
| Improvement over naive | **27.1%** |
| Congestion head MSE / MAE | 0.01198 / 0.0826 |

---

## 6. Why our absolute utility differs from the paper's Table 1

The paper reports total utility 95,823.79; ours is 10,145. **This is not because we use less data** (Section 3.11: 100% of the cleaned data builds every component).

### 6.1 The paper used 20 drivers, not 200 — recovered from Table 1

The paper never states its driver count. It is recoverable, because Total ÷ Mean = *n*:

| Method | Total | Mean | Implied *n* |
|---|---|---|---|
| Greedy | −1,514,736.24 | −75,736.81 | **20.000** |
| REASSIGN | 76,536.23 | 3,826.81 | **20.000** |
| LAF | 80,606.49 | 4,030.3245 | **20.000** |
| Balance Ride-Pooling | 85,923.68 | 4,296.18 | **20.000** |
| Proposed Method | 95,823.79 | 4,791.19 | **20.000** |

Exactly 20 on every row. With a 10× smaller fleet, the same workload concentrates into far fewer drivers, so per-driver means are much larger.

### 6.2 The paper's per-driver utility exceeds physical capacity by ~36×

Mean utility per driver per week = 4,791.19. At a realistic net utility of 2.49 km per trip (our measured value on the same city), that requires **≈1,924 trips per driver per week**.

But the paper's protocol allows a driver at most **14 h/week**. With a mean Manhattan trip of 11.3 min plus ~4.5 min deadhead, one driver can complete at most **≈53 trips**.

```
required 1,924 trips   vs   physically possible 53 trips   =   36× over capacity
```

This follows from the paper's own choice (Sec. 4.3): drivers *"accept multiple requests concurrently"* with no time dimension in the utility. Their totals aggregate trips a real driver could not perform. Our occupancy constraint is the main reason our totals are an order of magnitude smaller — and that constraint is what makes Gap 2 measurable at all.

### 6.3 Consequence

Absolute utilities are **not comparable** across the two papers, because *n*, the concurrency assumption and the node count all differ — and two of the three are unreported. Every conclusion here comes from comparisons **inside one harness**, where all six methods see identical drivers, requests, distances and travel times.

---

## 7. Limitations (reported, not hidden)

1. **The paper's prediction module gives no measurable benefit here.** Removing it *raises* utility (10,145 → 10,290) and marginally improves `Var(ρ)` (0.446 → 0.406). We cannot reproduce the paper's Table 2 claim of a 41% utility collapse without prediction (95,824 → 56,873). Likely cause: a tabular value function already observes all 48 time profiles in the real training week.
2. **Gap 1a's end-to-end efficiency gain is small** (~1%). Its real contribution is measurement correctness — 10.2% of utility signs flip. The `c ≡ 1` ablation even shows lower `Var(ρ)`; reported as-is.
3. `Var(total)` is worse for our method **by construction** (Section 5, Table 1).
4. Service rate is ~2 points below the paper method (62.7% vs 64.6%) — a real, small cost.
5. Group parity slightly overshoots (0.97 rather than 1.00); `λ_between` tunes this and the full curve is in Table 7.
6. The coordinate search does not certify a global optimum; it finds a good operating point and traces the trade-off curves.

---

## 8. Contributions

1. **Quantified both gaps on the paper's own dataset** rather than arguing them: 2.11× median within-OD travel-time variation, 10.2% utility sign flips, a 7.8× hidden hourly-rate gap.
2. **Showed the paper's objective causes the harm it cannot measure** — between-group rate variance rises 90× when its fairness term is switched on, and all five prior methods land at 1.62–1.91× group ratio.
3. **A group-decomposed rate-fairness objective** derived from the law of total variance, plus a utilisation objective with an opportunity-normalised variant that separates algorithmic starvation from driver shift choice.
4. **A time-aware utility that provably reduces to the paper's** at `c ≡ 1`, with congestion *predicted* by a second head on the paper's own forecasting module rather than read from ground truth.
5. **Demonstrated `ω = 0.6` does not transfer** and replaced it with measured calibration — without which the utilisation term is inert (roughly 270 times too small).
6. **Recovered the paper's unreported fleet size (n = 20) from its own Table 1** and showed its per-driver utility exceeds the physical capacity of its own protocol by ~36×.
7. **Showed the paper's protocol cannot measure either gap**: 0 of 200 drivers reach a full-time threshold, utilisation pins at 0.99.
8. **A reproducible harness**: 87-node graph from 10.3 M trips, 200-driver fleet, 5 baselines, 5 ablations, 20-point sweep, 3 seeds, 2 unit systems, both protocols.

---

## 9. How to reproduce

```bash
cd /Users/utkarshpandey/Downloads/Research_MTP

# instant (seconds): print every headline result from saved outputs
.venv/bin/python -m scripts.show_results

# instant: dataset usage accounting + why the paper's totals are larger
.venv/bin/python -m scripts.explain_scale

# verification: data pipeline (25 assertions) and driver fleet
.venv/bin/python -m scripts.verify_phase1
.venv/bin/python -m scripts.verify_phase2

# forecaster: demand head + congestion head            (~3 min, GPU)
.venv/bin/python -m scripts.run_forecaster

# main experiments: Tables 1-3, 5, 6                   (~25 min)
.venv/bin/python -m scripts.run_experiments

# weight sweep and Pareto fronts, both gaps active     (~60 min)
.venv/bin/python -m scripts.run_sweeps

# robustness: money units, paper protocol, seeds       (~60 min)
.venv/bin/python -m scripts.run_robustness

# figures
.venv/bin/python -m scripts.make_figures
```

### Code layout

```
ltf/config.py            all hyperparameters; Config.paper_faithful() reproduces the paper's setting
ltf/utility.py           UtilityModel: paper vs time_aware, distance vs money, reduction self-test
ltf/data/prepare.py      CSV -> 87 nodes + trips.parquet
ltf/data/graph.py        Geo distance matrix
ltf/data/traveltime.py   48-profile travel-time tensor + congestion multiplier
ltf/data/demand.py       demand tensor
ltf/sim/timeline.py      decision epochs, 3/1/3 phase split
ltf/sim/drivers.py       200-driver fleet, shifts, activity groups
ltf/sim/requests.py      stratified sampling, offered-load calibration
ltf/sim/env.py           occupancy-aware assignment environment
ltf/metrics/fairness.py  paper's metrics + Gap 1b + Gap 2, side by side
ltf/methods/            greedy, REASSIGN, LAF, Balance Ride-Pooling, MOMAQL
ltf/predict/            MLP forecaster, two heads
```

### Outputs

`outputs/`: `table1_methods.csv`, `table2_ablations.csv`, `fig4_horizon.csv`, `sweep_joint_grid.csv`, `robustness.csv`, `forecaster_results.json`, plus figures `fig_gaps_by_method.png`, `fig_parity_sweep.png`, `fig_gap1a_congestion.png`, `fig_sweep_pareto.png`, `fig_tradeoff.png`, `fig4_horizon.png`.

**Recommended figure for a first look:** `fig_gaps_by_method.png` — three panels showing hourly rate by group, between-group inequity, and utilisation fairness across all six methods.
