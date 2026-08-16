# Equations: The Base Paper, Our Modifications, and Correctness Verification

**Companion to `REPORT.md`**

**Base paper:** Y. Kang, J. Chan, W. Shao, F. D. Salim, C. Leckie, *"Long-term Fairness in Ride-Hailing Platform"*, ECML PKDD 2024 ([arXiv:2407.17839](https://arxiv.org/abs/2407.17839)).

This document has four parts:

- **Part A** — every equation in the base paper, explained term by term.
- **Part B** — every equation we modified or introduced to address Gap 1 and Gap 2.
- **Part C** — numerical verification of each equation as implemented.
- **Part D** — honest classification: which equations are provable identities and which are modelling choices.

---

## Notation

| Symbol | Meaning |
|---|---|
| `V` | set of drivers, `n = \|V\|` |
| `v` | a driver |
| `L` | set of locations (graph nodes), `K = \|L\|` |
| `r = (t_r, s_r, d_r)` | request raised at time `t_r`, pickup `s_r`, dropoff `d_r` |
| `g_v^t` | node where driver `v` is at time `t` (`−1` if traversing an edge) |
| `o_v^t` | cumulative utility of driver `v` up to time `t` |
| `c_v` | vehicle capacity |
| `m_v^t` | riders on board |
| `M` | an assignment (matching) of requests to drivers |
| `M(v)` | requests assigned to driver `v` under `M` |
| `Geo(a,b)` | shortest road distance from node `a` to node `b` |
| `T` | timeline, `t_n = max(T)` |
| `λ, ω` | fairness weight and fairness scaling factor |
| `γ` | discount factor |
| **ours** | |
| `τ̄(a,b)` | period-mean travel time (the paper's flattened value) |
| `τ_p(a,b)` | travel time in time profile `p` |
| `c_p(a,b)` | congestion multiplier |
| `H_v` | hours driver `v` was online |
| `ρ_v` | hourly rate of driver `v` |
| `ν_v` | utilisation of driver `v` |
| `O_v` | set of slots in which `v` was online |
| `p` | time profile index, `p ∈ {0..47}` = (weekday/weekend) × hour |

---

# Part A — The base paper's equations

## A.1 Utility of a request (Section 3.1, unnumbered but foundational)

```
U(r, v) = Geo(d_r, s_r) − Geo(s_r, g_v^t)
```

**Explanation.** Two terms:

- `Geo(d_r, s_r)` — the **profit** term: the distance of the paid trip from pickup to dropoff.
- `Geo(s_r, g_v^t)` — the **cost** term: the "deadhead" distance the driver drives *empty* from their current position to the pickup point.

Utility is the net distance benefit. Evaluated at the moment of assignment. The paper assumes drivers always take the shortest path, so two requests with the same endpoints always have the same value.

**Worked example.** Driver at node A. Request from B to C. `Geo(B,C) = 8 km`, `Geo(A,B) = 2 km`. Then `U = 8 − 2 = 6 km`. If the driver were 9 km away, `U = 8 − 9 = −1 km`, a loss-making assignment.

**Units.** Kilometres (distance). Note this is *not* money — the paper uses distance as a proxy for earnings.

## A.2 Travel-time flattening (Section 5.1, prose not an equation)

> "the shortest travel time from a certain pickup to a certain drop-off location is re-calculated as the mean travel time across the time period"

Formally, for every ordered node pair:

```
τ(a, b, t) := τ̄(a, b) = (1/|T|) Σ_{t∈T} τ(a, b, t)      for all t
```

**Explanation.** All time dependence in travel time is replaced by a single average per OD pair. Combined with A.1, this means **the value and the time cost of a trip do not depend on when it happens.** This one sentence is the origin of Gap 1a.

## A.3 Efficiency (Equation 1)

```
π(M) = Σ_{v∈V} o_v^{t_n}(M(v)),        t_n = max(T)
```

**Explanation.** Total utility summed over all drivers at the end of the horizon. `o_v^{t_n}(M(v))` is driver `v`'s accumulated utility from the requests assignment `M` gave them. Maximising `π` is the pure-efficiency objective.

**Worked example.** Three drivers ending with 100, 120, 80 km → `π = 300 km`.

## A.4 Long-term fairness (Equation 2)

```
F(M) = Var( o_v^{t_n}(M(v)) ),        t_n = max(T)
```

**Explanation.** The variance of *weekly total* utility across the `n` drivers. Expanded:

```
F(M) = (1/n) Σ_v ( o_v − ō )²,     ō = (1/n) Σ_v o_v
```

Lower is fairer; `F = 0` means every driver ended with exactly the same total. "Long-term" refers to the accumulation window (one week), not to a discounted sum.

**Worked example.** Totals (100, 120, 80): `ō = 100`, `F = (0 + 400 + 400)/3 = 266.7`. Totals (100, 100, 100): `F = 0`.

**Critical property.** `F` depends only on the totals `o_v`. It contains no reference to hours worked, trips received, or time online. This is the origin of Gap 1b and Gap 2.

## A.5 Combined objective (Equation 3)

```
max_M   π(M) − λ · F(M)
```

**Explanation.** A scalarised bi-objective: maximise total utility while penalising unequal totals. `λ ≥ 0` sets the price of unfairness. `λ = 0` is pure efficiency; large `λ` prioritises equality. The paper uses `λ = 1`.

**Dimensional note.** `π` is in km, `F` is in km². Adding them requires a scale factor, which is why `ω` appears in Equation 7 (see A.8). Equation 3 as written is dimensionally inhomogeneous.

## A.6 Assignment constraint (Equation 4)

```
Σ_{v∈V, r∈R} I_rv ≤ 1
```

**Explanation.** As printed, this sums the indicator over *all* driver–request pairs and bounds it by 1, which would permit only one assignment in total. The intended meaning — standard in matching formulations and consistent with the rest of the paper — is per-request:

```
Σ_{v∈V} I_rv ≤ 1      for every request r
```

i.e. each request goes to at most one driver. We implement the per-request reading. (Section 4.3 separately allows one *driver* to hold several requests concurrently, so there is no matching constraint on the driver side in the paper.)

## A.7 Assignment indicator (Equation 5)

```
I_rv = 1  if request r is assigned to vehicle v
     = 0  otherwise
```

**Explanation.** A binary decision variable. The matrix `I` fully describes assignment `M`.

## A.8 Instantaneous reward (Equation 6)

```
r^t_{s,A(v)} = Σ_{a ∈ A_v^t} [ Geo(d_a^t, s_a^t) − Geo(s_a^t, g_v^t) ]
```

**Explanation.** The MDP reward for driver-agent `v` at time `t`: the sum of A.1 utilities over **all** requests `A_v^t` assigned to `v` in that timestep. The summation over a *set* is what permits unlimited concurrent assignments (Section 4.3: *"allows an agent (driver) to accept multiple requests concurrently"*), and because no term depends on trip duration, nothing bounds how much work one driver can absorb in one timestep. This is the origin of the physical-capacity problem quantified in `REPORT.md` §6.2.

## A.9 Scalarisation function (Equation 7)

```
SR(M) = Σ_{v∈V} r_{s,A(v)} − λ · ω · Var( r_{s,A(v)} )

where   r_{s,A(v)} = Σ_{t∈T} r^t_{s,A(v)}
```

**Explanation.** Converts the multi-objective MDP into a single scalar for Q-learning. First term is total accumulated reward (≡ efficiency). Second term penalises its variance across drivers (≡ fairness).

- `0 ≤ λ ≤ 1` — preference weight for fairness.
- `0 < ω ≤ 1` — scale factor. The paper's stated purpose: *"to adjust fairness into the same range as utility"*, because variance grows mechanically as the horizon lengthens, so without `ω` the fairness term would progressively dominate.

Paper's values: `λ = 1`, `ω = 0.6`, `γ = 0.9`.

**Our finding on `ω`.** A variance carries squared units, so the ratio of the fairness marginal to a typical utility depends on fleet size, horizon length and *which quantity* the variance is taken over. A single hard-coded constant cannot transfer across settings. See N10.

## A.10 Normalised fairness (Equation 8)

```
F̂(M) = σ(U) / Ū
```

**Explanation.** The coefficient of variation of the driver-utility vector `U`: standard deviation divided by mean. Introduced because raw variance is not comparable across runs with different total utility. Scale-free and dimensionless.

**Worked example.** Totals (100, 120, 80): `σ = 16.33`, `Ū = 100`, `F̂ = 0.163`.

**Caveat.** Undefined/unstable when `Ū → 0`, and misleading if `Ū < 0` (the paper's own Greedy row reports `−0.0005` from a negative mean).

## A.11 Prediction module (Section 4.2, prose)

A three-layer MLP maps lagged request counts to future counts:

```
R̂_{t+1..t+h}(s,d) = MLP( R_t(s,d), R_{t−1}(s,d), …, R_{t−n}(s,d) )
```

at 1-hour steps, trained on one month, forecasting 7 days. Reported test MSE 94.69. Predicted requests are inserted into the MDP **action space** so the policy can anticipate demand.

---

# Part B — Our modified and new equations

## Gap 1a: making utility time- and traffic-aware

### N1. Congestion multiplier (new)

```
c_p(a,b) = τ_p(a,b) / τ̄(a,b)                                    ... (N1)

c_p(a,b) ← clip( c_p(a,b), c_min, c_max ),   c_min = 0.5, c_max = 3.0
```

**Explanation.** A dimensionless ratio measuring how much slower than typical the leg `a→b` is in time profile `p`.

- `c = 1` — this leg is moving at its own period-average speed.
- `c = 2` — it takes twice as long as usual (heavy congestion).
- `c = 0.5` — twice as fast as usual (empty roads at 04:00).

`τ̄(a,b)` in the denominator is **exactly the quantity the paper uses**, which is what makes `c` the precise measure of the information the paper discards.

**Estimation** (hierarchical, level recorded per cell):

| Level | Rule | Requirement |
|---|---|---|
| L0 | `τ_p(a,b)` = median observed duration for `(a,b,p)` | ≥ 30 trips |
| L1 | `τ_p(a,b) = τ̄(a,b) · m_p` where `m_p` = citywide multiplier | pair observed |
| L2 | `τ_p(a,b) = Geo(a,b) / speed_p` | always |

Note L1 and L2 both give `c = m_p`, so a data-poor cell degrades to *"average congestion at this hour"*, never to *"no congestion"*. L0 covers 13.5% of cells but **90.7% of trips**.

**Measured:** median peak/off-peak ratio **2.11×**, max 4.35×; clipping binds on only **0.08%** of cells.

### N2. Time-aware utility (modifies A.1) — **core Gap 1a equation**

```
U_p(r, v) = Geo(d_r, s_r) / c_p(s_r, d_r)  −  w · Geo(s_r, g_v) · c_p(g_v, s_r)     ... (N2)
```

with deadhead weight `w = 1` by default (recovering the paper's implicit 1:1 trade-off).

**Explanation, term by term.**

- `Geo(d_r,s_r) / c_p(s_r,d_r)` — the **profit** term is *divided* by congestion. A trip that takes twice as long delivers half the value per unit of the driver's clock. The driver is paid for distance but pays in time, so a congested paid leg is worth less to them.
- `w · Geo(s_r,g_v) · c_p(g_v,s_r)` — the **cost** term is *multiplied* by congestion. Driving empty through traffic is strictly worse than driving empty on clear roads: same distance, more time burned, still unpaid.

The asymmetry (divide the paid leg, multiply the unpaid leg) is deliberate: both adjustments move in the direction that hurts the driver, which is what the paper's time-blind formula cannot express.

**Reduction property (the key design constraint).**

```
c ≡ 1   ⟹   U_p(r,v) = Geo(d_r,s_r) − w·Geo(s_r,g_v)  ≡  A.1
```

So **the paper's utility is a strict special case of ours.** Verified numerically at `max |ΔU| = 0.000e+00` (Part C, E7). This is what makes the Gap 1a ablation clean rather than a comparison of two unrelated formulas.

**Worked example.** `Geo(B,C) = 8 km`, `Geo(A,B) = 2 km`.

| Scenario | `c(B,C)` | `c(A,B)` | Utility |
|---|---|---|---|
| Paper (any time) | — | — | `8 − 2 = 6.00 km` |
| Ours, 04:00 (clear) | 0.6 | 0.6 | `8/0.6 − 2·0.6 = 12.13 km` |
| Ours, 18:00 (jammed) | 2.0 | 2.0 | `8/2.0 − 2·2.0 = 0.00 km` |
| Ours, 18:00, far pickup 5 km | 2.0 | 2.0 | `8/2.0 − 5·2.0 = −6.00 km` |

The last row is the practical consequence: an assignment the paper scores at `8 − 5 = +3 km` (profitable) is in fact `−6 km` (loss-making). **Measured on real data, the sign flips on 10.2% of (driver, request, time) triples.**

**Units.** Kilometres, same as A.1 — `c` is dimensionless, so dimensional homogeneity is preserved.

**Money variant.** Setting `unit = "money"` multiplies both `Geo` terms by a calibrated fare rate (\$3.95/km measured from the data), giving utility in dollars. Reported as a robustness check.

### N3. Occupancy (new — the equation that makes Gap 2 possible)

```
occ_min(r,v) = τ_p(g_v, s_r) + τ_p(s_r, d_r)                              ... (N3a)

occ_slots(r,v) = max( 1, ⌈ occ_min(r,v) / Δ ⌉ ),    Δ = 5 min             ... (N3b)
```

**Explanation.** The number of decision epochs a driver is unavailable after accepting a request: time to reach the pickup, plus time to complete the trip. The driver becomes free again at slot `t + occ_slots`.

The paper has no equivalent — its A.8 reward sums over an unbounded set with no time cost, so a driver can absorb unlimited concurrent work. Without N3, *"was this driver busy at time t?"* has no answer and **Gap 2's metric cannot be defined at all.** N3 is therefore a prerequisite for Gap 2, not merely a refinement.

`max(1, ·)` ensures any assignment consumes at least one epoch. `⌈·⌉` (ceiling) is a conservative discretisation — a trip is never released early.

## Gap 1b: hourly-rate fairness with group decomposition

### N4. Hourly rate (replaces the fairness target in A.4)

```
ρ_v = o_v / H_v ,     H_v = |O_v| · Δ/60   (hours online)                 ... (N4)
```

Drivers with `H_v < 2 h` are excluded from rate statistics (one lucky trip in a 20-minute session is not a "rate").

**Explanation.** `o_v` is the same accumulated utility the paper uses; the change is dividing by hours online. `H_v` counts **all** online time, idle included — correct, because a driver is on the clock whether or not the platform gives them work.

**Worked example (the paper's blind spot made numeric).**

| Driver | `o_v` | `H_v` | `ρ_v` |
|---|---|---|---|
| A | 6000 | 60 h | **100** |
| B | 6000 | 15 h | **400** |

`Var(o) = 0` → the paper reports perfect fairness. `Var(ρ) = 22,500` → a 4× rate gap. Measured on our real 200-driver fleet, equalising totals produces rates spanning **47.9 to 375.7 (7.8×)**.

### N5. Fairness decomposition by group (new) — **core Gap 1b equation**

```
Var(ρ) = E_g[ Var(ρ | g) ]   +   Var_g( E[ρ | g] )                        ... (N5)
         └──── within ────┘       └──── between ────┘
```

Explicitly, with groups `g ∈ {full-time, part-time}` and population shares `w_g = n_g/n`:

```
Var_within(ρ)  = Σ_g w_g · Var(ρ | g)                                     ... (N5a)
Var_between(ρ) = Σ_g w_g · ( mean(ρ | g) − mean(ρ) )²                     ... (N5b)
```

**Explanation.** This is the **law of total variance**, a standard probability identity — not a constructed metric. It splits total rate inequality into:

- **within-group** — how unequal drivers are *compared to their own peers* (full-timer vs full-timer).
- **between-group** — how unequal the *group averages* are (full-timers as a class vs part-timers as a class).

This is exactly the "within and between these groups" comparison the gap analysis called for, obtained from an identity rather than invented. The identity guarantees `within + between = total` exactly, so the two components are a true partition with no overlap or residual — verified to `6.0e−16` relative error (Part C, E1).

**Worked example.** Full-timers `ρ = (100, 110)`, part-timers `ρ = (390, 410)`.

```
mean(ρ) = 252.5
within  = 0.5·Var(100,110) + 0.5·Var(390,410) = 0.5·25 + 0.5·100      = 62.5
between = 0.5·(105−252.5)² + 0.5·(400−252.5)²                        = 21,756.25
total   = 21,818.75  = within + between   ✓
```

The decomposition shows immediately that 99.7% of the inequality is *between* groups — a structural pay gap, not individual variation. The paper's `Var(o_v)` cannot distinguish these.

**Group definition.** Full-time `H_v ≥ 40 h/week`, part-time `H_v ≤ 20 h/week`, with a 40/60 population mix.

### N6. Group ratio (new, reporting)

```
Ratio = mean(ρ | part-time) / mean(ρ | full-time)                         ... (N6)
```

**Explanation.** A single interpretable number. `1.0` = exact parity. `>1` = part-timers out-earn full-timers per hour. Directly comparable across methods and unit systems. All five prior methods land at **1.62–1.91**; ours at **0.97**.

## Gap 2: utilisation / access fairness

### N7. Utilisation (new) — **core Gap 2 equation**

```
ν_v = (busy slots) / (online slots) = Σ_t 1[v busy at t] / |O_v|          ... (N7)
```

**Explanation.** The fraction of a driver's online time during which they were actually working. `ν = 1` means never idle; `ν = 0.25` means idle three-quarters of the time. Bounded in `[0,1]` by construction (verified, Part C, E11).

The paper has no analogue: it observes only `o_v`, so a driver who was online 8 hours and dispatched twice is indistinguishable from one dispatched ten times if their totals happen to match.

Fairness is then measured as `Var(ν)`, with the same N5 group decomposition applied.

**Worked example.** Driver X: 96 online slots, 24 busy → `ν = 0.25`. Driver Y: 96 online, 84 busy → `ν = 0.875`. `Var(ν) = 0.098` — large inequality that `Var(o_v)` would report as zero if their earnings matched.

### N8. Opportunity-normalised utilisation (new, refinement)

```
π_t          = (busy drivers at t) / (online drivers at t)                ... (N8a)

opp_v        = (1/|O_v|) Σ_{t ∈ O_v} π_t                                  ... (N8b)

ν_adj,v      = ν_v / max(opp_v, ε)                                        ... (N8c)
```

**Explanation and motivation.** Raw `Var(ν)` conflates two very different phenomena:

1. the platform starving an available driver — the algorithm's fault;
2. a driver choosing to be online at 04:00 when there is no demand — not the algorithm's fault.

`π_t` is the system-wide busy fraction in slot `t` — how much work was *available* at that moment. `opp_v` averages this over the slots driver `v` actually worked, giving the utilisation a typical peer working the same hours achieved. Dividing gives:

- `ν_adj = 1` — the driver did as well as their same-hours peers.
- `ν_adj < 1` — the algorithm under-served them relative to what those hours offered.

Reporting only the raw form would overstate the algorithm's culpability; reporting only the normalised form would hide genuine starvation. **Both are reported.**

**Worked example.** Driver X works only 03:00–05:00 where `π_t = 0.10`. X achieves `ν = 0.09`. Raw `ν = 0.09` looks like severe starvation, but `ν_adj = 0.09/0.10 = 0.90` — X did roughly as well as anyone could in those hours. Conversely a driver in peak hours with `π_t = 0.80` who achieves `ν = 0.40` has `ν_adj = 0.50`: genuinely under-served.

## Unified objective

### N9. Extended scalarisation (modifies A.9) — **the objective we optimise**

```
SR(M) = Σ_v u_v
        − ω_ρ · [ λ_w · Var_within(ρ) + λ_b · Var_between(ρ) ]
        − ω_ν · λ_ν · Var(ν)                                              ... (N9)
```

**Explanation.** Generalises Equation 7 in three ways:

1. The fairness target moves from **totals** `o_v` to **rate** `ρ_v` (Gap 1b).
2. The rate term is **split** into within- and between-group components with independent weights `λ_w, λ_b`, so structural group inequity can be priced separately from individual variation. This is the knob that produces group parity.
3. A **utilisation** term is added (Gap 2).

Setting `λ_b = λ_ν = 0` and switching the target back to totals recovers Equation 7 exactly, so the paper remains a special case of the full objective.

**Selected weights** (by the coordinate search in `scripts/run_sweeps.py`, both gap terms active in all 20 configurations): `λ_w = 1.0, λ_b = 16.0, λ_ν = 0.5`.

`λ_b = 16` is large only because `ω_ρ` is calibrated against the *total* rate variance, of which the between-group component is a small share; the weight must be correspondingly larger to move it.

### N10. Weight calibration (replaces the paper's fixed ω = 0.6)

For each fairness quantity `x` with per-assignment increment `dx`:

```
Δ Var(x) per assignment  ≈  2 · mean(|x|) · dx / n                        ... (N10a)

ω_x = mean(|u|) / [ 2 · mean(|x|) · dx / n ]                              ... (N10b)
```

with `mean(|u|)` the mean absolute utility of an assignment, both measured from an efficiency-only warm-up run.

**Explanation.** N10a is the first-order term of the exact variance marginal (N11): expanding `ΔVar = (2·x_j·dx + dx²)/n − (2·S·dx + dx²)/n²` and keeping the dominant term gives `≈ 2·x_j·dx/n`. N10b then chooses `ω_x` so that a typical assignment's fairness penalty is the same order of magnitude as a typical assignment's utility, leaving `λ` as the only meaningful trade-off knob.

**Why this was necessary.** With the paper's approach the utilisation penalty came out at ≈**0.009** against a typical trip utility of **2.4** — roughly 270 times too small. The Gap 2 term was effectively **inert**, and its ablation showed no effect (the ablation table came out backwards). After calibration: `ω_total = 1.64, ω_ρ = 1.11×10³, ω_ν = 5.96×10⁴`, and every ablation behaves correctly.

This is a substantive finding about the base paper: `ω = 0.6` is not transferable, and reporting it without the range it was calibrated against makes the result irreproducible.

### N11. Incremental variance marginals (new, exact)

For a greedy or Q-learning step we need the marginal effect of one assignment on a variance. With sufficient statistics `S = Σx_i`, `Q = Σx_i²`, and `Var = Q/n − (S/n)²`:

```
Δ Var_total(j, dx)   = (2·x_j·dx + dx²)/n  −  (2·S·dx + dx²)/n²           ... (N11a)

Δ Var_within(j, dx)  = (n_g/n) · [ (Q_g + 2x_j dx + dx²)/n_g − ((S_g+dx)/n_g)²
                                   − ( Q_g/n_g − (S_g/n_g)² ) ]           ... (N11b)

Δ Var_between(j, dx) = Δ Var_total − Δ Var_within                         ... (N11c)
```

**Explanation.** These are **algebraically exact**, not approximations — derived by substituting the updated statistics into the variance definition. They evaluate in `O(1)` (N11a) and `O(#groups)` (N11b) instead of `O(n)` per candidate, which is what makes greedy re-evaluation over all (request, driver) pairs tractable at every slot.

N11c is valid *because* N5 is an exact identity: since `total = within + between` always holds, the same must hold for their differences.

**Verified** against brute-force recomputation over 2,000 random cases: max absolute error `7.1e−15` (Part C, E3).

### N12. MOMAQL scoring with time discounting (modifies A.8/A.9)

```
score(r,v) = r_scal(r,v) + γ^{occ_slots(r,v)} · V[p', d_r, b']  −  V[p, g_v, b]   ... (N12)

r_scal(r,v) = U_p(r,v) − [fairness penalty from N9 marginals]
```

where `V[profile, node, deficit_bucket]` is a tabular value function (48 × 87 × 5 = 20,880 entries), `b` is the driver's discretised deficit bucket, and `p'` is the profile at trip completion.

**Explanation.** Three departures from the paper:

- **`γ^{occ_slots}` rather than `γ`.** The discount exponent is the number of slots the trip occupies, so a long trip is discounted for the driver-time it locks up. The paper's timeless reward cannot express this.
- **State includes a fairness deficit bucket.** A policy cannot optimise a term it cannot observe. The paper's state is location-only (Section 4.3), so with its formulation the fairness objective can be *measured* but never *improved* by the learned policy. Augmenting the state with rate-deficit and utilisation-deficit buckets is what makes N9 optimisable.
- **Advantage form** `+γ^k V[s'] − V[s]`, which scores the *improvement* from taking the assignment rather than the raw next-state value.

**Approximation, stated.** `p'` is computed by advancing the hour within the same day-type rather than re-deriving the exact wall-clock profile. At 48 profiles this is a negligible discretisation and it keeps the table small.

## Supporting equations

### N13. Rotated-L1 road-distance fallback (new, for unobserved OD pairs)

```
[x, y]  = [ lon · k_x , lat · k_y ] ,   k_x = 111.32·cos(lat₀), k_y = 111.32

Geo_L1(a,b) = α · ( |Δx·cosθ + Δy·sinθ| + |−Δx·sinθ + Δy·cosθ| ),  θ = 29°   ... (N13)
```

**Explanation.** Manhattan's avenue grid runs ≈29° east of true north. Rotating the coordinate frame by `θ` and taking an L1 (city-block) distance approximates driving distance far better than a great-circle line. `α` is fitted by trip-count-weighted least squares against the 80.8% of pairs with direct observations: `α = 0.9342`, weighted MAPE **14%**.

Used only for the 19.2% of pairs never observed, which carry 0.02% of trips. Validation: implied circuity (road ÷ great-circle) = **1.320**, matching the independently known NYC value of ≈1.3.

### N14. Offered-load calibration (new, experiment design)

```
Load = ( Σ_r occ_slots(r) ) / ( Σ_v |O_v| )                               ... (N14)
```

Sampling rate solved so `Load ≈ 0.60`.

**Explanation.** Demand pressure: driver-slots the requests *would* consume divided by driver-slots available. Distinct from realised utilisation (`Load × service rate`).

**Why this matters mathematically.** At `Load ≥ 1` every driver is busy every slot, so `ν_v → 1` for all `v` and `Var(ν) → 0` **identically for every method** — Gap 2's metric degenerates and can no longer discriminate. Calibrating to 0.60 keeps the metric informative. Confirmed empirically: under the paper's 0.05 rate our run gives `ν = 0.99` and 99/200 drivers idle.

### N15. Gini coefficient (supporting, reported alongside variance)

```
G(x) = Σ_i (2i − n − 1) · x_(i)  /  ( n · Σ_i x_i ) ,   x_(i) sorted ascending   ... (N15)
```

**Explanation.** Reported because variance is sensitive to outliers and unbounded, whereas Gini is bounded in `[0,1]` and scale-invariant. If both agree the conclusion is robust to the choice of inequality measure — they do agree here (Gini(ρ) 0.258 → 0.198, −23.4%).

### N16. Recovering the paper's unreported fleet size (new, diagnostic)

```
n = π(M) / mean_v( o_v )                                                  ... (N16)
```

**Explanation.** A trivial identity — total divided by mean is the count — but applied to the paper's Table 1 it recovers a setting the paper never reports. All five rows give **exactly 20.000**, so the paper used `n = 20` drivers.

Combined with the paper's own protocol (peak 2 h/day → ≤14 h/week) and its reported mean per-driver utility of 4,791.19, at a realistic 2.49 km net utility per trip this implies ≈1,924 trips per driver per week against a physical maximum of ≈53 — a factor of **36× over capacity**, explained by the unbounded concurrency in A.8.

---

# Part C — Correctness verification

Every equation was checked numerically. Reproduce with:

```bash
.venv/bin/python -m scripts.verify_phase1      # E7-E11 plus pipeline invariants
.venv/bin/python -m scripts.verify_phase2      # fleet + Gap 1b blind-spot demo
.venv/bin/python -m scripts.smoke_env          # N5 exactness inside a full run
```

| # | Property checked | Equations | Result |
|---|---|---|---|
| E1 | `within + between = total` over 500 random cases | N5 | **PASS**, max rel. err `6.0e−16` |
| E2 | Tracker `total/within/between` = direct computation | N5, N11 | **PASS**, max abs. err `5.3e−15` |
| E3 | Incremental marginals = brute-force recomputation (2,000 cases) | N11a–c | **PASS**, max abs. err `7.1e−15` |
| E4 | Vectorised marginals = scalar marginals | N11 | **PASS**, max abs. err `3.9e−16` |
| E5 | Sufficient statistics stay consistent over 50 sequential updates | N11 | **PASS**, max drift `1.1e−14` |
| E6 | `Gini(equal) = 0`; `Gini ∈ [0,1]`; scale-invariant; `CV = σ/μ` | N15, A.10 | **PASS** |
| E7 | `U_p = U_paper` when `c ≡ 1`; occupancy identical; test non-vacuous | N2, N3 | **PASS**, `max\|ΔU\| = 0.000e+00`; mean `\|ΔU\|` with real `c` = 3.487 km |
| E8 | `c > 0`; within clip bounds; centred near 1 on measured cells | N1 | **PASS**, mean `c` = 0.9841, clipping binds on 0.08% of cells |
| E9 | Congestion never benefits the driver (monotonicity); `occ ≥ 1` | N1, N2, N3 | **PASS** |
| E10 | Dimensional consistency: `U` in km, bounded by `(1+w)·max Geo / c_min` | N2 | **PASS**, `U ∈ [−28.20, 37.33]` km |
| E11 | `ρ = o/H` finite for `H > 0`; `ν ∈ [0,1]` | N4, N7 | **PASS** |

**Summary: all 11 equation groups pass.** The two most important results are E7 (the paper is a provable special case of our utility, to exact floating-point equality) and E1/E3 (the group decomposition and its marginals are exact identities, not approximations).

---

# Part D — Honest classification: what is provable and what is a choice

Not all equations carry the same epistemic status. Presenting them as uniformly "correct" would be misleading, so they are classified.

## D.1 Provable identities — correct by mathematics

| Equation | Status |
|---|---|
| **N5** decomposition | The law of total variance. A theorem. Verified to `6e−16`. |
| **N11a–c** marginals | Exact algebra from the variance definition, not approximations. Verified to `7e−15`. |
| **N2** reduction at `c ≡ 1` | Substituting `c = 1` reduces N2 to A.1 term by term. Verified to exact equality. |
| **N16** fleet recovery | `total/mean = count`. Arithmetic. |
| **A.10 / N15** | Standard definitions of CV and Gini. |

These cannot be wrong. They can only be implemented wrongly, and E1–E7 confirm the implementations match.

## D.2 Modelling choices — defensible, but not uniquely determined

**N2, the functional form of the time-aware utility.** Dividing the paid leg by `c` and multiplying the deadhead by `c` is a *choice*. Alternatives exist:

- `U = (Geo(d,s) − w·Geo(s,g)) / (τ(g,s) + τ(s,d))` — a strict utility-per-hour rate;
- an additive time penalty `U = Geo(d,s) − w·Geo(s,g) − β·occ_min`.

We chose the multiplicative form for one decisive reason: **it reduces exactly to the paper's utility at `c ≡ 1`**, making the ablation a clean isolation of the congestion effect. A rate form does not have that property and would confound "we changed the utility" with "we changed the units". The choice is justified but should be reported as a choice, and the sensitivity of conclusions to it is untested.

**N8, the opportunity normalisation.** Defining "what was achievable in those hours" as the system-wide busy fraction is reasonable but not unique; one could instead normalise by realised demand in the driver's vicinity, or by a matched-peer average. Two known imperfections:

- `opp_v` includes the driver's own contribution to the busy count, giving a slight self-bias — roughly `1/35 ≈ 3%` at our median concurrency of 35 online drivers.
- The `max(opp_v, ε)` guard is numerically fragile when supply is near zero. Under the paper's protocol (99/200 drivers idle, near-zero opportunity in some slots) it produced `ν_adj` variance of `1.4e23`. Those figures are reported as degenerate and explicitly marked non-interpretable rather than quietly dropped.

**N10, the `ω` calibration.** Setting the fairness penalty to the same order of magnitude as a typical utility is a sensible normalisation, but "same order of magnitude" is itself a convention. It is strictly better than an unexplained constant, and it makes `λ` the single interpretable knob, but it is not derived from first principles.

**Group thresholds** (`≥40 h` full-time, `≤20 h` part-time, 40/60 mix) and `c_min/c_max = 0.5/3.0` are parameter choices, documented in `ltf/config.py`. Clipping binds on only 0.08% of cells, so conclusions are insensitive to the latter.

## D.3 Approximations — small and bounded

| Where | Approximation | Impact |
|---|---|---|
| N3b | `⌈·⌉` rounds occupancy up to whole slots | ≤ 5 min per trip, conservative (never releases a driver early) |
| N12 | Next-state profile advanced within day-type | Negligible at 48 profiles |
| N10a | First-order term of the exact marginal | Only used to set a scale factor, not in any reported metric |
| N1 L1/L2 | Sparse cells fall back to citywide multiplier | Affects 9.3% of trips; degrades toward average congestion, never toward none |

## D.4 Where the base paper's equations are themselves problematic

Stated for completeness, since these motivated our changes:

1. **A.5** adds km to km² without a scale factor; Equation 3 is dimensionally inhomogeneous. Equation 7 patches this with `ω`, but `ω` is then reported as a single untransferable constant (see N10).
2. **A.6** as printed permits only one assignment in total. The per-request reading is clearly intended; we implement that.
3. **A.8** sums utility over an unbounded assignment set with no time term, which permits physically impossible workloads — quantified at 36× capacity in N16.
4. **A.10** is unstable near `Ū = 0` and meaningless for `Ū < 0`; the paper's own Greedy row reports `−0.0005`, a negative "normalised fairness" that cannot be interpreted as inequality.
5. **A.2 + A.4 together** are the substantive problem: a time-invariant utility combined with a fairness measure over totals means neither traffic nor hours worked can enter the objective.

## D.5 Overall assessment

**The equations we implemented are mathematically correct and appropriate for the stated purpose**, with three qualifications:

1. The core fairness machinery (**N5**, **N11**) rests on a probability theorem and exact algebra, verified to machine precision. These are not heuristics.
2. **N2**'s reduction property is proven and verified exactly, so the Gap 1a comparison is a genuine isolation of one effect. Its *functional form*, however, is a defensible modelling choice rather than a derived necessity.
3. **N8**'s normalisation is the least settled equation in the set — well-motivated, but with a known self-bias and a numerical fragility at low supply. It is reported alongside the raw metric precisely so no conclusion depends on it alone.

Every claim in `REPORT.md` follows from D.1 identities plus measurements, not from D.2 choices — with one exception: the magnitude of Gap 1a's utility effect depends on N2's form, which is why that result is reported as *measurement correctness* (10.2% sign flips) rather than as an efficiency gain.
