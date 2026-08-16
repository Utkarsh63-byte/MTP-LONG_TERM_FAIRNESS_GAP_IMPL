# Long-term Fairness in Ride-Hailing — Gap Implementation

MTP work extending **Kang, Chan, Shao, Salim & Leckie, *"Long-term Fairness in Ride-Hailing Platform"*, ECML PKDD 2024** ([arXiv:2407.17839](https://arxiv.org/abs/2407.17839)).

The base paper allocates ride requests to drivers while balancing total earnings against fairness, where fairness is the **variance of each driver's weekly total earnings**. This repository identifies two problems with that formulation, implements fixes, and evaluates them against five baselines under identical conditions.

---

## The two gaps

**Gap 1 — static utility and group-blind fairness.**

The paper's utility is distance-only, `U = Geo(d_r, s_r) − Geo(s_r, g_v)`, and Sec. 5.1 replaces travel time with a single period mean per OD pair. So a 5 km trip at 04:00 and at 18:00 are valued identically. Measured on the paper's own data, the same OD pair takes **2.11× longer** at peak (median; max 4.35×), and the paper's utility gets the profitable/unprofitable **sign wrong on 10.2%** of cases.

Its fairness measure `Var(total earnings)` is also blind to hours worked. A driver earning ₹6,000 over 60 h and one earning ₹6,000 over 15 h score a perfect 0, hiding a 4× gap in hourly rate.

**Gap 2 — utilisation / access fairness.**

The paper checks only final earnings, never whether an available driver was given work. A driver online 8 h and dispatched twice can match the earnings of one dispatched ten times, and the paper calls that fair while six hours of forced idleness go unrecorded.

---

## Headline results

7-day horizon, 200 simulated drivers, 87 graph nodes, identical scenario for every method.

| Metric | Kang et al. (reproduced) | Ours | Change |
|---|---|---|---|
| Total utility | 10,221 | 10,145 | −0.7% |
| Var(hourly rate) | 1.251 | **0.446** | −64.4% |
| between-group | 0.2473 | **0.0011** | −99.6% |
| Part-time / full-time rate ratio (parity = 1.00) | 1.62 | **0.97** | parity reached |
| Var(utilisation), opportunity-normalised | 0.2527 | **0.1550** | −38.6% |
| Drivers with no work all week | 1 | **0** | eliminated |

**A 0.7% reduction in total utility buys hourly-rate parity between activity groups and eliminates fully-idle drivers.**

Two further findings:

- **All five prior methods** (Greedy, REASSIGN, LAF, Balance Ride-Pooling, Kang et al.) land at a **1.62–1.91× part-time pay premium**. This is systematic, because all five optimise `Var(totals)`.
- Switching the paper's fairness term **on** makes between-group inequity **90× worse** (0.004 → 0.361). The objective does not merely fail to see the group gap — it creates it. Levelling totals across unequal hours can only be done by transferring hourly rate from long-hours to short-hours drivers.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# print every headline result from the committed outputs (seconds, no data needed)
.venv/bin/python -m scripts.show_results

# dataset accounting, and why the paper's totals are larger than ours
.venv/bin/python -m scripts.explain_scale
```

Both scripts read the CSVs in `outputs/`, so they work immediately after cloning.

## Reproducing from scratch

The raw dataset is not versioned (1.9 GB). Download `yellow_tripdata_2016-03.csv` from the [NYC TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) into the repository root, then:

```bash
.venv/bin/python -m ltf.data.prepare        # build 87 nodes + trips.parquet   (~2 min)
.venv/bin/python -m scripts.verify_phase1   # 25 assertions on the pipeline
.venv/bin/python -m scripts.verify_phase2   # driver fleet + Gap 1b demonstration
.venv/bin/python -m scripts.run_forecaster  # demand + congestion heads         (~3 min, GPU)
.venv/bin/python -m scripts.run_experiments # Tables 1-3, 5, 6                  (~25 min)
.venv/bin/python -m scripts.run_sweeps      # weight sweep, Pareto fronts       (~60 min)
.venv/bin/python -m scripts.run_robustness  # money units, seeds, paper protocol (~60 min)
.venv/bin/python -m scripts.make_figures    # all figures
```

`ltf/` is device-agnostic: the forecaster uses CUDA if present, then Apple MPS, then CPU. Override with `LTF_DEVICE=cuda`.

---

## What was implemented

| Component | Equation | Purpose |
|---|---|---|
| Congestion multiplier | `c_p(a,b) = τ_p(a,b) / τ̄(a,b)` | measures exactly the information the paper's period mean discards |
| Time-aware utility | `U_p = Geo(d,s)/c_p(s,d) − w·Geo(s,g)·c_p(g,s)` | **reduces exactly to the paper's utility at `c ≡ 1`** (verified, max \|ΔU\| = 0) |
| Occupancy | `occ = τ_p(g,s) + τ_p(s,d)` | a driver is busy for real time; prerequisite for Gap 2 |
| Hourly rate | `ρ_v = o_v / H_v` | replaces totals as the fairness target |
| Group decomposition | `Var(ρ) = E_g[Var(ρ\|g)] + Var_g(E[ρ\|g])` | law of total variance; within/between split is an identity, not a construction |
| Utilisation | `ν_v = busy slots / online slots` | Gap 2 metric, reported raw and opportunity-normalised |
| Unified objective | `Σu_v − ω_ρ[λ_w·Var_within + λ_b·Var_between] − ω_ν·λ_ν·Var(ν)` | what the agent optimises |

Two changes were required beyond metrics:

1. **MDP state augmentation.** A policy cannot optimise a term it cannot observe. The paper's state is location-only, so its fairness objective can be measured but never improved by the learned policy. The state now carries discretised rate-deficit and utilisation-deficit buckets: `V[profile, node, deficit]` = 48 × 87 × 5.
2. **Calibrated `ω` instead of the paper's fixed 0.6.** A variance carries squared units, so its scale relative to utility depends on fleet size and horizon. Uncalibrated, the utilisation penalty came out ≈0.009 against a typical trip utility of 2.4 — roughly **270× too small**, leaving the Gap 2 term inert and its ablation showing no effect. Calibrating from a measured run gives `ω_ν = 5.96e4`.

---

## Repository layout

```
ltf/
  config.py           all hyperparameters; Config.paper_faithful() reproduces the paper's setting
  utility.py          UtilityModel: paper vs time-aware, distance vs money, reduction self-test
  data/               CSV -> 87 nodes, distance matrix, 48-profile travel times, demand tensor
  sim/                timeline, driver fleet with shifts, request stream, occupancy-aware environment
  metrics/            the paper's metrics and the gap metrics, side by side
  methods/            Greedy, REASSIGN, LAF, Balance Ride-Pooling, MOMAQL
  predict/            MLP forecaster with demand and congestion heads
  experiments/        shared bench setup
scripts/              verification, experiment runners, figures, report generators
outputs/              result CSVs, figures, DOCX reports, run logs, terminal transcripts
REPORT.md             full technical report
EQUATIONS.md          every equation, with derivations and correctness verification
```

## Documents

| File | Contents |
|---|---|
| [`REPORT.md`](REPORT.md) | Full technical report: methodology, all result tables, comparison with the paper |
| [`EQUATIONS.md`](EQUATIONS.md) | Every equation in the paper and ours, term by term, plus numerical correctness checks |
| `outputs/MTP_Gap_Implementation_Report.docx` | Formal report with terminal transcripts as an appendix |
| `outputs/MTP_Progress_Update.docx` | Short plain-language update |

## Figures

`outputs/fig_gaps_by_method.png` is the most informative single view: hourly rate by group, between-group inequity, and utilisation fairness across all six methods. Also included: `fig_gap1a_congestion.png`, `fig_parity_sweep.png`, `fig_sweep_pareto.png`, `fig_tradeoff.png`, `fig4_horizon.png`.

---

## Notes on scope and honesty

- **Absolute utilities are not comparable to the paper's Table 1.** The paper used 20 drivers — unreported in the text, but recoverable since total ÷ mean = exactly 20.000 on all five rows — and permits unlimited concurrent trips per driver, implying ~1,924 trips per driver per week against a physical maximum of ~53 in its own 2 h/day window (≈36× over capacity). All comparisons here are made *inside one harness* where every method sees identical drivers, requests and distances. See `scripts/explain_scale.py`.
- **The paper's prediction module gave no measurable benefit** in this harness; removing it slightly *raises* utility. The paper's claimed 41% utility collapse without prediction could not be reproduced.
- **Gap 1a's efficiency gain is small (~1%).** Its contribution is measurement correctness (10.2% sign flips), not throughput, and is reported as such.
- **`Var(total)` is worse for our method by construction**, since equal hourly rates across unequal hours implies unequal totals. This incompatibility is the central finding, not a defect.

## Data licence

NYC TLC trip records are published by the NYC Taxi and Limousine Commission. The raw file is not redistributed here; download it from the [official source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
