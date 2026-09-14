# Conversation Transcript — MTP Gap Implementation

Project: `/Users/utkarshpandey/Downloads/Research_MTP`
Base paper: Kang, Chan, Shao, Salim, Leckie — *"Long-term Fairness in Ride-Hailing Platform"*, ECML PKDD 2024 (arXiv:2407.17839)
Dataset: NYC TLC Yellow Taxi, March 2016

---

## Turn 1 — Understanding the paper, the gaps, and the dataset

**User:**
> hey!
>
> this is my researcher paper which i have deep dived: `/Users/utkarshpandey/Downloads/Research_MTP/lncs14949217.pdf`
>
> and found the gaps of the paper after carefully understanding reading it!
>
> `/Users/utkarshpandey/Downloads/Research_MTP/GAP_Research -MTP.pdf`
>
> now, my task is to implement the gaps percetly!
>
> you task is to read the research paper line by line understand the complete paper, understand my gap!
>
> okay undertand complete paper, gap dayaset and let me know!!
>
> plsease deep dive and research of the gaps and also let me know the flow you gonna follow!

**Assistant:**

Read both PDFs (extracted via `pypdf` into `paper_text.txt` and `gap_text.txt`), fetched the arXiv HTML to cross-check equations, and profiled the raw dataset.

### The paper, precisely

| Quantity | Definition | Ref |
|---|---|---|
| Utility | `U = Geo(d_r, s_r) − Geo(s_r, g_v^t)` | Sec. 3.1 |
| Efficiency | `π(M) = Σ_v o_v^{t_n}(M(v))` | Eq. 1 |
| Long-term fairness | `F(M) = Var(o_v^{t_n}(M(v)))` | Eq. 2 |
| Combined objective | `max_M π(M) − λF(M)` | Eq. 3 |
| Reward | `r^t = Σ_{a∈A_v^t} Geo(d_a,s_a) − Geo(s_a,g_v^t)` | Eq. 6 |
| Scalarisation | `SR(M) = Σ_v r_v − λω·Var(r_v)` | Eq. 7 |
| Normalised fairness | `F̂ = σ(U)/Ū` | Eq. 8 |

Method: 3-layer MLP forecaster feeding predicted requests into the MDP **action space**, plus centralised MOMAQL. `λ=1, ω=0.6, γ=0.9`. Sec. 5.1 explicitly flattens travel time to a period mean.

**Not reported anywhere in the paper:** number of drivers, number of graph nodes, MLP hidden width, driver shift model, λ/ω sensitivity.

### The gaps, verified against data

Measured, not argued:

| Measurement | Value |
|---|---|
| Manhattan mean speed, 05:00 vs 15:00 | 16.69 vs 9.03 mph = **1.85×** |
| Same OD pair, peak/off-peak travel time | median **2.42×**, max 4.60× |
| Trips/day range | 278,805 (Easter Sunday) – 389,114 |

Key structural insight: because utility is distance-only and drivers accept requests concurrently, **occupancy time is unmodelled** in the paper. So Gap 1a is a *prerequisite* for Gap 2 — without time-dependent duration there is no well-defined "busy slot".

For Gap 1b, the law of total variance gives the within/between split as an identity:
```
Var(ρ) = E_g[Var(ρ|g)] + Var_g(E[ρ|g])      ρ_v = o_v / H_v
```

### Dataset check (full pass over 12.2M rows)

- 12,210,952 → **10,303,643 (84.4%)** after Manhattan bbox + sanity filters
- Raw lat/lon present (pre-July-2016 format), so granularity is ours to choose
- **No driver identifier** — drivers must be simulated
- Granularity trade-off measured: 0.5 km → 15.6% of (OD,hour) buckets usable; 1.5 km → 46.3%; 2 km → 58.0%

### Proposed flow

8 phases: data pipeline → driver simulator → faithful reproduction → Gap 1a → Gap 1b → Gap 2 → unified objective → experiments. Design principle: **every extension must reduce to the paper exactly when its knob is switched off.**

Six decisions requested: node granularity, driver count, group thresholds, utility units, baseline scope, compute budget.

---

## Turn 2 — Decisions

**User:**
> 1. i feel like 90 nodes are preety good
> 2. Driver count n = 200 seems to be bettter lets try with this , will modiefy later if required
>
> rest of your suggetion seems to be fine and also i have GPU but not configured yet but will provide it later on code it accordingly so we can use it any time

**Assistant:**

Locked in: 90 nodes, n=200, distance-primary utility, all four baselines, GPU-ready code.

**Flagged one config change:** the paper's peak-2h protocol caps a driver at 14 h/week, making a 40 h vs 20 h split unrepresentable — so Gap 1b would be untestable. Gap experiments run on a full-day timeline; the paper's protocol retained behind a flag.

Built Phase 1 (data pipeline):

- `ltf/config.py`, `ltf/utils.py` (device resolution: CUDA → MPS → CPU)
- `ltf/data/prepare.py` — 2-pass CSV → **87 nodes**, 99.94% endpoint coverage. (90 was *cells that appear*; only 87 have enough volume to estimate travel times.)
- `ltf/data/graph.py` — distance matrix from observed medians, 80.8% pairs direct (99.98% of trips). **Circuity 1.320**, matching the known NYC value ≈1.3.
- Rejected Floyd–Warshall closure: it shortened 74% of pairs by median 1.11 km — min-over-noise bias, not genuine routing.
- `ltf/data/traveltime.py` — 87×87×48 tensor, 3-level fallback. L0 covers 13.5% of cells but **90.7% of trips**. Peak/off-peak ratio median **2.11×**, max 4.35×.
- `ltf/utility.py` — `UtilityModel` with `assert_reduction()` self-test.

**Phase 1 verification: all checks pass.** Headline finding: switching to the time-aware utility **flips the sign of utility on 10.2%** of well-measured (driver, request, time) triples.

---

## Turn 3–4 — Phase 2 and Phase 3 core

**User:** `please continue !!` / `please continue !! once done then explain everything in hinglish`

**Assistant:**

**Phase 2 — driver simulator.** `ltf/sim/{timeline,requests,drivers,scenario}.py`

Two protocol corrections: test week shifted to **25–31 March** (01/04 is in the April file; 6 days would break the 3/1/3 split), and `target_utilisation` renamed `target_offered_load` for precision.

Three bugs found and fixed:
1. Night shifts painted by hour-of-day *within* a day wrapped a 23:00 shift into the same day's 00:00–05:00 slots, creating discontiguous runs. Fixed by painting on the absolute time axis.
2. `tsp.dayofweek.to_numpy()` failed on a numpy array.
3. Analytic sample-rate calibration overshot (0.855 vs 0.60 target). Added a measured correction loop → converges to 0.602 in one step.

Fleet: 200 drivers, 38% full-time (46.4 h/week), 62% part-time (14.0 h) — 3.32× separation. 825 contiguous shifts, 63,120 online driver-slots. 64/2016 zero-supply slots (3.2%), holding 1.70% of demand.

**Gap 1b demonstrated on the real fleet:** equalising totals gives `Var(total) = 0` ("perfectly fair") while `Var(rate) = 7,859` — hourly rates spanning 47.9 to 375.7, a **7.8× hidden gap**, with part-timers on 3.59× the full-time rate.

**Phase 3 core** — `ltf/sim/env.py` (occupancy-aware), `ltf/metrics/fairness.py`, `ltf/methods/{base,greedy}.py`. All invariants pass.

**Major new finding, stronger than "the metric is blind":**

| | FT rate | PT rate | between-group Var |
|---|---|---|---|
| efficiency-only | 2.28 | 2.42 | 0.004 |
| paper's Greedy (Eq. 3) | 1.89 | 3.13 | **0.361 (90× worse)** |

Optimising `Var(totals)` **actively creates** the group inequity its metric cannot see. Levelling totals across unequal hours can only transfer rate from long-hours to short-hours drivers.

Also corrected an overstatement in my own output: I had *not* reproduced the paper's negative-utility Greedy pathology, because our greedy stops when marginal gain ≤ 0 (using the paper's own no-action option). Documented as a deliberate choice.

Then explained everything in Hinglish across all phases.

---

## Turn 5 — Remaining phases, concluded

**User:**
> ok fine now please continue as your recommendation one by one also i dont want to drop any of the gaps(1 or 2 ) never assume anything please!

*(and earlier: proceed with remaining phases and conclude the research)*

**Assistant:**

torch 2.13.0 installed, **MPS (Apple GPU) available**.

**Forecaster** (`ltf/predict/forecaster.py`) — two heads (demand + congestion) sharing a trunk. First attempt diverged: the 336-hour seasonal lag left only 72 training hours, collapsing day-of-week feature variance and blowing up normalisation (MSE 12,129). Fixed by using lags ≥168 h only, a 48-hour validation window, an SD floor and feature clipping.

Final: demand MSE **6.455** vs seasonal-naive 8.856 (**27.1% better**); congestion head MSE 0.01198, MAE 0.0826.

**MOMAQL + baselines** — `ltf/methods/{momaql,classical}.py`: REASSIGN (two-stage bounded-loss), LAF (MDP re-weighting + Hungarian), Balance Ride-Pooling (RL, no prediction), Kang reproduced, Ours.

**Critical bug caught via ablation.** The first ablation table came out *backwards*. Cause: the paper's hard-coded `ω = 0.6` does not transfer — the utilisation penalty came out ≈0.009 against a typical trip utility of 2.4, roughly **270× too small**, leaving the Gap 2 term **inert**. Fixed with `calibrate_omegas()`, measuring scale from an efficiency-only warm-up: `ω_total=1.64, ω_ρ=1.11e3, ω_ν=5.96e4`. All ablations then behaved correctly.

**Sweep, with both gaps active in every configuration (enforced by assertion).** Coordinate search over 20 configs; parity found outside the obvious range (λ_between=16, λ_util=0.5). **12 of 20 configurations beat the paper on all six metrics simultaneously.**

**Robustness:** money units ($6.62/h vs $12.35/h for the paper → $8.20 vs $7.93 for ours), 3 seeds (paper's ratio 1.83–1.86 *every* seed — systematic), and the paper's own protocol, which confirmed **0 of 200 drivers** can reach the full-time threshold, 2.6% service rate, utilisation pinned at 0.99.

### Final results

| Method | Total Utility | Var(total) | Var(ρ) | between | FT/h | PT/h | ratio | Var(ν) | adj | idle |
|---|---|---|---|---|---|---|---|---|---|---|
| Greedy | 11,708 | 701 | 1.809 | 0.530 | 1.78 | 3.28 | 1.84 | 0.0202 | 0.426 | 1 |
| REASSIGN | 11,437 | 869 | 1.439 | 0.306 | 1.83 | 2.97 | 1.62 | 0.0214 | 0.401 | 2 |
| LAF | 11,704 | 635 | 1.641 | 0.597 | 1.75 | 3.34 | 1.91 | 0.0216 | 0.439 | 1 |
| Balance RP | 10,373 | 671 | 1.238 | 0.252 | 1.66 | 2.69 | 1.62 | 0.0140 | 0.275 | 0 |
| Kang (repro) | 10,221 | 657 | 1.251 | 0.247 | 1.64 | 2.67 | 1.62 | 0.0129 | 0.253 | 1 |
| **Ours** | 10,145 | 1,639 | **0.446** | **0.001** | 1.93 | 1.86 | **0.97** | **0.0081** | **0.155** | **0** |

Horizon: the paper's group inequity **grows 16.5×** over the week; ours stays flat, two orders of magnitude lower.

Honest negatives reported: prediction module gives no benefit (paper's 41% claim not reproduced); Gap 1a's efficiency gain is ~1%, its value is measurement correctness; `Var(total)` worse **by construction**.

---

## Turn 6 — Simple explanation

**User:**
> i dont understand a single word of results and what you achived/done!
> can you expalin me simplly what you have done how and why! with example as simple as possible please!
> ... also what changes you did in dataset why?! in hinglish pls

**Assistant:**

Explained from scratch in Hinglish with concrete examples: the ride-hailing allocation problem, the paper's two formulas, both gaps (with the ₹6,000/60h vs ₹6,000/15h example and the 2-trips-vs-10-trips idle example), all dataset changes with reasons, and the results reduced to: *"0.7% kamai ki qurbani de kar, humne per-hour barabari le aayi aur koi driver khaali nahi baitha."*

---

## Turn 7 — Professor documents

**User:** *(requested a concise but complete English doc — methodology, equations, results in tables, comparison with the paper)*

**Assistant:** Created `REPORT.md` (616 lines, 9 sections, 20+ tables) and `scripts/show_results.py` — an instant terminal summary reading saved CSVs.

---

## Turn 8 — The scale question

**User:**
> are you using the complete datasets as the utility number as presented in the paper, as i can see in the paper its a big number.. but in ours result its not that much ... why so can you explain what have you actually used as dataset and if not coplmplete why!?

**Assistant:**

This question exposed something hidden in the paper.

**All the cleaned data is used** — 100% of 10,300,738 trips build the graph, distance matrix, travel-time tensor, demand tensor and forecaster. Only the allocation *experiment* runs on a thinned stream (6,604 requests), which is what the paper does too (0.05 stratified sample).

**The paper used 20 drivers, not 200.** Never stated in the text, but Total ÷ Mean = *n*:

| Method | Total | Mean | Implied n |
|---|---|---|---|
| Greedy | −1,514,736.24 | −75,736.81 | **20.000** |
| REASSIGN | 76,536.23 | 3,826.81 | **20.000** |
| LAF | 80,606.49 | 4,030.3245 | **20.000** |
| Balance RP | 85,923.68 | 4,296.18 | **20.000** |
| Proposed | 95,823.79 | 4,791.19 | **20.000** |

**Their per-driver utility exceeds physical capacity by ~36×.** 4,791 utility ÷ 2.49 km per trip ≈ 1,924 trips/driver/week, against a maximum of ~53 in their own 2 h/day window. Explained by Sec. 4.3's unlimited concurrency.

**Thinning is deliberate:** capacity 11,006 trips vs 2,231,191 available = 203× oversubscribed, which would pin utilisation at 1.0 and make `Var(ν)` identically zero for every method — destroying Gap 2's metric.

Added `scripts/explain_scale.py` and Section 5 of `REPORT.md`.

---

## Turn 9 — Equations document

**User:** *(asked for all paper equations explained, all new equations explained, and confirmation of mathematical correctness)*

**Assistant:** Created `EQUATIONS.md` (586 lines): Part A (11 paper equations, term by term with worked examples), Part B (16 new equations N1–N16), Part C (verification table), Part D (**honest classification**).

Verification run — all pass:

| Property | Result |
|---|---|
| `within + between = total` (1,000 cases) | max rel. err **6.0e−16** |
| Incremental marginals vs brute force (3,000 cases) | max abs. err **7.1e−15** |
| `U_time = U_paper` at `c ≡ 1` (300k cases) | max \|ΔU\| = **0.000e+00** |
| Congestion monotonicity, `ν ∈ [0,1]`, Gini bounds, CV | PASS |

Part D separated **provable identities** (N5 law of total variance, N11 exact algebra, N2's reduction) from **modelling choices** (N2's functional form, N8's normalisation, N10's calibration convention) from **approximations** (occupancy ceiling, profile advance). Also recorded five defects in the paper's own equations.

---

## Turn 10–11 — DOCX deliverables

**User:** *(requested DOCX with results, comparison, methodology; then asked to add terminal output)*

**Assistant:** Created `scripts/make_docx.py` → `outputs/MTP_Gap_Implementation_Report.docx` (30 headings, 21 tables, every number read from CSVs at generation time). Key section: **3.4 "How the results improve as each gap is implemented"** — a 4-stage ladder attributing gains:

| Stage | Objective | Var(ρ) | between | ratio | Var(ν) adj | idle |
|---|---|---|---|---|---|---|
| 0 | No fairness | 1.284 | 0.0062 | 1.08 | 0.2484 | 4 |
| 1 | + Paper's Var(totals) | 1.251 | **0.2473** | **1.62** | 0.2527 | 1 |
| 2 | + Gap 1a & 1b | **0.426** | **0.0005** | **0.98** | 0.2224 | 0 |
| 3 | + Gap 2 (full) | 0.446 | 0.0011 | 0.97 | **0.1550** | 0 |

Then added **Appendix B** with 492 lines of real terminal output from four scripts, plus the commands to run them live.

---

## Turn 12 — Plain student-voice update

**User:**
> in concise, what we implemented, results, how we implemented and equtions and methdology! ... all terms explained ... keep examples and comaprison with current paper! provide a docx for that, not be damn ai generated be simple, human wriiten things and like complete student!

**Assistant:** Created `scripts/make_update_docx.py` → `outputs/MTP_Progress_Update.docx` (~8 pages, 18 tables, 3,800 words). First person, opens *"Sir, this is a short update on what I have implemented so far."* Every symbol explained in a table after each formula. Section 5.1 defines every result column *before* showing numbers. Section 6 handles the scale question. Section 7 lists what didn't work.

---

## Turn 13 — Explaining the objective, weights and state

**User:** *(asked for a simple explanation of the Score formula, the w1/w2/w3 weights and the ω scaling problem, and the state augmentation)*

**Assistant:**

First corrected my own overstatement: I had written "three orders of magnitude too small" in five places, but 2.4 / 0.009 = **267×**, i.e. 2.4 orders. Fixed in `REPORT.md`, `EQUATIONS.md`, `ltf/methods/base.py` and both DOCX generators.

Then explained:

1. **Score** turns earnings and three kinds of fairness into one comparable number, with a worked example where driver B wins a trip (Score 5.7) over driver A (Score 2.8) despite earning 1 km less.
2. **The weights** convert between units — earnings in km (~2.4) versus a variance that moves by ~0.00004 per trip. The paper's 0.6 made the penalty 267× too weak. Caught by ablation (removing the term changed nothing), fixed by measuring instead of guessing (→ 59,600).
3. **The state change** was necessary because a model that cannot *see* who is behind cannot *help* who is behind. Analogy: a teacher distributing extra classes fairly while only being told which classroom each student sits in. Added rate-deficit and utilisation-deficit buckets → 48 × 87 × 5 = 20,880 entries.

---

## Turn 14 — Git upload

**User:**
> i want to upload this on git with new branch on the REPO https://github.com/Utkarsh63-byte/MTP-LONG_TERM_FAIRNESS_GAP_IMPL

**Assistant:**

Checked state first: no git repo, no `gh` CLI, remote **exists, is public, and is empty** (0 refs).

Created `.gitignore`, `requirements.txt` (pinned versions), and `README.md` (repo front page with the two gaps, headline results, quick start, equation table, layout, and a scope/honesty section).

Three decisions flagged:

1. **Excluded the Springer LNCS PDF** and its extracted text — the repo is public, and redistributing the publisher version is a copyright risk. README links to arXiv instead.
2. **Excluded the raw CSV (1.8 GB) and `cache/trips.parquet` (132 MB)** — the latter exceeds GitHub's 100 MB hard limit. Both regenerable; documented.
3. **Kept `outputs/`** so `show_results` and `explain_scale` work immediately after cloning.

Verified before committing: 76 files, 1.9 MB, no `.venv`/`__pycache__`/data leakage, nothing over 50 MB.

Committed on branch `gap-implementation` (commit `afcd370`) and pushed successfully. Local and remote in sync, working tree clean.

**Branch:** https://github.com/Utkarsh63-byte/MTP-LONG_TERM_FAIRNESS_GAP_IMPL/tree/gap-implementation

Noted that since the repo was empty, they may want to set the branch as default or create `main` from it.

---

## Summary of deliverables

| Artifact | Description |
|---|---|
| `ltf/` | Package: config, utility, data pipeline, simulator, metrics, 5 methods, forecaster |
| `scripts/` | 12 scripts: verification, runners, figures, report generators |
| `outputs/` | 11 CSVs, 6 figures, 2 DOCX, 3 logs, 4 terminal transcripts |
| `README.md` | Repo front page |
| `REPORT.md` | Full technical report (9 sections) |
| `EQUATIONS.md` | All 27 equations + correctness verification |
| `outputs/MTP_Gap_Implementation_Report.docx` | Formal report with terminal appendix |
| `outputs/MTP_Progress_Update.docx` | Short plain-language update |

## Core scientific claims established

1. Gap 1a is real: same OD pair varies **2.11×** in travel time; the paper's utility mislabels **10.2%** of assignments.
2. Gap 1b is real *and the paper's objective causes it*: enabling its fairness term worsens between-group inequity **90×**; all five prior methods land at a **1.62–1.91×** part-time premium, on every seed.
3. Gap 2 was previously unmeasurable: the paper permits unbounded concurrency, so "busy" has no definition.
4. Our fix reaches group pay parity (**0.97** vs 1.62), cuts rate inequity **64%** and normalised utilisation inequity **39%**, eliminates idle drivers — for **0.7%** of total utility.
5. `Var(total)` and rate fairness are **mathematically incompatible**; the paper optimises the wrong one.
6. Reproducibility findings: the paper's fleet size (n=20) is unreported but recoverable; its per-driver utility exceeds its own protocol's capacity by **36×**; its `ω = 0.6` does not transfer and silently disables a fairness term.

## Open next steps

- Fleet-size sensitivity (100 / 200 / 400 drivers)
- Paper draft — the λ_between parity curve is already in `outputs/sweep_lambda_between.csv`
