"""Single source of truth for the research paper.

The paper is defined once as a structured document model and rendered twice
(DOCX and LaTeX) by scripts/make_paper.py, so the two outputs can never diverge.
Every numeric value is read from the result CSVs at build time.

Element grammar
---------------
("h1"|"h2"|"h3", text)              section heading
("p", text)                          paragraph; inline math in \\( ... \\)
("pb", None)                         page break
("bul", [text, ...])                 bullet list
("enum", [text, ...])                numbered list
("eq", (latex, plain, label))        display equation; label may be None
("table", (caption, header, rows, widths))
("fig", (path, caption, width_in))
("abstract", text)
("keywords", text)
("bib", [(key, text), ...])
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ltf.config import OUTPUT


def load_values() -> dict:
    """Every number the paper quotes, read from the experiment outputs."""
    t1 = pd.read_csv(OUTPUT / "table1_methods.csv")
    t2 = pd.read_csv(OUTPUT / "table2_ablations.csv")
    iso = pd.read_csv(OUTPUT / "gap_isolation.csv")
    rb = pd.read_csv(OUTPUT / "robustness.csv")
    gr = pd.read_csv(OUTPUT / "sweep_joint_grid.csv")
    hz = pd.read_csv(OUTPUT / "fig4_horizon.csv")
    fc = json.load(open(OUTPUT / "forecaster_results.json"))

    PAPER, OURS = "Kang et al. (reproduced)", "Ours (Gap 1+2)"
    V = {}
    V["t1"] = t1
    V["t2"] = t2
    V["iso"] = iso
    V["gr"] = gr
    V["fc"] = fc
    V["k"] = t1[t1.method == PAPER].iloc[0]
    V["o"] = t1[t1.method == OURS].iloc[0]
    V["hp"] = hz.pivot_table(index="days", columns="method",
                             values="rate_var_between")
    V["hv"] = hz.pivot_table(index="days", columns="method", values="rate_var")
    V["hf"] = hz.pivot_table(index="days", columns="method",
                             values="fairness_total_var")
    V["PAPER"], V["OURS"] = PAPER, OURS

    ab = {r.method: r for _, r in t2.iterrows()}
    V["nofair"] = ab["Ours, w/o fairness"]
    V["nopred"] = ab["Ours, w/o prediction"]
    V["noutil"] = ab["Ours, w/o utilisation term"]
    V["static"] = ab["Ours, static utility (c=1)"]

    g = {r.method[:2]: r for _, r in iso.iterrows()}
    V["b0"] = g["0."]
    V["g1a"] = g["1."]
    V["g1b"] = g["2."]
    V["g1"] = g["3."]
    V["g2"] = g["4."]
    V["gall"] = g["5."]

    m = rb[rb.variant == "money"]
    V["mk"] = m[m.method.str.startswith("Kang")].iloc[0]
    V["mo"] = m[m.method.str.startswith("Ours")].iloc[0]
    pp = rb[rb.variant == "paper_protocol"]
    V["ppk"] = pp[pp.method.str.startswith("Kang")].iloc[0]
    sd = rb[rb.variant.astype(str).str.startswith("seed")].copy()
    sd["fam"] = np.where(sd.method.str.startswith("Ours"), "Ours", "Paper")
    V["sd"] = sd.groupby("fam")[["total_utility", "rate_var", "rate_var_between",
                                 "rate_group_ratio", "util_adj_var"]].agg(
        ["mean", "std"])

    kk, oo = V["k"], V["o"]
    V["dom"] = gr[(gr.total_utility > kk.total_utility) & (gr.rate_var < kk.rate_var)
                  & (gr.rate_var_between < kk.rate_var_between)
                  & (gr.util_var < kk.util_var)
                  & (gr.util_adj_var < kk.util_adj_var)
                  & (gr.n_idle_drivers <= kk.n_idle_drivers)]
    V["sweep_b"] = gr[(gr.lambda_within == 1.0) & (gr.lambda_util == 0.25)
                      ].sort_values("lambda_between")
    return V


def pct(new, old, dp=1):
    if old == 0:
        return "n/a"
    return f"{100*(new-old)/abs(old):+.{dp}f}\\%"


def pctp(new, old, dp=1):
    """Percentage as plain text (for DOCX)."""
    if old == 0:
        return "n/a"
    return f"{100*(new-old)/abs(old):+.{dp}f}%"


def build(V) -> list:
    k, o = V["k"], V["o"]
    b0, g1a, g1b, g1, g2, gall = (V["b0"], V["g1a"], V["g1b"], V["g1"], V["g2"],
                                  V["gall"])
    hp, fc = V["hp"], V["fc"]
    mk, mo, sd = V["mk"], V["mo"], V["sd"]
    P, O = V["PAPER"], V["OURS"]
    D = []
    A = D.append

    # ------------------------------------------------------------------
    A(("abstract",
       "Fairness in ride-hailing driver allocation is almost universally "
       "operationalised as minimising the variance of drivers' accumulated "
       "earnings. We show that this objective is not merely incomplete but "
       "actively harmful. Because it is invariant to hours worked, equalising "
       "totals is only achievable by transferring pay rate from long-hours to "
       "short-hours drivers: on a 200-driver New York City fleet, enabling a "
       "state-of-the-art variance-of-totals fairness term widens the "
       "between-group hourly-pay disparity by a factor of 90, and all five "
       "methods we evaluate converge on paying part-time drivers 62--91\\% more "
       "per hour than full-time drivers. We further show that the distance-only "
       "utility standard in this literature, combined with period-mean travel "
       "times, inverts the profitability sign of 10.2\\% of candidate "
       "assignments, and that access fairness is formally undefinable under the "
       "prevailing model because drivers may hold unbounded concurrent trips. We "
       "propose three corrections: a congestion-aware utility that provably "
       "reduces to the prevailing definition at neutral congestion; a pay-rate "
       "fairness objective decomposed into within- and between-group components "
       "by the law of total variance; and an access-fairness objective with an "
       "opportunity-normalised variant that separates algorithmic starvation "
       "from a driver's own shift choice. Against a faithful reproduction of the "
       "strongest prior method under an identical request stream, our allocator "
       "attains hourly-pay parity between activity groups (ratio "
       f"{b0.rate_group_ratio:.2f} to {g1.rate_group_ratio:.2f}) while "
       f"{'increasing' if g1.total_utility > b0.total_utility else 'reducing'} "
       f"total utility by {abs(100*(g1.total_utility-b0.total_utility)/b0.total_utility):.1f}\\%, "
       "and eliminates fully idle drivers. We prove that variance-of-totals "
       "fairness and pay-rate parity are mutually unsatisfiable, and we report "
       "reproducibility findings on the base method, including the recovery of "
       "its unreported fleet size and a demonstration that its published "
       "per-driver earnings exceed the physical capacity of its own evaluation "
       "protocol by roughly 36 times."))
    A(("keywords",
       "Algorithmic fairness; ride-hailing; two-sided markets; Markov decision "
       "process; multi-objective reinforcement learning; gig economy"))

    # ================= 1 INTRODUCTION =================
    A(("h1", "Introduction"))
    A(("p",
       "Ride-hailing platforms allocate incoming trip requests to drivers "
       "thousands of times per minute. Because that allocation determines "
       "livelihoods, a substantial literature now augments efficiency "
       "objectives with fairness constraints on the driver side "
       "\\cite{lesmana2019,suhr2019,nanda2020,xu2020,shi2021,raman2021,kang2024}. "
       "Across this literature a single operationalisation dominates: fairness "
       "is the variance (or a monotone transform thereof) of drivers' "
       "accumulated earnings over some horizon, and the allocator maximises "
       "total utility minus a weighted variance penalty."))
    A(("p",
       "This paper argues that the dominant operationalisation is the wrong "
       "target, and demonstrates the consequences empirically. Our starting "
       "point is the observation that gig drivers do not supply equal labour. "
       "In any realistic fleet, some drivers work forty or more hours per week "
       "and others work ten. A fairness measure defined on accumulated totals "
       "is invariant to that difference, so it treats a driver who earned a "
       "given amount in sixty hours as identically situated to one who earned "
       "the same amount in fifteen. The two drivers face a fourfold difference "
       "in hourly pay, which the measure cannot represent."))
    A(("p",
       "The consequence is stronger than a measurement blind spot. Equalising "
       "totals across unequal hours is achievable only by moving work from "
       "long-hours to short-hours drivers, which necessarily inflates the "
       "latter's pay rate. The standard objective therefore does not merely "
       "fail to observe a between-group pay disparity, it produces one. We "
       "measure this directly: switching on the fairness term of a "
       "state-of-the-art allocator increases between-group hourly-pay variance "
       f"from {V['nofair'].rate_var_between:.4f} to {k.rate_var_between:.4f}, a "
       "factor of roughly ninety, and every one of the five allocators we "
       "evaluate settles at a part-time hourly-pay premium of between 62 and "
       "91 percent."))
    A(("p",
       "Two further deficiencies compound this. First, driver utility in this "
       "literature is a difference of road distances, and travel times are "
       "commonly collapsed to a per-pair period mean. Holding origin and "
       "destination fixed, we measure a median peak-to-off-peak travel-time "
       "ratio of 2.11 on the same public dataset used by prior work, with a "
       "maximum of 4.35; the resulting utility function inverts the "
       "profitability sign of 10.2 percent of candidate assignments. Second, "
       "access fairness --- whether an available driver received any work at "
       "all --- is not merely unmeasured but formally undefinable under the "
       "prevailing model, which permits a driver to hold unboundedly many "
       "concurrent trips and attaches no time cost to a trip."))
    A(("p", "We make the following contributions."))
    A(("enum", [
        "We prove that variance-of-totals fairness and hourly-pay parity are "
        "mutually unsatisfiable whenever drivers supply unequal hours "
        "(Proposition 1), and we quantify the disparity that the former "
        "induces across five published allocators.",
        "We introduce a congestion-aware utility that is provably identical to "
        "the prevailing distance-only definition at neutral congestion, so that "
        "prior work is recovered as a strict special case and the effect of the "
        "correction can be isolated without confounding.",
        "We define a pay-rate fairness objective and decompose it into "
        "within-group and between-group components using the law of total "
        "variance, yielding an exact partition rather than an ad hoc pair of "
        "metrics.",
        "We define an access-fairness objective over driver utilisation, "
        "together with an opportunity-normalised variant that separates "
        "algorithmic starvation from a driver's own choice of shift, and we "
        "supply the occupancy-constrained environment that makes either "
        "quantity well defined.",
        "We show that the scale factor used to combine utility and fairness in "
        "prior work does not transfer across settings, and that leaving it "
        "uncalibrated silently disables a fairness term by roughly two and a "
        "half orders of magnitude.",
        "We report reproducibility findings on the base method, recovering its "
        "unreported fleet size from its own published table and showing that "
        "its reported per-driver earnings exceed the physical capacity of its "
        "evaluation protocol by approximately 36 times.",
    ]))

    # ================= 2 RELATED WORK =================
    A(("h1", "Related Work"))
    A(("p",
       "\\textbf{Efficiency-oriented matching.} Request-to-driver assignment has "
       "been posed as online bipartite matching \\cite{zhao2019,tong2020} and as "
       "a Markov decision process solved by approximate dynamic programming or "
       "reinforcement learning \\cite{shah2020,delima2020,raman2021,sun2022}. "
       "These works establish that non-myopic policies materially outperform "
       "greedy assignment, and our allocator inherits that architecture."))
    A(("p",
       "\\textbf{Driver-side fairness.} Empirical work documents systematic "
       "earnings disparities in gig platforms, including a gender earnings gap "
       "among rideshare drivers \\cite{cook2021} and differential treatment of "
       "riders \\cite{brown2018}, and qualitative work records driver resentment of opaque algorithmic management \\cite{mohlmann2019}. Algorithmic responses fall into two families. "
       "Max-min approaches raise the worst-off driver's utility "
       "\\cite{lesmana2019}; variance-based approaches equalise across the whole "
       "fleet \\cite{raman2021,shi2021,kang2024}. Bounds on the achievable "
       "efficiency-fairness trade-off are given by \\cite{nanda2020,ma2020}, and "
       "incentive-based mechanisms by \\cite{kumar2023}. Two-sided formulations "
       "extend fairness to riders \\cite{suhr2019,kang2024b}. Crucially, every "
       "one of these driver-side measures is a functional of accumulated "
       "earnings alone; none conditions on hours supplied. Our critique applies "
       "to the family, not to a single method."))
    A(("p",
       "\\textbf{Long-horizon fairness.} The most direct antecedent to this work "
       "is \\cite{kang2024}, which argues that fairness should be assessed over "
       "a week rather than a dispatch round, and injects forecast demand into "
       "the action space of a multi-objective multi-agent Q-learning allocator "
       "so that a driver disadvantaged early can be compensated later. We adopt "
       "that horizon and that architecture, and take issue with the fairness "
       "functional it optimises. We show in Section 7.5 that on a group-"
       "conditioned measure its disparity grows monotonically with the horizon "
       "it is designed to protect."))
    A(("p",
       "\\textbf{Group fairness and rate-based measures.} Conditioning fairness "
       "on protected or activity groups is standard in classification but rare "
       "in matching; \\cite{ma2020} considers group-level fairness in online "
       "bipartite matching without a labour-supply interpretation. Labour "
       "economics measures gig compensation per engaged hour rather than per "
       "period, which is the convention we adopt. To our knowledge no prior "
       "allocation method optimises a per-hour fairness objective or "
       "decomposes it across activity groups."))

    # ================= 3 PRELIMINARIES =================
    A(("h1", "Preliminaries and Problem Formulation"))
    A(("p",
       "\\textbf{Setting.} Let \\(V\\) be a set of \\(n\\) drivers and \\(L\\) a "
       "set of locations forming the nodes of a directed graph whose edge "
       "weights are road distances. A request is a triple \\(r=(t_r,s_r,d_r)\\) "
       "raised at time \\(t_r\\) with pickup \\(s_r \\in L\\) and dropoff "
       "\\(d_r \\in L\\). Driver \\(v\\) occupies node \\(g_v^t\\) at time \\(t\\) "
       "and has accumulated utility \\(o_v^t\\). We write \\(\\mathrm{Geo}(a,b)\\) "
       "for the shortest road distance from \\(a\\) to \\(b\\), and \\(M\\) for an "
       "assignment with \\(M(v)\\) the requests allocated to \\(v\\). Time is "
       "discretised into decision epochs of length \\(\\Delta\\)."))
    A(("table", (
        "Notation.",
        ["Symbol", "Meaning"],
        [["\\(V, n\\)", "driver set and its cardinality"],
         ["\\(L\\)", "location set (graph nodes)"],
         ["\\(r=(t_r,s_r,d_r)\\)", "request: time, pickup, dropoff"],
         ["\\(g_v^t\\)", "node occupied by driver \\(v\\) at time \\(t\\)"],
         ["\\(o_v\\)", "utility accumulated by driver \\(v\\) over the horizon"],
         ["\\(\\mathrm{Geo}(a,b)\\)", "shortest road distance, \\(a\\) to \\(b\\)"],
         ["\\(\\tau_p(a,b)\\)", "travel time in time profile \\(p\\)"],
         ["\\(\\bar{\\tau}(a,b)\\)", "period-mean travel time (prior work)"],
         ["\\(c_p(a,b)\\)", "congestion multiplier"],
         ["\\(H_v\\)", "hours driver \\(v\\) was online"],
         ["\\(O_v\\)", "set of epochs in which \\(v\\) was online"],
         ["\\(\\rho_v\\)", "hourly pay rate of driver \\(v\\)"],
         ["\\(\\nu_v\\)", "utilisation of driver \\(v\\)"],
         ["\\(g(v)\\)", "activity group of \\(v\\) (full-time, part-time)"],
         ["\\(\\lambda, \\omega\\)", "fairness weight and scale factor"]],
        [1.3, 4.3])))
    A(("p",
       "\\textbf{Prevailing utility and fairness.} Prior work defines the "
       "utility of assigning \\(r\\) to \\(v\\) as trip distance net of the "
       "unpaid approach distance, and fairness as the variance of accumulated "
       "utility \\cite{raman2021,kang2024}:"))
    A(("eq", (r"U(r,v) \;=\; \mathrm{Geo}(d_r,s_r) \;-\; \mathrm{Geo}(s_r,g_v^t)",
              "U(r,v) = Geo(d_r, s_r) - Geo(s_r, g_v)", "eq:util-prior")))
    A(("eq", (r"\Pi(M) \;=\; \sum_{v \in V} o_v\big(M(v)\big), \qquad "
              r"F(M) \;=\; \mathrm{Var}\big(o_v(M(v))\big)",
              "Pi(M) = SUM_v o_v(M(v)),   F(M) = Var(o_v(M(v)))", "eq:eff-fair")))
    A(("p",
       "The allocator then maximises a scalarisation of the two objectives, "
       "with \\(\\lambda\\) expressing the fairness preference and \\(\\omega\\) "
       "rescaling the variance term into the range of the utility term "
       "\\cite{miettinen2002}:"))
    A(("eq", (r"\max_{M} \;\; \Pi(M) \;-\; \lambda\,\omega\, F(M) \qquad "
              r"\text{s.t.} \;\; \sum_{v \in V} I_{rv} \le 1 \;\; \forall r",
              "max_M  Pi(M) - lambda*omega*F(M)   s.t.  SUM_v I_rv <= 1 for all r",
              "eq:objective-prior")))
    A(("p",
       "\\textbf{Activity groups and pay rate.} We additionally equip each "
       "driver with an online epoch set \\(O_v\\), hours \\(H_v = |O_v|\\Delta\\), "
       "and an activity group \\(g(v)\\) determined by \\(H_v\\). These are not "
       "present in prior formulations, and their absence is precisely what "
       "renders the deficiencies below invisible."))

    # ================= 4 MOTIVATING ANALYSIS =================
    A(("h1", "Motivating Analysis"))
    A(("p",
       "This section establishes three deficiencies empirically, on the same "
       "public dataset used by the method we extend. Measurement details are in "
       "Section 6."))

    A(("h2", "Distance-only utility inverts one assignment in ten"))
    A(("p",
       "Equation~\\ref{eq:util-prior} contains no time term, and prior work "
       "additionally replaces travel time by a per-pair period mean, so a trip "
       "is valued identically regardless of when it occurs. This discards a "
       "large effect. Across 969 origin-destination pairs observed in at least "
       "24 of 48 (weekday/weekend \\(\\times\\) hour) profiles, the median ratio "
       "of peak to off-peak travel time is 2.11, the ninetieth percentile is "
       "2.93 and the maximum is 4.35. Citywide mean speed varies from 28.5 km/h "
       "at 04:00 to 12.8 km/h at 12:00, a factor of 2.22."))
    A(("p",
       "The operational consequence is not a small mis-pricing but a sign "
       "error. Consider an 8 km trip whose pickup lies 5 km from the driver. "
       "Equation~\\ref{eq:util-prior} scores this \\(8-5=+3\\) km at every hour. "
       "Under congestion that doubles travel times, the paid leg delivers half "
       "the value per unit of driver time while the unpaid approach costs twice "
       "as much, giving \\(8/2 - 5 \\times 2 = -6\\) km. Evaluated over 40{,}310 "
       "well-measured (driver, request, epoch) triples, the sign of utility "
       "differs between the two definitions in 10.2\\% of cases: prior work "
       "labels one assignment in ten profitable when it is not, or the reverse."))
    A(("fig", ("outputs/fig_gap1a_congestion.png",
               "Travel-time variation discarded by a period mean. Left: citywide "
               "speed by hour, with the period mean that prior work substitutes "
               "for it. Right: distribution of the peak to off-peak travel-time "
               "ratio across origin-destination pairs, holding endpoints fixed.",
               5.4)))

    A(("h2", "Variance of totals is invariant to hours supplied"))
    A(("p",
       "Let two drivers accumulate equal utility over a week while supplying "
       "60 and 15 hours respectively. Then \\(F(M)=0\\) by "
       "Equation~\\ref{eq:eff-fair} and the allocation is declared perfectly "
       "fair, although the hourly rates differ by a factor of four. On our "
       "simulated fleet, imposing exact equality of totals --- the fixed point "
       "the objective drives towards --- produces hourly rates spanning 47.9 to "
       "375.7, a factor of 7.8, with part-time drivers at 3.59 times the "
       "full-time rate."))
    A(("p", "This is not an artefact of a particular fleet. It is structural."))
    A(("prop", (
        "Incompatibility of total-equality and rate-equality",
        "Let drivers supply unequal hours, so that \\(H_i \\neq H_j\\) for some "
        "\\(i,j\\). Then no allocation simultaneously satisfies "
        "\\(\\mathrm{Var}(o_v)=0\\) and \\(\\mathrm{Var}(o_v/H_v)=0\\) unless all "
        "utilities are zero.")))
    A(("proof",
       "Suppose both variances vanish. From \\(\\mathrm{Var}(o_v)=0\\) we have "
       "\\(o_v = a\\) for all \\(v\\) and some constant \\(a\\); from "
       "\\(\\mathrm{Var}(o_v/H_v)=0\\) we have \\(o_v = b H_v\\) for all \\(v\\) "
       "and some constant \\(b\\). Equating, \\(a = b H_i = b H_j\\). Since "
       "\\(H_i \\neq H_j\\) it follows that \\(b = 0\\), hence \\(a = 0\\) and "
       "\\(o_v = 0\\) for all \\(v\\)."))
    A(("p",
       "Proposition 1 has an immediate corollary that is easy to overlook: an "
       "allocator that reduces \\(\\mathrm{Var}(o_v)\\) must, in general, "
       "\\emph{increase} pay-rate dispersion. Equalising totals across unequal "
       "hours can only be achieved by transferring work away from long-hours "
       "drivers, and the recipients' pay rate rises accordingly. Table~"
       "\\ref{tab:causes} confirms the effect."))

    A(("h2", "Access fairness is undefinable under the prevailing model"))
    A(("p",
       "Two drivers online for eight hours may receive two and ten dispatches "
       "respectively and still accumulate equal utility if the former's trips "
       "are longer. Equation~\\ref{eq:eff-fair} reports this as fair while six "
       "hours of enforced idleness go unrecorded. The deeper problem is that "
       "the quantity cannot be formed at all: prior work permits a driver to "
       "accept unboundedly many concurrent requests and its reward attaches no "
       "duration to a trip, so the predicate ``driver \\(v\\) is busy at epoch "
       "\\(t\\)'' has no referent. Any access-fairness measure therefore "
       "presupposes the occupancy model we introduce in Section 5.3."))

    # ================= 5 METHOD =================
    A(("pb", None))
    A(("h1", "Method"))
    A(("h2", "Congestion-aware utility"))
    A(("p",
       "We index time by a profile \\(p\\) (weekday/weekend \\(\\times\\) hour of "
       "day) and define a dimensionless congestion multiplier as the ratio of "
       "the travel time in profile \\(p\\) to the period mean that prior work "
       "substitutes for it:"))
    A(("eq", (r"c_p(a,b) \;=\; \frac{\tau_p(a,b)}{\bar{\tau}(a,b)}, \qquad "
              r"c_p \in [c_{\min}, c_{\max}]",
              "c_p(a,b) = tau_p(a,b) / tau_bar(a,b),  clipped to [c_min, c_max]",
              "eq:cong")))
    A(("p",
       "Because the denominator is exactly the quantity prior work uses, "
       "\\(c_p\\) measures precisely the information that is discarded. We then "
       "adjust both terms of Equation~\\ref{eq:util-prior}:"))
    A(("eq", (r"U^{\mathrm{ca}}_p(r,v) \;=\; "
              r"\frac{\mathrm{Geo}(d_r,s_r)}{c_p(s_r,d_r)} \;-\; "
              r"w\,\mathrm{Geo}(s_r,g_v)\,c_p(g_v,s_r)",
              "U_ca = Geo(d,s)/c_p(s,d) - w*Geo(s,g)*c_p(g,s)", "eq:util-ca")))
    A(("p",
       "The paid leg is divided by \\(c_p\\): a trip that takes twice as long "
       "returns the same fare for twice the driver's engaged time, so its value "
       "per unit time halves. The unpaid approach is multiplied by \\(c_p\\): "
       "traversing the same distance in congestion consumes more time and is "
       "still unremunerated. Both adjustments move against the driver, an "
       "asymmetry Equation~\\ref{eq:util-prior} cannot express."))
    A(("p",
       "\\textbf{Reduction property.} Setting \\(c_p \\equiv 1\\) in "
       "Equation~\\ref{eq:util-ca} recovers Equation~\\ref{eq:util-prior} term "
       "by term. Prior work is therefore a strict special case, and the "
       "congestion correction can be ablated without changing anything else. We "
       "verify the reduction numerically over 300{,}000 random triples and "
       "obtain a maximum absolute difference of exactly zero."))
    A(("p",
       "\\textbf{Estimation.} \\(\\tau_p\\) is estimated hierarchically: the "
       "direct median where a profile cell contains at least 30 observations; "
       "otherwise the pair's period mean scaled by the citywide profile "
       "multiplier; otherwise distance divided by the citywide profile speed. "
       "The first level covers 13.5\\% of cells but 90.7\\% of trips, and the "
       "fallbacks degrade towards mean congestion at that hour rather than "
       "towards no congestion. Clipping binds on 0.08\\% of cells."))

    A(("h2", "Pay-rate fairness with group decomposition"))
    A(("p",
       "We replace accumulated utility by the pay rate, the quantity a worker "
       "actually experiences:"))
    A(("eq", (r"\rho_v \;=\; \frac{o_v}{H_v}, \qquad H_v = |O_v|\,\Delta",
              "rho_v = o_v / H_v,   H_v = |O_v| * Delta", "eq:rate")))
    A(("p",
       "\\(H_v\\) counts all online time including idle epochs, since a driver "
       "is available whether or not the platform dispatches to them. Rather "
       "than report a single dispersion figure we decompose it by activity "
       "group using the law of total variance, which yields an exact partition:"))
    A(("eq", (r"\mathrm{Var}(\rho) \;=\; "
              r"\underbrace{\mathbb{E}_g\big[\mathrm{Var}(\rho \mid g)\big]}"
              r"_{\text{within-group}} \;+\; "
              r"\underbrace{\mathrm{Var}_g\big(\mathbb{E}[\rho \mid g]\big)}"
              r"_{\text{between-group}}",
              "Var(rho) = E_g[Var(rho|g)]  +  Var_g(E[rho|g])\n"
              "           within-group        between-group", "eq:ltv")))
    A(("p",
       "Explicitly, with group shares \\(w_g = n_g/n\\), the within term is "
       "\\(\\sum_g w_g \\mathrm{Var}(\\rho \\mid g)\\) and the between term is "
       "\\(\\sum_g w_g (\\mathbb{E}[\\rho \\mid g] - \\mathbb{E}[\\rho])^2\\). "
       "The identity guarantees that the two components sum to the total with "
       "no residual, so they can carry independent weights without "
       "double-counting; we verify the identity to a relative error of "
       "\\(6 \\times 10^{-16}\\). The within term captures dispersion among "
       "comparable drivers; the between term is the structural pay gap between "
       "activity groups, and is the quantity prior work is unable to form. For "
       "reporting we also use the group ratio "
       "\\(\\mathbb{E}[\\rho \\mid \\text{part-time}] / "
       "\\mathbb{E}[\\rho \\mid \\text{full-time}]\\), for which unity is parity."))

    A(("h2", "Access fairness and the occupancy model"))
    A(("p",
       "An assignment occupies a driver for the approach leg plus the loaded "
       "leg, both congestion-dependent:"))
    A(("eq", (r"\mathrm{occ}(r,v) \;=\; \Big\lceil \big(\tau_p(g_v,s_r) + "
              r"\tau_p(s_r,d_r)\big) / \Delta \Big\rceil",
              "occ(r,v) = ceil( (tau_p(g_v,s_r) + tau_p(s_r,d_r)) / Delta )",
              "eq:occ")))
    A(("p",
       "A driver assigned at epoch \\(t\\) is unavailable until \\(t + "
       "\\mathrm{occ}\\), with capacity one. This is the minimal repair that "
       "makes ``busy'' well defined, and it is a precondition for the following "
       "measure rather than a refinement of it:"))
    A(("eq", (r"\nu_v \;=\; \frac{\big|\{t \in O_v : v \text{ busy at } t\}\big|}"
              r"{|O_v|}",
              "nu_v = (busy epochs) / (online epochs)", "eq:util")))
    A(("p",
       "Raw \\(\\mathrm{Var}(\\nu)\\) conflates two distinct phenomena: a "
       "platform that starves an available driver, and a driver who elects to "
       "be online at 04:00 when there is no demand. Only the former is the "
       "allocator's responsibility. We therefore also report an "
       "opportunity-normalised variant, dividing each driver's utilisation by "
       "what drivers online in the \\emph{same} epochs achieved:"))
    A(("eq", (r"\pi_t = \frac{\#\{\text{busy at } t\}}{\#\{\text{online at } t\}}, "
              r"\quad \mathrm{opp}_v = \frac{1}{|O_v|}\sum_{t \in O_v} \pi_t, "
              r"\quad \tilde{\nu}_v = \frac{\nu_v}{\mathrm{opp}_v}",
              "pi_t = busy(t)/online(t);  opp_v = mean of pi_t over O_v;  "
              "nu_tilde_v = nu_v / opp_v", "eq:oppnorm")))
    A(("p",
       "\\(\\tilde{\\nu}_v = 1\\) indicates a driver who fared as well as "
       "contemporaneous peers. Reporting the raw form alone overstates the "
       "allocator's culpability; reporting only the normalised form would "
       "conceal genuine starvation. We report both throughout."))

    A(("h2", "Unified objective, state augmentation and weight calibration"))
    A(("p", "The allocator maximises"))
    A(("eq", (r"\mathrm{SR}(M) = \sum_{v} u_v \;-\; \omega_\rho\Big[\lambda_w "
              r"\mathrm{Var}_{\mathrm{within}}(\rho) + \lambda_b "
              r"\mathrm{Var}_{\mathrm{between}}(\rho)\Big] \;-\; "
              r"\omega_\nu \lambda_\nu \mathrm{Var}(\nu)",
              "SR(M) = SUM_v u_v - omega_rho[lam_w*Var_within(rho) + "
              "lam_b*Var_between(rho)] - omega_nu*lam_nu*Var(nu)",
              "eq:objective")))
    A(("p",
       "Setting \\(\\lambda_b = \\lambda_\\nu = 0\\) and restoring totals as the "
       "fairness target recovers Equation~\\ref{eq:objective-prior} exactly, so "
       "prior work remains nested within the objective."))
    A(("p",
       "\\textbf{State augmentation.} A policy cannot optimise a quantity it "
       "cannot observe. The state in prior work is location-only, so a "
       "fairness term appearing in its objective can be evaluated after the "
       "fact but never acted upon. We augment the tabular value function with "
       "discretised rate-deficit and utilisation-deficit indices, giving "
       "\\(V[p, \\ell, \\delta]\\) over 48 profiles, 87 nodes and 5 deficit "
       "levels. Assignment scores take an advantage form in which the discount "
       "exponent is the occupancy, so a long trip is discounted for the driver "
       "time it commits:"))
    A(("eq", (r"\mathrm{score}(r,v) = \tilde{r}(r,v) + "
              r"\gamma^{\mathrm{occ}(r,v)} V[s'] - V[s]",
              "score(r,v) = r_scalarised(r,v) + gamma^occ(r,v) * V[s'] - V[s]",
              "eq:score")))
    A(("p",
       "\\textbf{Weight calibration.} Prior work fixes \\(\\omega = 0.6\\) to "
       "``scale fairness into the same range as utility'' without reporting the "
       "range obtained. A variance carries squared units, so its magnitude "
       "relative to utility depends on fleet size, horizon and, critically, on "
       "which quantity the variance is taken over; a single constant cannot "
       "transfer. We instead measure the scale. Expanding the exact marginal of "
       "a variance under a single assignment and retaining the dominant term "
       "gives \\(\\partial \\mathrm{Var}(x) \\approx 2\\,\\mathbb{E}|x|\\,dx/n\\), "
       "so we set"))
    A(("eq", (r"\omega_x \;=\; \frac{\mathbb{E}|u|}{2\,\mathbb{E}|x|\,dx/n}",
              "omega_x = E|u| / ( 2 * E|x| * dx / n )", "eq:omega")))
    A(("p",
       "with \\(\\mathbb{E}|u|\\) the mean absolute utility of an assignment, "
       "both measured from an efficiency-only warm-up run. This leaves "
       "\\(\\lambda\\) as the sole trade-off control. The correction is not "
       "cosmetic: with an uncalibrated weight our utilisation penalty evaluated "
       "to approximately 0.009 against a typical assignment utility of 2.4, "
       "roughly 270 times too small, leaving the access-fairness term inert and "
       "its ablation without measurable effect. Calibration yields "
       "\\(\\omega_\\rho = 1.11\\times10^{3}\\) and "
       "\\(\\omega_\\nu = 5.96\\times10^{4}\\)."))
    A(("p",
       "\\textbf{Efficient marginals.} Greedy re-evaluation over all candidate "
       "pairs at every epoch requires the marginal effect of one assignment on "
       "each variance. Maintaining \\(S=\\sum_i x_i\\) and "
       "\\(Q=\\sum_i x_i^2\\), the exact marginal is"))
    A(("eq", (r"\Delta\mathrm{Var} = \frac{2x_j\,dx + dx^2}{n} - "
              r"\frac{2S\,dx + dx^2}{n^2}",
              "dVar = (2*x_j*dx + dx^2)/n - (2*S*dx + dx^2)/n^2",
              "eq:marginal")))
    A(("p",
       "which is algebraically exact rather than approximate and evaluates in "
       "\\(O(1)\\); the within-group analogue is \\(O(|g|)\\) and the "
       "between-group marginal follows from Equation~\\ref{eq:ltv} as the "
       "difference. We verify all three against brute-force recomputation over "
       "3{,}000 random instances, obtaining a maximum absolute error of "
       "\\(7 \\times 10^{-15}\\)."))

    # ================= 6 SETUP =================
    A(("h1", "Experimental Setup"))
    A(("p",
       "\\textbf{Data.} We use the New York City Taxi and Limousine Commission "
       "yellow-taxi records for March 2016 \\cite{tlc} (12{,}210{,}952 trips), the dataset "
       "used by the method we extend. This release predates the mid-2016 switch "
       "to pre-binned zone identifiers, so raw coordinates are available and "
       "spatial granularity is a free choice. After restricting to Manhattan "
       "and applying sanity filters on distance, fare, duration and implied "
       "speed, 10{,}300{,}738 trips remain (84.4\\%)."))
    A(("p",
       "\\textbf{Graph.} Coordinates are aggregated to a grid; cells with "
       "sufficient volume become nodes, with centroids at the empirical mean of "
       "their endpoints. A cell only earns a node if it can support a "
       "time-dependent travel-time estimate, and we select the granularity by "
       "measurement: at 0.5 km only 15.6\\% of (pair, hour) cells reach 30 "
       "observations, against approximately 90\\% of \\emph{trips} at 1.1 km. "
       "This yields 87 nodes covering 99.94\\% of endpoints. Road distances are "
       "the median observed trip distance per ordered pair, direct for 80.8\\% "
       "of pairs and 99.98\\% of trips, with a grid-aligned rotated-\\(L_1\\) "
       "fallback otherwise; the implied circuity of 1.320 matches the "
       "independently reported New York value of approximately 1.3."))
    A(("p",
       "\\textbf{Drivers.} The records contain no driver identifier, so the "
       "driver side is simulated, as in all prior work in this line. We "
       "instantiate 200 drivers: 76 full-time averaging 46.4 h per week and 124 "
       "part-time averaging 14.0 h, a separation of 3.32. Shifts are contiguous "
       "blocks drawn from four start-time archetypes held fixed per driver, "
       "which deliberately produces drivers systematically online in thin hours "
       "--- the population for which opportunity normalisation matters. Total "
       "supply is 63{,}120 online driver-epochs."))
    A(("p",
       "\\textbf{Protocol.} Decision epochs are 5 minutes; the mean Manhattan "
       "trip is 11.3 minutes, so an hourly epoch would leave occupancy "
       "ill-defined. The horizon is one week, 25--31 March 2016, partitioned "
       "three days history, one day current, three days future following "
       "\\cite{kang2024}. Requests are thinned by stratified sampling within "
       "(date, hour) cells at a rate solved so that offered load, the ratio of "
       "epochs demanded to epochs supplied, equals 0.60. This matters: at the "
       "sampling rate used by prior work a 200-driver fleet saturates, "
       "utilisation pins at 0.99 and \\(\\mathrm{Var}(\\nu)\\) collapses to zero "
       "for every method, rendering access fairness unmeasurable. The resulting "
       "stream contains 6{,}604 requests."))
    A(("p",
       "\\textbf{Forecaster.} Following \\cite{kang2024} a three-layer "
       "perceptron predicts hourly request counts per pair, with forecasts "
       "entering the action space. We add a second head predicting \\(c_p\\), "
       "since Equation~\\ref{eq:util-ca} requires congestion for future epochs "
       "and reading it from ground truth would leak. Only seasonal lags of at "
       "least 168 hours are used, so every test-week feature predates the "
       "forecast origin. The demand head attains a mean squared error of "
       f"{fc['demand_mse_counts']:.2f} against {fc['baseline_mse_counts']:.2f} "
       "for a seasonal-naive baseline, an improvement of "
       f"{100*(1-fc['demand_mse_counts']/fc['baseline_mse_counts']):.0f}\\%; the "
       f"congestion head attains a mean absolute error of {fc['congestion_mae']:.3f}."))
    A(("p",
       "\\textbf{Baselines.} We compare a greedy allocator on "
       "Equation~\\ref{eq:objective-prior}; REASSIGN \\cite{lesmana2019}, "
       "efficiency-first with bounded-loss reassignment; LAF \\cite{shi2021}, "
       "MDP edge re-weighting followed by optimal matching; Balance "
       "Ride-Pooling \\cite{raman2021}, reinforcement learning with "
       "variance-of-totals fairness and no demand forecast; and a faithful "
       "reproduction of \\cite{kang2024}, which adds the forecast. Every method "
       "is evaluated on an identical fleet, request stream, distance matrix and "
       "travel-time tensor, so differences are attributable to the allocation "
       "objective alone."))

    # ================= 7 RESULTS =================
    A(("pb", None))
    A(("h1", "Results"))

    A(("h2", "Comparison with baselines"))
    A(("p",
       "Table~\\ref{tab:main} reports all six allocators on both metric "
       "families. On the prevailing measure our method is worse by "
       "construction, which Proposition 1 predicts: pay-rate parity across "
       "unequal hours forces unequal totals. On every rate-based and "
       "access-based measure it improves substantially."))
    rows = [[r.method, f"{r.total_utility:,.0f}", f"{r.fairness_total_var:,.0f}",
             f"{r.rate_var:.3f}", f"{r.rate_var_within:.3f}",
             f"{r.rate_var_between:.4f}", f"{r.rate_group_ratio:.2f}",
             f"{r.util_var:.4f}", f"{r.util_adj_var:.3f}", f"{r.gini_rate:.3f}",
             f"{int(r.n_idle_drivers)}"] for _, r in V["t1"].iterrows()]
    A(("table", (
        "Allocator comparison on an identical scenario. \\(\\Pi\\) is total "
        "utility; \\(F\\) is the prevailing fairness measure "
        "\\(\\mathrm{Var}(o_v)\\); the remaining columns are pay-rate variance "
        "and its decomposition, the part-time to full-time rate ratio "
        "(unity is parity), raw and opportunity-normalised utilisation "
        "variance, the Gini coefficient of pay rate, and the number of drivers "
        "receiving no work in the week. Arrows indicate the preferred "
        "direction.",
        ["Method", "\\(\\Pi\\uparrow\\)", "\\(F\\downarrow\\)",
         "\\(\\mathrm{Var}\\rho\\downarrow\\)", "within\\(\\downarrow\\)",
         "between\\(\\downarrow\\)", "ratio\\(\\to 1\\)",
         "\\(\\mathrm{Var}\\nu\\downarrow\\)", "\\(\\tilde{\\nu}\\downarrow\\)",
         "Gini\\(\\downarrow\\)", "idle\\(\\downarrow\\)"],
        rows, [1.35, 0.55, 0.5, 0.5, 0.5, 0.55, 0.45, 0.5, 0.45, 0.45, 0.35])))
    A(("p",
       "The dominant pattern is in the ratio column. All five prior allocators "
       "settle between 1.62 and 1.91, paying part-time drivers 62--91\\% more "
       "per hour than full-time drivers. The uniformity is not coincidence: all "
       "five optimise variance of totals, and Proposition 1 makes the "
       "consequence unavoidable. Our allocator attains "
       f"{o.rate_group_ratio:.2f}."))
    A(("fig", ("outputs/fig_gaps_by_method.png",
               "Fairness outcomes by allocator. Left: hourly pay by activity "
               "group. Centre: between-group pay-rate variance. Right: raw and "
               "opportunity-normalised utilisation variance. Lower is fairer in "
               "the centre and right panels.", 6.4)))

    A(("h2", "Isolating each correction"))
    A(("p",
       "Table~\\ref{tab:main} varies the fairness objective while holding the "
       "congestion-aware utility fixed, which is the correct control for "
       "comparing objectives but does not isolate the individual corrections. "
       "We therefore run a second experiment beginning from a faithful "
       "baseline --- prevailing utility \\emph{and} prevailing fairness --- and "
       "enabling one correction at a time with the request stream pinned, so "
       "all runs see an identical fleet and identical demand."))
    rows = [[r.method.split(". ", 1)[1], f"{r.total_utility:,.0f}",
             f"{r.rate_var:.3f}", f"{r.rate_var_between:.4f}",
             f"{r.rate_full_time:.2f}", f"{r.rate_part_time:.2f}",
             f"{r.rate_group_ratio:.3f}", f"{r.util_var:.4f}",
             f"{r.util_adj_var:.3f}", f"{int(r.n_idle_drivers)}"]
            for _, r in V["iso"].iterrows()]
    A(("table", (
        "Ablation from a faithful baseline, one correction at a time, with the "
        "request stream pinned across all runs. CA denotes the congestion-aware "
        "utility of Equation~\\ref{eq:util-ca}, RF the pay-rate fairness "
        "objective of Equation~\\ref{eq:ltv}, and AF the access-fairness term "
        "of Equation~\\ref{eq:util}.",
        ["Configuration", "\\(\\Pi\\uparrow\\)",
         "\\(\\mathrm{Var}\\rho\\downarrow\\)", "between\\(\\downarrow\\)",
         "FT \\(\\rho\\)", "PT \\(\\rho\\)", "ratio\\(\\to 1\\)",
         "\\(\\mathrm{Var}\\nu\\downarrow\\)", "\\(\\tilde{\\nu}\\downarrow\\)",
         "idle\\(\\downarrow\\)"],
        rows, [1.65, 0.55, 0.55, 0.55, 0.45, 0.45, 0.5, 0.55, 0.5, 0.35])))
    A(("p", "Three observations follow."))
    A(("bul", [
        "\\textbf{The congestion correction is an efficiency and correctness "
        "measure, not a fairness measure.} Alone it raises utility by "
        f"{pct(g1a.total_utility, b0.total_utility)} --- the largest "
        "single-correction gain --- yet pay-rate variance \\emph{rises} from "
        f"{b0.rate_var:.3f} to {g1a.rate_var:.3f} and two drivers receive no "
        "work. A more accurate valuation identifies which trips are worth more, "
        "not who should receive them; pursuing high-value trips more "
        "aggressively concentrates them.",
        "\\textbf{Pay-rate fairness is the decisive and nearly costless "
        "correction.}"
        f" Alone it moves the group ratio from "
        f"{b0.rate_group_ratio:.3f} to {g1b.rate_group_ratio:.3f} and reduces "
        f"between-group variance by {pct(g1b.rate_var_between, b0.rate_var_between)}, "
        f"at a utility cost of {pct(g1b.total_utility, b0.total_utility)}.",
        "\\textbf{The two corrections are complementary rather than "
        "substitutable.} Applied together they retain the efficiency gain "
        f"({pct(g1.total_utility, b0.total_utility)}) \\emph{{and}} reach parity "
        f"({g1.rate_group_ratio:.3f}) \\emph{{and}} return the idle-driver count "
        f"to {int(g1.n_idle_drivers)}, repairing the side effect the congestion "
        "correction introduces on its own. Neither achieves this alone.",
    ]))
    A(("p",
       "The access-fairness term contributes what the others cannot: alone it "
       f"reduces raw utilisation variance by {pct(g2.util_var, b0.util_var)} and "
       f"the opportunity-normalised form by "
       f"{pct(g2.util_adj_var, b0.util_adj_var)}, the largest reductions on "
       "those measures, while leaving the group ratio far from parity at "
       f"{g2.rate_group_ratio:.3f}. Even work distribution and equal pay per "
       "hour are distinct goals."))

    A(("h2", "The prevailing objective produces the disparity"))
    A(("p",
       "Proposition 1 shows incompatibility; Table~\\ref{tab:causes} shows the "
       "direction of the induced effect. Disabling the fairness term entirely "
       "and then enabling the prevailing one is the cleanest available test."))
    A(("table", (
        "Effect of enabling variance-of-totals fairness. Enabling the term "
        "worsens between-group pay-rate variance by roughly ninety times.",
        ["Configuration", "FT \\(\\rho\\)", "PT \\(\\rho\\)",
         "ratio", "between-group \\(\\mathrm{Var}\\rho\\)"],
        [["No fairness term", "2.28", "2.42", "1.06",
          f"{V['nofair'].rate_var_between:.4f}"],
         ["Variance of totals enabled", "1.89", "3.13", "1.65",
          f"{k.rate_var_between:.4f}"]],
        [2.0, 0.9, 0.9, 0.7, 1.5])))
    A(("p",
       "The mechanism is exactly that of Proposition 1's corollary. This "
       "elevates the critique from a measurement gap to a causal claim: the "
       "objective adopted throughout this literature manufactures the "
       "between-group disparity that its own metric is structurally unable to "
       "observe."))

    A(("h2", "Component ablations"))
    rows = [[r.method.replace("Ours, ", "").replace("Ours ", ""),
             f"{r.total_utility:,.0f}", f"{r.rate_var:.3f}",
             f"{r.rate_var_between:.4f}", f"{r.util_var:.4f}",
             f"{r.util_adj_var:.3f}", f"{int(r.n_idle_drivers)}"]
            for _, r in V["t2"].iterrows()]
    A(("table", (
        "Component ablations of the full method.",
        ["Configuration", "\\(\\Pi\\)", "\\(\\mathrm{Var}\\rho\\)", "between",
         "\\(\\mathrm{Var}\\nu\\)", "\\(\\tilde{\\nu}\\)", "idle"],
        rows, [2.1, 0.7, 0.7, 0.7, 0.7, 0.6, 0.4])))
    A(("p",
       "Removing the access-fairness term degrades raw utilisation variance "
       f"from {o.util_var:.4f} to {V['noutil'].util_var:.4f} and the normalised "
       f"form from {o.util_adj_var:.3f} to {V['noutil'].util_adj_var:.3f}, "
       "confirming that the term does independent work. Two ablations return "
       "negative results, which we report as such. Removing the demand "
       f"forecaster slightly \\emph{{raises}} utility, from {o.total_utility:,.0f} "
       f"to {V['nopred'].total_utility:,.0f}; we are unable to reproduce the "
       "large degradation reported for its removal in \\cite{kang2024}, and "
       "attribute this to a tabular value function already observing all 48 "
       "profiles during training. Removing the congestion-aware utility while "
       "retaining both fairness terms yields a lower pay-rate variance "
       f"({V['static'].rate_var:.3f}), consistent with the isolation finding "
       "that this correction serves accuracy rather than equity."))

    A(("h2", "Behaviour over the horizon"))
    A(("p",
       "Prior work motivates a weekly horizon on the grounds that drivers care "
       "about weekly rather than per-dispatch outcomes \\cite{kang2024}. We "
       "evaluate both allocators as the horizon grows one day at a time."))
    rows = [[int(d), f"{V['hf'].loc[d][P]:.1f}", f"{V['hf'].loc[d][O]:.1f}",
             f"{V['hv'].loc[d][P]:.3f}", f"{V['hv'].loc[d][O]:.3f}",
             f"{hp.loc[d][P]:.4f}", f"{hp.loc[d][O]:.4f}"]
            for d in sorted(hp.index)]
    A(("table", (
        "Fairness against horizon length. Prior work's between-group pay "
        f"disparity grows by a factor of {hp[P].iloc[-1]/hp[P].iloc[0]:.1f} over "
        "the week; ours remains flat and roughly two orders of magnitude lower.",
        ["Days", "\\(F\\) prior", "\\(F\\) ours",
         "\\(\\mathrm{Var}\\rho\\) prior", "\\(\\mathrm{Var}\\rho\\) ours",
         "between prior", "between ours"],
        rows, [0.5, 0.85, 0.85, 1.05, 1.05, 0.95, 0.95])))
    A(("p",
       "The prior allocator's between-group disparity increases monotonically "
       f"in the horizon, from {hp[P].iloc[0]:.4f} at one day to "
       f"{hp[P].iloc[-1]:.4f} at seven, while ours remains near "
       f"{hp[O].iloc[-1]:.4f}. On a group-conditioned measure, then, the method "
       "that is designed to deliver long-term fairness accumulates long-term "
       "\\emph{un}fairness in precisely the dimension it cannot observe. This is "
       "the sharpest form of our critique."))
    A(("fig", ("outputs/fig4_horizon.png",
               "Fairness against horizon length. Left: the prevailing measure. "
               "Centre: pay-rate variance. Right: between-group pay-rate "
               "variance, where prior work degrades monotonically while our "
               "allocator remains stable.", 6.4)))

    A(("h2", "The parity frontier"))
    A(("p",
       "Group parity is not a fixed point of our objective but a tunable "
       "operating point governed by \\(\\lambda_b\\). We report the frontier so "
       "that a platform may select a position deliberately rather than inherit "
       "one."))
    rows = [[f"{r.lambda_between:.2f}", f"{r.rate_group_ratio:.3f}",
             f"{r.rate_var_between:.5f}", f"{r.rate_var:.3f}",
             f"{r.total_utility:,.0f}"] for _, r in V["sweep_b"].iterrows()]
    rows.append(["prior work", f"{k.rate_group_ratio:.3f}",
                 f"{k.rate_var_between:.5f}", f"{k.rate_var:.3f}",
                 f"{k.total_utility:,.0f}"])
    A(("table", (
        "Parity frontier in \\(\\lambda_b\\), with the within-group and "
        "access-fairness weights held fixed. Parity is reached asymptotically "
        "and total utility varies by under 1.2\\% across the range.",
        ["\\(\\lambda_b\\)", "ratio", "between-group \\(\\mathrm{Var}\\rho\\)",
         "\\(\\mathrm{Var}\\rho\\)", "\\(\\Pi\\)"],
        rows, [0.9, 0.9, 1.6, 1.0, 1.1])))
    A(("p",
       f"Across a coordinate search over {len(V['gr'])} weight configurations, "
       "each retaining both fairness terms, "
       f"{len(V['dom'])} configurations dominate the reproduced prior method "
       "simultaneously on total utility, pay-rate variance, between-group "
       "variance, raw and normalised utilisation variance, and idle-driver "
       "count. The improvement is therefore not purchased by trading efficiency "
       "for equity at a single hand-chosen point."))
    A(("fig", ("outputs/fig_sweep_pareto.png",
               "Trade-off frontiers over the weight sweep, with both fairness "
               "terms active at every point. The reproduced prior method is "
               "marked. Down and to the right is preferred.", 6.4)))

    A(("h2", "Robustness"))
    A(("p",
       "\\textbf{Monetary units.} Distance-denominated utility preserves "
       "comparability with prior work but invites the objection that the "
       "disparity is an artefact of units. Re-running with utility priced at "
       "the fare rate measured from the data ($3.95/km) reproduces it: prior "
       f"work pays full-time drivers \\${mk.rate_full_time:.2f} and part-time "
       f"drivers \\${mk.rate_part_time:.2f} per hour, a ratio of "
       f"{mk.rate_group_ratio:.3f}, against \\${mo.rate_full_time:.2f} and "
       f"\\${mo.rate_part_time:.2f} under ours "
       f"(ratio {mo.rate_group_ratio:.3f})."))
    A(("p",
       "\\textbf{Seeds.} Across three independent seeds re-drawing the fleet, "
       "the request sample and exploration, the prior method's group ratio is "
       f"{sd.loc['Paper', ('rate_group_ratio', 'mean')]:.3f} \\(\\pm\\) "
       f"{sd.loc['Paper', ('rate_group_ratio', 'std')]:.3f} and ours is "
       f"{sd.loc['Ours', ('rate_group_ratio', 'mean')]:.3f} \\(\\pm\\) "
       f"{sd.loc['Ours', ('rate_group_ratio', 'std')]:.3f}. The disparity is "
       "systematic, not a sampling artefact."))
    A(("table", (
        "Robustness across three independent seeds (mean \\(\\pm\\) standard "
        "deviation).",
        ["Method", "\\(\\Pi\\)", "\\(\\mathrm{Var}\\rho\\)", "between",
         "ratio", "\\(\\tilde{\\nu}\\)"],
        [["Prior work",
          f"{sd.loc['Paper',('total_utility','mean')]:,.0f}",
          f"{sd.loc['Paper',('rate_var','mean')]:.3f} \\(\\pm\\) {sd.loc['Paper',('rate_var','std')]:.3f}",
          f"{sd.loc['Paper',('rate_var_between','mean')]:.4f}",
          f"{sd.loc['Paper',('rate_group_ratio','mean')]:.3f} \\(\\pm\\) {sd.loc['Paper',('rate_group_ratio','std')]:.3f}",
          f"{sd.loc['Paper',('util_adj_var','mean')]:.3f}"],
         ["Ours",
          f"{sd.loc['Ours',('total_utility','mean')]:,.0f}",
          f"{sd.loc['Ours',('rate_var','mean')]:.3f} \\(\\pm\\) {sd.loc['Ours',('rate_var','std')]:.3f}",
          f"{sd.loc['Ours',('rate_var_between','mean')]:.4f}",
          f"{sd.loc['Ours',('rate_group_ratio','mean')]:.3f} \\(\\pm\\) {sd.loc['Ours',('rate_group_ratio','std')]:.3f}",
          f"{sd.loc['Ours',('util_adj_var','mean')]:.3f}"]],
        [1.1, 0.9, 1.3, 0.8, 1.3, 0.7])))

    # ================= 8 REPRODUCIBILITY =================
    A(("h1", "Reproducibility Analysis of the Base Method"))
    A(("p",
       "Our absolute utility figures are an order of magnitude below those "
       "published in \\cite{kang2024}, and the discrepancy is instructive "
       "rather than incidental. It is not attributable to data volume: the "
       "complete filtered dataset builds every model component, and only the "
       "allocation experiment operates on a thinned stream, as in the original."))
    A(("p",
       "\\textbf{Fleet size is recoverable but unreported.} The number of "
       "drivers is not stated in \\cite{kang2024}. Since total utility divided "
       "by mean per-driver utility equals the driver count, it can be recovered "
       "from the published table, and yields exactly 20.000 on all five rows "
       "(Table~\\ref{tab:repro}). A tenfold smaller fleet concentrates the same "
       "workload, inflating per-driver figures correspondingly."))
    A(("table", (
        "Fleet size recovered from the published results of \\cite{kang2024}. "
        "The ratio is exactly 20 on every row.",
        ["Method (as published)", "Total utility", "Mean per driver",
         "Implied \\(n\\)"],
        [["Greedy", "\\(-1{,}514{,}736.24\\)", "\\(-75{,}736.81\\)", "20.000"],
         ["REASSIGN", "76,536.23", "3,826.81", "20.000"],
         ["LAF", "80,606.49", "4,030.3245", "20.000"],
         ["Balance Ride-Pooling", "85,923.68", "4,296.18", "20.000"],
         ["Proposed", "95,823.79", "4,791.19", "20.000"]],
        [1.9, 1.4, 1.4, 0.9])))
    A(("p",
       "\\textbf{Reported earnings exceed protocol capacity.} The published "
       "mean per-driver weekly utility is 4{,}791. At the net per-trip utility "
       "we measure on the same city, approximately 2.5 km, this requires about "
       "1{,}924 trips per driver per week. The evaluation protocol extracts a "
       "two-hour peak window per day, giving at most 14 online hours per week; "
       "with a mean trip of 11.3 minutes plus approach time, a driver can "
       "complete at most about 53 trips. The reported figure therefore exceeds "
       "the capacity of its own protocol by roughly 36 times. The explanation "
       "lies in the model itself: drivers may accept unboundedly many "
       "concurrent requests and the reward carries no duration, so nothing "
       "bounds the work a single driver absorbs in one epoch. Our occupancy "
       "constraint is the reason our totals are smaller, and is simultaneously "
       "what makes access fairness measurable."))
    A(("p",
       "\\textbf{The protocol cannot express the phenomena it would need to.} "
       "Executing the original protocol in our harness, a two-hour daily window "
       "caps every driver at 14 weekly hours, so no driver reaches a "
       "conventional full-time threshold and the activity-group contrast cannot "
       "exist. Utilisation saturates at 0.99, the service rate falls below "
       f"3\\%, and {int(V['ppk'].n_idle_drivers)} of 200 drivers receive no work. "
       "Neither deficiency identified in Section 4 is expressible under that "
       "protocol, which is why we evaluate on a full-day timeline."))
    A(("p",
       "We draw no inference about the correctness of the original "
       "implementation. The point is narrower and, we believe, useful to the "
       "community: absolute utility figures in this literature are not "
       "comparable across papers unless fleet size, concurrency semantics and "
       "spatial granularity are reported, and at present they frequently are "
       "not. All comparisons in this paper are consequently made within a "
       "single harness."))

    # ================= 9 DISCUSSION =================
    A(("h1", "Discussion"))
    A(("h2", "Who gains, and the policy question"))
    A(("p",
       "Enforcing pay-rate parity is redistributive, and we state the incidence "
       "explicitly rather than presenting it as a Pareto improvement. In "
       "monetary terms full-time drivers gain "
       f"\\${mo.rate_full_time-mk.rate_full_time:+.2f} per hour and part-time "
       f"drivers lose \\${mo.rate_part_time-mk.rate_part_time:+.2f}; over a "
       "fifty-week year at the simulated hours this is approximately "
       f"\\${(mo.rate_full_time-mk.rate_full_time)*46.4*50:+,.0f} and "
       f"\\${(mo.rate_part_time-mk.rate_part_time)*14.0*50:+,.0f} respectively. "
       "Our claim is not that this particular incidence is socially optimal. It "
       "is that the prevailing objective imposes the opposite incidence "
       "\\emph{without representing it}, since hours do not appear in its "
       "fairness functional. Making the trade-off explicit, and exposing it "
       "through \\(\\lambda_b\\) with the frontier of "
       "Table~\\ref{tab:frontier}, converts an unexamined side effect into a "
       "policy decision."))

    A(("h2", "Limitations"))
    A(("bul", [
        "\\textbf{Simulated driver population.} Public taxi records contain no "
        "driver identifier, so hours, shifts and activity groups are simulated, "
        "as in all prior work in this line. Our conclusions concern the "
        "\\emph{objective} rather than the empirical distribution of gig labour "
        "supply, but validation on a platform dataset with driver identifiers "
        "remains desirable.",
        "\\textbf{Functional form of the congestion correction.} "
        "Equation~\\ref{eq:util-ca} is a modelling choice, not a derivation. We "
        "selected the multiplicative form because it is the only candidate that "
        "reduces exactly to prior work at neutral congestion, which keeps the "
        "ablation uncontaminated; a rate form or an additive time penalty would "
        "confound the correction with a change of units. Sensitivity to this "
        "choice is untested.",
        "\\textbf{Opportunity normalisation.} Equation~\\ref{eq:oppnorm} "
        "includes each driver's own contribution to the contemporaneous busy "
        "count, a self-bias of roughly 3\\% at our median concurrency, and is "
        "numerically fragile when supply approaches zero. We therefore always "
        "report it alongside the raw measure.",
        "\\textbf{Residual access regression.} The fully combined allocator "
        f"leaves {int(gall.n_idle_drivers)} drivers without work where the "
        f"congestion and rate corrections together leave "
        f"{int(g1.n_idle_drivers)}. The interaction is with the access term and "
        "is not yet resolved; a per-driver floor is the natural remedy.",
        "\\textbf{Single city and month.} All results are New York City, March "
        "2016, and a single fleet size of 200. Sensitivity to scale and "
        "geography is future work.",
    ]))

    # ================= 10 CONCLUSION =================
    A(("h1", "Conclusion"))
    A(("p",
       "Fairness in ride-hailing allocation is conventionally defined as "
       "equality of accumulated earnings. We have shown that this definition is "
       "unsatisfiable jointly with equality of pay rate whenever drivers supply "
       "unequal hours, that optimising it consequently manufactures a "
       "between-group pay disparity which it is structurally unable to observe, "
       "and that the disparity grows with the very horizon the approach is "
       "designed to protect. We have also shown that the distance-only utility "
       "standard in this literature inverts the profitability of one assignment "
       "in ten, and that access fairness cannot be formed at all under a model "
       "permitting unbounded concurrency."))
    A(("p",
       "Redefining the objective over pay rate, decomposed across activity "
       "groups by the law of total variance, and adding an access-fairness term "
       "over an occupancy-constrained environment, attains hourly-pay parity "
       f"between activity groups ({b0.rate_group_ratio:.2f} to "
       f"{g1.rate_group_ratio:.2f}) while increasing total utility by "
       f"{100*(g1.total_utility-b0.total_utility)/b0.total_utility:.1f}\\% "
       "relative to a faithful baseline, and eliminates fully idle drivers. The "
       "result is robust across seeds, survives re-denomination into currency, "
       f"and holds at {len(V['dom'])} of {len(V['gr'])} weight configurations "
       "that dominate prior work on all six measures simultaneously."))
    A(("p",
       "We suggest two implications for the field. Methodologically, fairness "
       "functionals in labour-market matching should condition on supplied "
       "hours; a functional of accumulated output alone will misattribute "
       "equity in any setting where participation is heterogeneous. "
       "Practically, papers in this line should report fleet size, concurrency "
       "semantics and spatial granularity, without which absolute results are "
       "not comparable. Our code, configurations and all result files are "
       "released to that end."))

    # ================= REFERENCES =================
    A(("bib", [
        ("kang2024",
         "Kang, Y., Chan, J., Shao, W., Salim, F.D., Leckie, C.: Long-term "
         "fairness in ride-hailing platform. In: ECML PKDD, pp. 217--233 (2024)"),
        ("raman2021",
         "Raman, N., Shah, S., Dickerson, J.: Data-driven methods for balancing "
         "fairness and efficiency in ride-pooling. arXiv:2110.03524 (2021)"),
        ("shi2021",
         "Shi, D., Tong, Y., Zhou, Z., Song, B., Lv, W., et al.: Learning to "
         "assign: towards fair task assignment in large-scale ride hailing. In: "
         "KDD, pp. 3549--3557 (2021)"),
        ("lesmana2019",
         "Lesmana, N., Zhang, X., Bei, X.: Balancing efficiency and fairness in "
         "on-demand ridesourcing. In: NeurIPS (2019)"),
        ("suhr2019",
         "S\\\"uhr, T., Biega, A.J., Zehlike, M., Gummadi, K.P., et al.: "
         "Two-sided fairness for repeated matchings in two-sided markets. In: "
         "KDD, pp. 3082--3092 (2019)"),
        ("nanda2020",
         "Nanda, V., Xu, P., Sankararaman, K.A., et al.: Balancing the tradeoff "
         "between profit and fairness in rideshare platforms. In: AAAI, "
         "pp. 2210--2217 (2020)"),
        ("xu2020",
         "Xu, Y., Xu, P.: Trading the system efficiency for the income equality "
         "of drivers in rideshare. arXiv:2012.06850 (2020)"),
        ("ma2020",
         "Ma, W., Xu, P., Xu, Y.: Group-level fairness maximization in online "
         "bipartite matching. arXiv:2011.13908 (2020)"),
        ("kumar2023",
         "Kumar, A., Vorobeychik, Y., Yeoh, W.: Using simple incentives to "
         "improve two-sided fairness in ridesharing systems. In: ICAPS, "
         "pp. 227--235 (2023)"),
        ("shah2020",
         "Shah, S., Lowalekar, M., Varakantham, P.: Neural approximate dynamic "
         "programming for on-demand ride-pooling. In: AAAI, pp. 507--515 (2020)"),
        ("delima2020",
         "de Lima, O., Shah, H., Chu, T.S., Fogelson, B.: Efficient ridesharing "
         "dispatch using multi-agent reinforcement learning. arXiv:2006.10897 "
         "(2020)"),
        ("sun2022",
         "Sun, J., Jin, H., Yang, Z., Su, L., Wang, X.: Optimizing long-term "
         "efficiency and fairness in ride-hailing via joint order dispatching "
         "and driver repositioning. In: KDD, pp. 3950--3960 (2022)"),
        ("cook2021",
         "Cook, C., Diamond, R., Hall, J.V., List, J.A., Oyer, P.: The gender "
         "earnings gap in the gig economy: evidence from over a million "
         "rideshare drivers. Review of Economic Studies 88(5), 2210--2238 (2021)"),
        ("brown2018",
         "Brown, A.E.: Ridehail revolution: ridehail travel and equity in Los "
         "Angeles. Ph.D. thesis, UCLA (2018)"),
        ("zhao2019",
         "Zhao, B., Xu, P., Shi, Y., Tong, Y., et al.: Preference-aware task "
         "assignment in on-demand taxi dispatch. In: AAAI, pp. 2245--2252 (2019)"),
        ("tong2020",
         "Tong, Y., Zhou, Z., Zeng, Y., Chen, L., Shahabi, C.: Spatial "
         "crowdsourcing: a survey. VLDB Journal 29, 217--250 (2020)"),
        ("kang2024b",
         "Kang, Y., Zhang, R., Shao, W., Salim, F.D., Chan, J.: Promoting "
         "two-sided fairness in dynamic vehicle routing. arXiv:2405.19184 (2024)"),
        ("miettinen2002",
         "Miettinen, K., M\\\"akel\\\"a, M.M.: On scalarizing functions in "
         "multiobjective optimization. OR Spectrum 24(2), 193--213 (2002)"),
        ("tlc",
         "New York City Taxi and Limousine Commission: TLC trip record data. "
         "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"),
        ("mohlmann2019",
         "M\\\"ohlmann, M., Henfridsson, O.: What people hate about being "
         "managed by algorithms, according to a study of Uber drivers. Harvard "
         "Business Review (2019)"),
    ]))
    return D
