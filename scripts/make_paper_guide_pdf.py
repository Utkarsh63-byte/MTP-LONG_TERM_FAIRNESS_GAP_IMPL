"""Build a PDF companion guide to the research paper.

Explains every section, equation, table and figure of
"Equal Earnings Are Not Equal Pay", section by section, for a reader who has the
paper in front of them. Rendered with reportlab, so no LaTeX toolchain is needed.

Run:  .venv/bin/python -m scripts.make_paper_guide_pdf
Output: outputs/paper/MTP_Paper_Explained.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platforms import *  # noqa: F401,F403  (no-op; keeps linters quiet)
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               ListFlowable, ListItem, PageBreak, PageTemplate,
                               Paragraph, Spacer, Table, TableStyle)

from ltf.config import OUTPUT, ROOT
from scripts.paper_content import build, load_values

NAVY = colors.HexColor("#102540")
TEAL = colors.HexColor("#0F3D4C")
GREY = colors.HexColor("#5A6470")
LIGHT = colors.HexColor("#E8EEF4")
ACCENT = colors.HexColor("#B01C1C")
GREEN = colors.HexColor("#1B6B2F")

PAPER_DIR = OUTPUT / "paper"
TITLE = ("Equal Earnings Are Not Equal Pay: Hours-Aware and Access-Aware "
         "Fairness for Driver Allocation in Ride-Hailing")

ss = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15, textColor=NAVY, spaceBefore=14,
                         spaceAfter=6, leading=18),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, textColor=TEAL, spaceBefore=10,
                         spaceAfter=4, leading=14),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                         fontSize=10, textColor=TEAL, spaceBefore=8,
                         spaceAfter=3, leading=12),
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica",
                           fontSize=9.2, leading=12.8, alignment=TA_JUSTIFY,
                           spaceAfter=5),
    "small": ParagraphStyle("small", parent=ss["BodyText"], fontName="Helvetica",
                            fontSize=8.2, leading=11, alignment=TA_JUSTIFY,
                            spaceAfter=4, textColor=GREY),
    "code": ParagraphStyle("code", parent=ss["BodyText"], fontName="Courier",
                           fontSize=8, leading=10.5, leftIndent=8,
                           spaceBefore=3, spaceAfter=5,
                           backColor=colors.HexColor("#F4F6F9")),
    "eq": ParagraphStyle("eq", parent=ss["BodyText"], fontName="Courier-Bold",
                         fontSize=8.6, leading=11.5, leftIndent=10,
                         spaceBefore=3, spaceAfter=5,
                         textColor=NAVY),
    "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontName="Helvetica",
                           fontSize=7.6, leading=9.6, spaceAfter=0),
    "cellb": ParagraphStyle("cellb", parent=ss["BodyText"],
                            fontName="Helvetica-Bold", fontSize=7.6,
                            leading=9.6, spaceAfter=0),
    "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontName="Helvetica-Oblique",
                          fontSize=8, leading=10.5, alignment=TA_CENTER,
                          textColor=GREY, spaceAfter=8),
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=19, textColor=NAVY, leading=23,
                            alignment=TA_CENTER, spaceAfter=4),
    "sub": ParagraphStyle("sub", parent=ss["BodyText"], fontSize=11,
                          alignment=TA_CENTER, textColor=GREY, leading=14,
                          spaceAfter=3),
    "flag": ParagraphStyle("flag", parent=ss["BodyText"], fontName="Helvetica-Bold",
                           fontSize=9.2, leading=12.5, textColor=ACCENT,
                           spaceBefore=3, spaceAfter=3),
    "good": ParagraphStyle("good", parent=ss["BodyText"], fontName="Helvetica-Bold",
                           fontSize=9.2, leading=12.5, textColor=GREEN,
                           spaceBefore=3, spaceAfter=3),
}


def P(t, st="body"):
    return Paragraph(t, S[st])


def BUL(items, st="body"):
    return ListFlowable(
        [ListItem(Paragraph(i, S[st]), leftIndent=12) for i in items],
        bulletType="bullet", start="circle", leftIndent=14,
        bulletFontSize=6, spaceAfter=5)


def TBL(header, rows, widths, fs=7.6, hdr_bg=LIGHT, zebra=True):
    data = [[Paragraph(str(h), S["cellb"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), S["cell"]) for c in r])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), hdr_bg),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i),
                              colors.HexColor("#FAFBFC")))
    t.setStyle(TableStyle(style))
    return t


def FIG(path, width_mm, caption=None):
    src = ROOT / path
    out = []
    if src.exists():
        ir = ImageReader(str(src))
        iw, ih = ir.getSize()
        w = width_mm * mm
        out.append(Image(str(src), width=w, height=w * ih / iw))
    if caption:
        out.append(Paragraph(caption, S["cap"]))
    return out


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 287 * mm,
                      "Companion guide - Equal Earnings Are Not Equal Pay")
    canvas.drawRightString(192 * mm, 287 * mm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#C8D2DC"))
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 285 * mm, 192 * mm, 285 * mm)
    canvas.restoreState()


def main() -> int:
    V = load_values()
    DOC = build(V)
    k, o = V["k"], V["o"]
    b0, g1a, g1b, g1, g2, gall = (V["b0"], V["g1a"], V["g1b"], V["g1"], V["g2"],
                                  V["gall"])
    hp, fc = V["hp"], V["fc"]
    mk, mo, sd = V["mk"], V["mo"], V["sd"]
    nofair, nopred, noutil, static = (V["nofair"], V["nopred"], V["noutil"],
                                      V["static"])
    F = []

    # ============================== COVER ==============================
    F += [Spacer(1, 22 * mm),
          P("Companion Guide", "title"),
          P("A section-by-section explanation of the paper", "sub"),
          Spacer(1, 6 * mm),
          P(f"<b>{TITLE}</b>", "sub"),
          Spacer(1, 8 * mm)]
    F.append(TBL(["Item", "Detail"], [
        ["What this document is",
         "An explanation of the paper: what every section does, what every equation "
         "means, how to read every table and figure, and the questions a reviewer is "
         "likely to ask about each part."],
        ["What it is not",
         "It is not the paper. Read it alongside main.tex or "
         "MTP_Research_Paper.docx, both in this folder."],
        ["Paper structure",
         "10 sections, 14 numbered equations, 9 tables, 4 figures, 1 proposition "
         "with proof, ~4,450 words of prose"],
        ["Core claim in one line",
         "Equalising drivers' total earnings is the wrong fairness target: it is "
         "blind to hours worked, it provably cannot coexist with equal pay per "
         "hour, and optimising it actively manufactures a full-time versus "
         "part-time pay gap."],
    ], [34, 138]))
    F += [Spacer(1, 6 * mm),
          P("How to use this guide", "h2"),
          BUL([
              "<b>Section 2</b> is a map of the paper. Read it first to see how the "
              "argument is assembled.",
              "<b>Section 3</b> walks the paper in order. Each entry states the "
              "purpose of that section, what it claims, and the likely challenge.",
              "<b>Sections 4, 5 and 6</b> are reference material: every equation, "
              "every table and every figure explained individually. Jump to them "
              "when you need one specific item.",
              "<b>Section 7</b> is the reviewer question bank with prepared answers. "
              "This is the most useful section before a viva or a rebuttal.",
              "<b>Section 8</b> is a pre-submission checklist.",
          ])]
    F.append(PageBreak())

    # ============================== 2. MAP ==============================
    F.append(P("1. The paper at a glance", "h1"))
    F.append(P(
        "Every section of a paper has a job. If a reader cannot tell what a "
        "section is for, it is usually because it is doing two jobs or none. "
        "Here is what each of the ten sections is for, and how many words it "
        "spends doing it.", "body"))
    F.append(TBL(["Sec", "Title", "Its job in the argument", "Contains"], [
        ["1", "Introduction",
         "State the problem, state that the standard answer is wrong, and promise "
         "six specific contributions.",
         "6 numbered contributions"],
        ["2", "Related Work",
         "Establish that our critique applies to a whole family of methods, not one "
         "paper. This is what makes the contribution significant rather than "
         "incremental.",
         "3 strands: efficiency matching, driver fairness, long-horizon fairness"],
        ["3", "Preliminaries",
         "Define notation and restate prior work formally, so the critique has a "
         "precise target.",
         "Table 1, Equations 1-3"],
        ["4", "Motivating Analysis",
         "Prove the three deficiencies are real, before proposing any fix. This "
         "ordering matters: evidence first, method second.",
         "Figure 1, Proposition 1 and its proof"],
        ["5", "Method",
         "Introduce the three corrections plus the machinery needed to make them "
         "work.",
         "Equations 4-14"],
        ["6", "Experimental Setup",
         "Make the experiment reproducible and pre-empt objections about the data, "
         "the granularity and the sampling.",
         "Data, graph, drivers, protocol, forecaster, baselines"],
        ["7", "Results",
         "Seven subsections, each answering one question. Deliberately includes "
         "results that are unfavourable to us.",
         "Tables 2-8, Figures 2-4"],
        ["8", "Reproducibility Analysis",
         "Explain why our absolute numbers differ from the published ones, and turn "
         "that into a finding rather than an excuse.",
         "Table 9"],
        ["9", "Discussion",
         "State who gains and who loses, then list five limitations honestly.",
         "Policy incidence, 5 limitations"],
        ["10", "Conclusion",
         "Restate the claim and give two implications for the field.",
         "-"],
    ], [10, 30, 84, 48]))

    F.append(P("1.1 The argument chain", "h2"))
    F.append(P(
        "The paper is one argument in six steps. If any step fails the paper "
        "fails, so it is worth knowing the chain by heart.", "body"))
    F.append(TBL(["Step", "Claim", "Where it is established", "Evidence type"], [
        ["1", "Prior work values a trip by distance and averages away travel time.",
         "Section 3, Equations 1-2", "Citation"],
        ["2", "That inverts the profit sign on 10.2% of assignments.",
         "Section 4.1, Figure 1", "Measurement"],
        ["3", "Prior fairness is a function of totals, so it is blind to hours.",
         "Section 4.2", "Definition"],
        ["4", "Total-equality and rate-equality are mutually unsatisfiable.",
         "Proposition 1", "Proof"],
        ["5", "Therefore optimising totals must widen the pay-rate gap - and it "
              "does, by about 90 times.",
         "Section 7.3, Table 4", "Measurement"],
        ["6", "Re-targeting the objective fixes it at almost no efficiency cost.",
         "Sections 7.1-7.2, Tables 2-3", "Measurement"],
    ], [10, 74, 50, 38]))
    F.append(P(
        "Notice that step 4 is a proof and step 5 is a measurement that the proof "
        "predicts. That pairing is the strongest part of the paper: theory says "
        "the disparity is unavoidable, and the experiment shows it appearing.",
        "small"))
    F.append(PageBreak())

    # ============================== 3. WALKTHROUGH ==============================
    F.append(P("2. Section-by-section walkthrough", "h1"))

    def sec_block(num, title, purpose, says, detail, question, answer):
        out = [P(f"2.{num}  Paper Section {num} - {title}", "h2"),
               P(f"<b>Purpose.</b> {purpose}", "body"),
               P(f"<b>What it claims.</b> {says}", "body")]
        if detail:
            out.append(BUL(detail))
        out += [P(f"<b>Likely challenge.</b> {question}", "flag"),
                P(f"<b>Answer.</b> {answer}", "body"),
                Spacer(1, 2 * mm)]
        return out

    F += sec_block(
        1, "Introduction",
        "Convince a reader in one page that the standard fairness objective in "
        "this literature is not merely incomplete but harmful, and that we have "
        "the evidence and the fix.",
        "Gig drivers supply unequal hours; a fairness measure on accumulated "
        "totals cannot see that; equalising totals therefore transfers pay rate "
        "from long-hours to short-hours drivers; and this is measurable.",
        ["The six contributions are ordered from theory to engineering: the "
         "impossibility proof first, then the three corrections, then the "
         "calibration finding, then the reproducibility findings.",
         "The phrase 'not merely a measurement blind spot' appears early and "
         "deliberately. It signals that this is a causal claim, which is a "
         "stronger and more publishable position than an observational one."],
        "Is this really a problem with the field, or just with one paper?",
        "Section 2 positions the critique against the family, and Table 2 shows "
        "all five evaluated allocators landing at a 1.62 to 1.91 part-time "
        "premium. The uniformity is the evidence: they share an objective, so "
        "they share its consequence.")

    F += sec_block(
        2, "Related Work",
        "Establish scope. A critique of one paper is a comment; a critique of an "
        "objective shared by a literature is a contribution.",
        "Three strands are surveyed. The key sentence is that every driver-side "
        "fairness measure in the cited work is a functional of accumulated "
        "earnings alone, and none conditions on hours supplied.",
        ["Efficiency matching gives us the architecture we inherit.",
         "Driver-side fairness is where the critique lands.",
         "Long-horizon fairness is the direct antecedent, and Section 7.5 turns "
         "its own motivation against it by showing its disparity grows with the "
         "horizon it was designed to protect."],
        "Has nobody really used a rate-based fairness measure before?",
        "Labour economics measures compensation per engaged hour routinely, and "
        "the paper says so. The gap is specifically in allocation methods, where "
        "we found no prior work optimising a per-hour objective or decomposing it "
        "across activity groups. That is a narrow, defensible novelty claim.")

    F += sec_block(
        3, "Preliminaries and Problem Formulation",
        "Give the critique a precise target. You cannot disprove an informal "
        "claim, so prior work is restated formally first.",
        "Sets up the graph, requests, drivers and the assignment; then states "
        "prior work's utility (Eq 1), efficiency and fairness (Eq 2), and its "
        "objective (Eq 3).",
        ["Table 1 is the notation table. Reviewers use it constantly, so it is "
         "placed before the first equation rather than in an appendix.",
         "The last paragraph introduces the online epoch set, hours and activity "
         "group, and states plainly that these are absent from prior "
         "formulations. That single sentence is the hinge of the whole paper: the "
         "deficiencies are invisible precisely because these objects are never "
         "constructed."],
        "Are you representing prior work fairly?",
        "Equations 1 to 3 are transcribed from the cited paper, and Section 5 "
        "shows our utility reduces to Equation 1 exactly when congestion is "
        "neutral. Prior work is a special case of ours, which is the fairest "
        "possible representation.")

    F += sec_block(
        4, "Motivating Analysis",
        "Prove the problems exist before proposing anything. A method section "
        "that arrives before its motivation invites the question 'why should I "
        "care'.",
        "Three subsections, one per deficiency: the utility inverts one "
        "assignment in ten (4.1); variance of totals is hours-blind and provably "
        "incompatible with rate parity (4.2); access fairness cannot even be "
        "defined under the prevailing model (4.3).",
        ["4.1 leads with the sign inversion rather than the 2.11x travel-time "
         "spread. A sign error is a decision error, which is far more serious "
         "than a calibration error, so it belongs first.",
         "4.2 contains Proposition 1 and its two-line proof. This converts our "
         "apparently unfavourable result - that our Var(total) is worse - into a "
         "theorem. A reviewer can no longer read it as a deficiency.",
         "4.3 makes a subtle point: the problem is not that access fairness is "
         "unmeasured but that it is unmeasurable, because unbounded concurrency "
         "means 'busy' has no referent."],
        "Is 10.2% of assignments a lot?",
        "It is one in ten decisions where the allocator believes a trip is "
        "profitable when it is a loss, or the reverse. Not a small mispricing: a "
        "sign error. The worked example in the text makes the mechanism concrete "
        "with an 8 km trip and a 5 km approach.")

    F += sec_block(
        5, "Method",
        "Introduce the three corrections and the machinery they require, in the "
        "order the deficiencies were raised.",
        "5.1 congestion-aware utility; 5.2 pay-rate fairness with group "
        "decomposition; 5.3 access fairness plus the occupancy model it needs; "
        "5.4 the unified objective, the state augmentation and the weight "
        "calibration.",
        ["The reduction property in 5.1 is the most important design decision in "
         "the paper. Because our utility becomes prior work's utility exactly at "
         "c = 1, the congestion correction can be ablated without confounding.",
         "The decomposition in 5.2 is the law of total variance, a theorem. This "
         "matters rhetorically: within and between are an exact partition, not "
         "two metrics we invented and hoped were complementary.",
         "5.3 explains why the occupancy model is a precondition rather than a "
         "refinement. Without it there is no Gap 2 metric at all.",
         "5.4 contains two things a reader may skim but a reviewer will not: the "
         "state augmentation, without which the objective cannot be optimised, "
         "and the weight calibration, without which one term is inert."],
        "Why divide the paid leg by c and multiply the approach leg by c?",
        "Because they are economically different. The paid leg returns a fixed "
        "fare for more of the driver's time, so its value per unit time falls. "
        "The approach leg is unpaid, so more time is pure loss. Both adjustments "
        "move against the driver. The paper also states openly that the "
        "multiplicative form is a modelling choice, selected because it is the "
        "only candidate preserving the exact reduction.")

    F += sec_block(
        6, "Experimental Setup",
        "Make the work reproducible and pre-empt the three objections a careful "
        "reviewer will raise: the granularity, the simulated drivers and the "
        "sampling rate.",
        "Data and cleaning; graph construction with measured granularity "
        "selection; the simulated fleet; the protocol; the forecaster; and the "
        "five baselines.",
        ["The granularity choice is justified by measurement, not preference: at "
         "0.5 km only 15.6% of cells are estimable, against about 90% of trips at "
         "1.1 km.",
         "The circuity check of 1.320 against the known New York value of about "
         "1.3 is an external validation the paper did not fit to. Worth "
         "mentioning aloud in a viva.",
         "The sampling rate is solved rather than inherited, and the paper "
         "explains that the inherited rate would saturate the fleet and collapse "
         "the access-fairness metric to zero for every method."],
        "Simulated drivers undermine the whole result.",
        "Public taxi records contain no driver identifier, so every paper in this "
        "line simulates drivers - including the one we extend. Our claims concern "
        "the objective, not the empirical distribution of labour supply. The paper "
        "lists this first among its limitations rather than burying it.")

    F += sec_block(
        7, "Results",
        "Answer seven distinct questions, each in its own subsection, and include "
        "the answers that are unflattering.",
        "7.1 baseline comparison; 7.2 isolating each correction; 7.3 the "
        "objective causes the disparity; 7.4 ablations; 7.5 horizon behaviour; "
        "7.6 the parity frontier; 7.7 robustness.",
        ["7.2 exists because 7.1 cannot isolate corrections: in Table 2 every "
         "method runs with the congestion-aware utility, so only the fairness "
         "objective varies. Table 3 starts from a faithful baseline instead.",
         "7.3 is the causal claim and the single most important table in the "
         "paper.",
         "7.5 is the sharpest rhetorical turn: the method designed for long-term "
         "fairness accumulates long-term unfairness.",
         "7.4 reports two negative results, including our inability to reproduce a "
         "claim from the base paper about its forecaster."],
        "You report your own method as worse on the standard metric. Why should "
        "we accept that?",
        "Because Proposition 1 predicts it. Equal pay rates across unequal hours "
        "forces unequal totals. If our Var(total) had improved, one of the two "
        "measurements would be wrong.")

    F += sec_block(
        8, "Reproducibility Analysis of the Base Method",
        "Convert an apparent weakness - our absolute utility is an order of "
        "magnitude below the published figure - into a contribution.",
        "Two findings: the fleet size is recoverable from the published table and "
        "equals exactly 20 on every row; and the published per-driver earnings "
        "exceed the physical capacity of the paper's own protocol by roughly 36 "
        "times.",
        ["The 36x argument is arithmetic a reader can check in their head: 4,791 "
         "utility divided by about 2.5 km per trip is about 1,924 trips, against "
         "at most about 53 in a 14-hour week.",
         "The section closes by explicitly declining to allege error, and "
         "reframes the issue as a reporting standard for the field. That "
         "restraint is deliberate and makes the finding harder to dismiss as an "
         "attack."],
        "This looks like you are attacking the authors to excuse your own numbers.",
        "The section states that it draws no inference about the correctness of "
        "their implementation, and the actual recommendation is a community norm: "
        "report fleet size, concurrency semantics and spatial granularity. Also "
        "note the fleet-size recovery is arithmetic, not an accusation: total "
        "divided by mean is a count.")

    F += sec_block(
        9, "Discussion",
        "Say who loses before a reviewer finds out, and list the limitations "
        "honestly.",
        "9.1 gives the redistribution incidence in currency. 9.2 lists five "
        "limitations.",
        ["Enforcing parity moves money: full-time drivers gain about "
         f"${(mo.rate_full_time-mk.rate_full_time)*46.4*50:,.0f} a year and "
         f"part-time drivers lose about "
         f"${abs((mo.rate_part_time-mk.rate_part_time)*14.0*50):,.0f}.",
         "The argument is not that this incidence is optimal. It is that prior "
         "work imposes the opposite incidence without representing it, since "
         "hours never enter its fairness functional.",
         "Limitation 2 concedes that the functional form of the congestion "
         "correction is a choice, not a derivation, and that sensitivity to it is "
         "untested."],
        "You are just choosing a different set of winners. Why is yours better?",
        "We do not claim it is. We claim it is visible and tunable. The frontier "
        "in Table 7 lets a platform pick a point deliberately, whereas the "
        "prevailing objective picks one silently.")

    F += sec_block(
        10, "Conclusion",
        "Restate the claim compactly and leave the reader with two implications.",
        "Methodologically, fairness functionals in labour matching should "
        "condition on supplied hours. Practically, papers should report fleet "
        "size, concurrency semantics and granularity.",
        [],
        "What is the single sentence a reader should remember?",
        "A functional of accumulated output alone will misattribute equity in any "
        "setting where participation is heterogeneous.")
    F.append(PageBreak())

    # ============================== 4. EQUATIONS ==============================
    F.append(P("3. Every equation explained", "h1"))
    F.append(P(
        "Fourteen numbered equations. Equations 1 to 3 are prior work; 4 to 14 "
        "are ours. For each: what it says in words, what every symbol means, and "
        "why it is in the paper.", "body"))

    eqs = [
        (1, "U(r,v) = Geo(d_r, s_r) - Geo(s_r, g_v)",
         "Prior work's trip value",
         "The value of giving request r to driver v is the distance of the paid "
         "trip minus the distance the driver must drive empty to reach the "
         "passenger.",
         [("U", "utility, in kilometres"),
          ("Geo(a,b)", "shortest road distance from a to b"),
          ("s_r", "pickup point of the request"),
          ("d_r", "dropoff point"),
          ("g_v", "where driver v currently is")],
         "It is the target of the first criticism: there is no time in it, so a "
         "trip is worth the same at 04:00 and at 18:00."),
        (2, "Pi(M) = SUM_v o_v(M(v));   F(M) = Var(o_v(M(v)))",
         "Prior work's efficiency and fairness",
         "Efficiency is the total utility summed over all drivers. Fairness is "
         "the variance of those totals: if every driver ends the week with the "
         "same amount, variance is zero and the allocation is called perfectly "
         "fair.",
         [("Pi", "total utility across the fleet"),
          ("o_v", "utility driver v accumulated over the horizon"),
          ("M", "the assignment; M(v) is what driver v received"),
          ("Var", "variance, a measure of spread")],
         "This is the equation the paper argues against. Everything in Section 4.2 "
         "and Proposition 1 is about this definition of F."),
        (3, "max_M  Pi(M) - lambda*omega*F(M)   s.t.  SUM_v I_rv <= 1",
         "Prior work's objective",
         "Choose the assignment maximising total utility minus a weighted "
         "fairness penalty, subject to each request going to at most one driver.",
         [("lambda", "how much you care about fairness"),
          ("omega", "a scale factor converting the variance into utility units"),
          ("I_rv", "1 if request r goes to driver v, else 0")],
         "Note omega. Prior work fixes it at 0.6 without saying what range that "
         "produced, and Section 5.4 shows this does not transfer."),
        (4, "c_p(a,b) = tau_p(a,b) / tau_bar(a,b),  clipped to [0.5, 3.0]",
         "Congestion multiplier (ours)",
         "How much slower than usual this route is right now. c = 1 means normal "
         "speed, c = 2 means twice as slow, c = 0.5 means twice as fast.",
         [("tau_p(a,b)", "travel time from a to b in time profile p"),
          ("tau_bar(a,b)", "the period-mean travel time - exactly the number "
                           "prior work uses"),
          ("p", "time profile: weekday or weekend, crossed with hour of day, "
                "giving 48")],
         "Because the denominator is precisely the quantity prior work "
         "substitutes for reality, c measures exactly the information that was "
         "discarded. That framing is why the correction is principled rather "
         "than arbitrary."),
        (5, "U_ca = Geo(d,s)/c_p(s,d) - w*Geo(s,g)*c_p(g,s)",
         "Congestion-aware utility (ours)",
         "Same as Equation 1, but the paid leg is divided by congestion and the "
         "unpaid approach is multiplied by it.",
         [("w", "weight on the approach cost; 1 by default, matching prior work"),
          ("division", "a congested paid trip returns the same fare for more of "
                       "the driver's time, so its value per unit time falls"),
          ("multiplication", "a congested unpaid approach burns more time for "
                             "nothing, so it costs more")],
         "The key property: set c = 1 and this becomes Equation 1 exactly, "
         "verified to a maximum absolute difference of zero over 300,000 random "
         "cases. Prior work is a strict special case, so the ablation isolates "
         "the congestion effect alone."),
        (6, "rho_v = o_v / H_v,   H_v = |O_v| * Delta",
         "Hourly pay rate (ours)",
         "What a driver earned divided by how long they were online. This is the "
         "quantity a worker actually experiences.",
         [("rho_v", "driver v's pay rate"),
          ("H_v", "hours driver v was online"),
          ("O_v", "the set of epochs in which v was online"),
          ("Delta", "length of one decision epoch, 5 minutes here")],
         "H_v counts all online time including idle epochs, because a driver is "
         "on the clock whether or not the platform sends them work. This is the "
         "single substitution that makes the hidden disparity visible."),
        (7, "Var(rho) = E_g[Var(rho|g)] + Var_g(E[rho|g])",
         "Group decomposition (ours)",
         "Total pay-rate inequality splits exactly into inequality within groups "
         "plus inequality between group averages.",
         [("within-group", "how unequal drivers are compared with others of their "
                           "own type"),
          ("between-group", "how far apart the two group averages are - the "
                            "structural pay gap"),
          ("g", "activity group: full-time or part-time")],
         "This is the law of total variance, a standard identity, so the two "
         "components sum to the total with no residual and can carry independent "
         "weights without double-counting. Verified to a relative error of about "
         "6e-16."),
        (8, "occ(r,v) = ceil( (tau_p(g_v,s_r) + tau_p(s_r,d_r)) / Delta )",
         "Occupancy (ours)",
         "How many decision epochs an assignment takes up: time to reach the "
         "passenger plus time to complete the trip.",
         [("ceil", "round up, so a trip never releases a driver early"),
          ("capacity", "one trip at a time in our environment")],
         "Prior work has no equivalent. Its reward sums over an unbounded set "
         "with no duration, so a driver can absorb unlimited work and 'busy' is "
         "undefined. This equation is a precondition for Equation 9, not a "
         "refinement of it."),
        (9, "nu_v = (busy epochs) / (online epochs)",
         "Utilisation (ours)",
         "The fraction of a driver's online time actually spent working. 0.25 "
         "means idle three-quarters of the time.",
         [("nu_v", "driver v's utilisation, bounded in [0,1] by construction")],
         "This is the access-fairness quantity. Fairness is then the variance of "
         "nu across drivers, with the same group decomposition applied."),
        (10, "pi_t = busy(t)/online(t);  opp_v = mean of pi_t over O_v;  "
             "nu_adj = nu_v / opp_v",
         "Opportunity normalisation (ours)",
         "Divide each driver's utilisation by what drivers working the same hours "
         "actually achieved. A result of 1 means the driver did as well as "
         "contemporaneous peers.",
         [("pi_t", "system-wide busy fraction at epoch t - how much work was "
                   "available"),
          ("opp_v", "average availability over the epochs v actually worked"),
          ("nu_adj", "utilisation relative to opportunity")],
         "Raw variance conflates two different things: a platform starving a "
         "driver, and a driver choosing to work 04:00 when there is no demand. "
         "Only the first is the allocator's fault. Both forms are reported so no "
         "conclusion rests on this normalisation alone."),
        (11, "SR(M) = SUM_v u_v - omega_rho[lam_w*Var_within + lam_b*Var_between] "
             "- omega_nu*lam_nu*Var(nu)",
         "The unified objective (ours)",
         "Total utility, minus a weighted penalty for within-group pay "
         "inequality, minus a weighted penalty for the between-group gap, minus a "
         "weighted penalty for uneven idle time.",
         [("lam_w, lam_b, lam_nu", "the three fairness weights"),
          ("omega_rho, omega_nu", "scale factors, measured not guessed "
                                  "(Equation 13)")],
         "Setting lam_b and lam_nu to zero and switching the target back to "
         "totals recovers Equation 3 exactly, so prior work stays nested inside "
         "our objective. Selected values: lam_w = 1, lam_b = 16, lam_nu = 0.5."),
        (12, "score(r,v) = r_scalarised + gamma^occ(r,v) * V[s'] - V[s]",
         "Assignment score in the MDP (ours)",
         "The value of an assignment is its immediate scalarised reward plus the "
         "discounted value of where it leaves the driver, minus the value of "
         "where they are now.",
         [("gamma^occ", "the discount exponent is the occupancy, so a long trip "
                        "is discounted for the driver time it commits"),
          ("V[s]", "tabular value function over 48 profiles x 87 nodes x 5 "
                   "deficit levels"),
          ("advantage form", "scores the improvement from taking the assignment")],
         "The state includes a discretised fairness deficit. Without it the "
         "policy could evaluate the fairness term after the fact but never act on "
         "it, because prior work's state is location-only."),
        (13, "omega_x = E|u| / ( 2 * E|x| * dx / n )",
         "Weight calibration (ours)",
         "Choose each scale factor so that a typical assignment's fairness "
         "penalty is the same order of magnitude as a typical assignment's "
         "utility.",
         [("E|u|", "mean absolute utility of an assignment, measured from a "
                   "warm-up run"),
          ("dx", "the increment one assignment makes to the fairness quantity"),
          ("n", "number of drivers")],
         "Without this the access-fairness penalty came to about 0.009 against a "
         "typical utility of 2.4, roughly 270 times too small, so the term was "
         "inert and its ablation showed nothing. This is a reproducibility "
         "finding about prior work's fixed omega, not just an implementation "
         "detail."),
        (14, "dVar = (2*x_j*dx + dx^2)/n - (2*S*dx + dx^2)/n^2",
         "Exact incremental variance marginal (ours)",
         "The exact change in a variance when one driver's value increases by dx, "
         "computed in constant time from running sums.",
         [("S", "running sum of the quantity"),
          ("x_j", "the affected driver's current value"),
          ("exact", "this is algebra, not an approximation")],
         "Greedy re-evaluation over all candidate pairs at every epoch needs this "
         "marginal thousands of times per epoch. Verified against brute-force "
         "recomputation over 3,000 random instances to a maximum error of about "
         "7e-15."),
    ]
    for n, form, name, words, syms, why in eqs:
        blk = [P(f"Equation {n} - {name}", "h3"),
               P(form.replace("<", "&lt;"), "eq"),
               P(f"<b>In words.</b> {words}", "body")]
        blk.append(TBL(["Symbol", "Meaning"],
                       [[s, m] for s, m in syms], [26, 146]))
        blk.append(P(f"<b>Why it is in the paper.</b> {why}", "body"))
        F.append(KeepTogether(blk))
        F.append(Spacer(1, 1.5 * mm))
    F.append(PageBreak())

    # ============================== 5. TABLES ==============================
    F.append(P("4. Every table explained", "h1"))
    F.append(P(
        "Nine tables. For each: what question it answers, what the columns mean, "
        "the number to look at first, and what a reviewer may probe.", "body"))

    F.append(P("4.0 Column glossary used across the results tables", "h2"))
    F.append(TBL(["Column", "Meaning", "Good direction"], [
        ["Pi", "Total utility summed over all drivers for the week, in "
               "kilometre-equivalent units", "higher"],
        ["F", "Prior work's fairness measure: variance of weekly total utility",
         "lower, but see Proposition 1"],
        ["Var rho", "Variance of hourly pay rate across drivers - our headline "
                    "fairness measure", "lower"],
        ["within", "Part of Var rho arising among drivers of the same activity "
                   "group", "lower"],
        ["between", "Part of Var rho arising between the group averages - the "
                    "structural pay gap", "lower"],
        ["FT rho / PT rho", "Mean hourly pay rate of full-time and part-time "
                            "drivers", "should converge"],
        ["ratio", "PT rho divided by FT rho. 1.00 is exact parity", "towards 1.00"],
        ["Var nu", "Variance of utilisation - how unevenly busy time is shared",
         "lower"],
        ["nu tilde", "Same, after removing the effect of which hours a driver "
                     "chose to work", "lower"],
        ["Gini", "Gini coefficient of pay rate; a bounded second opinion on "
                 "inequality", "lower"],
        ["idle", "Number of drivers who received no work in the entire week",
         "zero"],
    ], [24, 116, 32]))

    tables = [
        (1, "Notation", "Section 3",
         "Defines every symbol before it is used.",
         "Nothing to read; it is a lookup.",
         "Reviewers return to it constantly. Its presence before the first "
         "equation is a courtesy that improves review outcomes."),
        (2, "Allocator comparison on an identical scenario", "Section 7.1",
         "How do six allocators compare when everything except the objective is "
         "held fixed?",
         "The <b>ratio</b> column. All five prior methods lie between 1.62 and "
         f"1.91; ours is {o.rate_group_ratio:.2f}. Also note that our F column is "
         f"worse ({k.fairness_total_var:,.0f} to {o.fairness_total_var:,.0f}), "
         "which Proposition 1 predicts.",
         "A reviewer may ask why our F is worse. The answer is the proposition. A "
         "second likely probe: every method here uses the congestion-aware "
         "utility, so this table compares objectives, not corrections - which is "
         "exactly why Table 3 exists."),
        (3, "Ablation from a faithful baseline, one correction at a time",
         "Section 7.2",
         "What does each correction contribute on its own, starting from true "
         "prior work?",
         f"Row 1 versus row 3. The congestion correction alone raises utility by "
         f"{100*(g1a.total_utility-b0.total_utility)/b0.total_utility:+.1f}% but "
         f"raises pay-rate variance from {b0.rate_var:.3f} to {g1a.rate_var:.3f}. "
         f"Rate fairness alone reaches ratio {g1b.rate_group_ratio:.3f} at almost "
         f"no cost. Together they achieve both.",
         "This is the most informative table in the paper and also the most "
         "honest: it shows one of our own corrections making fairness worse in "
         "isolation. The request stream is pinned across all rows, so the "
         "comparison is exact."),
        (4, "Effect of enabling variance-of-totals fairness", "Section 7.3",
         "Does prior work's objective cause the disparity, or merely fail to see "
         "it?",
         f"The between-group column: {nofair.rate_var_between:.4f} with no "
         f"fairness term, {k.rate_var_between:.4f} with prior work's term "
         f"enabled. About ninety times worse.",
         "Small table, largest claim in the paper. It elevates the critique from "
         "observational to causal, and it is the empirical counterpart of "
         "Proposition 1's corollary."),
        (5, "Component ablations", "Section 7.4",
         "Does each component of our own method earn its place?",
         f"The access-fairness row: removing it moves Var nu from "
         f"{o.util_var:.4f} to {noutil.util_var:.4f}. Also the prediction row, "
         f"where removing the forecaster slightly raises utility - a negative "
         f"result we report rather than omit.",
         "Reviewers look here for unearned components. The negative results "
         "improve credibility rather than damaging it."),
        (6, "Fairness against horizon length", "Section 7.5",
         "What happens as the evaluation window grows from one day to seven?",
         f"The two 'between' columns. Prior work grows from {hp[V['PAPER']].iloc[0]:.4f} "
         f"to {hp[V['PAPER']].iloc[-1]:.4f}, about "
         f"{hp[V['PAPER']].iloc[-1]/hp[V['PAPER']].iloc[0]:.0f} times; ours stays "
         f"near {hp[V['OURS']].iloc[-1]:.4f}.",
         "The rhetorical peak of the paper. Prior work is explicitly motivated by "
         "long-horizon fairness, and on a group-conditioned measure its disparity "
         "accumulates with exactly the horizon it was designed to protect."),
        (7, "Parity frontier in lambda_b", "Section 7.6",
         "Is parity a lucky operating point, or a controllable one?",
         "The ratio column rising towards 1.00 as lambda_b increases, while total "
         "utility varies by under about 1.2% across the whole range.",
         "Pre-empts 'you tuned to one favourable point'. It also supports the "
         "policy argument in Section 9.1, because a platform can choose a "
         "position on this curve deliberately."),
        (8, "Robustness across three independent seeds", "Section 7.7",
         "Is the disparity systematic or a sampling artefact?",
         f"Prior work's ratio is {sd.loc['Paper',('rate_group_ratio','mean')]:.3f} "
         f"plus or minus {sd.loc['Paper',('rate_group_ratio','std')]:.3f} on every "
         f"seed. The standard deviation is tiny, which is the point.",
         "The narrow spread is the evidence. A systematic effect reproduces; noise "
         "does not."),
        (9, "Fleet size recovered from published results", "Section 8",
         "Why are our absolute numbers an order of magnitude below the published "
         "ones?",
         "The last column: exactly 20.000 on all five rows.",
         "Arithmetic, not accusation - total divided by mean is a count. Combined "
         "with the 36x capacity argument it explains the discrepancy without "
         "alleging error."),
    ]
    for n, name, where, question, first, note in tables:
        blk = [P(f"Table {n} - {name}  <font size=8 color='#5A6470'>"
                 f"({where})</font>", "h3"),
               P(f"<b>Question it answers.</b> {question}", "body"),
               P(f"<b>Read this first.</b> {first}", "body"),
               P(f"<b>Note.</b> {note}", "small")]
        F.append(KeepTogether(blk))
        F.append(Spacer(1, 1.5 * mm))
    F.append(PageBreak())

    # ============================== 6. FIGURES ==============================
    F.append(P("5. Every figure explained", "h1"))
    F.append(P(
        "Four figures. Each is reproduced below at reduced size with an "
        "explanation of the axes and what to look for.", "body"))

    figs = [
        ("outputs/fig_gap1a_congestion.png", 1,
         "Travel-time variation discarded by a period mean", "Section 4.1",
         "Two panels. <b>Left:</b> horizontal axis is hour of day, vertical axis "
         "is citywide mean speed in km/h; the solid line is weekdays, the dashed "
         "line weekends, and the dotted horizontal line is the single period mean "
         "that prior work substitutes for all of it. <b>Right:</b> a histogram; "
         "horizontal axis is the ratio of peak to off-peak travel time for the "
         "same origin-destination pair, vertical axis is how many pairs fall in "
         "each bin.",
         "In the left panel, the gap between the curves and the dotted line is "
         "the error prior work accepts at every hour. In the right panel, note "
         "that the distribution sits well to the right of 1.0 - the median is "
         f"{V['fc'] and 2.11:.2f} - so this is not a handful of outlier routes "
         "but the typical case.",
         "It establishes that Gap 1a is real before any method is proposed. A "
         "reviewer who accepts this figure has accepted the premise of Section "
         "5.1."),
        ("outputs/fig_gaps_by_method.png", 2,
         "Fairness outcomes by allocator", "Section 7.1",
         "Three panels, six allocators along each horizontal axis. <b>Left:</b> "
         "paired bars giving mean hourly pay for full-time and part-time drivers. "
         "<b>Centre:</b> between-group pay-rate variance. <b>Right:</b> raw and "
         "opportunity-normalised utilisation variance. Lower is fairer in the "
         "centre and right panels.",
         "In the left panel the first five allocators show a tall part-time bar "
         "beside a short full-time bar; in ours the two are level. That visual "
         "levelling is the paper's central result, and it is the one image to "
         "show someone who will not read the tables.",
         "This is the figure to put on a slide. It carries Table 2's message "
         "without requiring the reader to parse eleven columns."),
        ("outputs/fig4_horizon.png", 3,
         "Fairness against horizon length", "Section 7.5",
         "Three panels sharing a horizontal axis of horizon length in days, one "
         "to seven. Vertical axes are logarithmic. <b>Left:</b> prior work's "
         "fairness measure. <b>Centre:</b> pay-rate variance. <b>Right:</b> "
         "between-group pay-rate variance.",
         "The right panel. Prior work's line climbs steadily while ours stays "
         "flat and roughly two orders of magnitude lower. The divergence widening "
         "with the horizon is the point: this is not a fixed offset but an "
         "accumulating one.",
         "Supports the strongest rhetorical claim in the paper. Prior work is "
         "named for long-term fairness; this panel shows long-term unfairness "
         "accumulating in the dimension its metric cannot observe."),
        ("outputs/fig_sweep_pareto.png", 4,
         "Trade-off frontiers over the weight sweep", "Section 7.6",
         "Three scatter panels. Horizontal axis in each is total utility; "
         "vertical axes are the three fairness measures on log scales. Each point "
         "is one weight configuration with both fairness terms active; colour "
         "encodes the between-group weight. The cross marks the reproduced prior "
         "method.",
         "Points lying below and to the right of the cross dominate prior work: "
         "more utility and more fairness at once. There are many such points, "
         f"{len(V['dom'])} of {len(V['gr'])} configurations dominating on all six "
         "metrics simultaneously.",
         "Answers the standard objection that fairness gains are simply purchased "
         "with efficiency. If that were true, no point could sit down and to the "
         "right of the cross."),
    ]
    for path, n, name, where, axes, look, why in figs:
        blk = [P(f"Figure {n} - {name}  <font size=8 color='#5A6470'>"
                 f"({where})</font>", "h3")]
        blk += FIG(path, 150)
        blk += [P(f"<b>Axes.</b> {axes}", "body"),
                P(f"<b>What to look for.</b> {look}", "body"),
                P(f"<b>Why it is in the paper.</b> {why}", "small")]
        F.append(KeepTogether(blk))
        F.append(Spacer(1, 2 * mm))
    F.append(PageBreak())

    # ============================== 7. Q&A ==============================
    F.append(P("6. Reviewer question bank", "h1"))
    F.append(P(
        "Twelve questions a referee or examiner is likely to ask, with the answer "
        "and where in the paper it is supported. Rehearse these.", "body"))
    F.append(TBL(["Question", "Answer", "Support"], [
        ["Your Var(total) is worse than the baseline. Is your method not simply "
         "less fair?",
         "No - it is provably impossible to be better on both. Equal pay rates "
         "across unequal hours forces unequal totals. If our Var(total) had "
         "improved, one of the two measurements would be wrong.",
         "Proposition 1, Section 4.2"],
        ["Your total utility is 10,145 against the published 95,824.",
         "Different experimental settings, not less data. Prior work used 20 "
         "drivers (recoverable from its own table) and permits unbounded "
         "concurrency, implying about 36 times more trips per driver than its "
         "protocol allows. All our comparisons are within one harness.",
         "Section 8, Table 9"],
        ["The drivers are simulated, so the fairness results are invented.",
         "Public taxi records contain no driver identifier, so every paper in this "
         "line simulates drivers, including the one we extend. Our claims concern "
         "the objective, not the empirical distribution of labour supply.",
         "Section 6, Limitation 1"],
        ["Why divide the paid leg by c but multiply the approach leg?",
         "They are economically different: the paid leg returns a fixed fare for "
         "more time, the approach leg is unpaid so extra time is pure loss. The "
         "form is also the only one that reduces exactly to prior work at c = 1.",
         "Section 5.1, Limitation 2"],
        ["Is 10.2% sign inversion actually important?",
         "It is a decision error, not a pricing error: one assignment in ten is "
         "labelled profitable when it loses money, or the reverse.",
         "Section 4.1"],
        ["You tuned lambda to a favourable point.",
         f"The full frontier is reported, and {len(V['dom'])} of "
         f"{len(V['gr'])} swept configurations dominate prior work on all six "
         f"metrics at once.",
         "Section 7.6, Table 7"],
        ["Could the disparity be an artefact of distance-denominated utility?",
         f"No. Re-run in currency at the measured fare rate, prior work pays "
         f"${mk.rate_full_time:.2f} against ${mk.rate_part_time:.2f} per hour "
         f"(ratio {mk.rate_group_ratio:.3f}); ours is ${mo.rate_full_time:.2f} "
         f"against ${mo.rate_part_time:.2f}.",
         "Section 7.7"],
        ["Could it be one unlucky random draw?",
         f"Prior work's ratio is "
         f"{sd.loc['Paper',('rate_group_ratio','mean')]:.3f} plus or minus "
         f"{sd.loc['Paper',('rate_group_ratio','std')]:.3f} across three "
         f"independent seeds.",
         "Section 7.7, Table 8"],
        ["Which of your three corrections actually does the work?",
         "Rate fairness does the fairness work and is nearly free; the congestion "
         "correction does the efficiency work but hurts fairness alone; access "
         "fairness does the idle-time work. Table 3 separates them.",
         "Section 7.2, Table 3"],
        ["You changed the evaluation protocol from the original.",
         "Necessarily. A two-hour daily window caps a driver at 14 weekly hours, "
         "so no driver reaches full-time and the activity-group contrast cannot "
         "exist. We ran the original protocol to confirm: 0 of 200 drivers "
         "reached the threshold.",
         "Sections 6 and 8"],
        ["Part-time drivers lose income under your method. Is that fair?",
         "We do not claim our incidence is optimal, only that it is visible and "
         "tunable. Prior work imposes the opposite incidence without representing "
         "it, because hours never enter its fairness functional.",
         "Section 9.1"],
        ["The forecaster gives no benefit. Does that not undermine the method?",
         "It undermines a component we inherited, not one we introduced, and we "
         "report it plainly. Our contributions are the objective and the "
         "measurement corrections, none of which depend on the forecaster.",
         "Section 7.4"],
    ], [50, 90, 32]))
    F.append(PageBreak())

    # ============================== 8. CHECKLIST ==============================
    F.append(P("7. Before you submit", "h1"))
    F.append(P("7.1 Strengths to make sure a reviewer notices", "h2"))
    F.append(BUL([
        "A <b>proof</b> plus the measurement it predicts (Proposition 1 and "
        "Table 4). Few applied fairness papers have both.",
        "An <b>exact reduction</b> to prior work, verified to zero difference, so "
        "the ablation genuinely isolates one effect.",
        "A fairness decomposition that is a <b>theorem</b>, not an invented pair "
        "of metrics.",
        "<b>Negative results reported</b>: the forecaster does not help, and one "
        "of our own corrections hurts fairness in isolation.",
        "<b>Robustness on three axes</b>: seeds, currency units, and a 20-point "
        "weight sweep.",
        "<b>Reproducibility findings</b> that are useful to the community rather "
        "than merely critical.",
    ]))
    F.append(P("7.2 Weakest points, and what to do about them", "h2"))
    F.append(TBL(["Weak point", "Mitigation available now", "Best fix"], [
        ["Simulated drivers",
         "Stated as Limitation 1; prior work does the same",
         "A fleet-size sensitivity study (100 / 200 / 400) - the harness supports "
         "it with a one-line config change"],
        ["Single city and single month",
         "Stated as Limitation 5",
         "Repeat on a second month; the pipeline rebuilds in about two minutes"],
        ["Functional form of the congestion correction is a choice",
         "Stated as Limitation 2, with the reason for the choice",
         "Report one alternative form and show conclusions are unchanged"],
        ["Residual access regression in the fully combined method",
         "Stated as Limitation 4",
         "Add a per-driver floor so no driver can be skipped for a whole week"],
        ["Opportunity normalisation has a small self-bias",
         "Stated as Limitation 3; always reported beside the raw measure",
         "Leave-one-out denominator"],
    ], [40, 62, 70]))
    F.append(P("7.3 Mechanical checklist", "h2"))
    F.append(BUL([
        "Compile main.tex and read the PDF - the tables are wide and may need "
        "column trimming in a two-column layout.",
        "Swap the documentclass line for the target venue and re-check table "
        "widths; llncs is narrower than the generic article class.",
        "Add author names, affiliations and an acknowledgements line.",
        "Confirm the anonymity requirements of the venue before including the "
        "repository URL.",
        "Check the page limit. At 10 sections, 9 tables and 4 figures the paper "
        "will run long for an 8-page limit; Table 5 and Table 8 are the first "
        "candidates for an appendix.",
        "Decide whether Section 8 stays in the main body. It is a strong "
        "contribution but some venues prefer it as an appendix or a footnote.",
    ]))

    F.append(P("7.4 Files in this folder", "h2"))
    F.append(TBL(["File", "What it is"], [
        ["main.tex", "The paper source, self-contained, compiles with pdflatex"],
        ["references.bib", "Bibliography, 20 entries, no orphans"],
        ["figures/", "The four figures referenced by main.tex"],
        ["MTP_Research_Paper.docx", "The same paper as a Word document with "
                                    "figures embedded"],
        ["MTP_Paper_Explained.pdf", "This companion guide"],
        ["README_COMPILE.txt", "Compile instructions and venue-switching notes"],
    ], [46, 126]))

    # ============================== BUILD ==============================
    out = PAPER_DIR / "MTP_Paper_Explained.pdf"
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=22 * mm, bottomMargin=16 * mm,
                          title="Companion Guide - Equal Earnings Are Not Equal Pay",
                          author="Utkarsh Pandey")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="normal")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=header_footer)])
    doc.build(F)
    print(f"wrote {out}")
    print(f"size: {out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
