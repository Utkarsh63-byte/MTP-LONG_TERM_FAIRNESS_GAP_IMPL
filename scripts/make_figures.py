"""Paper-ready figures from the experiment outputs.

  fig_gap1a_congestion.png  travel-time spread the paper's period-mean discards
  fig_gap1b_rate_gap.png    between-group hourly-rate gap by method
  fig_gap2_utilisation.png  utilisation fairness by method
  fig4_horizon.png          fairness vs horizon length, the paper's Fig. 4 view

Run:  .venv/bin/python -m scripts.make_figures
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ltf.config import Config, OUTPUT
from ltf.data import graph as graph_mod
from ltf.data import traveltime as tt_mod
from ltf.utils import LOG, output_path

PAPER_ORDER = ["Greedy", "REASSIGN", "LAF", "Balance Ride-Pooling",
               "Kang et al. (reproduced)", "Ours (Gap 1+2)"]


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__o"] = df.method.apply(
        lambda m: PAPER_ORDER.index(m) if m in PAPER_ORDER else 99)
    return df.sort_values("__o").drop(columns="__o")


def fig_gap1a(cfg: Config) -> None:
    tt = tt_mod.load_or_build(cfg)
    g = graph_mod.load_or_build(cfg)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    hours = np.arange(24)
    wd = tt.city_speed[:24]
    we = tt.city_speed[24:]
    ax[0].plot(hours, wd, "o-", label="weekday", lw=2)
    ax[0].plot(hours, we, "s--", label="weekend", lw=2)
    ax[0].axhline(np.average(tt.city_speed), color="k", ls=":",
                  label="period mean (paper's assumption)")
    ax[0].set_xlabel("hour of day")
    ax[0].set_ylabel("citywide speed (km/h)")
    ax[0].set_title("Speed the paper collapses to one number")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    rep = tt_mod.report(tt, g)
    ax[1].hist(rep.ratio, bins=40, color="steelblue", edgecolor="white")
    ax[1].axvline(1.0, color="k", ls=":", label="no variation (paper)")
    ax[1].axvline(rep.ratio.median(), color="crimson", ls="-",
                  label=f"median {rep.ratio.median():.2f}x")
    ax[1].set_xlabel("peak / off-peak travel time, same OD pair")
    ax[1].set_ylabel("number of OD pairs")
    ax[1].set_title(f"Within-OD time variation ({len(rep)} pairs)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path("fig_gap1a_congestion.png"), dpi=150)
    plt.close(fig)
    LOG.info("wrote fig_gap1a_congestion.png")


def fig_gaps(t1: pd.DataFrame) -> None:
    d = _order(t1)
    x = np.arange(len(d))
    short = [m.replace(" (reproduced)", "").replace(" (Gap 1+2)", "*")
             for m in d.method]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    w = 0.38
    ax[0].bar(x - w / 2, d.rate_full_time, w, label="full-time")
    ax[0].bar(x + w / 2, d.rate_part_time, w, label="part-time")
    ax[0].set_ylabel("hourly rate (km-equiv / h)")
    ax[0].set_title("Gap 1b: hourly rate by group")
    ax[0].legend(fontsize=8)

    ax[1].bar(x, d.rate_var_between, color="indianred")
    ax[1].set_ylabel("between-group Var(hourly rate)")
    ax[1].set_title("Gap 1b: between-group inequity\n(lower is fairer)")

    ax[2].bar(x - w / 2, d.util_var, w, label="raw")
    ax[2].bar(x + w / 2, d.util_adj_var, w, label="opportunity-normalised")
    ax[2].set_ylabel("Var(utilisation)")
    ax[2].set_title("Gap 2: utilisation fairness\n(lower is fairer)")
    ax[2].legend(fontsize=8)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path("fig_gaps_by_method.png"), dpi=150)
    plt.close(fig)
    LOG.info("wrote fig_gaps_by_method.png")


def fig_tradeoff(t1: pd.DataFrame) -> None:
    d = _order(t1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for _, r in d.iterrows():
        mark = "*" if "Ours" in r.method else "o"
        size = 260 if "Ours" in r.method else 110
        ax[0].scatter(r.total_utility, r.fairness_total_var, s=size, marker=mark)
        ax[0].annotate(r.method.replace(" (reproduced)", ""),
                       (r.total_utility, r.fairness_total_var),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax[1].scatter(r.total_utility, r.rate_var, s=size, marker=mark)
        ax[1].annotate(r.method.replace(" (reproduced)", ""),
                       (r.total_utility, r.rate_var),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("total utility (Eq. 1)")
    ax[0].set_ylabel("Var(total utility)  [paper Eq. 2]")
    ax[0].set_title("Trade-off on the paper's own metric")
    ax[1].set_xlabel("total utility (Eq. 1)")
    ax[1].set_ylabel("Var(hourly rate)  [Gap 1b]")
    ax[1].set_title("Trade-off on the rate metric")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path("fig_tradeoff.png"), dpi=150)
    plt.close(fig)
    LOG.info("wrote fig_tradeoff.png")


def fig_horizon(h: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, grp in h.groupby("method"):
        grp = grp.sort_values("days")
        style = "-*" if "Ours" in label else "--o"
        ax[0].plot(grp.days, grp.fairness_total_var, style, label=label, lw=2)
        ax[1].plot(grp.days, grp.rate_var, style, label=label, lw=2)
        ax[2].plot(grp.days, grp.rate_var_between, style, label=label, lw=2)
    ax[0].set_ylabel("Var(total utility)  [paper Eq. 2]")
    ax[0].set_title("Paper's fairness vs horizon")
    ax[1].set_ylabel("Var(hourly rate)  [Gap 1b]")
    ax[1].set_title("Rate fairness vs horizon")
    ax[2].set_ylabel("between-group Var(rate)")
    ax[2].set_title("Group inequity vs horizon")
    for a in ax:
        a.set_xlabel("length of time horizon (days)")
        a.set_yscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path("fig4_horizon.png"), dpi=150)
    plt.close(fig)
    LOG.info("wrote fig4_horizon.png")


def main() -> int:
    cfg = Config()
    fig_gap1a(cfg)
    p1 = OUTPUT / "table1_methods.csv"
    if p1.exists():
        t1 = pd.read_csv(p1)
        fig_gaps(t1)
        fig_tradeoff(t1)
    ph = OUTPUT / "fig4_horizon.csv"
    if ph.exists():
        fig_horizon(pd.read_csv(ph))
    pg = OUTPUT / "sweep_joint_grid.csv"
    if pg.exists():
        grid = pd.read_csv(pg)
        kang = None
        if p1.exists():
            t1 = pd.read_csv(p1)
            row = t1[t1.method.str.startswith("Kang")]
            if len(row):
                kang = row.iloc[0].to_dict()
        fig_sweep(grid, kang)
        fig_parity(grid)
    print("figures written to outputs/")
    return 0


def fig_sweep(grid: pd.DataFrame, kang: dict | None = None) -> None:
    """Pareto view of the weight sweep, with the reproduced paper method marked.

    Every point here has both gap terms active, so the frontier is a frontier for
    the joint objective rather than for one gap at a time.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        ("rate_var", "Var(hourly rate)  [Gap 1b]"),
        ("rate_var_between", "between-group Var(rate)  [Gap 1b]"),
        ("util_adj_var", "opportunity-normalised Var(util)  [Gap 2]"),
    ]
    for a, (col, lab) in zip(ax, panels):
        sc = a.scatter(grid.total_utility, grid[col],
                       c=np.log10(grid.lambda_between.clip(lower=0.05)),
                       s=70, cmap="viridis", edgecolor="k", linewidth=0.4)
        if kang and col in kang:
            a.scatter([kang["total_utility"]], [kang[col]], marker="X", s=220,
                      color="crimson", edgecolor="k", zorder=5,
                      label="Kang et al. (reproduced)")
            a.legend(fontsize=8)
        a.set_xlabel("total utility (Eq. 1)")
        a.set_ylabel(lab)
        a.set_yscale("log")
        a.grid(alpha=0.3)
        plt.colorbar(sc, ax=a, label="log10 lambda_between")
    fig.suptitle("Weight sweep: down-and-right is better; both gaps active in "
                 "every point", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path("fig_sweep_pareto.png"), dpi=150)
    plt.close(fig)
    LOG.info("wrote fig_sweep_pareto.png")


def fig_parity(grid: pd.DataFrame) -> None:
    """How the between-group weight moves the fleet towards group parity."""
    g = grid[(grid.lambda_within == 1.0) & (grid.lambda_util == 0.25)]
    if len(g) < 3:
        g = grid.sort_values("lambda_between")
    g = g.sort_values("lambda_between")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(g.lambda_between.clip(lower=0.05), g.rate_group_ratio, "o-", lw=2)
    ax.axhline(1.0, color="k", ls="--", label="group parity")
    ax.axhline(1.624, color="crimson", ls=":",
               label="Kang et al. (reproduced) = 1.62")
    ax.set_xlabel("lambda_between")
    ax.set_ylabel("part-time / full-time hourly rate")
    ax.set_title("Gap 1b: driving the fleet to group parity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path("fig_parity_sweep.png"), dpi=150)
    plt.close(fig)
    LOG.info("wrote fig_parity_sweep.png")


if __name__ == "__main__":
    sys.exit(main())
