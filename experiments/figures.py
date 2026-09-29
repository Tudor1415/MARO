"""The five figures of the README.

    python -m experiments.figures          # writes figures/fig{1..5}_*.png

Each figure answers one question with two or three simple panels. Colour always means
the same method: random restarts violet, GRASP blue, ILS orange, Tabu Search green.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from experiments import analyse as A  # noqa: E402
from experiments import config as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

INK, INK2, MUTED, GRID, SURFACE = "#1f1f1e", "#52514e", "#a3a29c", "#e6e5e0", "#ffffff"
METHODS = ["rrls", "grasp", "ils", "tabu"]
NAME = {"rrls": "Random restarts", "grasp": "GRASP", "ils": "ILS", "tabu": "Tabu Search"}
COLOR = {"rrls": "#4a3aa7", "grasp": "#2a78d6", "ils": "#eb6834", "tabu": "#1baf7a"}
LIGHT, DARK = "#c9c8c2", "#52514e"  # neutral pair for before/after bars
SIZE = (11, 2.7)  # every figure is one short, wide strip

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 8, "axes.labelsize": 12, "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "legend.frameon": False, "lines.linewidth": 2.6, "text.color": INK,
})


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def plain_log(ax, axis="x"):
    """Log axis with plain-number tick labels (0.1, 1, 10 rather than powers of ten)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(mticker.NullFormatter())


def hbars(ax, labels, values, colors, fmt="{:.2g}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=11, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


# --------------------------------------------------------------------------- Figure 1
def fig1_local_search(e1, e2):
    """Construction versus local search: what gets you close to the optimum?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={"width_ratios": [1.25, 1]})
    s1 = A.analyse_e1(e1)[0]
    rows = [("random", "Random start"), ("grasp α=0.3", "Greedy start (GRASP)"), ("becker", "Smart start (Becker)")]
    y = np.arange(len(rows))
    before = [s1[k]["gap_construct"] for k, _ in rows]
    after = [s1[k]["gap_ls"] for k, _ in rows]
    ax1.barh(y - 0.19, before, height=0.36, color=LIGHT, label="start")
    ax1.barh(y + 0.19, after, height=0.36, color=DARK, label="after improving it")
    for yi, b, a in zip(y, before, after):
        ax1.text(b, yi - 0.19, f"  {b:.1f}%", va="center", fontsize=11)
        ax1.text(a, yi + 0.19, f"  {a:.2f}%", va="center", fontsize=11)
    ax1.set_yticks(y, [lab for _, lab in rows])
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlim(0.05, 600)
    plain_log(ax1)
    ax1.set_xlabel("distance from the best answer (%)")
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=11)
    ax1.set_title("a   A good start barely matters")

    s2 = A.analyse_e2(e2)[0]
    keys = [("swap/best", "Swap two neighbours"), ("insert/element", "Move one item anywhere")]
    hbars(ax2, [lab for _, lab in keys], [s2[k]["gap"] for k, _ in keys], [LIGHT, DARK], "{:.2g}%")
    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 600)
    plain_log(ax2)
    ax2.set_xlabel("distance from the best answer (%)")
    ax2.set_title("b   Which moves are allowed matters a lot")
    fig.tight_layout(w_pad=3)
    save(fig, "fig1_local_search")


# --------------------------------------------------------------------------- Figure 2
def fig4_why_ils_works(e6, e6h):
    """Why does perturbing a good solution (ILS) work? The landscape is a big valley."""
    land = A.analyse_e6(e6)[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={"width_ratios": [1, 1.1]})
    name = "be75eec"
    o = land[name]
    d, g = np.array(o["scatter"]["dist"]), np.array(o["scatter"]["gap"])
    ax1.scatter(100 * d[g > 0], g[g > 0], s=10, color=COLOR["ils"], alpha=0.35, lw=0)
    ax1.set_xlabel("how different from the best order (%)")
    ax1.set_ylabel("score gap (%)")
    ax1.set_title("a   Good answers look like the best one")
    ax1.text(0.03, 0.95, f"one problem, 1000 improved random orders",
             transform=ax1.transAxes, va="top", fontsize=10.5, color=INK2)

    zero = defaultdict(list)
    for o in land.values():
        zero["IO" if o["family"] == "t" else "other"].append(100 * o["zero_move_fraction"])
    for r in e6h:
        zero[r["family"]].append(100 * float(np.mean(r["zero_move_fraction"])))
    rows = [("IO", "Economy tables (main set)"), ("other", "Economy tables (other sets)"), ("xlolib", "Large economy-like"), ("mb", "Large random-like")]
    hbars(ax2, [lab for _, lab in rows], [float(np.median(zero[k])) for k, _ in rows],
          [DARK, DARK, LIGHT, LIGHT], "{:.2g}%")
    ax2.set_xscale("log")
    ax2.set_xlim(0.005, 80)
    plain_log(ax2)
    ax2.set_xlabel("moves that change nothing (%)")
    ax2.set_title("b   Real economy tables have many ties")
    fig.tight_layout(w_pad=3)
    save(fig, "fig4_why_ils_works")


# --------------------------------------------------------------------------- Figure 3
def fig5_what_helps(e3):
    """How sensitive is each method to its main parameter?"""
    summ = A.summarise_configs(A.run_stats(e3))

    def pick(algo, **kw):
        out = []
        for cfg, s in summ.items():
            a, _, rest = cfg.partition(":")
            params = dict(x.split("=", 1) for x in rest.split(","))
            if a == algo and all(params.get(k) == str(v) for k, v in kw.items()):
                out.append((params, s["success"]))
        return out

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    pts = sorted((float(p["alpha"]), v) for p, v in pick("grasp"))
    ax1.plot([a for a, _ in pts], [100 * v for _, v in pts], "-o", color=COLOR["grasp"], ms=6)
    ax1.set_xlabel("0 = fully greedy, 1 = fully random")
    ax1.set_ylabel("solved within 1 s (%)")
    ax1.set_title("a   GRASP: how greedy?")

    pts = sorted((int(p["strength"]), v) for p, v in pick("ils", acceptance="better_equal")
                 if "start" not in p and "pivot" not in p)
    ax2.plot([k for k, _ in pts], [100 * v for _, v in pts], "-o", color=COLOR["ils"], ms=6)
    ax2.set_xscale("log")
    ax2.set_xticks([1, 3, 8, 30], ["1", "3", "8", "30"])
    ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.set_xlabel("size of the shake (random moves)")
    ax2.set_title("b   ILS: how big a shake?")
    for ax in (ax1, ax2):
        ax.set_ylim(60, 100)

    runs = defaultdict(list)
    for r in e3:
        p = r["params"]
        if r["algo"] == "tabu" and p.get("null_moves") is False:
            runs[(p["diversification"] > 0, p["restart_after"] is not None)].append(r["time_to_target"] is not None)
    rows = [((False, False), "short memory only"), ((True, False), "+ long memory"),
            ((False, True), "+ restarts"), ((True, True), "+ both")]
    hbars(ax3, [lab for _, lab in rows], [100 * float(np.mean(runs[k])) for k, _ in rows],
          [LIGHT, LIGHT, COLOR["tabu"], COLOR["tabu"]], "{:.0f}%")
    ax3.set_xlim(0, 118)
    ax3.set_xlabel("solved within 1 s (%)")
    ax3.set_title("c   Tabu Search: what helps?")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_what_helps")


# --------------------------------------------------------------------------- Figure 4
def _solved_by(rows, cfg, instances, grid):
    """Fraction of runs that reached the optimum by each time (each instance weighted equally)."""
    per = defaultdict(list)
    for r in rows:
        if r["config"] == cfg and r["instance"] in instances:
            per[r["instance"]].append(np.inf if r["time_to_target"] is None else r["time_to_target"])
    return np.mean([(np.array(t)[None, :] <= grid[:, None]).mean(axis=1) for t in per.values()], axis=0)


def fig4_main(e4):
    """Main comparison on the 35 held-out LOLIB instances."""
    held = sorted({r["instance"] for r in e4} - set(C.DEVELOPMENT_INSTANCES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={"width_ratios": [1.2, 1]})
    grid = np.logspace(-2.5, 1, 300)
    for m in METHODS:
        ax1.plot(grid, 100 * _solved_by(e4, m, set(held), grid), color=COLOR[m], label=NAME[m])
    ax1.set_xscale("log")
    plain_log(ax1)
    ax1.set_xlim(grid[0], grid[-1])
    ax1.set_ylim(0, 101)
    ax1.set_xlabel("time (seconds)")
    ax1.set_ylabel("best answer found (%)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("a   Who finds the best answer fastest?")

    st = A.run_stats([r for r in e4 if r["instance"] in held])
    logp = {m: {i: math.log10(v["par10"]) for i, v in st[m].items()} for m in METHODS}
    others = ["tabu", "rrls", "grasp"]
    ratios = [10 ** np.mean([logp[m][i] - logp["ils"][i] for i in held]) for m in others]
    hbars(ax2, [NAME[m] for m in others], ratios, [COLOR[m] for m in others], "×{:.1f}")
    ax2.axvline(1, color=COLOR["ils"], lw=2, ls="--")
    ax2.set_xlim(0, 2.8)
    ax2.set_xlabel("time needed, compared with ILS (dashed line)")
    ax2.set_title("b   How much slower than ILS?")
    fig.tight_layout(w_pad=3)
    save(fig, "fig2_main_comparison")


# --------------------------------------------------------------------------- Figure 5
def fig5_hard(e5):
    """Larger instances never used for tuning: MB (proven optima) and xLOLIB (best-known values)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZE)
    mb = e5["E5mb"]
    succ = [100 * float(np.mean([r["time_to_target"] is not None for r in mb if r["config"] == m])) for m in METHODS]
    hbars(ax1, [NAME[m] for m in METHODS], succ, [COLOR[m] for m in METHODS], "{:.0f}%")
    ax1.set_xlim(0, 118)
    ax1.set_xlabel("best answer found within 20 s (%)")
    ax1.set_title("a   Large random-like problems")

    rows = e5["E5x250"]
    gaps = []
    for m in METHODS:
        per = defaultdict(list)
        for r in rows:
            if r["config"] == m:
                per[r["instance"]].append(A.rgap(r))
        gaps.append(float(np.mean([np.mean(v) for v in per.values()])))
    hbars(ax2, [NAME[m] for m in METHODS], gaps, [COLOR[m] for m in METHODS], "{:.2f}%")
    ax2.set_xlim(0, 1.8)
    ax2.set_xlabel("distance from best known answer after 30 s (%)")
    ax2.set_title("b   Large economy-like problems")
    fig.tight_layout(w_pad=3)
    save(fig, "fig3_large_problems")


def main():
    fig1_local_search(A.load("E1"), A.load("E2"))
    fig4_why_ils_works(A.load("E6"), A.load("E6h"))
    fig5_what_helps(A.load("E3"))
    fig4_main(A.load("E4"))
    fig5_hard({e: A.load(e) for e in C.E5_SETS})


if __name__ == "__main__":
    main()
