"""Turn the raw experiment logs into tables and statistics.

    python -m experiments.analyse            # writes results/tables/*.md and results/summary.json

Conventions:
* ``gap`` is the relative deviation from the reference value on the *normalised*
  objective, in percent: ``100 (f* - f) / (f* - LB)``, where
  ``LB = sum_{i<j} min(m_ij, m_ji)`` is earned by every ordering. This is the gap
  one would compute on the Optsicom normalised matrices, so it does not depend on
  the arbitrary offset that separates raw and normalised LOLIB values. It is
  larger than the raw relative gap by a factor ``f* / (f* - LB)`` (median about 1.3
  on LOLIB IO). ``f*`` is the proven optimum for LOLIB IO and MB, and the
  literature best-known value for xLOLIB, which can therefore give negative gaps.
* A run *succeeds* when it reaches ``f*`` within its CPU budget. Time-to-target
  (TTT) is the CPU time of the first improvement that reached it.
* ``PAR-10`` is the penalised average runtime: TTT for successful runs and 10x the
  budget for failed runs. It is the standard single-number summary of runtime
  distributions under a cutoff.
* Per-instance quantities are averaged over runs first, and then over instances,
  so every instance weighs the same whatever its number of runs.
"""

from __future__ import annotations

import json
import lzma
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from experiments import config as C

import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("LOP_RESULTS", ROOT / "results"))
RAW = RESULTS / "raw"
TABLES = RESULTS / "tables"

ALGO_ORDER = ["rrls", "grasp", "ils", "ils_random", "ils_scaled", "tabu", "tabu_random", "tabu_nofreq",
              "tabu_short", "tabu_scaled"]
MAIN = ["rrls", "grasp", "ils", "tabu"]  # the four methods ranked against each other
ALGO_LABEL = {
    "rrls": "RRLS (control)", "grasp": "GRASP", "ils": "ILS", "ils_random": "ILS, random start",
    "ils_scaled": "ILS, kick scaled with n", "tabu": "Tabu Search", "tabu_random": "TS, random start",
    "tabu_nofreq": "TS, restarts without frequency memory", "tabu_short": "TS, short-term memory only",
    "tabu_scaled": "TS, tenure and kick scaled with n",
}
FAMILY_ORDER = ["be75", "stabu", "t", "tiw56"]
FAMILY_LABEL = {"be75": "SGB", "stabu": "Stabu", "t": "IO family (t…)", "tiw56": "TIW", "rand": "Random",
                "xlolib": "xLOLIB", "mb": "MB"}


# --------------------------------------------------------------------------- IO helpers
def load(exp: str) -> list[dict]:
    """Records of one experiment, from ``<exp>.jsonl`` or its committed ``<exp>.jsonl.xz``."""
    path = RAW / f"{exp}.jsonl"
    if path.exists():
        opener = path.open
    elif (xz := RAW / f"{exp}.jsonl.xz").exists():
        opener = lambda: lzma.open(xz, "rt")  # noqa: E731
    else:
        return []
    rows, seen = [], set()
    with opener() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by a killed job
            if r["id"] in seen:  # the E3 grid listed one configuration twice; its runs are identical
                continue
            seen.add(r["id"])
            rows.append(r)
    return rows


def gap(value: float, ref: float, lower: float = 0.0) -> float:
    return 100.0 * (ref - value) / (ref - lower)


def rgap(r: dict, value: float | None = None) -> float:
    """Normalised gap of a result record (or of ``value`` on that record's instance)."""
    return gap(r["value"] if value is None else value, r["reference"], r["lower_bound"])


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" if a == "l" else ":---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if 0 < abs(x) < 10 ** -digits:  # never round a small non-zero value to an apparent exact zero
        return f"{x:.1e}"
    return f"{x:.{digits}f}"


def instance_bootstrap(per_instance: np.ndarray, stat=np.mean, reps: int = 4000, seed: int = 0):
    """95% percentile interval of ``stat`` over instances, resampling instances (the statistical units)."""
    x = np.asarray(per_instance, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(reps, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down adjustment of a family of p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out


# Studentised range quantiles q_{0.05}(k, inf) / sqrt(2) for the Nemenyi test (Demsar, 2006).
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def rank_tests(scores: dict[str, dict[str, float]], algos: list[str], lower_is_better: bool = True) -> dict:
    """Friedman test, mean ranks, Nemenyi CD and Holm-corrected Wilcoxon over instances.

    ``scores[algo][instance]`` is one number per (algorithm, instance) pair.
    """
    insts = sorted(set.intersection(*(set(scores[a]) for a in algos)))
    x = np.array([[scores[a][i] for a in algos] for i in insts], dtype=float)
    if not lower_is_better:
        x = -x
    ranks = np.array([stats.rankdata(row) for row in x])
    mean_ranks = dict(zip(algos, ranks.mean(axis=0).tolist()))
    k, n = len(algos), len(insts)
    try:
        fr = stats.friedmanchisquare(*x.T)
        friedman = {"statistic": float(fr.statistic), "p": float(fr.pvalue)}
    except ValueError:  # all identical
        friedman = {"statistic": 0.0, "p": 1.0}
    cd = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6.0 * n))
    raw = {}
    for a, b in combinations(algos, 2):
        d = x[:, algos.index(a)] - x[:, algos.index(b)]
        if np.all(d == 0):
            raw[(a, b)] = 1.0
        else:
            raw[(a, b)] = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    adj = holm(raw)
    pairs = []
    for (a, b), p in raw.items():
        d = x[:, algos.index(a)] - x[:, algos.index(b)]
        pairs.append({"a": a, "b": b, "p": p, "p_holm": adj[(a, b)],
                      "a_better": int((d < 0).sum()), "b_better": int((d > 0).sum()), "ties": int((d == 0).sum())})
    return {"n_instances": n, "mean_ranks": mean_ranks, "friedman": friedman, "cd": cd, "pairs": pairs}


# --------------------------------------------------------------------------- E0
def analyse_e0(rows):
    """Throughput (gain evaluations per CPU second) alone versus with every core busy."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["instance"]][r["concurrency"]].append(r["evaluations"] / r["elapsed"])
    levels = sorted({r["concurrency"] for r in rows})
    out = {i: {c: float(np.mean(v)) for c, v in d.items()} for i, d in per.items()}
    table = md_table(["instance", *[f"{c} concurrent run(s): M evaluations / CPU s" for c in levels], "ratio"],
                     [[i, *[fmt(out[i][c] / 1e6, 3) for c in levels], fmt(out[i][levels[-1]] / out[i][levels[0]], 3)]
                      for i in sorted(out)])
    return out, table


# --------------------------------------------------------------------------- E1
def analyse_e1(rows):
    per = defaultdict(lambda: defaultdict(list))  # method -> instance -> [(gap0, gap1, t)]
    for r in rows:
        key = r["method"] if r["method"] != "grasp" else f"grasp α={r['alpha']:g}"
        per[key][r["instance"]].append((rgap(r, r["value_construct"]), rgap(r, r["value_ls"]),
                                        r["construct_time"] + r["ls_time"], r["value_ls"] == r["optimum"],
                                        r["ls_moves"]))
    out = {}
    for method, by_inst in per.items():
        arr = {i: np.array(v, dtype=float) for i, v in by_inst.items()}
        out[method] = {
            "gap_construct": float(np.mean([a[:, 0].mean() for a in arr.values()])),
            "gap_ls": float(np.mean([a[:, 1].mean() for a in arr.values()])),
            "p_opt_ls": float(np.mean([a[:, 3].mean() for a in arr.values()])),
            "ls_moves": float(np.mean([a[:, 4].mean() for a in arr.values()])),
            "time_ms": 1e3 * float(np.mean([a[:, 2].mean() for a in arr.values()])),
            "per_instance": {i: [float(a[:, 0].mean()), float(a[:, 1].mean())] for i, a in arr.items()},
            "samples": {i: a[:, :2].tolist() for i, a in arr.items()},
        }
    order = ["random", "becker"] + [f"grasp α={a:g}" for a in C.E1_ALPHAS]
    table = md_table(
        ["Construction", "gap after construction (%)", "gap after insert-LS (%)", "P(LS hits optimum)",
         "LS moves", "time (ms)"],
        [[m, fmt(out[m]["gap_construct"]), fmt(out[m]["gap_ls"]), fmt(out[m]["p_opt_ls"], 3),
          fmt(out[m]["ls_moves"], 1), fmt(out[m]["time_ms"], 2)] for m in order if m in out])
    return {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in out.items()}, table, out


# --------------------------------------------------------------------------- E2
def analyse_e2(rows):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[(r["neighbourhood"], r["pivot"])][r["instance"]].append(
            (rgap(r), r["moves"], r["evaluations"], r["cpu"], r["value"] == r["optimum"]))
    out = {}
    for key, by_inst in per.items():
        arr = [np.array(v, dtype=float) for v in by_inst.values()]
        out[f"{key[0]}/{key[1]}"] = {
            "gap": float(np.mean([a[:, 0].mean() for a in arr])),
            "gap_median": float(np.median(np.concatenate([a[:, 0] for a in arr]))),
            "moves": float(np.mean([a[:, 1].mean() for a in arr])),
            "evaluations": float(np.mean([a[:, 2].mean() for a in arr])),
            "cpu_ms": 1e3 * float(np.mean([a[:, 3].mean() for a in arr])),
            "p_opt": float(np.mean([a[:, 4].mean() for a in arr])),
            "per_instance_gap": {i: float(np.mean(np.array(v)[:, 0])) for i, v in by_inst.items()},
        }
    order = ["swap/first", "swap/best", "insert/first", "insert/element", "insert/best"]
    table = md_table(
        ["Neighbourhood / pivot", "mean gap (%)", "median gap (%)", "P(optimum)", "moves", "gain evaluations",
         "CPU (ms)"],
        [[k, fmt(out[k]["gap"]), fmt(out[k]["gap_median"]), fmt(out[k]["p_opt"], 4), fmt(out[k]["moves"], 1),
          f"{out[k]['evaluations']:.3g}", fmt(out[k]["cpu_ms"], 2)] for k in order if k in out])
    return out, table


# --------------------------------------------------------------------------- E3/E4 runtime statistics
def run_stats(rows, budget_key="budget"):
    """Per (config, instance): success rate, mean final gap, PAR-10, TTTs."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["config"]][r["instance"]].append(r)
    out = {}
    for cfg, by_inst in per.items():
        inst_stats = {}
        for inst, runs in by_inst.items():
            budget = runs[0][budget_key]
            ttt = [r["time_to_target"] for r in runs]
            par10 = [t if t is not None else 10 * budget for t in ttt]
            inst_stats[inst] = {
                "success": float(np.mean([t is not None for t in ttt])),
                "gap": float(np.mean([rgap(r) for r in runs])),
                "par10": float(np.mean(par10)),
                "ttt": [t for t in ttt if t is not None],
                "runs": len(runs),
                "family": runs[0]["family"],
            }
        out[cfg] = inst_stats
    return out


def summarise_configs(st, instances=None):
    res = {}
    for cfg, by_inst in st.items():
        vals = [v for i, v in by_inst.items() if instances is None or i in instances]
        if not vals:
            continue
        all_ttt = np.concatenate([np.array(v["ttt"]) for v in vals]) if vals else np.array([])
        res[cfg] = {
            "success": float(np.mean([v["success"] for v in vals])),
            "gap": float(np.mean([v["gap"] for v in vals])),
            "par10": float(np.mean([v["par10"] for v in vals])),
            "median_ttt": float(np.median(all_ttt)) if all_ttt.size else float("nan"),
            "instances_all_solved": int(sum(v["success"] == 1.0 for v in vals)),
            "n_instances": len(vals),
        }
    return res


def analyse_e3(rows):
    st = run_stats(rows)
    summ = summarise_configs(st)
    by_algo = defaultdict(list)
    for cfg, s in summ.items():
        by_algo[cfg.split(":")[0]].append((cfg, s))
    tables, best = [], {}
    for algo in ("grasp", "ils", "tabu"):
        items = sorted(by_algo[algo], key=lambda kv: (-kv[1]["success"], kv[1]["par10"]))
        best[algo] = items[0][0] if items else None
        tables.append(f"**{algo.upper()}** (sorted by success rate, then PAR-10; top {min(12, len(items))} of {len(items)})\n\n" + md_table(
            ["configuration", "success", "mean gap (%)", "PAR-10 (s)", "median TTT (ms)"],
            [[cfg.split(":", 1)[1], fmt(s["success"], 3), fmt(s["gap"], 4), fmt(s["par10"], 3),
              fmt(1e3 * s["median_ttt"], 1)] for cfg, s in items[:12]]))
    return {"configs": summ, "best": best, "per_instance": st}, "\n\n".join(tables)


def anytime(rows, grid):
    """Mean gap of the best-so-far value at each time in ``grid``, per config.

    Instances are weighted equally: runs are averaged within an instance first.
    Before a run's first trace point, its gap is undefined and it is excluded.
    """
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tr = np.array([t[:2] for t in r["trace"]], dtype=float)
        g = np.full(len(grid), np.nan)
        if tr.size:
            # a run that stopped at the target keeps its final (zero) gap for the rest of the grid
            idx = np.searchsorted(tr[:, 0], grid, side="right") - 1
            ok = idx >= 0
            g[ok] = gap(tr[idx[ok], 1], r["reference"], r["lower_bound"])
        per[r["config"]][r["instance"]].append(g)
    out = {}
    for cfg, by_inst in per.items():
        inst_curves = np.array([np.nanmean(np.array(v), axis=0) for v in by_inst.values()])
        out[cfg] = {"mean": np.nanmean(inst_curves, axis=0).tolist(),
                    "q25": np.nanpercentile(inst_curves, 25, axis=0).tolist(),
                    "q75": np.nanpercentile(inst_curves, 75, axis=0).tolist()}
    return out


def comparison_tables(rows, instances, budget):
    """Runtime statistics and paired tests of the configurations on ``instances``.

    Success rates carry a 95% interval from resampling *instances* (runs of one instance
    are not independent evidence about the next instance). ERT is the total CPU time spent
    per success (failed runs count their full budget) and is infinite on an instance never
    solved. The primary pairwise procedure is the Wilcoxon signed-rank test on per-instance
    log PAR-10 with Holm's correction over the six pairs of the four main methods. The
    Nemenyi critical difference is reported alongside and may disagree.
    """
    sub = [r for r in rows if r["instance"] in instances]
    st = run_stats(sub)
    summ = summarise_configs(st)
    configs = [c for c in ALGO_ORDER if c in st]
    extra = {}
    for c in configs:
        succ = np.array([st[c][i]["success"] for i in instances if i in st[c]])
        ert = []
        for i in instances:
            runs = [r for r in sub if r["config"] == c and r["instance"] == i]
            if not runs:
                continue
            hits = sum(r["time_to_target"] is not None for r in runs)
            spent = sum(r["time_to_target"] if r["time_to_target"] is not None else min(r["elapsed"], budget)
                        for r in runs)
            ert.append(spent / hits if hits else math.inf)
        extra[c] = {"ci": instance_bootstrap(succ), "ert_median": float(np.median(ert)),
                    "never_solved": int(sum(math.isinf(e) for e in ert))}
    main = [c for c in MAIN if c in st]
    par10 = {a: {i: v["par10"] for i, v in st[a].items()} for a in configs}
    tests = rank_tests({a: {i: math.log10(v) for i, v in par10[a].items()} for a in main}, main)
    header = ["Algorithm", "success rate [95% CI over instances]", "instances solved in every run",
              "instances never solved", "mean gap (%)", "PAR-10 (s)", "median ERT (ms)", "median TTT of successes (ms)"]
    table = md_table(header, [[
        ALGO_LABEL[a], f"{fmt(summ[a]['success'], 3)} [{fmt(extra[a]['ci'][0], 3)}, {fmt(extra[a]['ci'][1], 3)}]",
        f"{summ[a]['instances_all_solved']}/{summ[a]['n_instances']}", extra[a]["never_solved"],
        fmt(summ[a]["gap"], 4), fmt(summ[a]["par10"], 3), fmt(1e3 * extra[a]["ert_median"], 1),
        fmt(1e3 * summ[a]["median_ttt"], 1)] for a in configs])
    pairs = md_table(["pair", "instances: first faster / slower / tied (PAR-10)", "Wilcoxon p (log PAR-10)",
                      "Holm-adjusted p", "geometric-mean PAR-10 ratio [95% CI]", "Nemenyi: rank gap vs CD"],
                     [[f"{ALGO_LABEL[p['a']]} vs {ALGO_LABEL[p['b']]}", f"{p['a_better']} / {p['b_better']} / {p['ties']}",
                       f"{p['p']:.2g}", f"{p['p_holm']:.2g}", _ratio_ci(par10[p['a']], par10[p['b']]),
                       f"{abs(tests['mean_ranks'][p['a']] - tests['mean_ranks'][p['b']]):.2f} vs {tests['cd']:.2f}"]
                      for p in tests["pairs"]], "llrrrr")
    ranks = md_table(["Algorithm", "mean rank (log PAR-10)"],
                     [[ALGO_LABEL[a], fmt(tests["mean_ranks"][a], 2)] for a in main])
    ranks += (f"\n\nFriedman chi2 = {tests['friedman']['statistic']:.1f}, p = {tests['friedman']['p']:.2g}, "
              f"{tests['n_instances']} instances; Nemenyi critical difference (alpha = 0.05) = {tests['cd']:.2f}.")
    return {"summary": summ, "extra": extra, "tests": tests, "per_instance": st}, table, pairs, ranks


def _ratio_ci(a: dict, b: dict) -> str:
    """Geometric mean over instances of a/b with an instance-bootstrap 95% interval."""
    common = sorted(set(a) & set(b))
    logr = np.log10([a[i] / b[i] for i in common])
    lo, hi = instance_bootstrap(logr)
    return f"{10 ** logr.mean():.2f} [{10 ** lo:.2f}, {10 ** hi:.2f}]"


def paired_variant_tests(rows, instances, pairs, metric):
    """Holm-corrected paired Wilcoxon tests for variant-vs-base comparisons (one family per set).

    ``metric`` is ``"par10"`` (log PAR-10; proven-optimum sets) or ``"gap"`` (mean final gap; xLOLIB).
    """
    st = run_stats([r for r in rows if r["instance"] in instances])
    raw, info = {}, {}
    for a, b in pairs:
        if a not in st or b not in st:
            continue
        common = sorted(set(st[a]) & set(st[b]))
        xa = np.array([st[a][i][metric] for i in common])
        xb = np.array([st[b][i][metric] for i in common])
        if metric == "par10":
            xa, xb = np.log10(xa), np.log10(xb)
        d = xa - xb
        raw[(a, b)] = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
        info[(a, b)] = (int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum()),
                        float(np.mean([st[a][i]["success"] for i in common])),
                        float(np.mean([st[b][i]["success"] for i in common])),
                        float(np.mean([st[a][i]["gap"] for i in common])),
                        float(np.mean([st[b][i]["gap"] for i in common])))
    adj = holm(raw)
    return [{"a": a, "b": b, "p": raw[(a, b)], "p_holm": adj[(a, b)], "better": info[(a, b)][0],
             "worse": info[(a, b)][1], "ties": info[(a, b)][2], "success": info[(a, b)][3:5],
             "gap": info[(a, b)][5:7]} for (a, b) in raw]


VARIANT_PAIRS = [("ils", "ils_random"), ("tabu", "tabu_random"), ("tabu", "tabu_nofreq"), ("tabu", "tabu_short"),
                 ("ils", "ils_scaled"), ("tabu", "tabu_scaled")]


def variant_table(sets: dict) -> str:
    """One row per (set, base vs variant) pair; ``sets`` maps a label to (rows, instances, metric)."""
    out = []
    for label, (rows, insts, metric) in sets.items():
        for t in paired_variant_tests(rows, insts, VARIANT_PAIRS, metric):
            out.append([label, f"{ALGO_LABEL[t['a']]} vs {ALGO_LABEL[t['b']]}",
                        f"{fmt(t['success'][0], 3)} vs {fmt(t['success'][1], 3)}",
                        f"{fmt(t['gap'][0], 4)} vs {fmt(t['gap'][1], 4)}",
                        f"{t['better']} / {t['worse']} / {t['ties']}", f"{t['p_holm']:.2g}"])
    return md_table(["set", "comparison", "success", "mean gap (%)",
                     "instances: base better / worse / tied" , "Holm-adjusted Wilcoxon p"], out, "llrrrr")


def ts_diagnostics_table(sets: dict) -> str:
    """Median trajectory statistics of the Tabu Search runs: restarts, walk length, cycling."""
    out = []
    for label, rows in sets.items():
        for cfg in ("tabu", "tabu_nofreq", "tabu_short", "tabu_scaled"):
            runs = [r for r in rows if r["config"] == cfg]
            if not runs:
                continue
            failed = [r for r in runs if r["time_to_target"] is None]
            its = np.array([r["iterations"] for r in runs], dtype=float)
            rest = np.array([r["diagnostics"]["restarts"] for r in runs], dtype=float)
            out.append([label, ALGO_LABEL[cfg], f"{np.median(its):,.0f}", f"{np.median(rest):,.0f}",
                        f"{np.median(its / np.maximum(rest, 1)):,.0f}",
                        fmt(float(np.median([r["diagnostics"]["worsening_moves"] for r in runs])), 2),
                        fmt(float(np.median([r["diagnostics"]["distinct_solutions"] for r in failed])), 2) if failed else "–"])
    return md_table(["set", "configuration", "iterations per run", "restarts per run",
                     "iterations per restart phase", "share of worsening moves",
                     "distinct solutions / iterations (runs that failed or ran out of budget)"], out, "llrrrrr")


def analyse_e4(rows):
    all_inst = sorted({r["instance"] for r in rows})
    held = [i for i in all_inst if i not in C.DEVELOPMENT_INSTANCES]
    res_h, t_h, p_h, r_h = comparison_tables(rows, held, C.E4_BUDGET)
    res_a, t_a, p_a, r_a = comparison_tables(rows, all_inst, C.E4_BUDGET)
    configs = [c for c in ALGO_ORDER if c in res_a["per_instance"]]
    fam_rows = []
    for f in FAMILY_ORDER:
        insts = [i for i in held if rows and next(r["family"] for r in rows if r["instance"] == i) == f]
        if not insts:
            continue
        s = summarise_configs(run_stats([r for r in rows if r["instance"] in insts]))
        fam_rows.append([f"{FAMILY_LABEL[f]} ({len(insts)})",
                         *[f"{fmt(s[a]['success'], 2)} · {fmt(s[a]['par10'], 2)} s" for a in configs]])
    t_fam = md_table(["Family (held-out)", *[ALGO_LABEL[a] for a in configs]], fam_rows)
    hardest = sorted(held, key=lambda i: -np.mean([res_h["per_instance"][a][i]["par10"] for a in MAIN]))[:8]
    t_hard = md_table(["instance", *[ALGO_LABEL[a] for a in configs]],
                      [[i, *[f"{fmt(res_h['per_instance'][a][i]['success'], 2)} · {fmt(res_h['per_instance'][a][i]['par10'], 2)} s"
                             for a in configs]] for i in hardest])
    grid = np.logspace(-3, math.log10(C.E4_BUDGET), 60)
    return ({"held_out": {k: v for k, v in res_h.items() if k != "per_instance"},
             "all": {k: v for k, v in res_a.items() if k != "per_instance"},
             "anytime_grid": grid.tolist(), "anytime_held_out": anytime([r for r in rows if r["instance"] in held], grid)},
            {"held_out": t_h, "held_out_pairs": p_h, "held_out_ranks": r_h, "all": t_a, "all_pairs": p_a,
             "all_ranks": r_a, "family": t_fam, "hardest": t_hard})


# --------------------------------------------------------------------------- E5
def analyse_e5(rows_by_exp: dict):
    """xLOLIB (gap to the literature best-known value) and MB (proven optima, runtime statistics)."""
    out, tables = {}, {}
    for exp, rows in rows_by_exp.items():
        if not rows:
            continue
        spec = C.E5_SETS[exp]
        configs = [c for c in ALGO_ORDER if any(r["config"] == c for r in rows)]
        per = defaultdict(lambda: defaultdict(list))
        for r in rows:
            per[r["config"]][r["instance"]].append(r)
        insts = sorted({r["instance"] for r in rows})
        gaps = {c: {i: float(np.mean([rgap(r) for r in per[c][i]])) for i in insts if per[c][i]} for c in configs}
        best = {c: {i: float(np.mean([r["value"] >= r["reference"] for r in per[c][i]])) for i in insts if per[c][i]}
                for c in configs}
        main = [c for c in MAIN if c in configs]
        tests = rank_tests({c: gaps[c] for c in main}, main)
        entry = {"gaps": gaps, "tests": tests, "instances": len(insts)}
        if spec["use_target"]:  # MB: proven optima, same runtime statistics as E4
            res, t, p, rk = comparison_tables(rows, insts, spec["budget"])
            entry["runtime"] = {k: v for k, v in res.items() if k != "per_instance"}
            tables[exp] = t
            tables[exp + "_pairs"] = p
            tables[exp + "_ranks"] = rk
            by_n = defaultdict(list)
            for i in insts:
                by_n[next(r["n"] for r in rows if r["instance"] == i)].append(i)
            tables[exp + "_by_size"] = md_table(
                ["n (instances)", *[ALGO_LABEL[c] for c in configs]],
                [[f"{n} ({len(v)})", *[fmt(np.mean([res['per_instance'][c][i]['success'] for i in v]), 2) for c in configs]]
                 for n, v in sorted(by_n.items())])
        else:  # xLOLIB: fixed budget, quality only
            tables[exp] = md_table(
                ["Algorithm", "mean gap to best-known (%)", "median gap (%)", "instances where it matched or beat best-known",
                 "mean rank (gap)"],
                [[ALGO_LABEL[c], fmt(np.mean(list(gaps[c].values())), 4), fmt(np.median(list(gaps[c].values())), 4),
                  f"{sum(v > 0 for v in best[c].values())}/{len(best[c])}",
                  fmt(tests["mean_ranks"][c], 2) if c in tests["mean_ranks"] else "–"] for c in configs])
            tables[exp] += (f"\n\nFriedman over {tests['n_instances']} instances (four main methods): "
                            f"p = {tests['friedman']['p']:.2g}; Nemenyi CD = {tests['cd']:.2f}.")
            tables[exp + "_pairs"] = md_table(
                ["pair", "instances: first better / worse / tied (mean gap)", "Wilcoxon p", "Holm-adjusted p"],
                [[f"{ALGO_LABEL[p['a']]} vs {ALGO_LABEL[p['b']]}", f"{p['a_better']} / {p['b_better']} / {p['ties']}",
                  f"{p['p']:.2g}", f"{p['p_holm']:.2g}"] for p in tests["pairs"]], "llrr")
        grid = np.logspace(-2, math.log10(spec["budget"]), 60)
        entry["anytime_grid"] = grid.tolist()
        entry["anytime"] = anytime(rows, grid)
        out[exp] = entry
    return out, tables


# --------------------------------------------------------------------------- E6
def analyse_e6(rows):
    from lop.objective import kendall_distance

    per = defaultdict(lambda: {"values": [], "dist": [], "zero": [], "perms": []})
    meta = {}
    for r in rows:
        d = per[r["instance"]]
        d["values"] += r["values"]
        d["dist"] += r["dist_to_ref"]
        d["zero"] += r["zero_move_fraction"]
        d["perms"] += r["perms"]
        meta[r["instance"]] = r
    out = {}
    for inst, d in per.items():
        m = meta[inst]
        opt, n = m["optimum"], m["n"]
        v = np.array(d["values"], dtype=float)
        perms = np.array(d["perms"])
        # Many instances have several optimal orderings. Measure distance to the NEAREST
        # known optimal ordering: the official one plus every optimal local optimum sampled.
        optimal = [p for p, val in zip(perms, v) if val == opt]
        uniq_opt = {p.tobytes(): p for p in optimal}
        ref = np.array(json.loads((ROOT / "instances" / "optimal_orderings.json").read_text())[inst])
        uniq_opt.setdefault(ref.tobytes(), ref)
        refs = list(uniq_opt.values())[:50]
        dist_near = np.array([min(kendall_distance(p, q) for q in refs) for p in perms], dtype=float)
        pairs = n * (n - 1) / 2
        gaps = 100.0 * (opt - v) / opt
        fdc = float(stats.pearsonr(gaps, dist_near)[0]) if np.std(dist_near) > 0 and np.std(gaps) > 0 else float("nan")
        fdc_ref = float(stats.pearsonr(gaps, d["dist"])[0]) if np.std(gaps) > 0 else float("nan")
        sub = gaps > 0  # zero-gap points (distance 0 to themselves) would inflate the correlation
        fdc_nonopt = (float(stats.pearsonr(gaps[sub], dist_near[sub])[0])
                      if sub.sum() > 2 and np.std(dist_near[sub]) > 0 else float("nan"))
        out[inst] = {
            "family": m["family"], "n": n,
            "p_opt": float(np.mean(v == opt)),
            "distinct_local_optima": len({p.tobytes() for p in perms}),
            "samples": len(v),
            "distinct_optima_found": len(uniq_opt),
            "mean_gap": float(gaps.mean()),
            "mean_dist_nearest": float(dist_near.mean() / pairs),
            "fdc_nearest": fdc,
            "fdc_reference": fdc_ref,
            "fdc_non_optimal": fdc_nonopt,
            "zero_move_fraction": float(np.mean(d["zero"])),
            "ref_zero_move_fraction": m["ref_zero_move_fraction"],
            "scatter": {"gap": gaps.tolist(), "dist": (dist_near / pairs).tolist()},
        }
    fams = defaultdict(list)
    for inst, o in out.items():
        fams[o["family"]].append(o)
    table = md_table(
        ["Family", "instances", "P(LS from random start = optimum)", "distinct local optima / 1000",
         "mean normalised distance to nearest optimum", "FDC", "FDC, non-optimal local optima only",
         "zero-gain insertion moves at local optima"],
        [[FAMILY_LABEL[f], len(fams[f]), fmt(np.mean([o["p_opt"] for o in fams[f]]), 3),
          fmt(np.mean([o["distinct_local_optima"] for o in fams[f]]), 0),
          fmt(np.mean([o["mean_dist_nearest"] for o in fams[f]]), 3),
          fmt(np.nanmean([o["fdc_nearest"] for o in fams[f]]), 2),
          fmt(np.nanmean([o["fdc_non_optimal"] for o in fams[f]]), 2),
          f"{100 * np.mean([o['zero_move_fraction'] for o in fams[f]]):.1f}%"] for f in FAMILY_ORDER if f in fams])
    return out, table


# --------------------------------------------------------------------------- verification
def verify_final_values(exps=("E3", "E3b", "E4", *C.E5_SETS)) -> dict:
    """Recompute f(perm) for every stored final permutation; any mismatch aborts the analysis."""
    from experiments.run import instance
    from lop.objective import objective

    checked = {}
    for exp in exps:
        n = 0
        for r in load(exp):
            if "perm" not in r:
                continue
            perm = np.array(r["perm"])
            m = instance(r["instance"]).matrix
            if sorted(perm.tolist()) != list(range(len(m))) or objective(m, perm) != r["value"]:
                raise AssertionError(f"{exp}: stored value of {r['id']} does not match its permutation")
            n += 1
        checked[exp] = n
    return checked


# --------------------------------------------------------------------------- driver
def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = {"verified_final_values": verify_final_values()}
    print("re-verified final values:", summary["verified_final_values"])
    if rows := load("E0"):
        s, t = analyse_e0(rows)
        summary["E0"] = s
        (TABLES / "E0_calibration.md").write_text(t + "\n")
    if rows := load("E1"):
        s, t, _ = analyse_e1(rows)
        summary["E1"] = s
        (TABLES / "E1_constructive.md").write_text(t + "\n")
    if rows := load("E2"):
        s, t = analyse_e2(rows)
        summary["E2"] = s
        (TABLES / "E2_local_search.md").write_text(t + "\n")
    if rows := load("E3"):
        s, t = analyse_e3(rows)
        summary["E3"] = {"configs": s["configs"], "best": s["best"]}
        (TABLES / "E3_sensitivity.md").write_text(t + "\n")
    if rows := load("E4"):
        s, t = analyse_e4(rows)
        summary["E4"] = s
        for name, tab in t.items():
            (TABLES / f"E4_{name}.md").write_text(tab + "\n")
    e5 = {e: load(e) for e in C.E5_SETS}
    if any(e5.values()):
        s, t = analyse_e5(e5)
        summary["E5"] = {e: {k: v for k, v in d.items() if k != "gaps"} for e, d in s.items()}
        for name, tab in t.items():
            (TABLES / f"{name}.md").write_text(tab + "\n")
    if rows := load("E6"):
        s, t = analyse_e6(rows)
        summary["E6"] = {i: {k: v for k, v in o.items() if k != "scatter"} for i, o in s.items()}
        (TABLES / "E6_landscape.md").write_text(t + "\n")
    # attribution of the variants and Tabu Search trajectory statistics, across every set
    e4 = load("E4")
    held = sorted({r["instance"] for r in e4} - set(C.DEVELOPMENT_INSTANCES))
    sets = {"LOLIB held-out (35)": (e4, held, "par10")}
    for e, label in (("E5mb", "MB (30)"), ("E5x150", "xLOLIB 150 (39)"), ("E5x250", "xLOLIB 250 (39)")):
        rows = load(e)
        if rows:
            sets[label] = (rows, sorted({r["instance"] for r in rows}), "par10" if C.E5_SETS[e]["use_target"] else "gap")
    if e4:
        (TABLES / "variants.md").write_text(variant_table(sets) + "\n")
        (TABLES / "ts_diagnostics.md").write_text(
            ts_diagnostics_table({k: [r for r in v[0] if r["instance"] in v[1]] for k, v in sets.items()}) + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for p in sorted(TABLES.glob("*.md")):
        print(f"\n## {p.stem}\n{p.read_text()}")


if __name__ == "__main__":
    main()
