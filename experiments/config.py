"""Experimental protocol: instance splits, budgets and algorithm configurations.

Every number reported in the README comes from a configuration defined here.
"""

# Development instances. Every instance that influenced a design decision (the
# pilot runs that exposed the Tabu Search stagnation, plus the tuning subset of
# E3) is excluded from the primary, held-out evaluation. The 10 tuning instances
# span the four LOLIB families and the range of observed difficulty.
PILOT_INSTANCES = ["be75eec", "stabu1", "t59b11xx", "tiw56n54"]
TUNING_INSTANCES = [
    "be75np", "be75oi",
    "stabu2",
    "t59f11xx", "t65w11xx", "t69r11xx", "t70x11xx", "t75e11xx",
    "tiw56n62", "tiw56r66",
]
DEVELOPMENT_INSTANCES = sorted(set(PILOT_INSTANCES) | set(TUNING_INSTANCES))

# E0 - timing calibration: does running 40 single-threaded jobs side by side slow each one down?
E0_INSTANCES = ["be75np", "stabu2", "t65w11xx", "tiw56r66"]
E0_BUDGET = 2.0

# E1 - constructive heuristics (each construction followed by the standard insert local search)
E1_SAMPLES = 30
E1_ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

# E2 - local search: all variants start from the same 30 random permutations per instance
E2_STARTS = 30
E2_VARIANTS = [("swap", "first"), ("swap", "best"), ("insert", "first"), ("insert", "element"), ("insert", "best")]

# E3 - parameter sensitivity, on the tuning instances only
E3_RUNS = 20
E3_BUDGET = 1.0  # CPU seconds, stop at the proven optimum
_TS_FULL = {"diversification": 30.0, "restart_after": 100, "restart_strength": 10}
E3_GRID = {
    "grasp": [{"alpha": a} for a in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]],
    "ils": [
        {"strength": k, "acceptance": acc}
        for k in [1, 2, 3, 5, 8, 12, 20, 30]
        for acc in ["better", "better_equal", "always"]
    ]
    + [{"strength": 5, "acceptance": "better_equal", "start": "random"},
       {"strength": 5, "acceptance": "better_equal", "pivot": "first"},
       {"strength": 5, "acceptance": "better_equal", "pivot": "best"}],
    "tabu": [  # tenure x attribute, full method
        {"tenure": t, "attribute": att, **_TS_FULL}
        for t in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        for att in ["element", "position", "pair"]
    ]
    + [  # component ablation: null moves x frequency memory x restarts
        {"tenure": 7, "attribute": "pair", "null_moves": nm, "diversification": d, "restart_after": r,
         "restart_strength": 10}
        for nm in [True, False]
        for d in [0.0, 30.0]
        for r in [None, 100]
    ]
    + [  # long-term memory strength
        {"tenure": 7, "attribute": "pair", "diversification": d, "restart_after": r, "restart_strength": 10}
        for d in [0.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        for r in [25, 50, 100, 200, 400, 800]
    ]
    + [  # restart kick strength
        {"tenure": 7, "attribute": "pair", "diversification": 30.0, "restart_after": 100, "restart_strength": s}
        for s in [2, 5, 20, 40]
    ],
}

# E3b - development-only follow-up requested by the reviewers: R = 25 was at the edge of the E3 grid,
# and tenure / restart kick had only been swept at R = 100.
E3B_GRID = {
    "tabu": [
        {"tenure": 7, "attribute": "pair", "diversification": d, "restart_after": r, "restart_strength": 10}
        for d in [0.0, 30.0]
        for r in [5, 10, 25, 50]
    ]
    + [{"tenure": 7, "attribute": "pair", "diversification": 30.0, "restart_after": 25, "restart_strength": s}
       for s in [20, 40]]
    + [{"tenure": t, "attribute": "pair", "diversification": 30.0, "restart_after": 25, "restart_strength": 10}
       for t in [3, 15]]
    + [{"tenure": 20, "attribute": "element", "diversification": 30.0, "restart_after": 25, "restart_strength": 10}],
}

# E4 - main comparison on the 49 LOLIB IO instances (primary: the 35 held-out ones).
# Frozen from E3/E3b by taking a robust region, not the single best cell (see README, "Tuning"):
#   GRASP alpha = 0.3 (0.2-0.5 plateau); ILS k = 12, accept if not worse (k = 8-30 plateau);
#   TS element attribute, tenure 20 (15-20 plateau), frequency weight 30, restart after R = 25
#   (interior optimum of 5-800), restart kick 10 (10-20 plateau; 40 collapses).
E4_RUNS = 30
E4_BUDGET = 10.0
_ILS = {"strength": 12, "acceptance": "better_equal"}
_TS_SHORT = {"tenure": 20, "attribute": "element"}
_TS = {**_TS_SHORT, "diversification": 30.0, "restart_after": 25, "restart_strength": 10}
E4_CONFIGS = {
    "rrls": ("grasp", {"alpha": 1.0}),  # random-restart local search: the control
    "grasp": ("grasp", {"alpha": 0.3}),
    "ils": ("ils", _ILS),
    "ils_random": ("ils", {**_ILS, "start": "random"}),  # same start distribution as GRASP/RRLS
    "tabu": ("tabu", _TS),
    "tabu_random": ("tabu", {**_TS, "start": "random"}),
    "tabu_nofreq": ("tabu", {**_TS, "diversification": 0.0}),  # attribution: restarts without frequency memory
    "tabu_short": ("tabu", _TS_SHORT),  # short-term memory only: the classical core of TS
}
# E5 adds variants whose move-count parameters grow with n, calibrated so they equal the frozen
# values at n = 48, the mean size of the tuning instances (k = 12 = 0.25 n, tenure 20 ~ 0.42 n, kick 10 ~ 0.21 n).
E5_CONFIGS = {
    **E4_CONFIGS,
    "ils_scaled": ("ils", {**_ILS, "strength": "0.25n"}),
    "tabu_scaled": ("tabu", {**_TS, "tenure": "0.42n", "restart_strength": "0.21n"}),
}

# E5 - harder instances, parameters frozen from E3 (a genuinely held-out test of the tuning):
# xLOLIB (IO-like, n = 150 and 250; open instances, gap to the literature best-known value, no
# target) and MB (Mitchell-Borchers, n = 100-250, proven optima, stop at the optimum).
E5_SETS = {
    "E5x150": {"set": "xlolib", "n": 150, "budget": 10.0, "runs": 5, "use_target": False},
    "E5x250": {"set": "xlolib", "n": 250, "budget": 30.0, "runs": 5, "use_target": False},
    "E5mb": {"set": "mb", "n": None, "budget": 20.0, "runs": 10, "use_target": True},
}

# E6 - landscape analysis
E6_LOCAL_OPTIMA = 1000
E6_HARD_LOCAL_OPTIMA = 50  # per MB and xLOLIB-150 instance
