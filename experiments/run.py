"""Run one experiment of the study in parallel and append one JSON line per task.

    python -m experiments.run E4 --workers 96
    python -m experiments.run all --workers 96

Runs are resumable: a task whose id is already in ``results/raw/<EXP>.jsonl`` is
skipped. Each task gets a seed derived from its id, so a rerun reproduces it
exactly apart from timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

# One thread per run: parallelism comes from running tasks side by side, and CPU-time budgets
# are only comparable if no run gets extra BLAS threads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from experiments import config as C  # noqa: E402
from lop import (  # noqa: E402
    ALGORITHMS, Budget, becker, grasp_construct, insert_gains, kendall_distance,
    load_instances, local_search, objective, read_instance,
)
from lop.generate import random_instance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("LOP_RESULTS", ROOT / "results")) / "raw"


def seed_of(task_id: str) -> int:
    return zlib.crc32(task_id.encode())


_INSTANCES: dict = {}


def instance(name: str):
    """All instances are loaded once per worker process."""
    if not _INSTANCES:
        _load_all()
    return _INSTANCES[name]


def _load_all() -> None:
    for inst in load_instances(ROOT / "instances", ROOT / "instances" / "optima.json"):
        _INSTANCES[inst.name] = inst
    _load_hard_sets()


_BEST_KNOWN = json.loads((ROOT / "instances" / "xlolib" / "best_known.json").read_text())


def reference(inst) -> int | None:
    """Proven optimum if known, otherwise the literature best-known value (xLOLIB)."""
    return inst.optimum if inst.optimum is not None else _BEST_KNOWN.get(inst.name)


def _load_hard_sets() -> None:
    """xLOLIB and MB instances (Optsicom normalised files, kept unchanged in instances/)."""
    mb_opt = json.loads((ROOT / "instances" / "mb" / "optima_normalised.json").read_text())
    for p in sorted((ROOT / "instances" / "xlolib").glob("N-*")):
        inst = read_instance(p)
        _INSTANCES[inst.name] = inst
    for p in sorted((ROOT / "instances" / "mb" / "optsicom_normalised").glob("N-*")):
        inst = read_instance(p)
        _INSTANCES[inst.name] = read_instance(p, mb_opt[inst.name])


def hard_names(which: str, n: int | None) -> list[str]:
    if not _INSTANCES:
        _load_all()
    return sorted(k for k, v in _INSTANCES.items() if v.family == which and (n is None or v.n == n))


def io_names() -> list[str]:
    return sorted(p.stem for p in (ROOT / "instances").glob("*.mat"))


def base_record(task: dict) -> dict:
    inst = instance(task["instance"])
    return {"id": task["id"], "exp": task["exp"], "instance": inst.name, "family": inst.family,
            "n": inst.n, "optimum": inst.optimum, "reference": reference(inst),
            "lower_bound": inst.symmetric_lower_bound}


# --------------------------------------------------------------------------- E1 constructive
def tasks_e1():
    for name in io_names():
        for s in range(C.E1_SAMPLES):
            yield {"exp": "E1", "instance": name, "method": "becker", "alpha": None, "sample": s}
            yield {"exp": "E1", "instance": name, "method": "random", "alpha": None, "sample": s}
            for a in C.E1_ALPHAS:
                yield {"exp": "E1", "instance": name, "method": "grasp", "alpha": a, "sample": s}


def run_e1(task):
    inst, rng = instance(task["instance"]), np.random.default_rng(task["seed"])
    t0 = time.process_time()
    if task["method"] == "becker":
        perm = becker(inst.matrix)
    elif task["method"] == "random":
        perm = rng.permutation(inst.n)
    else:
        perm = grasp_construct(inst.matrix, rng, task["alpha"])
    t1 = time.process_time()
    v0 = objective(inst.matrix, perm)
    _, v1, st = local_search(inst.matrix, perm, rng, "insert", "element", v0)
    t2 = time.process_time()
    return {**base_record(task), "method": task["method"], "alpha": task["alpha"], "sample": task["sample"],
            "value_construct": v0, "value_ls": v1, "construct_time": t1 - t0, "ls_time": t2 - t1,
            "ls_moves": st.moves}


# --------------------------------------------------------------------------- E2 local search
def tasks_e2():
    for name in io_names():
        for s in range(C.E2_STARTS):
            for nb, pivot in C.E2_VARIANTS:
                yield {"exp": "E2", "instance": name, "neighbourhood": nb, "pivot": pivot, "start": s}


def run_e2(task):
    inst = instance(task["instance"])
    # the start depends on the start index only, so all four local searches share it
    start = np.random.default_rng(seed_of(f"E2/{inst.name}/{task['start']}")).permutation(inst.n)
    rng = np.random.default_rng(task["seed"])
    v0 = objective(inst.matrix, start)
    t0 = time.process_time()
    perm, v, st = local_search(inst.matrix, start, rng, task["neighbourhood"], task["pivot"], v0)
    cpu = time.process_time() - t0
    return {**base_record(task), "neighbourhood": task["neighbourhood"], "pivot": task["pivot"],
            "start": task["start"], "start_value": v0, "value": v, "moves": st.moves,
            "evaluations": st.evaluations, "cpu": cpu}


# --------------------------------------------------------------------------- E3/E4/E5 metaheuristics
def _meta_record(task, res):
    return {**base_record(task), "config": task["config"], "algo": task["algo"], "params": task["params"],
            "run": task["run"], "budget": task["budget"], "value": res.value, "elapsed": res.elapsed,
            "iterations": res.iterations, "evaluations": res.evaluations, "time_to_target": res.time_to_target,
            "diagnostics": res.diagnostics, "perm": res.perm.tolist(),
            "trace": [[round(t, 5), v, e] for t, v, e in res.trace]}


def resolve_params(params: dict, n: int) -> dict:
    """Parameters written as ``"<c>n"`` scale with the instance size: ``round(c * n)``, at least 1."""
    return {k: max(1, round(float(v[:-1]) * n)) if isinstance(v, str) and v.endswith("n") else v
            for k, v in params.items()}


def run_meta(task):
    inst = instance(task["instance"])
    rng = np.random.default_rng(task["seed"])
    target = inst.optimum if task.get("use_target", True) else None
    params = resolve_params(task["params"], inst.n)
    res = ALGORITHMS[task["algo"]](inst.matrix, rng, Budget(task["budget"], target), **params)
    return _meta_record(task, res)


def tasks_e3(grid_spec=None, exp="E3"):
    for algo, grid in (grid_spec or C.E3_GRID).items():
        for params in grid:
            label = algo + ":" + ",".join(f"{k}={v}" for k, v in sorted(params.items()))
            for name in C.TUNING_INSTANCES:
                for r in range(C.E3_RUNS):
                    yield {"exp": exp, "instance": name, "config": label, "algo": algo, "params": params,
                           "run": r, "budget": C.E3_BUDGET}


def tasks_e4():
    for name in io_names():
        for label, (algo, params) in C.E4_CONFIGS.items():
            for r in range(C.E4_RUNS):
                yield {"exp": "E4", "instance": name, "config": label, "algo": algo, "params": params,
                       "run": r, "budget": C.E4_BUDGET}


def tasks_e5(exp: str):
    spec = C.E5_SETS[exp]
    for name in hard_names(spec["set"], spec["n"]):
        for label, (algo, params) in C.E5_CONFIGS.items():
            for r in range(spec["runs"]):
                yield {"exp": exp, "instance": name, "config": label, "algo": algo, "params": params,
                       "run": r, "budget": spec["budget"], "use_target": spec["use_target"]}


# --------------------------------------------------------------------------- E6 landscape
def tasks_e6():
    chunk = 100
    for name in io_names():
        for c in range(C.E6_LOCAL_OPTIMA // chunk):
            yield {"exp": "E6", "instance": name, "chunk": c, "size": chunk}


def run_e6(task):
    """Sample insert-local optima from uniform random starts and describe each one.

    For every local optimum: its value, Kendall distance to the official optimal
    ordering, and the plateau measure, i.e. the fraction of its (n-1)^2
    insertion moves whose gain is exactly zero. The permutations are kept so the
    analysis can also measure distance to the *nearest* optimal ordering found.
    """
    inst, rng = instance(task["instance"]), np.random.default_rng(task["seed"])
    orderings = json.loads((ROOT / "instances" / "optimal_orderings.json").read_text())
    ref = np.array(orderings[inst.name]) if inst.name in orderings else None
    n = inst.n
    off = ~np.eye(n, dtype=bool)
    values, dists, zero_frac, zero_frac_random, perms = [], [], [], [], []
    for _ in range(task["size"]):
        start = rng.permutation(n)
        zero_frac_random.append(float((insert_gains(inst.matrix, start)[off] == 0).mean()))
        perm, v, _ = local_search(inst.matrix, start, rng, "insert", "element")
        g = insert_gains(inst.matrix, perm)
        values.append(v)
        dists.append(kendall_distance(perm, ref) if ref is not None else None)
        zero_frac.append(float((g[off] == 0).mean()))
        perms.append(perm.tolist())
    ref_zero = float((insert_gains(inst.matrix, ref)[off] == 0).mean()) if ref is not None else None
    return {**base_record(task), "chunk": task["chunk"], "values": values, "dist_to_ref": dists,
            "zero_move_fraction": zero_frac, "zero_move_fraction_random": zero_frac_random, "perms": perms,
            "ref_zero_move_fraction": ref_zero}


# --------------------------------------------------------------------------- E0 calibration
def tasks_e0(concurrency: int, copies: int):
    for name in C.E0_INSTANCES:
        for r in range(copies):
            yield {"exp": "E0", "instance": name, "config": "ils", "algo": "ils",
                   "params": {"strength": 5, "acceptance": "better_equal"}, "run": r, "budget": C.E0_BUDGET,
                   "use_target": False, "concurrency": concurrency}


def run_e0(task):
    rec = run_meta(task)
    rec["concurrency"] = task["concurrency"]
    rec.pop("trace"), rec.pop("perm")
    return rec


def tasks_e6h():
    """Plateau measure on the harder sets (no optimal orderings, so no distances)."""
    for name in hard_names("mb", None) + hard_names("xlolib", 150):
        yield {"exp": "E6h", "instance": name, "chunk": 0, "size": C.E6_HARD_LOCAL_OPTIMA}


EXPERIMENTS = {
    "E0": (None, run_e0),
    "E1": (tasks_e1, run_e1),
    "E2": (tasks_e2, run_e2),
    "E3": (tasks_e3, run_meta),
    "E3b": (lambda: tasks_e3(C.E3B_GRID, "E3b"), run_meta),
    "E4": (tasks_e4, run_meta),
    **{e: (lambda e=e: tasks_e5(e), run_meta) for e in C.E5_SETS},
    "E6": (tasks_e6, run_e6),
    "E6h": (tasks_e6h, run_e6),
}


def task_id(task: dict) -> str:
    return json.dumps(task, sort_keys=True)


def _execute(task):
    exp_run = EXPERIMENTS[task["exp"]][1]
    return exp_run(task)


def run_experiment(exp: str, workers: int, limit: int | None = None) -> None:
    if exp == "E0":
        # the same ILS runs, first one at a time, then with every worker busy
        _run_tasks("E0", list(tasks_e0(1, 1)), 1, limit)
        _run_tasks("E0", list(tasks_e0(workers, max(1, workers // len(C.E0_INSTANCES)))), workers, limit)
        return
    _run_tasks(exp, list(EXPERIMENTS[exp][0]()), workers, limit)


def _run_tasks(exp: str, raw_tasks: list[dict], workers: int, limit: int | None) -> None:
    tasks, ids = [], set()
    for t in raw_tasks:
        t["id"] = task_id(t)
        if t["id"] in ids:  # a grid that lists the same configuration twice runs it once
            continue
        ids.add(t["id"])
        t["seed"] = seed_of(t["id"])
        tasks.append(t)
    if limit is not None:
        tasks = tasks[:limit]
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{exp}.jsonl"
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut short by a killed job; its task simply reruns
    todo = [t for t in tasks if t["id"] not in done]
    # random order, so that slow and fast configurations share the node's load evenly over time
    np.random.default_rng(0).shuffle(todo)
    print(f"[{exp}] {len(tasks)} tasks, {len(done)} already done, {len(todo)} to run on {workers} workers",
          flush=True)
    t0, finished = time.time(), 0
    with out.open("a") as f, ProcessPoolExecutor(max_workers=workers) as pool:
        # bounded submission keeps memory flat and lets progress be reported as results arrive
        pending, it = set(), iter(todo)
        for t in it:
            pending.add(pool.submit(_execute, t))
            if len(pending) >= 4 * workers:
                break
        while pending:
            complete, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in complete:
                f.write(json.dumps(fut.result()) + "\n")
                finished += 1
                nxt = next(it, None)
                if nxt is not None:
                    pending.add(pool.submit(_execute, nxt))
            f.flush()
            if finished % max(1, len(todo) // 20) < len(complete):
                print(f"[{exp}] {finished}/{len(todo)} after {time.time() - t0:.0f}s", flush=True)
    print(f"[{exp}] done in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help=f"any of {', '.join(EXPERIMENTS)}, or 'all'")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=None, help="run only the first N tasks (smoke test)")
    a = p.parse_args(argv)
    exps = list(EXPERIMENTS) if a.experiments == ["all"] else a.experiments
    for e in exps:
        run_experiment(e, a.workers, a.limit)


if __name__ == "__main__":
    sys.exit(main())
