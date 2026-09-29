"""Synthetic LOP instances for the scaling study.

LOLIB's input-output matrices are small (n <= 60) and every method in this
study solves them to proven optimality within a second. To see how the methods
separate when the optimum is out of easy reach, the scaling study adds dense
random instances in the style of the RandA1 set of Reinelt and Martí: off-diagonal
entries i.i.d. uniform on ``{0, ..., 100}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .instance import Instance


def random_instance(n: int, seed: int, high: int = 100) -> Instance:
    rng = np.random.default_rng([n, seed])
    m = rng.integers(0, high + 1, size=(n, n), dtype=np.int64)
    np.fill_diagonal(m, 0)
    return Instance(name=f"rand_n{n}_s{seed}", matrix=m, header=f"uniform[0,{high}] n={n} seed={seed}")


def write_instance(inst: Instance, path: str | Path) -> None:
    """Write ``inst`` in LOLIB ``.mat`` format (header, order, rows)."""
    rows = "\n".join(" ".join(str(int(v)) for v in row) for row in inst.matrix)
    Path(path).write_text(f"{inst.header}\n{inst.n}\n{rows}\n")
