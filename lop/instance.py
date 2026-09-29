"""Reading LOLIB instances.

A LOLIB ``.mat`` file is plain text: a free-form header (one or more lines), a
line holding only the matrix order ``n``, then the ``n * n`` integer entries in
row-major order, wrapped over as many lines as the author liked. The Optsicom
distributions of xLOLIB and MB have no header and start directly with ``n``;
their files are named ``N-<name>`` (``N`` for "normalised").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Instance families of the LOLIB input-output set, keyed by file-name prefix.
FAMILIES = {
    "be75": "SGB (Belgium 1975)",
    "stabu": "Stabu (Germany 1970)",
    "t": "IO (Europe 1959-75)",
    "tiw56": "TIW (Germany 1954)",
    "rand": "Synthetic uniform (RandA1-like)",
    "xlolib": "xLOLIB (IO-like, n = 150, 250)",
    "mb": "MB (Mitchell-Borchers, n = 100-250)",
}


def family_of(name: str) -> str:
    """Return the short family key of an instance (``be75``, ``stabu``, ``t``, ``tiw56``, ``xlolib``, ``mb``)."""
    if re.fullmatch(r".+_(150|250)", name):
        return "xlolib"
    if re.fullmatch(r"r\d{3}[a-e]\d", name):
        return "mb"
    for prefix in ("be75", "stabu", "tiw56"):
        if name.startswith(prefix):
            return prefix
    if re.fullmatch(r"t\d\d[a-z]\d\d[a-z]{2}", name):
        return "t"
    if name.startswith("rand_"):
        return "rand"
    raise ValueError(f"unknown LOLIB family for instance {name!r}")


@dataclass(frozen=True)
class Instance:
    """A Linear Ordering Problem instance.

    Attributes:
        name: File stem, e.g. ``"be75eec"``.
        matrix: ``(n, n)`` int64 weight matrix. The objective of a permutation is the
            sum of the entries that end up above the diagonal once rows and columns
            are both reordered by it.
        header: The descriptive first line of the file.
        optimum: Proven optimal value, if known.
    """

    name: str
    matrix: np.ndarray = field(repr=False)
    header: str = ""
    optimum: int | None = None

    @property
    def n(self) -> int:
        return self.matrix.shape[0]

    @property
    def family(self) -> str:
        return family_of(self.name)

    @property
    def diagonal_free_total(self) -> int:
        """Sum of all off-diagonal entries, a trivial upper bound on the objective."""
        return int(self.matrix.sum() - np.trace(self.matrix))

    @property
    def symmetric_lower_bound(self) -> int:
        """Value any ordering is guaranteed to reach: ``sum_{i<j} min(m_ij, m_ji)``."""
        m = self.matrix
        return int(np.triu(np.minimum(m, m.T), 1).sum())


def read_instance(path: str | Path, optimum: int | None = None) -> Instance:
    path = Path(path)
    lines = path.read_text().splitlines()
    size_line = next(k for k, line in enumerate(lines) if re.fullmatch(r"\s*\d+\s*", line))
    header = " ".join(line.strip() for line in lines[:size_line])
    n = int(lines[size_line])
    tokens = " ".join(lines[size_line + 1 :]).split()
    # Some SGB files repeat the header after the matrix; only trailing text is tolerated.
    count = next((k for k, tok in enumerate(tokens) if not re.fullmatch(r"-?\d+", tok)), len(tokens))
    if count != n * n:
        raise ValueError(f"{path.name}: expected {n * n} entries for n={n}, found {count}")
    values = np.array(tokens[:count], dtype=np.int64)
    name = path.stem[2:] if path.stem.startswith("N-") else path.stem
    return Instance(name=name, matrix=values.reshape(n, n), header=header, optimum=optimum)


def load_optima(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if not path.exists():
        return {}
    return {k: int(v) for k, v in json.loads(path.read_text()).items()}


def load_instances(
    directory: str | Path = "instances",
    optima_file: str | Path | None = "instances/optima.json",
    names: list[str] | None = None,
) -> list[Instance]:
    """Load every ``*.mat`` instance in ``directory``, sorted by name.

    ``names`` restricts the selection; optimal values are attached when
    ``optima_file`` lists them.
    """
    directory = Path(directory)
    optima = load_optima(optima_file) if optima_file else {}
    paths = sorted(directory.glob("*.mat"))
    if names is not None:
        wanted = set(names)
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"instances not found: {sorted(missing)}")
    return [read_instance(p, optima.get(p.stem)) for p in paths]
