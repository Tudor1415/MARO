"""Dependency-free correctness check, for machines without pytest (e.g. the cluster image).

    python -m tests.selfcheck

Runs every test function of ``tests/test_lop.py``, expanding its ``pytest.mark.parametrize``
decorators by hand, and exits non-zero on the first failure.
"""

from __future__ import annotations

import itertools
import sys
import types

# A tiny stand-in for the parts of pytest the test module uses.
_fake = types.ModuleType("pytest")


def _parametrize(names, values):
    names = [n.strip() for n in names.split(",")]

    def deco(fn):
        fn._params = getattr(fn, "_params", []) + [(names, list(values))]
        return fn

    return deco


_fake.mark = types.SimpleNamespace(parametrize=_parametrize)
sys.modules.setdefault("pytest", _fake)

from tests import test_lop  # noqa: E402


def main() -> int:
    count = 0
    for name in sorted(dir(test_lop)):
        fn = getattr(test_lop, name)
        if not (name.startswith("test_") and callable(fn)):
            continue
        axes = getattr(fn, "_params", [])
        for combo in itertools.product(*[vals for _, vals in axes]):
            kwargs = {}
            for (names, _), val in zip(axes, combo):
                val = val if len(names) > 1 else (val,)
                kwargs.update(zip(names, val))
            fn(**kwargs)
            count += 1
    print(f"selfcheck: {count} test cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
