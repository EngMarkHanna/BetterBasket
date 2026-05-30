"""Load System 1's modules from disk by file path under unique names so
they coexist with SYSTEM 2's identically-named `solution` package without
a sys.path-order collision.

Why this exists: both systems use `solution/` as their package root, so
`from solution.retrieve import t1_strict_blocks` resolves ambiguously
depending on which root sits first in sys.path. We side-step that by
loading System 1's modules directly via `importlib`.

Public surface:
  system1_retrieve.t1_strict_blocks(a, b, top_per_a)
  system1_retrieve.t3_tfidf_topk(a, b, k, cosine_floor)
  system1_retrieve.t5_private_label(a, b, bridge_csv, top_per_a)
  system1_score.score_candidates(df, a_index, b_index)
  system1_score.build_lookup_index(df)
  system1_select.select_one_b_per_a(df)
  system1_validate.validate_against_eval(matches_csv, candidates, labels)
  system1_validate.write_report(report, md_path)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_SYSTEM1_SOLUTION = (
    Path(__file__).resolve().parents[2] / "SYSTEM 1 MVP" / "solution"
)


def _load(name: str, file: Path, *, dependencies: dict[str, ModuleType] | None = None) -> ModuleType:
    """Load a file as a fresh module under a unique name, optionally
    pre-registering dependency modules so relative imports resolve.
    """
    if not file.exists():
        raise FileNotFoundError(f"System 1 module missing: {file}")
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {file}")
    module = importlib.util.module_from_spec(spec)
    if dependencies:
        for dep_name, dep_mod in dependencies.items():
            sys.modules[dep_name] = dep_mod
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# parse.py has no relative imports - safe to load directly.
system1_parse = _load("system1_solution_parse", _SYSTEM1_SOLUTION / "parse.py")

# score.py imports `from .parse import normalize_text` - we need to make
# `system1_solution_parse` available as a relative target. Easiest path:
# register a synthetic package `system1_solution` with `parse` attribute.
_pkg = ModuleType("system1_solution")
_pkg.__path__ = [str(_SYSTEM1_SOLUTION)]
_pkg.parse = system1_parse
sys.modules["system1_solution"] = _pkg
sys.modules["system1_solution.parse"] = system1_parse

system1_retrieve = _load(
    "system1_solution.retrieve", _SYSTEM1_SOLUTION / "retrieve.py"
)
system1_score = _load(
    "system1_solution.score", _SYSTEM1_SOLUTION / "score.py"
)
system1_select = _load(
    "system1_solution.select", _SYSTEM1_SOLUTION / "select.py"
)
system1_validate = _load(
    "system1_solution.validate", _SYSTEM1_SOLUTION / "validate.py"
)
