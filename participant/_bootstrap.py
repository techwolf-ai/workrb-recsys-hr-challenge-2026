"""Make the entry-point scripts importable regardless of editable-install state.

Why this exists
---------------

The project ships as an editable install (``src/workrb_challenge`` on the
import path via a ``.pth`` file written by ``uv sync``). That install
occasionally goes stale: the ``.pth`` is registered but not yet effective,
so ``import workrb_challenge`` raises ``ModuleNotFoundError`` even though
nothing is actually wrong with your code. The documented fix
(``uv sync --reinstall-package workrb-challenge-2026``) works, but it is a
trap to hit on your very first run.

The fix is to put the repo root (for ``import participant.*``) and ``src/``
(for ``import workrb_challenge.*``) on ``sys.path`` directly, the same two
entries the pytest config adds via ``pythonpath``. Idempotent and cheap; if
the editable install is healthy it is a no-op.

The entry points (``participant/train.py``, ``participant/test.py``) inline
those few lines at the very top rather than importing this module. They have
to: when you run ``python participant/train.py`` as a path, Python only puts
``participant/`` on ``sys.path``, so ``import participant._bootstrap`` would
itself fail with ``ModuleNotFoundError`` before the path is fixed. This file
is the canonical explanation of that snippet, and a convenience for any of
your own scripts launched with the repo root already importable (e.g. a sweep
driver run as ``python -m ...``)::

    import participant._bootstrap  # noqa: F401  (import for side effect)
"""

from __future__ import annotations

import sys
from pathlib import Path

# __file__ = .../participant/_bootstrap.py  ->  parents[1] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC):
    _entry = str(_path)
    if _path.is_dir() and _entry not in sys.path:
        # Prepend so the in-tree source always wins over a stale build.
        sys.path.insert(0, _entry)
