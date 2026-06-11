"""CLI overrides for the ``TrainConfig`` literal: ``--set dotted.key=value``.

The training recipe in ``participant/train.py`` is a single ``TrainConfig``
literal, and that stays the source of truth. This module lets you nudge a few
fields from the command line without editing the file, which is what makes
shell-loop sweeps practical::

    uv run python participant/train.py --set optim.learning_rate=1e-4
    uv run python participant/train.py --set data.batch_size=128 --epochs 3 --seed 1

    for lr in 1e-5 2e-5 5e-5; do
        uv run python participant/train.py --set optim.learning_rate=$lr
    done

How it works
-----------

``--set`` takes a dotted path into the config and a value. The path walks
nested dataclasses (``optim.learning_rate``) and into the ``init`` kwargs dict
of any target-resolved component (``model.init.max_length``,
``loss.init.temperature``). A few common scalars also get their own short
flags (``--epochs``, ``--seed``, ``--batch-size``, ...), which are pure sugar
for the equivalent ``--set``.

Values are parsed with a JSON-first rule: ``1e-4`` -> float, ``128`` -> int,
``true`` -> bool, ``null`` -> None, ``"foo"`` or bare ``foo`` -> str,
``[1,2]`` -> list. So you rarely have to quote anything in the shell.

Nothing here is hidden: every override is logged before training starts, and
the resolved ``TrainConfig`` is still snapshotted to ``config.json`` in the
run folder exactly as before, so a swept run remains fully reproducible.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Short flags that are sugar for ``--set <dest>=<value>``. Keep this to the
# handful of scalars people actually sweep; everything else goes through
# ``--set``.
_SHORTCUT_FLAGS: dict[str, str] = {
    "--epochs": "epochs",
    "--seed": "seed",
    "--log-every": "log_every",
    "--learning-rate": "optim.learning_rate",
    "--lr": "optim.learning_rate",
    "--weight-decay": "optim.weight_decay",
    "--batch-size": "data.batch_size",
    "--num-workers": "data.num_workers",
    "--output-dir": "output_dir",
}


def add_override_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--set`` plus the shortcut flags on an existing parser."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help=(
            "Override a TrainConfig field by dotted path, e.g. "
            "--set optim.learning_rate=1e-4 --set loss.init.temperature=0.07. "
            "Repeatable. Values parse as JSON (numbers, true/false/null, "
            "lists), falling back to a plain string."
        ),
    )
    for flag, dest in _SHORTCUT_FLAGS.items():
        parser.add_argument(
            flag,
            dest=f"shortcut::{dest}",
            default=None,
            metavar="VALUE",
            help=f"Shortcut for --set {dest}=<value>.",
        )


def collect_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Merge ``--set`` entries and shortcut flags into one {path: raw} dict.

    ``--set`` wins over a shortcut for the same path, so an explicit
    ``--set optim.learning_rate=...`` after ``--lr`` is the value that lands.
    Later ``--set`` entries win over earlier ones (last-wins), matching how a
    repeated flag normally reads.
    """
    collected: dict[str, str] = {}

    for dest_attr, raw in vars(args).items():
        if dest_attr.startswith("shortcut::") and raw is not None:
            collected[dest_attr[len("shortcut::"):]] = raw

    for item in getattr(args, "overrides", []) or []:
        if "=" not in item:
            raise ValueError(
                f"--set expects 'dotted.key=value', got {item!r} (no '=')."
            )
        path, _, raw = item.partition("=")
        path = path.strip()
        if not path:
            raise ValueError(f"--set has an empty key in {item!r}.")
        collected[path] = raw

    return collected


def _parse_value(raw: str) -> Any:
    """JSON-first scalar parse; fall back to the raw string.

    ``1e-4`` -> float, ``128`` -> int, ``true`` -> True, ``null`` -> None,
    ``[1,2]`` -> list, ``foo`` -> ``"foo"`` (bare strings need no quoting).
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _set_by_path(config: Any, path: str, value: Any) -> None:
    """Walk ``path`` into nested dataclasses / ``init`` dicts and assign.

    Dataclass fields are set as attributes; once the walk steps into a dict
    (e.g. a component's ``init`` kwargs), the remaining segments are dict keys.
    Raises with the offending path if a segment does not exist on a dataclass,
    which catches typos like ``--set optin.lr=...`` instead of silently
    creating a dead field.
    """
    parts = path.split(".")
    cursor: Any = config
    for i, part in enumerate(parts[:-1]):
        traversed = ".".join(parts[: i + 1])
        if isinstance(cursor, dict):
            cursor = cursor.setdefault(part, {})
            continue
        if dataclasses.is_dataclass(cursor) and not isinstance(cursor, type):
            if part not in {f.name for f in dataclasses.fields(cursor)}:
                raise KeyError(
                    f"--set path {path!r}: {traversed!r} is not a field of "
                    f"{type(cursor).__name__}."
                )
            cursor = getattr(cursor, part)
            continue
        raise KeyError(
            f"--set path {path!r}: cannot descend into {traversed!r} "
            f"(a {type(cursor).__name__} is neither a dataclass nor a dict)."
        )

    leaf = parts[-1]
    if isinstance(cursor, dict):
        cursor[leaf] = value
        return
    if dataclasses.is_dataclass(cursor) and not isinstance(cursor, type):
        if leaf not in {f.name for f in dataclasses.fields(cursor)}:
            raise KeyError(
                f"--set path {path!r}: {leaf!r} is not a field of "
                f"{type(cursor).__name__}."
            )
        setattr(cursor, leaf, value)
        return
    raise KeyError(
        f"--set path {path!r}: cannot assign onto a {type(cursor).__name__}."
    )


def apply_overrides(config: Any, overrides: dict[str, str]) -> Any:
    """Mutate ``config`` in place from {dotted.path: raw_value}, then return it.

    Logs every applied override at INFO so a swept run is self-documenting in
    the console as well as in the ``config.json`` snapshot the loop writes.
    """
    for path in sorted(overrides):
        value = _parse_value(overrides[path])
        _set_by_path(config, path, value)
        logger.info("override: %s = %r", path, value)
    return config


def apply_cli_overrides(config: Any, argv: list[str] | None = None) -> Any:
    """One-call helper for an entry point: parse argv, apply, return config.

    Builds a small parser, reads ``--set`` and the shortcut flags off
    ``argv`` (defaults to ``sys.argv[1:]``), and applies them to ``config``.
    Unknown args raise the usual argparse error so typos surface loudly.
    """
    parser = argparse.ArgumentParser(
        description="Override the TrainConfig recipe from the command line.",
    )
    add_override_args(parser)
    args = parser.parse_args(argv)
    return apply_overrides(config, collect_overrides(args))
