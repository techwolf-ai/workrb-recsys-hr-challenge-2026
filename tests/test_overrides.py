"""Does the command-line sweep mechanism actually move the knobs it claims to?

The training recipe in ``participant/train.py`` is a single ``TrainConfig``
literal, and ``apply_cli_overrides`` is what lets a shell loop nudge it without
editing the file::

    for lr in 1e-5 2e-5 5e-5; do
        uv run python participant/train.py --lr $lr
    done

If that override path quietly drops a flag, mistypes a nested key into a dead
field, or parses ``1e-4`` as the string ``"1e-4"``, an entire sweep runs with
the wrong (or unchanged) hyperparameters and you only notice from the
leaderboard. These checks make a broken sweep fail in a second instead.

They exercise the contract the module promises: shortcut flags and ``--set``
land with JSON-parsed types, dotted paths reach into nested dataclasses and
``init`` kwarg dicts, ``--set`` wins over a shortcut for the same path, typos
raise loudly, and a swept config still round-trips through the JSON snapshot
that makes the run reproducible.
"""

from __future__ import annotations

import json

import pytest

from workrb_challenge.training.config import TrainConfig, snapshot_config
from workrb_challenge.training.overrides import apply_cli_overrides


def test_shortcut_flags_land_with_parsed_types():
    """``--lr``/``--batch-size``/``--epochs`` are sugar for the matching --set."""
    cfg = apply_cli_overrides(
        TrainConfig(), ["--lr", "1e-4", "--batch-size", "128", "--epochs", "3"]
    )
    assert cfg.optim.learning_rate == 1e-4
    assert isinstance(cfg.optim.learning_rate, float)
    assert cfg.data.batch_size == 128
    assert isinstance(cfg.data.batch_size, int)
    assert cfg.epochs == 3


def test_set_reaches_nested_dataclass_and_init_dicts():
    """``--set`` walks dataclass fields and into a component's ``init`` kwargs."""
    cfg = apply_cli_overrides(
        TrainConfig(),
        [
            "--set", "optim.weight_decay=0.0",
            "--set", "loss.init.temperature=0.07",
            "--set", "model.init.leaderboard_name=sweep-run",
        ],
    )
    assert cfg.optim.weight_decay == 0.0
    assert cfg.loss.init["temperature"] == 0.07
    assert cfg.model.init["leaderboard_name"] == "sweep-run"


def test_value_parsing_is_json_first():
    """Numbers/bools/null/lists parse as JSON; a bare word stays a string."""
    cfg = apply_cli_overrides(
        TrainConfig(),
        [
            "--set", "seed=7",          # int
            "--set", "data.drop_last=false",  # bool
            "--set", "output_dir=null",       # None
            "--set", "model.target=participant.my_model:MyModel",  # bare str
        ],
    )
    assert cfg.seed == 7 and isinstance(cfg.seed, int)
    assert cfg.data.drop_last is False
    assert cfg.output_dir is None
    assert cfg.model.target == "participant.my_model:MyModel"


def test_explicit_set_wins_over_shortcut_for_same_path():
    """An explicit ``--set optim.learning_rate=...`` overrides the ``--lr`` sugar."""
    cfg = apply_cli_overrides(
        TrainConfig(), ["--lr", "1e-5", "--set", "optim.learning_rate=9e-9"]
    )
    assert cfg.optim.learning_rate == 9e-9


def test_typo_into_unknown_field_raises_not_silently_dropped():
    """A misspelled path must fail loudly instead of creating a dead field."""
    with pytest.raises(KeyError):
        apply_cli_overrides(TrainConfig(), ["--set", "optin.learning_rate=1e-4"])


def test_malformed_set_entries_raise():
    """``--set`` without ``=`` or with an empty key is a usage error."""
    with pytest.raises(ValueError):
        apply_cli_overrides(TrainConfig(), ["--set", "epochs"])
    with pytest.raises(ValueError):
        apply_cli_overrides(TrainConfig(), ["--set", "=5"])


def test_unknown_flag_raises():
    """A typo'd flag surfaces via argparse rather than being ignored."""
    with pytest.raises(SystemExit):
        apply_cli_overrides(TrainConfig(), ["--learnign-rate", "1e-4"])


def test_swept_config_round_trips_through_snapshot(tmp_path):
    """A swept run stays reproducible: the override shows up in config.json."""
    cfg = apply_cli_overrides(
        TrainConfig(),
        ["--lr", "1e-4", "--set", "loss.init.temperature=0.07"],
    )
    path = snapshot_config(cfg, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["optim"]["learning_rate"] == 1e-4
    assert payload["loss"]["init"]["temperature"] == 0.07
