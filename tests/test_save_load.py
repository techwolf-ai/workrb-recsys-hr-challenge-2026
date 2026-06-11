"""Does your model survive a save/reload round-trip?

``participant/test.py`` and the submission script never score the model you
just trained in memory. They call ``WorkrbSaveable.from_pretrained(path)`` to
rebuild it from disk. If your save/load is wrong, training succeeds, then the
reloaded model is silently broken (e.g. a classifier head re-initialized at
random) and your leaderboard number is garbage with no error.

This catches the most common version of that: you added trainable state
beyond the backbone and forgot to persist it in ``_save_extra``. Reloaded
weights must equal saved weights.

If you override ``_save_extra`` in ``participant/my_model.py``, keep this green.
"""

from __future__ import annotations

import torch
from workrb_challenge.models import WorkrbSaveable


def test_round_trip_returns_same_subclass(model, tmp_path):
    """``from_pretrained`` must rebuild the concrete subclass, not a base class."""
    save_dir = tmp_path / "ckpt"
    model.save_pretrained(save_dir)
    reloaded = WorkrbSaveable.from_pretrained(save_dir)
    assert type(reloaded) is type(model), (
        f"from_pretrained returned {type(reloaded).__name__}, expected {type(model).__name__}. "
        "The dotted-path target in workrb_model.json must resolve to your class."
    )


def test_config_file_written(model, tmp_path):
    """The convention file that makes reload work must exist next to the weights."""
    save_dir = tmp_path / "ckpt"
    model.save_pretrained(save_dir)
    assert (save_dir / "workrb_model.json").exists(), (
        "save_pretrained must write workrb_model.json (dotted path + init kwargs). "
        "from_pretrained reads it to rebuild your model."
    )


def test_reloaded_model_gives_identical_scores(model, tmp_path, queries, targets):
    """The whole point: a reloaded model must score identically to the original.

    If you add a head/projection/learned-temperature and don't persist it in
    ``_save_extra``, this test fails because the reloaded copy has fresh random
    weights. That is exactly the silent failure we want to surface.
    """
    save_dir = tmp_path / "ckpt"
    model.save_pretrained(save_dir)
    reloaded = WorkrbSaveable.from_pretrained(save_dir)

    model.eval()
    reloaded.eval()
    before = model.compute_rankings(queries, targets)
    after = reloaded.compute_rankings(queries, targets)
    assert torch.allclose(before, after, atol=1e-5), (
        "Reloaded model scores differ from the original. You likely have trainable "
        "state beyond self.backbone/self.tokenizer that save_pretrained did not "
        "persist. Override _save_extra in participant/my_model.py to write it and "
        "return the init kwargs needed to rebuild it."
    )
