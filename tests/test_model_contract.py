"""Does YOUR model still satisfy the WorkRB contract?

These are the checks that decide whether the leaderboard can score you at
all. WorkRB calls five things on your model (see the table in
``participant/my_model.py``); if any of them is the wrong shape or type, you
get a crash or a zero *after* a slow upload, against a capped submission
budget. Catch it here in a second instead.

If you change ``participant/my_model.py`` (new architecture, new scoring),
re-run ``uv run pytest`` and keep these green.
"""

from __future__ import annotations

import torch
from workrb.models import ModelInterface


def test_model_is_a_workrb_model(model):
    """Your model must be a ``ModelInterface``; that is what WorkRB scores."""
    assert isinstance(model, ModelInterface), (
        "MyModel must inherit from workrb.models.ModelInterface so WorkRB can "
        "score it with no adapter. Check the class declaration in "
        "participant/my_model.py."
    )


def test_name_and_description_are_strings(model):
    """``name`` is your leaderboard label; ``description`` rides along with it."""
    assert isinstance(model.name, str) and model.name, (
        "model.name must be a non-empty string (it is your leaderboard display name)."
    )
    assert isinstance(model.description, str), "model.description must be a string."


def test_classification_label_space_is_none_or_list(model):
    """``None`` for bi-encoders; a list of label strings for a fixed classifier head."""
    space = model.classification_label_space
    assert space is None or isinstance(space, list), (
        "classification_label_space must be None (bi-encoder) or a list of label "
        "strings (fixed classifier head). Got: "
        f"{type(space).__name__}."
    )


def test_compute_rankings_shape(model, queries, targets):
    """The one shape WorkRB hard-requires: ``(num_queries, num_targets)``.

    Called through the public ``compute_rankings`` wrapper, which is what
    WorkRB actually invokes (it wraps your ``_compute_rankings`` in
    ``torch.no_grad``).
    """
    scores = model.compute_rankings(queries, targets)
    assert isinstance(scores, torch.Tensor), "_compute_rankings must return a torch.Tensor."
    assert scores.shape == (len(queries), len(targets)), (
        f"_compute_rankings returned shape {tuple(scores.shape)}, but WorkRB needs "
        f"(len(queries), len(targets)) = ({len(queries)}, {len(targets)}). "
        "Higher score = more relevant."
    )
    assert torch.isfinite(scores).all(), (
        "_compute_rankings produced non-finite scores (nan/inf). WorkRB cannot rank with those."
    )


def test_compute_classification_shape(model, queries, targets):
    """``(num_texts, num_classes)``. For a bi-encoder the class axis is the targets."""
    if model.classification_label_space is None:
        logits = model.compute_classification(queries, targets, input_type=None)
        expected = (len(queries), len(targets))
    else:
        labels = model.classification_label_space
        logits = model.compute_classification(queries, labels, input_type=None)
        expected = (len(queries), len(labels))
    assert isinstance(logits, torch.Tensor), "_compute_classification must return a torch.Tensor."
    assert logits.shape == expected, (
        f"_compute_classification returned shape {tuple(logits.shape)}, expected {expected} "
        "= (num_texts, num_classes)."
    )


def test_single_query_single_target(model):
    """Degenerate 1x1 case still produces a well-formed matrix, not a scalar."""
    scores = model.compute_rankings(["one sentence"], ["one skill"])
    assert scores.shape == (1, 1), (
        f"With one query and one target, expected shape (1, 1), got {tuple(scores.shape)}. "
        "Make sure your scoring keeps both axes even for size-1 inputs."
    )
