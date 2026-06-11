"""Does your data path produce a Batch your loss can read?

The dataset yields per-index items, the collate function turns a list of them
into a ``Batch``, and the loss reads fields off that ``Batch``. The framework
treats ``Batch`` as opaque and never validates it. If you add a field to
``Batch`` (hard negatives, labels, teacher logits) you must produce it in the
collate too, or the loss reads a missing attribute mid-training.

These checks exercise the collate seam without downloading the 138K-row HF
dataset: they feed ``default_collate`` synthetic ``(sentence, skill)`` pairs,
exactly the shape ``SkillSentenceDataset.__getitem__`` returns.

If you change ``participant/data.py`` (Batch shape, collate, or columns),
update these tests to match the new contract.
"""

from __future__ import annotations

from dataclasses import fields

from participant.data import Batch, default_collate


def test_default_collate_builds_aligned_batch():
    """Collate must preserve pair alignment: row i query lines up with row i target."""
    rows = [
        ("operate a sewing machine", "sewing"),
        ("diagnose engine faults", "engine repair"),
        ("write React components", "front-end development"),
    ]
    batch = default_collate(rows)
    assert isinstance(batch, Batch), "default_collate must return a Batch instance."
    assert batch.queries == [r[0] for r in rows], "Queries must keep the input order."
    assert batch.targets == [r[1] for r in rows], "Targets must keep the input order."
    assert len(batch.queries) == len(batch.targets), (
        "queries and targets must be the same length: the diagonal is the positives."
    )


def test_batch_fields_are_populated():
    """Every field declared on Batch must actually be produced by the collate.

    If you add a field to the Batch dataclass and forget to set it in the
    collate, this fails. That is the mismatch that otherwise surfaces as an
    AttributeError deep inside your loss, mid-training.
    """
    rows = [("a sentence", "a skill"), ("another", "skill two")]
    batch = default_collate(rows)
    for f in fields(Batch):
        value = getattr(batch, f.name, None)
        assert value is not None, (
            f"Batch field {f.name!r} is None after default_collate. If you added it to "
            "the Batch dataclass, also populate it in default_collate in participant/data.py."
        )
