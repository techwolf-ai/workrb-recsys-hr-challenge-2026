"""Does your loss return something the training loop can actually use?

The framework calls ``loss_fn(model, batch)`` once per step, then
``loss.backward()`` and ``optimizer.step()``. It checks nothing about the
result. If your loss returns the wrong shape, a detached tensor, or a nan,
you find out after a full training run (or not at all). These checks make a
loss/model/batch shape mismatch fail in a second instead.

The loss and the model are co-designed (see ``participant/loss.py``): the loss
decides which methods to call on the model. If you swap the loss, swap the
model methods it reads, and update these tests to build the batch your loss
expects.
"""

from __future__ import annotations

import torch

from participant.data import Batch
from participant.loss import InfoNCELoss


def _batch(queries, targets):
    """The shipped Batch the default InfoNCE reads. Pairs must be aligned (diagonal = positives)."""
    n = min(len(queries), len(targets))
    return Batch(queries=list(queries[:n]), targets=list(targets[:n]))


def test_loss_returns_scalar(model, queries, targets):
    """A training step needs a 0-dim scalar to call ``.backward()`` on."""
    loss = InfoNCELoss()
    value = loss(model, _batch(queries, targets))
    assert isinstance(value, torch.Tensor), "The loss must return a torch.Tensor."
    assert value.dim() == 0, (
        f"The loss must return a scalar (0-dim) tensor, got shape {tuple(value.shape)}. "
        "The training loop calls .backward() on it directly."
    )


def test_loss_is_finite(model, queries, targets):
    """nan/inf here means a silently dead training run."""
    loss = InfoNCELoss()
    value = loss(model, _batch(queries, targets))
    assert torch.isfinite(value), (
        "The loss is not finite (nan/inf). Common causes: temperature too small, "
        "un-normalized embeddings, or a batch of size < 2."
    )


def test_loss_is_differentiable(model, queries, targets):
    """The loss must stay attached to the graph, or training does nothing.

    Runs an actual backward pass and confirms at least one model parameter
    received a gradient. A detached tensor (e.g. a stray ``torch.no_grad`` or
    ``.detach()`` in your forward path) is caught here.
    """
    model.train()
    loss = InfoNCELoss()
    value = loss(model, _batch(queries, targets))
    assert value.requires_grad, (
        "The loss does not require grad: it is detached from the model. Check for "
        "torch.no_grad()/.detach() on the training-side forward in participant/my_model.py."
    )
    value.backward()
    got_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert got_grad, "No model parameter received a finite gradient after loss.backward()."
