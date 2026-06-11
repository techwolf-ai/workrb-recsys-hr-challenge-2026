"""Your loss: the training objective.

The training loop calls ``loss_fn(model, batch)`` once per step. The framework
does not know what shape the loss expects, what fields the batch carries, or
which methods the model exposes for training. All it knows is "the loss
returns a scalar tensor, then we backprop and step."

This means your loss and your model are **co-designed**. If you change the
loss to expect ``model.classifier_logits(...)`` and ``batch.labels``, then
your model and your dataset/collate need to provide them. The framework does
not mediate.

The contract
------------

A loss class implements:

    def forward(self, model, batch: Batch) -> torch.Tensor

That is the whole interface. ``model`` is whatever your participant model is
(in the baseline, a ``MyModel`` that combines ``nn.Module``, WorkRB's
``ModelInterface``, and ``WorkrbSaveable``). The loss is the only thing
that reads training-side methods on it, so it decides what those methods
are called and what they return. Subclass ``nn.Module`` so optimizer
state, ``.to(device)``, and ``.train()`` semantics behave like any other
module.

The shipped default
-------------------

``InfoNCELoss`` is symmetric in-batch InfoNCE with cosine similarity. It
treats the diagonal of the in-batch ``(queries, targets)`` similarity matrix
as the positives, every off-diagonal entry as a negative, and averages the
query-to-target and target-to-query cross entropies.

Common swaps (and what else moves with them)
--------------------------------------------

* **Triplet / pairwise hinge.**
  Loss reads ``model.score_pairs(queries, positives, negatives)``. Add a
  ``negatives`` field on the batch. Add a ``score_pairs`` method on the
  model.

* **Multi-positive in-batch NCE.**
  Loss reads a positives mask on the batch (multiple skills can be valid
  for one sentence). Same encode path; different label mask.

* **Distillation / alignment.**
  Loss reads ``model.encode_query(...)`` plus ``batch.teacher_logits`` (or
  a teacher's embeddings). The teacher is precomputed offline and packed
  into the batch by your collate, so training-time forward stays cheap.

* **Classifier cross-entropy.**
  Loss reads ``model.classifier_logits(batch.texts)`` against ``batch.labels``
  over a fixed label space. The model overrides ``classification_label_space``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from participant.data import Batch


# ============================================================================
# >>> SWAP: loss <<<
# ----------------------------------------------------------------------------
# Replace the body, or add new loss classes, then point ``LossConfig.target``
# at the new dotted path from ``participant/train.py``.
#
# Sketches:
#
#   class TripletLoss(nn.Module):
#       def __init__(self, margin: float = 0.2):
#           super().__init__()
#           self.margin = margin
#       def forward(self, model, batch):
#           pos = model.score_pairs(batch.queries, batch.positives)
#           neg = model.score_pairs(batch.queries, batch.negatives)
#           return F.relu(self.margin - pos + neg).mean()
#
#   class ClassifierCELoss(nn.Module):
#       def forward(self, model, batch):
#           logits = model.classifier_logits(batch.texts)
#           return F.cross_entropy(logits, batch.labels)
# ============================================================================


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE with cosine similarity.

    The trick: in a batch of B (sentence, skill) positive pairs, the
    similarity matrix is ``B x B``. The diagonal entries are the true
    positives; every off-diagonal is treated as a negative. Doing cross
    entropy in both directions ("which target matches this query" plus
    "which query matches this target") and averaging gives a stable,
    symmetric objective.

    Hyperparameters
    ---------------
    temperature:
        Divides the logits before cross entropy. Smaller = sharper
        distribution = more aggressive pull/push. 0.05 is a common
        starting point for sentence-level cosine spaces.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, model, batch: Batch) -> torch.Tensor:
        # 1. Encode both sides. These two methods are the model's
        # training-side surface; you choose what they do (one shared
        # backbone, two backbones, projection heads, whatever).
        q_emb = model.encode_query(batch.queries)
        t_emb = model.encode_target(batch.targets)

        # 2. Normalize to make dot product = cosine similarity.
        q_norm = F.normalize(q_emb, p=2, dim=-1)
        t_norm = F.normalize(t_emb, p=2, dim=-1)

        # 3. Build the B x B similarity matrix and temperature-scale it.
        logits = (q_norm @ t_norm.T) / self.temperature

        # 4. Positives are on the diagonal: row i should pick column i.
        labels = torch.arange(logits.size(0), device=logits.device)

        # 5. Symmetric cross entropy: average the "which target for this
        # query" loss with the "which query for this target" loss.
        loss_q2t = F.cross_entropy(logits, labels)
        loss_t2q = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_q2t + loss_t2q)
