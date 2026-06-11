"""Alt architecture: frozen backbone + trainable linear head, trained with cross entropy.

Why this example exists
-----------------------

A bi-encoder treats every skill as a free-form text string. That generalizes
beautifully to unseen skills, but it pays for that flexibility every time
you score: you have to encode the label text.

A classifier head fixes a **closed label space** up front. The model maps
any input sentence directly to a probability over that fixed set of labels
through a single linear projection on top of the encoder. Inference is one
forward pass per query, regardless of how many labels exist.

The trade-off: you can only score sentences against labels you trained on.
For ESCO that's plenty (the label space is fixed and well-defined). For
open-ended skill discovery, the bi-encoder wins.

This file demonstrates the **end-to-end change** required to swap to a
classifier head:

  1. A model with a fixed ``classification_label_space`` and a
     ``classifier_logits`` method the loss can read.
  2. A loss that reads ``classifier_logits`` plus ``batch.labels``.
  3. The corresponding tweak to your dataset and batch: the targets are no
     longer free-form strings but indices into the label space.

The model and loss live here. The batch change (``LabeledBatch`` with a
``labels`` field) is in this file too, kept small so you can see all the
moving pieces at once.

Quick smoke test::

    uv run python -m participant.examples.classifier_head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Put the repo root + ``src/`` on the import path so ``workrb_challenge``
# resolves even when the editable install is stale (the trap the other entry
# points guard against). Safe here because ``-m participant.examples.*`` already
# has the repo root importable, so this module import resolves. Idempotent.
import participant._bootstrap  # noqa: F401  (import for side effect)

# WorkRB contract (scored through ModelInterface) + challenge save/load mixin.
from workrb.models import ModelInterface
from workrb_challenge.models import WorkrbSaveable


# ============================================================================
# A custom batch shape: this loss needs labels, not target strings.
# ============================================================================


@dataclass
class LabeledBatch:
    texts: list[str]
    labels: torch.Tensor   # shape (B,), long dtype, values in [0, num_labels)


def labeled_collate(rows: list[tuple[str, int]]) -> LabeledBatch:
    """Pack a list of (text, label_idx) into one ``LabeledBatch``.

    Wire this up in ``participant/train.py`` with::

        collate="participant.examples.classifier_head:labeled_collate"

    Pair it with a dataset whose ``__getitem__`` returns ``(text, label_idx)``
    instead of ``(sentence, skill)``.
    """
    texts, labels = zip(*rows, strict=True)
    return LabeledBatch(texts=list(texts), labels=torch.tensor(labels, dtype=torch.long))


# ============================================================================
# The model
# ============================================================================


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class FrozenClassifierModel(nn.Module, ModelInterface, WorkrbSaveable):
    """Frozen encoder + trainable linear head over a fixed label space.

    The backbone is frozen, so only the linear head trains. That makes
    training fast and stable, at the cost of capacity: the encoder cannot
    adapt to your domain. Unfreeze ``self.backbone`` to lift that ceiling.
    """

    def __init__(
        self,
        labels: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 128,
        leaderboard_name: str = "FrozenClassifier-example",
        leaderboard_description: str = "Frozen encoder + linear head + CE.",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.labels = list(labels)
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, len(self.labels))

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def _encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.backbone(**inputs)
        return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    # ----- training surface (read by the loss) ------------------------------

    def classifier_logits(self, texts: list[str]) -> torch.Tensor:
        """The loss reads this. Returns shape (B, num_labels)."""
        return self.head(self._encode(texts))

    # ----- inference surface (read by WorkRB) -------------------------------

    @property
    def classification_label_space(self) -> list[str]:
        return list(self.labels)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type=None,
        target_input_type=None,
    ) -> torch.Tensor:
        # WorkRB hands us a list of target labels to score against. Compute
        # the full softmax over the model's label space, then gather the
        # columns corresponding to the requested targets.
        logits = self.classifier_logits(texts)
        probs = F.softmax(logits, dim=-1)

        # Map each requested target to its column index. Targets not in the
        # label space get a score of 0 (a clear signal of "I never saw this").
        label_to_idx = {label: i for i, label in enumerate(self.labels)}
        cols = []
        for t in targets:
            if t in label_to_idx:
                cols.append(probs[:, label_to_idx[t]])
            else:
                cols.append(torch.zeros(probs.size(0), device=probs.device))
        return torch.stack(cols, dim=1)

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type=None,
        target_input_type=None,
    ) -> torch.Tensor:
        # For ranking tasks we reuse classification: score the queries
        # against the requested targets in the fixed label space.
        return self._compute_classification(queries, targets)

    @property
    def name(self) -> str:
        return self._leaderboard_name

    @property
    def description(self) -> str:
        return self._leaderboard_description


# ============================================================================
# The matching loss
# ============================================================================


class ClassifierCELoss(nn.Module):
    """Vanilla cross entropy over the model's label space.

    Reads ``model.classifier_logits(batch.texts)`` and ``batch.labels``.
    Requires the dataset + collate to produce ``LabeledBatch`` (see above)
    or any object with ``texts: list[str]`` and ``labels: LongTensor``.
    """

    def forward(self, model: FrozenClassifierModel, batch: LabeledBatch) -> torch.Tensor:
        logits = model.classifier_logits(batch.texts)
        return F.cross_entropy(logits, batch.labels.to(logits.device))


# ============================================================================
# How to wire it up
# ============================================================================
#
# This swap is bigger than the cross-encoder one because the batch shape
# changes. You'll need to:
#
#   1. Build a label index from your dataset (e.g. all unique skills).
#      Pass it as ``init["labels"]`` on the model.
#   2. Write or wrap a dataset whose ``__getitem__`` returns
#      ``(text, label_idx)`` instead of ``(text, label_text)``.
#   3. Point ``DataConfig.collate`` at ``labeled_collate``.
#
# Example ``TrainConfig`` block (assuming you already built ``LABELS``,
# a list of label strings, and ``MyLabeledDataset`` somewhere in
# ``participant/data.py``):
#
#     model=ModelConfig(
#         target="participant.examples.classifier_head:FrozenClassifierModel",
#         init={"labels": LABELS, "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
#     ),
#     data=DataConfig(
#         dataset=TargetConfig(
#             target="participant.data:MyLabeledDataset",
#             init={"label_to_idx": {l: i for i, l in enumerate(LABELS)}},
#         ),
#         sampler=SamplerConfig(target="participant.sampler:RandomBatchSampler"),
#         collate="participant.examples.classifier_head:labeled_collate",
#         batch_size=64,
#     ),
#     loss=LossConfig(
#         target="participant.examples.classifier_head:ClassifierCELoss",
#         init={},
#     ),
#
# ============================================================================


if __name__ == "__main__":
    # Dummy smoke test: tiny label space, 4 random rows.
    LABELS = ["python", "java", "spanish", "project management"]
    model = FrozenClassifierModel(labels=LABELS)
    loss_fn = ClassifierCELoss()

    batch = LabeledBatch(
        texts=[
            "writes python services",
            "responsible for sprint planning",
            "fluent in spanish",
            "experience with the java ecosystem",
        ],
        labels=torch.tensor([0, 3, 2, 1], dtype=torch.long),
    )
    loss = loss_fn(model, batch)
    print(f"ClassifierCELoss on a 4-row dummy batch = {loss.item():.4f}")
