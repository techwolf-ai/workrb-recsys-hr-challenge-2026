"""Alt architecture: cross-encoder + pairwise hinge loss.

Why this example exists
-----------------------

The default ``MyModel`` is a bi-encoder: it encodes the query and the target
independently, then takes their cosine similarity. That makes inference
fast (you can pre-encode all targets once and reuse), but it can be weaker
than a cross-encoder when the query and target interact in subtle ways.

A cross-encoder concatenates query and target into a single text input
and runs the backbone over the pair. The model can then attend across the
two sides at every layer, which is more expressive but more expensive at
inference (you re-encode every (query, target) pair).

This file is **self-contained**. The model class lives here, the matching
loss class lives here, and the comment block at the bottom shows the
exact ``TrainConfig`` snippet to wire it up from ``participant/train.py``.
Nothing in ``participant/my_model.py`` or the rest of the participant
files needs to change.

Quick smoke test::

    uv run python -m participant.examples.cross_encoder

This runs a 4-row dummy batch through the model + loss and prints the loss
value, so you know it builds end-to-end before you try a real training run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Put the repo root + ``src/`` on the import path so ``workrb_challenge``
# resolves even when the editable install is stale (the trap the other entry
# points guard against). Safe here because ``-m participant.examples.*`` already
# has the repo root importable, so this module import resolves. Idempotent.
import participant._bootstrap  # noqa: F401  (import for side effect)

# WorkRB contract (the leaderboard scores through ModelInterface) +
# challenge save/load mixin.
from workrb.models import ModelInterface
from workrb_challenge.models import WorkrbSaveable


# ============================================================================
# The model
# ============================================================================


class CrossEncoderModel(nn.Module, ModelInterface, WorkrbSaveable):
    """Score (query, target) pairs jointly through one HF backbone.

    Inference path (called by WorkRB):
        For every (q, t) in the Nq x Nt grid, build "[CLS] q [SEP] t [SEP]",
        run the backbone, take the [CLS] embedding, project it to a single
        relevance scalar with a linear head.

    Training path (called by ``PairwiseHingeLoss``):
        ``score_pairs(queries, targets) -> Tensor of shape (B,)``. The loss
        invokes this twice, once for positives and once for negatives.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 128,
        leaderboard_name: str = "CrossEncoder-example",
        leaderboard_description: str = "Cross-encoder + pairwise hinge loss example.",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        # One linear head turning [CLS] into a single relevance scalar.
        self.head = nn.Linear(hidden, 1)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _score(self, queries: list[str], targets: list[str]) -> torch.Tensor:
        """Score a flat list of (q, t) pairs. Returns shape (len(queries),)."""
        assert len(queries) == len(targets)
        inputs = self.tokenizer(
            queries,
            targets,                          # second-text arg = pair encoding
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        # [CLS] is position 0; take its embedding for each row in the batch.
        cls = outputs.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)

    # ----- training surface (read by the loss) ------------------------------

    def score_pairs(self, queries: list[str], targets: list[str]) -> torch.Tensor:
        """The loss reads this. One scalar per (query, target) pair."""
        return self._score(queries, targets)

    # ----- inference surface (read by WorkRB) -------------------------------

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type=None,
        target_input_type=None,
    ) -> torch.Tensor:
        # WorkRB asks for a (Nq, Nt) score matrix. We score every pair.
        # This is O(Nq * Nt) backbone forwards, which is the trade-off you
        # accept when you choose a cross-encoder.
        Nq, Nt = len(queries), len(targets)
        # Build the full pair list with each query repeated Nt times and
        # the target list tiled Nq times, so positions line up.
        q_flat = [q for q in queries for _ in range(Nt)]
        t_flat = targets * Nq
        scores = self._score(q_flat, t_flat)
        return scores.view(Nq, Nt)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type=None,
        target_input_type=None,
    ) -> torch.Tensor:
        # For label-space scoring we reuse the pair scorer with each label
        # as a target. Same shape semantics as a bi-encoder.
        return self._compute_rankings(texts, targets)

    @property
    def name(self) -> str:
        return self._leaderboard_name

    @property
    def description(self) -> str:
        return self._leaderboard_description

    @property
    def classification_label_space(self) -> list[str] | None:
        return None


# ============================================================================
# The matching loss
# ============================================================================


class PairwiseHingeLoss(nn.Module):
    """Hinge loss over (positive, negative) pairs.

    For each row in the batch, the in-batch contrastive trick gives us one
    positive (the row's own target) and B-1 candidate negatives (every other
    row's target). We average a margin-based hinge over all of them.

    The loss expects ``batch.queries`` and ``batch.targets`` and reads
    ``model.score_pairs(...)``. The default ``Batch`` and ``default_collate``
    in ``participant/data.py`` already produce both fields, so no other
    participant file needs to change.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, model: CrossEncoderModel, batch) -> torch.Tensor:
        B = len(batch.queries)
        # Score the B positives in one backbone pass.
        pos_scores = model.score_pairs(batch.queries, batch.targets)  # (B,)

        # Score the B*(B-1) negative pairs (each query x every other target).
        q_flat, t_flat = [], []
        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                q_flat.append(batch.queries[i])
                t_flat.append(batch.targets[j])
        neg_scores = model.score_pairs(q_flat, t_flat).view(B, B - 1)

        # Hinge: want pos > neg + margin. Average the loss over all
        # (positive, negative) pairs in the batch.
        margin_violation = F.relu(self.margin - pos_scores.unsqueeze(-1) + neg_scores)
        return margin_violation.mean()


# ============================================================================
# How to wire it up
# ============================================================================
#
# In ``participant/train.py``, replace the model + loss blocks with:
#
#     model=ModelConfig(
#         target="participant.examples.cross_encoder:CrossEncoderModel",
#         init={
#             "model_name": "sentence-transformers/all-MiniLM-L6-v2",
#             "max_length": 128,
#             "leaderboard_name": "CrossEncoder-baseline",
#         },
#     ),
#     loss=LossConfig(
#         target="participant.examples.cross_encoder:PairwiseHingeLoss",
#         init={"margin": 0.2},
#     ),
#
# Everything else (data, sampler, optimizer, schedule, callbacks) stays the
# same because the cross-encoder reads ``batch.queries`` and ``batch.targets``
# just like the bi-encoder does.
#
# ============================================================================


if __name__ == "__main__":
    # Build the model + loss and run a single dummy batch through them.
    # If you see a scalar print, the wiring is fine; you can switch
    # ``train.py`` over to this example with confidence.
    model = CrossEncoderModel()
    loss_fn = PairwiseHingeLoss(margin=0.2)

    from participant.data import Batch
    batch = Batch(
        queries=[
            "the candidate must be able to write SQL queries",
            "responsible for managing project schedules",
            "deep knowledge of distributed systems is required",
            "should be familiar with English and Spanish",
        ],
        targets=["SQL", "project management", "distributed computing", "Spanish"],
    )
    loss = loss_fn(model, batch)
    print(f"PairwiseHingeLoss on a 4-row dummy batch = {loss.item():.4f}")
