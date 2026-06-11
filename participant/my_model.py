"""Your model. THE one place to change architecture or inference scoring.

What your model inherits from (and why)
---------------------------------------

::

    class MyModel(nn.Module, ModelInterface, WorkrbSaveable):

Three parents, each adding one capability:

  * ``torch.nn.Module``
        Lets PyTorch optimizers, ``.to(device)``, ``.train()``,
        ``.state_dict()``, and everything else just work.

  * ``workrb.models.ModelInterface``
        The contract WorkRB scores you on. It declares the methods WorkRB
        will call when it builds your leaderboard numbers. The whole
        ``ModelInterface`` class lives in the workrb library; this is
        not a wrapper we added. The full source is at
        ``.venv/lib/python3.12/site-packages/workrb/models/base.py``.

        The five things ``ModelInterface`` requires you to provide:

        ====================================  ======================================
        ``_compute_rankings(queries, targets, ...)``  Tensor of shape ``(Nq, Nt)``, higher = more relevant
        ``_compute_classification(texts, targets, ...)``  Tensor of shape ``(Nt, Nc)`` over a fixed label space
        ``name`` (property)                  Leaderboard display name
        ``description`` (property)           Leaderboard description
        ``classification_label_space``       List of labels, or ``None`` for bi-encoder-style models
        ====================================  ======================================

  * ``WorkrbSaveable``
        A challenge-side mixin (not part of WorkRB) that gives you
        ``save_pretrained(path)`` and ``WorkrbSaveable.from_pretrained(path)``.
        Override ``_save_extra`` if your model has trainable state beyond
        ``self.backbone`` / ``self.tokenizer``.

The two SWAP banners
--------------------

You change two regions in this file:

  1. ``ARCHITECTURE``  what modules exist, how they compose.
  2. ``INFERENCE``     how ``_compute_rankings`` turns text into the
                       ``(Nq, Nt)`` matrix WorkRB consumes.

The training-side methods (``encode_query``, ``encode_target`` in the
baseline) are *not* part of the WorkRB contract. They exist because the
loss in ``participant/loss.py`` reads them. Name them whatever you want,
return whatever shape you need; the loss is the only consumer.

Worked architecture examples
----------------------------

* **Tied bi-encoder** (this file): one HF backbone, encode both sides,
  cosine similarity.
* **Untied bi-encoder**: two backbones in ``__init__``, separate
  ``encode_query`` / ``encode_target`` paths.
* **Cross-encoder**: tokenize ``(q, t)`` pairs jointly, run a backbone
  with a regression head, take the scalar logit per pair. See
  ``participant/examples/cross_encoder.py``.
* **Classifier on frozen encoder**: freeze the backbone, add an
  ``nn.Linear`` head, override ``classification_label_space`` to the
  fixed label list. See ``participant/examples/classifier_head.py``.
* **Late interaction / ColBERT**: keep per-token query embeddings, mean
  pool the target, attention-weighted dot product.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# The WorkRB contract. This is the library that scores your model on the
# leaderboard; the abstract methods below are *its* abstract methods.
from workrb.models import ModelInterface
from workrb.types import ModelInputType

# Project save/load mixin. Not part of WorkRB.
from workrb_challenge.models import WorkrbSaveable


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(N, D) x (M, D)`` -> ``(N, M)`` cosine similarity matrix."""
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return a @ b.T


class MyModel(nn.Module, ModelInterface, WorkrbSaveable):
    """The starter. Tied bi-encoder with mean pooling and cosine similarity.

    Rename this class, change its modules, change its scoring. Anything is
    allowed as long as ``ModelInterface``'s five methods stay consistent.
    """

    # ==================================================================
    # >>> SWAP: ARCHITECTURE <<<
    # ------------------------------------------------------------------
    # Define your modules. The starter is a tied bi-encoder: one HF
    # transformer + tokenizer used for both query and target sides.
    #
    # Common variations:
    #   * untied bi-encoder: two backbones, one per side
    #   * cross-encoder:     one backbone over [CLS] q [SEP] t
    #   * classifier head:   frozen backbone + nn.Linear(d, num_classes)
    #   * alignment model:   two encoders + projection matrices
    # ==================================================================

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        max_length: int = 128,
        leaderboard_name: str = "MyModel",
        leaderboard_description: str = "Custom model for the workrb challenge.",
        encode_batch_size: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Tokenize, run the backbone, mean-pool one batch. Returns ``(B, D)``."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        return _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode ``texts`` in chunks of ``encode_batch_size``, then concat.

        WorkRB ranking tasks pass the entire target corpus (tens of thousands
        of sentences) in a single call. Encoding that as one forward pass
        exhausts GPU/MPS memory, so we tile it. The result is identical to a
        single pass; only the peak memory differs. During training the loss
        passes batch-sized lists, so this is a no-op there.
        """
        chunk = self.encode_batch_size
        if chunk <= 0 or len(texts) <= chunk:
            return self._encode_batch(texts)
        embeddings = [self._encode_batch(texts[i : i + chunk]) for i in range(0, len(texts), chunk)]
        return torch.cat(embeddings, dim=0)

    # ------------------------------------------------------------------
    # Training-side forwards. The loss in participant/loss.py calls these.
    # Rename / split / merge freely. The loss is the only consumer. To
    # untie the bi-encoder, give each method its own backbone here.
    # ------------------------------------------------------------------

    def encode_query(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    def encode_target(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    # ==================================================================
    # >>> SWAP: INFERENCE <<<
    # ------------------------------------------------------------------
    # WorkRB calls ``_compute_rankings`` (wrapped in torch.no_grad by
    # ModelInterface) when it scores your model. The MUST is the shape:
    # ``(num_queries, num_targets)``, higher = more relevant.
    #
    # Match this body to your architecture:
    #   * bi-encoder (default):  encode both sides, cosine
    #   * cross-encoder:         tokenize (q, t) pairs, take pair logits;
    #                            reshape to (Nq, Nt)
    #   * classifier head:       encode queries, project to logits over
    #                            label space, gather columns at indices
    #                            of `targets` in the label space
    #   * late interaction:      per-token query embs vs. mean target,
    #                            attention-weighted dot
    # ==================================================================

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        q_emb = self.encode_query(queries)
        t_emb = self.encode_target(targets)
        return _cosine_similarity(q_emb, t_emb)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        # Bi-encoder default: classification = rank texts against label texts.
        # Override this if you train a real classifier head with fixed labels.
        return self._compute_rankings(
            queries=texts,
            targets=targets,
            query_input_type=input_type,
            target_input_type=target_input_type or input_type,
        )

    # ------------------------------------------------------------------
    # WorkRB leaderboard metadata. ``name`` is what shows up.
    # ``classification_label_space`` defaults to ``None`` on
    # ``ModelInterface`` already, which is the right answer for any
    # bi-encoder-style model that scores against arbitrary label text.
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._leaderboard_name

    @property
    def description(self) -> str:
        return self._leaderboard_description

    @property
    def classification_label_space(self) -> list[str] | None:
        # Bi-encoder: any text can be a label, no fixed space.
        return None
