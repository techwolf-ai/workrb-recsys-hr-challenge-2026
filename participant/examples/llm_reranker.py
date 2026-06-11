"""Alt route: bi-encoder retrieval + a local instruct LLM as re-ranker.

Why this example exists
-----------------------

The other two examples still train something. This one trains nothing.
It shows the third route through the challenge: build a better *inference
pipeline* and skip ``participant/train.py`` entirely.

The idea is the classic two-stage setup:

1. **Retrieve.** A bi-encoder scores every (query, target) pair with
   cosine similarity, exactly like the default ``MyModel``. Cheap, runs
   over the full ESCO corpus.
2. **Re-rank.** For each query, the ``top_k`` retrieved skills are
   re-scored by an instruction-tuned LLM. The LLM sees the sentence and
   one candidate skill, and we read the logit margin between "Yes" and
   "No" on the question "does this sentence demonstrate this skill?".
   The margin is blended with the retrieval score (``retriever_weight``),
   because a small LLM is a noisy judge of near-ties. Expensive per pair,
   so we only spend it where it matters.

The leaderboard metric is nDCG@100, so the ordering of the first ~100
candidates is all that counts. Re-ranking the top 50 with a stronger
scorer attacks exactly that region at a tiny fraction of the cost of
running the LLM over the whole corpus.

This composes with route 1: if you fine-tuned a bi-encoder first, point
``retriever_name`` at your checkpoint folder
(``data/runs/MyModel-baseline/latest/best``); the folder contains plain
HF backbone files, so ``AutoModel.from_pretrained`` loads it directly.

This file is **self-contained** and there is nothing to wire into
``TrainConfig``: there is no loss, no sampler, no training run. The
comment block at the bottom shows how to evaluate and submit it.

One pipeline among very many
----------------------------

Do not read this file as "the" LLM recipe. It is one point in an enormous
design space: swap the yes/no margin for listwise prompting, distill the
LLM's judgments into a small encoder, use the LLM to generate training
data instead of scores, replace the retriever with late interaction,
chain more stages, or do something nobody has published yet.
``knowledge_sharing/skill_extraction_references.md`` lists the method
families the field has tried so far; treat that list as a starting
point, not a boundary.

Quick smoke test::

    uv run python -m participant.examples.llm_reranker

This retrieves and re-ranks a 3-query dummy corpus and prints the score
matrix. The smoke run uses the 0.5B Qwen variant to keep the download
small; the default for real evaluation is the 1.5B variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Put the repo root + ``src/`` on the import path so ``workrb_challenge``
# resolves even when the editable install is stale (the trap the other entry
# points guard against). Safe here because ``-m participant.examples.*`` already
# has the repo root importable, so this module import resolves. Idempotent.
import participant._bootstrap  # noqa: F401  (import for side effect)

# WorkRB contract (the leaderboard scores through ModelInterface) +
# challenge save/load mixin.
from workrb.models import ModelInterface
from workrb_challenge.models import WorkrbSaveable

__all__ = ["LLMRerankerModel"]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class LLMRerankerModel(nn.Module, ModelInterface, WorkrbSaveable):
    """Two-stage scorer: bi-encoder retrieval, instruct-LLM re-ranking.

    Inference path (called by WorkRB):
        ``_compute_rankings`` first builds the full cosine matrix with the
        retriever, then re-scores each query's ``top_k`` candidates with
        the LLM and lifts them above the rest of the ranking.

    There is no training path. Nothing in this class has a matching loss.
    """

    def __init__(
        self,
        retriever_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        reranker_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        top_k: int = 50,
        retriever_weight: float = 1.0,
        max_length: int = 128,
        encode_batch_size: int = 256,
        rerank_batch_size: int = 16,
        leaderboard_name: str = "LLMReranker-example",
        leaderboard_description: str = "Bi-encoder retrieval + local instruct LLM re-ranking.",
    ):
        super().__init__()
        self.retriever_name = retriever_name
        self.reranker_name = reranker_name
        self.top_k = top_k
        self.retriever_weight = retriever_weight
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size
        self.rerank_batch_size = rerank_batch_size
        self._leaderboard_name = leaderboard_name
        self._leaderboard_description = leaderboard_description

        # Stage 1: the retriever. Named ``backbone``/``tokenizer`` so the
        # WorkrbSaveable defaults persist it (useful when it is a fine-tuned
        # checkpoint rather than the stock MPNet).
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.backbone = AutoModel.from_pretrained(retriever_name)

        # Stage 2: the re-ranker. Reloaded by name, never saved: it carries
        # no trained state of ours. ``torch_dtype="auto"`` keeps the LLM in
        # its checkpoint dtype (bf16 for Qwen) instead of blowing it up to
        # fp32.
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self.reranker = AutoModelForCausalLM.from_pretrained(reranker_name, torch_dtype="auto")
        # We score at the last position of each row, so shorter prompts must
        # be padded on the left to line everything up there.
        self.reranker_tokenizer.padding_side = "left"
        if self.reranker_tokenizer.pad_token is None:
            self.reranker_tokenizer.pad_token = self.reranker_tokenizer.eos_token
        self._yes_id = self.reranker_tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id = self.reranker_tokenizer.encode("No", add_special_tokens=False)[0]

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # ----- stage 1: retrieval (same shape as MyModel) ------------------------

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
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
        """Encode in ``encode_batch_size`` chunks; full ESCO does not fit in one pass."""
        chunk = self.encode_batch_size
        if chunk <= 0 or len(texts) <= chunk:
            return self._encode_batch(texts)
        embeddings = [self._encode_batch(texts[i : i + chunk]) for i in range(0, len(texts), chunk)]
        return torch.cat(embeddings, dim=0)

    # ----- stage 2: LLM scoring ----------------------------------------------

    def _build_prompt(self, query: str, target: str) -> str:
        """One chat-formatted yes/no question per (sentence, skill) pair."""
        user_msg = (
            f"Sentence: {query}\n"
            f"Skill: {target}\n\n"
            "Does the sentence demonstrate or require this skill? "
            "Answer with exactly one word: Yes or No."
        )
        return self.reranker_tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def _llm_score_pairs(self, queries: list[str], targets: list[str]) -> torch.Tensor:
        """Score a flat list of (q, t) pairs. Returns shape (len(queries),).

        The score is ``logit("Yes") - logit("No")`` at the first generated
        position: one forward pass per pair, no sampling, deterministic.
        """
        assert len(queries) == len(targets)
        prompts = [self._build_prompt(q, t) for q, t in zip(queries, targets)]
        margins: list[torch.Tensor] = []
        for i in range(0, len(prompts), self.rerank_batch_size):
            inputs = self.reranker_tokenizer(
                prompts[i : i + self.rerank_batch_size],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.reranker(**inputs).logits[:, -1, :]
            margins.append((logits[:, self._yes_id] - logits[:, self._no_id]).float())
        return torch.cat(margins, dim=0)

    # ----- inference surface (read by WorkRB) ---------------------------------

    @torch.no_grad()
    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type=None,
        target_input_type=None,
    ) -> torch.Tensor:
        # Stage 1: full cosine matrix, exactly like the bi-encoder baseline.
        q_emb = F.normalize(self._encode(queries), p=2, dim=-1)
        t_emb = F.normalize(self._encode(targets), p=2, dim=-1)
        base = q_emb @ t_emb.T  # (Nq, Nt)

        # Stage 2: re-score each query's top_k candidates with the LLM.
        k = min(self.top_k, len(targets))
        if k == 0:
            return base
        top_idx = base.topk(k, dim=1).indices  # (Nq, k)

        final = base.clone()
        for qi, query in enumerate(queries):
            candidates = [targets[ti] for ti in top_idx[qi].tolist()]
            margins = self._llm_score_pairs([query] * k, candidates)
            # Blend the LLM margin with the retrieval prior. A small LLM is
            # a noisy judge of near-ties; the cosine score carries real
            # signal, so we keep it as a weighted vote instead of letting
            # the LLM ordering overrule it outright (set retriever_weight=0
            # for pure LLM ordering).
            blended = margins + self.retriever_weight * base[qi, top_idx[qi]]
            # Lift the re-ranked block above every retriever-only score:
            # shift the blend to start at 0, then offset past the row max.
            # Below the block the retriever ordering is untouched.
            lifted = base[qi].max() + 1.0 + (blended - blended.min())
            final[qi, top_idx[qi]] = lifted.to(final.dtype)
        return final

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type=None,
        target_input_type=None,
    ) -> torch.Tensor:
        # Label-space scoring reuses the same two-stage ranking.
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

    # ----- save/load -----------------------------------------------------------

    def _save_extra(self, path) -> dict:
        # The mixin already wrote the retriever backbone + tokenizer into
        # ``path``; point the reload there. The LLM re-downloads by name.
        return {
            "retriever_name": str(path),
            "reranker_name": self.reranker_name,
            "top_k": self.top_k,
            "retriever_weight": self.retriever_weight,
            "max_length": self.max_length,
            "encode_batch_size": self.encode_batch_size,
            "rerank_batch_size": self.rerank_batch_size,
            "leaderboard_name": self._leaderboard_name,
            "leaderboard_description": self._leaderboard_description,
        }


# ============================================================================
# How to evaluate and submit it
# ============================================================================
#
# No training run, so ``participant/train.py`` is not involved. Two ways in:
#
# 1. Local matrix (``participant/test.py``): save an instance once so the
#    checkpoint loader can find it, then point ``--checkpoint`` at it:
#
#        uv run python - <<'PY'
#        from participant.examples.llm_reranker import LLMRerankerModel
#        LLMRerankerModel().save_pretrained("data/runs/LLMReranker/manual")
#        PY
#        uv run python participant/test.py --checkpoint data/runs/LLMReranker/manual
#
# 2. Leaderboard submission: this file exports exactly one ModelInterface
#    subclass, so the no-weights path of the submission script works as is:
#
#        SPLIT          = "validation"
#        MODEL_DEF_FILE = "participant/examples/llm_reranker.py"
#        WEIGHTS_PATH   = ""
#
# Knobs worth sweeping: ``top_k`` (how deep the LLM looks),
# ``retriever_weight`` (how much the cosine prior counts inside the
# re-ranked block; 0 = pure LLM ordering), the prompt in ``_build_prompt``,
# and ``reranker_name`` (any instruct LLM with a chat template works;
# bigger is slower and usually better).
#
# ============================================================================


if __name__ == "__main__":
    # Retrieval + re-ranking over a tiny dummy corpus. The 0.5B variant keeps
    # the smoke download small; evaluation quality comes from the 1.5B default.
    model = LLMRerankerModel(reranker_name="Qwen/Qwen2.5-0.5B-Instruct", top_k=2)

    queries = [
        "the candidate must be able to write SQL queries",
        "responsible for managing project schedules",
        "should be familiar with English and Spanish",
    ]
    targets = ["SQL", "project management", "distributed computing", "Spanish"]

    scores = model._compute_rankings(queries, targets)
    print(f"score matrix shape = {tuple(scores.shape)}")
    for qi, query in enumerate(queries):
        best = targets[int(scores[qi].argmax())]
        print(f"  {query!r:>55} -> {best}")
