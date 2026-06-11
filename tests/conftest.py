"""Shared fixtures for the participant sanity-check suite.

The whole point of this suite is that it runs offline, in seconds, after
every edit. The real baseline downloads ``paraphrase-mpnet-base-v2`` (~400 MB)
from the Hugging Face Hub the moment you construct ``MyModel``. We do not want
that in a smoke test: it would make the suite slow and network-bound, and you
would stop running it.

So this file installs a **tiny offline stub** for the two Hugging Face calls
``participant/my_model.py`` makes:

  * ``AutoTokenizer.from_pretrained`` -> a trivial whitespace tokenizer
  * ``AutoModel.from_pretrained``     -> a ~few-thousand-parameter transformer

The stub mimics only the surface the participant code touches: the tokenizer
returns ``input_ids`` / ``attention_mask`` tensors, the model returns an object
with ``.last_hidden_state``, and both expose ``save_pretrained`` so the
save/load round-trip works. The numbers are meaningless; these tests check
*shapes and contracts*, never model quality.

The ``patch_hf_backbone`` fixture is ``autouse=True``, so every test in this
directory gets the stub automatically. Nothing you write in ``participant/``
needs to know it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

HIDDEN_SIZE = 16


class _StubTokenizer:
    """Whitespace tokenizer. Maps each unique token to a small integer id.

    Returns the same dict shape Hugging Face tokenizers return for the call
    ``my_model.py`` makes: ``tokenizer(texts, padding=True, truncation=True,
    max_length=..., return_tensors="pt")`` -> ``{"input_ids", "attention_mask"}``.
    """

    def __init__(self, max_length: int = 128):
        self.max_length = max_length

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        token_lists = [t.split()[:max_length] or ["<empty>"] for t in texts]
        width = max(len(toks) for toks in token_lists)
        input_ids, attention_mask = [], []
        for toks in token_lists:
            ids = [(abs(hash(tok)) % 99) + 1 for tok in toks]
            pad = width - len(ids)
            input_ids.append(ids + [0] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "stub_tokenizer.json").write_text(json.dumps({"max_length": self.max_length}))

    @classmethod
    def from_pretrained(cls, name_or_path, *args, **kwargs):
        return cls()


class _StubOutput:
    """Stand-in for a Hugging Face model output. Only ``.last_hidden_state`` is read."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _StubModel(nn.Module):
    """A tiny embedding + linear transformer-ish module.

    Real parameters (so optimizers and ``.backward()`` behave), but only a
    few thousand of them. ``forward`` returns an object with
    ``.last_hidden_state`` of shape ``(B, T, HIDDEN_SIZE)``, matching what
    ``_mean_pool`` in ``my_model.py`` expects.
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.proj(self.embeddings(input_ids))
        return _StubOutput(hidden)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "stub_model.pt")
        (path / "config.json").write_text(json.dumps({"stub": True, "hidden_size": HIDDEN_SIZE}))

    @classmethod
    def from_pretrained(cls, name_or_path, *args, **kwargs):
        model = cls()
        weights = Path(name_or_path) / "stub_model.pt"
        if weights.exists():
            model.load_state_dict(torch.load(weights, weights_only=True))
        return model


@pytest.fixture(autouse=True)
def patch_hf_backbone(monkeypatch):
    """Swap the two Hub-downloading calls for offline stubs, suite-wide.

    Patches the names as imported in ``participant/my_model.py``
    (``from transformers import AutoModel, AutoTokenizer``), so any model the
    tests construct gets the tiny stub instead of a 400 MB download.
    """
    import participant.my_model as mm

    monkeypatch.setattr(mm, "AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr(mm, "AutoModel", _StubModel)
    yield


@pytest.fixture
def model():
    """A fresh ``MyModel`` wired to the offline stub backbone."""
    from participant.my_model import MyModel

    return MyModel(leaderboard_name="TestModel")


@pytest.fixture
def queries():
    return ["operate a sewing machine", "diagnose engine faults", "write React components"]


@pytest.fixture
def targets():
    return ["sewing", "engine repair", "front-end development", "carpentry"]
