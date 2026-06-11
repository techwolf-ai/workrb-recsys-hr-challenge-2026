"""Your data: where training pairs come from, and what a batch looks like.

This file owns three things:

  1. ``Batch``             the dataclass your loss reads. Add fields here if
                           your loss needs more than (queries, targets).
  2. ``SkillSentenceDataset`` a torch ``Dataset`` that wraps one or more
                           HuggingFace datasets and yields one positive pair
                           per index.
  3. ``default_collate``   how a list of ``__getitem__`` results becomes a
                           ``Batch``.

The training framework in ``src/workrb_challenge/`` knows none of this.
It only knows that ``DataConfig`` will hand back a torch DataLoader that
yields *something* the loss can consume. The "something" is whatever you
define here.

The default flow
----------------

The shipped pipeline is:

    SkillSentenceDataset.__getitem__(i)  ->  (sentence, skill)
    default_collate(list of pairs)       ->  Batch(queries=..., targets=...)
    InfoNCELoss(model, batch)            ->  scalar

Each row is one positive pair. With symmetric in-batch InfoNCE, every
other row in the same batch is treated as a negative for both sides.

What you typically change
-------------------------

* **Different dataset**: pass another HF dataset name to
  ``SkillSentenceDataset(dataset_names=[...])`` from ``participant/train.py``.
  Anything with a ``sentence`` and ``skill`` column works out of the box.
* **Mix multiple datasets**: pass a list. They get concatenated.
* **Filter rows** (language, skill type, sentence length, ...): wrap or
  subclass ``SkillSentenceDataset`` and add the filter in ``__post_init__``
  before storing ``self._data``.
* **Richer batches** (e.g. hard negatives appended as extra columns,
  teacher logits for distillation, class labels for a classifier head):
  add fields to ``Batch``, return them from ``__getitem__``, pack them in
  ``default_collate``. Your loss is the only consumer; the framework never
  inspects the shape.

Source dataset
--------------

Default: `TechWolf/Synthetic-ESCO-skill-sentences`_, 138K rows, one
``train`` split, two columns (``sentence`` and ``skill``). Covers about
99.5% of ESCO v1.1.0 skills.

.. _TechWolf/Synthetic-ESCO-skill-sentences: https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset


# Project-local HuggingFace cache. Lives under data/ (gitignored), so the
# first run downloads from the Hub and every subsequent run reads from disk.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data"

# Default HF dataset(s). Add more names to the list in ``participant/train.py``
# to mix sources. Each must expose ``sentence`` and ``skill`` columns.
DEFAULT_DATASETS: tuple[str, ...] = ("TechWolf/Synthetic-ESCO-skill-sentences",)

# Column names the dataset must expose. If your source uses different names,
# rename via ``ds.rename_columns(...)`` inside ``__post_init__`` below, or
# subclass and override.
SENTENCE_COLUMN = "sentence"
SKILL_COLUMN = "skill"


# ============================================================================
# >>> SWAP: batch shape <<<
# ----------------------------------------------------------------------------
# The contract between the dataset, the collate function, and the loss.
# The framework treats ``Batch`` as opaque: it hands the object to the loss
# unchanged. So if you add a field here, also produce it in ``default_collate``
# (or your own collate) and read it from your loss.
#
# Common extensions:
#
#   * hard negatives as extra columns
#         queries: list[str]
#         targets: list[str]
#         extra_negatives: list[list[str]]   # per-anchor neighbour skills
#
#   * classifier head (label space is fixed in the model)
#         texts: list[str]
#         labels: torch.LongTensor          # shape (B,)
#
#   * distillation (teacher precomputed)
#         queries: list[str]
#         targets: list[str]
#         teacher_logits: torch.Tensor      # shape (B, B)
# ============================================================================


@dataclass
class Batch:
    """One mini-batch passed to your loss.

    The default loss (``participant/loss.py``) reads ``queries`` and
    ``targets``. Add fields as needed. Keep field names in sync with both
    your collate and your loss.
    """

    queries: list[str]
    targets: list[str]


# ============================================================================
# >>> SWAP: dataset <<<
# ----------------------------------------------------------------------------
# A torch ``Dataset`` that yields one positive (sentence, skill) pair per
# index. The default wraps any HF dataset with ``sentence``/``skill`` columns.
#
# To plug in your own data source:
#
#   * If it's an HF dataset with the same two columns: just pass its name
#     via ``DataConfig`` in ``participant/train.py``.
#   * If it's an HF dataset with different column names: rename inside
#     ``__post_init__`` before the missing-columns check.
#   * If it's something exotic (parquet on disk, a SQL query, a streaming
#     iterator): replace this class entirely. The only contract is
#     ``__len__`` + ``__getitem__`` returning whatever your collate expects.
# ============================================================================


@dataclass
class SkillSentenceDataset(Dataset):
    """Wraps one or more HuggingFace datasets and yields (sentence, skill) pairs.

    Parameters
    ----------
    dataset_names:
        HF dataset identifiers. All are concatenated into one virtual dataset.
    split:
        Which split to pull from each ``dataset_names`` entry.
    cache_dir:
        Where the HuggingFace ``datasets`` library caches downloads. Re-runs
        with the same ``cache_dir`` read from disk instead of re-fetching.
    """

    dataset_names: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    split: str = "train"
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)

    def __post_init__(self):
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        parts = []
        for name in self.dataset_names:
            ds = load_dataset(name, split=self.split, cache_dir=str(cache_dir))

            # Required-column check. Adapt here if you need to rename
            # incoming columns to match SENTENCE_COLUMN / SKILL_COLUMN.
            present = set(ds.column_names)
            missing = {SENTENCE_COLUMN, SKILL_COLUMN} - present
            if missing:
                raise ValueError(
                    f"Dataset {name!r} is missing required column(s) {sorted(missing)}. "
                    f"Found columns: {ds.column_names}"
                )

            # Drop everything except the two columns we care about, so
            # concatenation across heterogeneous datasets works.
            keep = [SENTENCE_COLUMN, SKILL_COLUMN]
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
            parts.append(ds)

        self._data = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        row = self._data[int(idx)]
        return row[SENTENCE_COLUMN], row[SKILL_COLUMN]


# ============================================================================
# >>> SWAP: collate <<<
# ----------------------------------------------------------------------------
# Receives a list of ``__getitem__`` results and returns a ``Batch``. Replace
# this function (and point ``DataConfig.collate`` at the new dotted path)
# whenever you change ``Batch`` or ``__getitem__``.
# ============================================================================


def default_collate(rows: list[tuple[str, str]]) -> Batch:
    """Turn ``[(sent_0, skill_0), (sent_1, skill_1), ...]`` into one ``Batch``."""
    sentences, skills = zip(*rows, strict=True)
    return Batch(queries=list(sentences), targets=list(skills))
