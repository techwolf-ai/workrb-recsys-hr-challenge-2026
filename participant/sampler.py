"""Your batch sampler: which examples land in the same batch.

The sampler is the single biggest knob for contrastive learning. The dataset
yields one positive pair per index. The sampler picks which indices co-occur
in a batch, which decides which examples become *negatives* for each other
when you run InfoNCE (or any in-batch contrastive loss).

The shipped default, ``RandomBatchSampler``, is uniform random: it shuffles
the indices and yields them. That gives you a working baseline but mostly
easy negatives. Almost any improvement strategy you read about (hard
negatives, class-balanced batches, curriculum, cluster-batching) lives here.

What a sampler is, mechanically
-------------------------------

It's a ``torch.utils.data.Sampler[int]`` that yields integer indices into
your dataset. The DataLoader chunks the stream into batches of ``batch_size``.
With ``drop_last=True`` (the default in the framework) you always get
full-size batches, which keeps InfoNCE's negative count constant.

The dataset is index-addressable
--------------------------------

The framework instantiates your sampler with one positional argument:
``num_samples = len(dataset)``. So a custom sampler signature is:

    def __init__(self, num_samples: int, ..., **kwargs): ...

Any other kwargs come from ``SamplerConfig(init=...)`` in
``participant/train.py``.

Hard negatives: pick one home
-----------------------------

There are two places hard negatives can live:

  1. **In this file (recommended).** Pre-mine k nearest-neighbour skills per
     anchor, then build a sampler that always co-locates an anchor and its
     k neighbours in the same batch. InfoNCE picks them up automatically.
     One swap point.

  2. **In ``participant/loss.py`` (more flexible).** Pass an
     ``extra_negatives`` field on the batch, encode them in the loss, and
     concatenate as extra columns of the score matrix. Touches ``data.py``
     (Batch + collate) *and* ``loss.py``.

Default to (1) unless you have a specific reason to break the batch shape.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


# ============================================================================
# >>> SWAP: batch sampler <<<
# ----------------------------------------------------------------------------
# Replace this class (or write a new ``Sampler``) to control which dataset
# indices land in the same batch. Examples:
#
#   * hard negatives:    for each anchor, pre-mine its k nearest-neighbour
#                        skills (offline) and force them into the same batch
#   * class-balanced:    one row per skill before shuffling
#   * curriculum:        easy -> hard ordering by sentence/skill similarity
#   * cluster-batched:   group by an ESCO sub-tree so all in-batch negatives
#                        share semantic structure with the positive
#
# Anything subclassing ``torch.utils.data.Sampler[int]`` plugs in by pointing
# ``SamplerConfig.target`` at it from ``participant/train.py``.
# ============================================================================


class RandomBatchSampler(Sampler[int]):
    """Uniform-random index sampler. Skill-identity-blind.

    With in-batch InfoNCE this means most negatives are easy: a randomly
    sampled skill rarely looks anything like the positive. Good as a
    baseline, weak as a final recipe.
    """

    def __init__(self, num_samples: int, shuffle: bool = True, seed: int = 0):
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Hook the training loop can use to reseed each epoch."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Seed-per-epoch so a re-shuffle is reproducible across runs and
            # different across epochs.
            generator = torch.Generator()
            generator.manual_seed(self.seed + self._epoch)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = list(range(self.num_samples))
        self._epoch += 1
        yield from indices

    def __len__(self) -> int:
        return self.num_samples
