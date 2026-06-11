"""Training configuration: one dataclass per concern, every component swap-by-string.

The big idea: the framework should not import any concrete model, dataset,
sampler, or loss. Those are all participant-owned and live under
``participant/``. The framework only learns about them through dotted-path
strings on the config.

Each sub-config has a ``.build(...)`` method. ``train(config)`` calls them
in the right order with whatever bootstrap arguments they need (seed for the
sampler, model parameters for the optimizer). Swapping any component means
changing one ``target`` string in ``participant/train.py``; you never edit
this file to try a new model or a new loss.

The dotted-path convention
--------------------------

A target looks like ``"package.module:ClassName"``. We resolve it with
``importlib.import_module`` plus ``getattr``. Pair it with an ``init``
kwargs dict that is splatted into the class:

    cls = _resolve_target("participant.loss:InfoNCELoss")
    obj = cls(**{"temperature": 0.05})

That's it. No registry, no plugin system, no decorators.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Where run artifacts go when ``TrainConfig.output_dir`` is left as None.
# The training loop appends ``{model.name}/{timestamp}/`` so different model
# names and different runs each get their own folder.
# __file__ = .../src/workrb_challenge/training/config.py
# parents:    0=training  1=workrb_challenge  2=src  3=repo root
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[3] / "data" / "runs"

if TYPE_CHECKING:
    import torch.nn as nn

    from workrb_challenge.training.callbacks import Callback


# ---------------------------------------------------------------------------
# Target resolution: the one mechanism that lets the framework stay ignorant
# of concrete classes.
# ---------------------------------------------------------------------------


def _resolve_target(target: str) -> Any:
    """``"package.module:ClassName"`` -> the class object.

    Also accepts module-level functions (e.g. for ``collate``).
    """
    if ":" not in target:
        raise ValueError(
            f"target must be of the form 'module.path:Name', got {target!r}"
        )
    module_name, _, attr_name = target.partition(":")
    return getattr(import_module(module_name), attr_name)


@dataclass
class TargetConfig:
    """Generic ``{target, init}`` pair. Used for model, loss, dataset, sampler.

    Build the configured object by calling ``.build(**extra_kwargs)``. Any
    extra kwargs are merged into ``init`` at build time so the framework can
    inject things the participant should not have to know about (e.g.
    ``num_samples`` for a sampler, ``seed`` for shuffling).
    """

    target: str
    init: dict[str, Any] = field(default_factory=dict)

    def build(self, **extra: Any) -> Any:
        cls = _resolve_target(self.target)
        kwargs = {**self.init, **extra}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Model config: which class, which init kwargs. That's all the framework
# needs to spin up your model.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig(TargetConfig):
    """Which model to instantiate.

    Default points at the bi-encoder baseline in ``participant/my_model.py``.
    Swap by changing ``target`` to any class that (a) is an ``nn.Module``
    (so training works) and (b) implements WorkRB's ``ModelInterface`` (so
    validation/test/leaderboard scoring works). The framework itself only
    relies on the ``nn.Module`` half; ``ModelInterface`` is what
    ``workrb.evaluate(...)`` reads in ``validate.py`` and ``test.py``.
    """

    target: str = "participant.my_model:MyModel"

    def build(self, **extra: Any) -> "nn.Module":
        return super().build(**extra)


# ---------------------------------------------------------------------------
# Loss config: same {target, init} shape. The loss only needs to implement
# ``forward(model, batch) -> Tensor``.
# ---------------------------------------------------------------------------


@dataclass
class LossConfig(TargetConfig):
    """Which loss to instantiate.

    Default is symmetric in-batch InfoNCE. Pair the loss with a model that
    exposes the methods it reads. See ``participant/loss.py`` for the
    contract and worked examples.
    """

    target: str = "participant.loss:InfoNCELoss"
    init: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.05})

    def build(self, **extra: Any) -> "nn.Module":
        return super().build(**extra)


# ---------------------------------------------------------------------------
# Sampler config: how dataset indices flow into batches. Often the single
# biggest knob for contrastive learning quality.
# ---------------------------------------------------------------------------


@dataclass
class SamplerConfig(TargetConfig):
    """Which sampler to instantiate.

    The framework injects ``num_samples=len(dataset)`` automatically when
    building, so your sampler's signature is just
    ``def __init__(self, num_samples: int, **your_kwargs)``.

    Default uniform-random sampler lives in ``participant/sampler.py``.
    """

    target: str = "participant.sampler:RandomBatchSampler"
    init: dict[str, Any] = field(default_factory=lambda: {"shuffle": True})


# ---------------------------------------------------------------------------
# Data config: dataset + sampler + collate + DataLoader knobs all bundled.
# ``build(seed)`` returns a ready-to-iterate DataLoader.
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Where training data comes from and how it is batched.

    Each piece is target-resolved. The participant edits ``participant/data.py``
    (dataset and collate) and ``participant/sampler.py`` (sampler), then
    only flips ``target`` strings here to use them.
    """

    dataset: TargetConfig = field(
        default_factory=lambda: TargetConfig(
            target="participant.data:SkillSentenceDataset",
        )
    )
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    collate: str = "participant.data:default_collate"

    batch_size: int = 64
    num_workers: int = 0
    drop_last: bool = True

    def build(self, seed: int) -> DataLoader:
        # Build the dataset first; the sampler needs its length.
        dataset = self.dataset.build()

        # The framework injects ``num_samples`` and ``seed`` so participants
        # never have to thread these through manually.
        sampler = self.sampler.build(num_samples=len(dataset), seed=seed)

        collate_fn = _resolve_target(self.collate)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=self.drop_last,
        )


# ---------------------------------------------------------------------------
# Optimizer config: AdamW with two knobs. Add a SchedulerConfig here if you
# want a built-in scheduler; for now, schedulers can be implemented as a
# callback (on_step_end -> step the LR).
# ---------------------------------------------------------------------------


@dataclass
class OptimConfig:
    """Optimizer hyperparameters. AdamW only, by design."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    def build(self, model: "nn.Module") -> torch.optim.Optimizer:
        return AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


# ---------------------------------------------------------------------------
# Top-level config: everything the training loop needs, in one object.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Everything ``train(config)`` reads, in one place.

    The training loop reads from this object exclusively; no globals, no env
    vars, no hidden defaults. To freeze "this exact recipe ran on this
    exact day," the loop also writes a JSON snapshot of this dataclass into
    the run folder (see :func:`snapshot_config`).
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    epochs: int = 1
    seed: int = 0
    log_every: int = 50

    # Run-folder for this training run. ``None`` (default) auto-resolves at
    # ``train()`` time to ``data/runs/{model.name}/{YYYY-MM-DD_HH-MM-SS}/``,
    # so different model names and different runs each get their own folder
    # and never overwrite. Set to a concrete path string to override.
    output_dir: str | Path | None = None

    # Pass any list of ``Callback`` instances. ``None`` (default) falls back
    # to the sensible defaults in
    # :func:`workrb_challenge.training.train.default_callbacks`.
    callbacks: list["Callback"] | None = None


# ---------------------------------------------------------------------------
# Config snapshot: dump the resolved TrainConfig to JSON so the run folder
# is self-documenting.
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """``dataclasses.asdict`` but path-safe and recursion-tolerant."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def snapshot_config(config: TrainConfig, output_dir: Path) -> Path:
    """Write the resolved ``TrainConfig`` to ``{output_dir}/config.json``.

    Skips the ``callbacks`` field because callbacks are live Python objects
    and not always JSON-friendly. Stamps the resolved ``output_dir`` into the
    snapshot so the file is self-contained when inspected later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _dataclass_to_dict(config)
    payload.pop("callbacks", None)
    payload["output_dir"] = str(output_dir)
    path = output_dir / "config.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
