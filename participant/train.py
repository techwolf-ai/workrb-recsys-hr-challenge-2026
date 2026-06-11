"""Training entry point. Read this once, then change one line at a time.

Run it::

    uv run python participant/train.py

Or override a few knobs from the command line without editing the file,
which is what makes shell-loop sweeps practical::

    uv run python participant/train.py --set optim.learning_rate=1e-4
    uv run python participant/train.py --lr 5e-5 --batch-size 128 --epochs 3
    uv run python participant/train.py --help     # full list of --set targets

``--set dotted.key=value`` reaches any field on the ``TrainConfig`` below,
including a component's ``init`` kwargs (``--set loss.init.temperature=0.07``,
``--set model.init.max_length=256``). Every override is logged before training
and lands in the run's ``config.json`` snapshot, so a swept run stays
reproducible. With no CLI args the recipe runs exactly as written.

What this file is
-----------------

A single ``TrainConfig(...)`` literal that names every knob the training
loop will read. Nothing is hidden. Every component (model, data, sampler,
loss) is selected by a dotted-path ``target`` string plus an ``init`` dict.
Swapping any component is one ``target`` change here. The framework code in
``src/workrb_challenge/`` never needs to be opened.

How the files relate
--------------------

``train.py`` is the recipe; it points at the participant-owned components::

    participant/my_model.py    architecture + inference  (ModelConfig.target)
    participant/data.py        dataset + Batch + collate (DataConfig.dataset / collate)
    participant/sampler.py     which examples co-occur in a batch (SamplerConfig.target)
    participant/loss.py        the training objective              (LossConfig.target)
    participant/validate.py    WorkRB validation + wandb during training
    participant/test.py        WorkRB evaluation matrix after training

Each one of those files is editable. The framework in
``src/workrb_challenge/`` is the loop, the callbacks, the config dataclasses,
the model base class, and the loggers. You do not need to edit anything
under ``src/`` to do research.

Defaults shipped
----------------

* Model      ``MyModel``: tied bi-encoder, mean-pool, cosine similarity
* Backbone   ``sentence-transformers/paraphrase-mpnet-base-v2``
* Dataset    ``TechWolf/Synthetic-ESCO-skill-sentences`` (138K pairs)
* Sampler    ``RandomBatchSampler`` (uniform random)
* Loss       symmetric in-batch InfoNCE @ temperature=0.05
* Optimizer  AdamW, lr=2e-5, weight_decay=0.01
* Schedule   1 epoch, seed=0
* Logging    loss every 50 steps, metrics to the local console by default
* Validation WorkRB every 500 steps + at epoch end (macro ndcg@100)
* Stopping   early stop on ``ndcg@100_macro`` (patience 2), keep best checkpoint
* Checkpoints into ``data/runs/{model.name}/{timestamp}/``

Where each knob lives
---------------------

================================ ==============================================
What you want to change          File to edit
================================ ==============================================
Architecture, inference scoring  ``participant/my_model.py``
Loss objective                   ``participant/loss.py``  (or add another class)
Batch sampler / hard negatives   ``participant/sampler.py``
Dataset, columns, filtering      ``participant/data.py``
Batch shape (extra fields)       ``participant/data.py``  (the ``Batch`` dataclass)
Validation tasks / metrics       ``participant/validate.py``
Test tasks / baselines           ``participant/test.py``
Training hyperparameters         this file (``TrainConfig(...)`` below)
Run folder layout                this file (``output_dir=...``)
================================ ==============================================
"""

from __future__ import annotations

# Make ``participant.*`` and ``workrb_challenge.*`` importable even when the
# editable install is stale and even when this file is run as a path
# (``python participant/train.py``), which only puts ``participant/`` on the
# path, not the repo root. Must run before any project import below. See
# ``participant/_bootstrap.py`` for the why.
import sys as _sys
from pathlib import Path as _Path

for _p in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1] / "src"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging

from participant.validate import callback as validation_callback
from workrb_challenge.training import (
    DataConfig,
    EarlyStopping,
    LossConfig,
    LossLogger,
    ModelCheckpoint,
    ModelConfig,
    OptimConfig,
    SamplerConfig,
    TargetConfig,
    TrainConfig,
    apply_cli_overrides,
    train,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    config = TrainConfig(
        # ----- model ---------------------------------------------------------
        # ``target`` resolves a class; ``init`` is the kwargs dict passed
        # to it. Point ``target`` at any class that combines ``nn.Module``
        # with WorkRB's ``workrb.models.ModelInterface`` (and, if you want
        # the project's save/load helper, ``WorkrbSaveable``). See
        # ``participant/my_model.py`` for the contract, and
        # ``participant/examples/`` for alternative architectures.
        model=ModelConfig(
            target="participant.my_model:MyModel",
            init={
                "model_name": "sentence-transformers/paraphrase-mpnet-base-v2",
                "max_length": 128,
                "leaderboard_name": "MyModel-baseline",
                "leaderboard_description": "Tied bi-encoder + cosine + symmetric InfoNCE.",
            },
        ),

        # ----- data ----------------------------------------------------------
        # ``dataset`` and ``collate`` resolve to participant-side code, so the
        # framework never needs to know your dataset shape. Mix multiple HF
        # datasets by passing a list to ``init["dataset_names"]``; each one
        # must expose ``sentence`` and ``skill`` columns. The sampler decides
        # which examples co-occur in a batch (see
        # ``participant/sampler.py``).
        data=DataConfig(
            dataset=TargetConfig(
                target="participant.data:SkillSentenceDataset",
                init={
                    "dataset_names": ["TechWolf/Synthetic-ESCO-skill-sentences"],
                    "split": "train",
                },
            ),
            sampler=SamplerConfig(
                target="participant.sampler:RandomBatchSampler",
                init={"shuffle": True},
            ),
            collate="participant.data:default_collate",
            batch_size=64,
            num_workers=0,
        ),

        # ----- loss ----------------------------------------------------------
        # Any class implementing ``forward(model, batch) -> Tensor``. Pair it
        # with a model that exposes the methods the loss reads.
        loss=LossConfig(
            target="participant.loss:InfoNCELoss",
            init={"temperature": 0.05},
        ),

        # ----- optimizer -----------------------------------------------------
        optim=OptimConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
        ),

        # ----- schedule ------------------------------------------------------
        epochs=1,
        seed=0,
        log_every=50,

        # ``output_dir=None`` auto-resolves to
        # ``data/runs/{model.name}/{YYYY-MM-DD_HH-MM-SS}/``. Override with a
        # concrete path to write somewhere specific.
        output_dir=None,

        # ----- callbacks -----------------------------------------------------
        # The training loop calls each callback at lifecycle events
        # (begin / step / epoch / end). Everything below is optional; pass
        # ``callbacks=None`` to fall back to the framework's defaults.
        #
        #   ``LossLogger``           prints the running training loss.
        #   ``validation_callback``  runs WorkRB and logs the metrics (console
        #                            by default; see ``participant/validate.py``).
        #                            It writes ``ndcg@100_macro`` into
        #                            ``state.metrics`` for the callbacks below.
        #   ``ModelCheckpoint``      writes checkpoints into the run folder and,
        #                            with ``monitor`` set, keeps a ``best/`` copy
        #                            of the highest macro-ndcg checkpoint.
        #   ``EarlyStopping``        stops training once macro ndcg@100 stops
        #                            improving across validation evaluations.
        #
        # Order matters: ``validation_callback`` must run before the checkpoint
        # and early-stopping callbacks so the metric is fresh when they read it.
        callbacks=[
            LossLogger(log_every=50),
            validation_callback,
            ModelCheckpoint(
                every_epochs=1,
                save_last=True,
                monitor="ndcg@100_macro",
                mode="max",
            ),
            EarlyStopping(
                monitor="ndcg@100_macro",
                mode="max",
                patience=2,
                min_delta=0.0,
            ),
        ],
    )

    # Apply any ``--set dotted.key=value`` (or shortcut flags like
    # ``--lr``, ``--epochs``, ``--batch-size``) from the command line on top of
    # the literal above, then run. With no CLI args this is a no-op and the
    # recipe runs exactly as written. Lets you sweep without editing the file::
    #
    #     uv run python participant/train.py --set optim.learning_rate=1e-4
    #     for lr in 1e-5 2e-5 5e-5; do
    #         uv run python participant/train.py --lr $lr --seed 1
    #     done
    config = apply_cli_overrides(config)

    train(config)


if __name__ == "__main__":
    main()
