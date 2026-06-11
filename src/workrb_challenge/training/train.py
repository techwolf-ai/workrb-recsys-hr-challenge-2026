"""Training loop. Short, architecture-agnostic, loss-agnostic.

What it knows
-------------

Exactly three abstractions:

  1. **Model**: an ``nn.Module`` that (for WorkRB scoring to work) also
     implements ``workrb.models.ModelInterface``. The training loop only
     calls torch-side things on it (``.to(device)``, ``.train()``,
     ``.eval()``); the WorkRB methods are read by the validation callback
     and ``test.py``, never by the training loop.
  2. **Loss**: any callable that satisfies ``loss(model, batch) -> Tensor``.
     The loop does not inspect ``batch``.
  3. **DataLoader**: any iterable of batches. The default
     ``DataConfig.build()`` produces one, but the loop accepts any iterable.

That's it. Models can be bi-encoders, cross-encoders, or classifier heads.
Losses can be InfoNCE, triplet, distillation, or CE. The batch can carry
extra fields. The loop doesn't change.

How to run
----------

::

    uv run python participant/train.py

Or programmatically::

    from workrb_challenge.training import TrainConfig, train
    train(TrainConfig(epochs=3))

What happens
------------

1. Build model, loss, dataloader, optimizer from the config.
2. Resolve the output directory (``data/runs/{model.name}/{timestamp}/``)
   and snapshot the resolved config to ``config.json`` inside it.
3. For each epoch, for each batch: forward, backward, step, then fire all
   the callbacks. Any callback can set ``state.should_stop = True`` to
   stop training mid-epoch (early stopping, sigterm handlers, etc).
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from pathlib import Path

import torch

from workrb_challenge.training.callbacks import (
    Callback,
    LossLogger,
    ModelCheckpoint,
    TrainerState,
)
from workrb_challenge.training.config import (
    DEFAULT_RUNS_ROOT,
    TrainConfig,
    snapshot_config,
)

logger = logging.getLogger(__name__)


def default_callbacks(output_dir: str | Path, log_every: int = 50) -> list[Callback]:
    """Sensible default callbacks: log loss + save every epoch + save final."""
    return [
        LossLogger(log_every=log_every),
        ModelCheckpoint(output_dir=output_dir, every_epochs=1, save_last=True),
    ]


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(name: str) -> str:
    return _SAFE_NAME.sub("-", name).strip("-") or "model"


def _resolve_output_dir(config: TrainConfig, model_name: str) -> Path:
    """Pick the run folder. Honor explicit override; otherwise auto-version."""
    if config.output_dir is not None:
        return Path(config.output_dir)
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_RUNS_ROOT / _slug(model_name) / ts


def _update_latest_symlink(run_dir: Path) -> None:
    """Point ``{run_dir.parent}/latest`` at this run. Best-effort: skip on failure."""
    link = run_dir.parent / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
    except OSError as e:
        logger.warning("Could not update 'latest' symlink at %s: %s", link, e)


def train(config: TrainConfig | None = None) -> Path:
    """Run training and return the output directory.

    Builds model / loss / data / optimizer from ``config``. The loop itself
    is architecture- and loss-agnostic: it calls ``loss_fn(model, batch)``
    and trusts that the loss knows how to read the model.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = _resolve_device()
    logger.info("Device: %s", device)

    # --- build everything from config ---------------------------------------
    model = config.model.build().to(device)
    loss_fn = config.loss.build().to(device)
    loader = config.data.build(seed=config.seed)
    optimizer = config.optim.build(model)

    # --- resolve output dir from model.name + freeze a config snapshot -----
    output_dir = _resolve_output_dir(config, model.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = output_dir  # so callbacks reading it see the resolved path
    snapshot_config(config, output_dir)
    _update_latest_symlink(output_dir)
    logger.info("Run folder: %s", output_dir)

    callbacks = (
        config.callbacks
        if config.callbacks is not None
        else default_callbacks(output_dir, log_every=config.log_every)
    )
    state = TrainerState(model=model, optimizer=optimizer, run_dir=output_dir)

    # --- loop ----------------------------------------------------------------
    for cb in callbacks:
        cb.on_train_begin(state)

    model.train()
    for epoch in range(config.epochs):
        state.epoch = epoch
        for batch in loader:
            loss = loss_fn(model, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state.step += 1
            state.loss = loss.item()
            for cb in callbacks:
                cb.on_step_end(state)
            if state.should_stop:
                break

        for cb in callbacks:
            cb.on_epoch_end(state)
        if state.should_stop:
            break

    for cb in callbacks:
        cb.on_train_end(state)

    return Path(config.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
