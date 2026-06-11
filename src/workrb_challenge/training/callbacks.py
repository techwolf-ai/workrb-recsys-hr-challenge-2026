"""Training callbacks.

A callback is anything that responds to a small set of training events:

    on_train_begin(state)
    on_step_end(state)         # after each optimizer step
    on_epoch_end(state)        # after each epoch
    on_train_end(state)

The training loop in :mod:`workrb_challenge.training.train` calls these
hooks. ``state`` is a mutable :class:`TrainerState` carrying everything a
callback might need: the model, optimizer, current step/epoch, the latest
loss, and a ``metrics`` dict that callbacks read from and write to.

Stop training early by setting ``state.should_stop = True`` from any hook —
the loop checks after every step.

Built-in callbacks
------------------
* :class:`LossLogger` — prints the running training loss.
* :class:`ModelCheckpoint` — saves the model. Versioned by step or epoch,
  optionally filtered to "best so far" on a chosen metric.
* :class:`EarlyStopping` — stops training when a monitored metric stops
  improving for ``patience`` evaluations.
* :class:`Evaluator` — periodically runs a user-supplied callable that
  returns metric dict(s); merges into ``state.metrics`` so other callbacks
  (checkpoint, early stopping) can react.

Anything else is a few lines of code: subclass :class:`Callback` and override
the hooks you care about.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

_PLACEHOLDER_RUN_DIR = Path(".")

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Mutable state passed to every callback hook."""

    model: "nn.Module"              # the model being trained (a WorkRB ModelInterface in practice)
    optimizer: "torch.optim.Optimizer"
    step: int = 0
    epoch: int = 0
    loss: float = math.nan          # latest training-step loss
    metrics: dict[str, float] = field(default_factory=dict)
    # Step at which each metric was last written. Lets downstream callbacks
    # (e.g. EarlyStopping, ModelCheckpoint) tell a fresh value from a stale one.
    metric_steps: dict[str, int] = field(default_factory=dict)
    should_stop: bool = False
    # This run's output folder (data/runs/{model.name}/{ts}/). Set by train().
    # Callbacks write per-run artifacts (workrb outputs, plots, ...) here.
    run_dir: Path = field(default_factory=lambda: _PLACEHOLDER_RUN_DIR)

    def log(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)
        self.metric_steps[name] = self.step


class Callback:
    """Base class. Override the hooks you need; the rest are no-ops."""

    def on_train_begin(self, state: TrainerState) -> None: ...
    def on_step_end(self, state: TrainerState) -> None: ...
    def on_epoch_end(self, state: TrainerState) -> None: ...
    def on_train_end(self, state: TrainerState) -> None: ...


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class LossLogger(Callback):
    """Print training loss every ``log_every`` steps.

    Add additional signals by writing to ``state.metrics`` from your own
    callback (or by passing a custom ``extra_keys`` list to also surface those
    keys in the same line).
    """

    def __init__(self, log_every: int = 50, extra_keys: list[str] | None = None):
        self.log_every = log_every
        self.extra_keys = extra_keys or []

    def on_step_end(self, state: TrainerState) -> None:
        if state.step % self.log_every != 0:
            return
        parts = [f"epoch={state.epoch}", f"step={state.step}", f"loss={state.loss:.4f}"]
        for key in self.extra_keys:
            if key in state.metrics:
                parts.append(f"{key}={state.metrics[key]:.4f}")
        logger.info(" ".join(parts))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


EvalFn = Callable[[TrainerState], dict[str, float]]


class Evaluator(Callback):
    """Run an evaluation function periodically and merge its metrics into ``state``.

    ``eval_fn`` receives the current :class:`TrainerState` and must return a
    flat ``dict[str, float]``. The keys land in ``state.metrics`` so other
    callbacks (checkpoint, early stopping) can monitor them.

    Use ``every_steps`` for mid-epoch evaluation, ``every_epochs`` for end-of-
    epoch evaluation, or both. Set the value to 0 to disable that schedule.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        every_steps: int = 0,
        every_epochs: int = 1,
        run_at_end: bool = True,
    ):
        self.eval_fn = eval_fn
        self.every_steps = every_steps
        self.every_epochs = every_epochs
        self.run_at_end = run_at_end

    def _run(self, state: TrainerState) -> None:
        results = self.eval_fn(state)
        if not isinstance(results, dict):
            raise TypeError(f"eval_fn must return dict[str, float], got {type(results)!r}")
        for k, v in results.items():
            state.log(k, v)
        logger.info("eval @ step=%d: %s", state.step, results)

    def on_step_end(self, state: TrainerState) -> None:
        if self.every_steps and state.step > 0 and state.step % self.every_steps == 0:
            self._run(state)

    def on_epoch_end(self, state: TrainerState) -> None:
        if self.every_epochs and (state.epoch + 1) % self.every_epochs == 0:
            self._run(state)

    def on_train_end(self, state: TrainerState) -> None:
        # Skip the final eval when training stopped early — the metric was
        # just computed and is still fresh in ``state.metrics``.
        if self.run_at_end and not state.should_stop:
            self._run(state)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


Mode = Literal["min", "max"]


def _is_better(current: float, best: float, mode: Mode) -> bool:
    return current < best if mode == "min" else current > best


class ModelCheckpoint(Callback):
    """Save the model during training.

    Two save modes:

    - ``every_epochs > 0`` (default 1): save at the end of every Nth epoch
      under ``output_dir/epoch-{epoch}/``.
    - ``monitor`` set: also save a ``best/`` copy whenever the monitored
      metric improves. Pair with :class:`Evaluator` so ``state.metrics``
      contains the metric.

    Saves go through ``state.model.save_pretrained(path)`` — the model
    owns its own serialization. The checkpoint can then be reloaded with
    :func:`workrb_challenge.models.WorkrbSaveable.from_pretrained`.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        every_epochs: int = 1,
        monitor: str | None = None,
        mode: Mode = "max",
        save_last: bool = True,
    ):
        # ``output_dir=None`` (default) means "write into ``state.run_dir``" so
        # the participant doesn't have to know the auto-resolved run folder.
        self.output_dir: Path | None = Path(output_dir) if output_dir is not None else None
        self.every_epochs = every_epochs
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self._best: float = math.inf if mode == "min" else -math.inf

    def _save(self, state: TrainerState, subdir: str) -> Path:
        base = self.output_dir if self.output_dir is not None else state.run_dir
        path = base / subdir
        state.model.save_pretrained(path)
        logger.info("Saved checkpoint to %s", path)
        return path

    def on_epoch_end(self, state: TrainerState) -> None:
        if self.every_epochs and (state.epoch + 1) % self.every_epochs == 0:
            self._save(state, f"epoch-{state.epoch}")
        self._maybe_save_best(state)

    def on_step_end(self, state: TrainerState) -> None:
        # Also react to mid-epoch evaluations by Evaluator(every_steps=...).
        self._maybe_save_best(state)

    def _maybe_save_best(self, state: TrainerState) -> None:
        if not self.monitor or self.monitor not in state.metrics:
            return
        # Only react when the metric is fresh at this step.
        if state.metric_steps.get(self.monitor, -1) != state.step:
            return
        value = state.metrics[self.monitor]
        if _is_better(value, self._best, self.mode):
            self._best = value
            self._save(state, "best")
            logger.info("New best %s=%.4f at epoch=%d", self.monitor, value, state.epoch)

    def on_train_end(self, state: TrainerState) -> None:
        if self.save_last:
            self._save(state, "last")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Pair with :class:`Evaluator` so the metric is computed and written to
    ``state.metrics``. Improvement is checked after every epoch, and also
    after every Evaluator step that produced the metric.
    """

    def __init__(
        self,
        monitor: str,
        mode: Mode = "max",
        patience: int = 1,
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = math.inf if mode == "min" else -math.inf
        self._misses = 0
        self._last_seen_step = -1

    def _check(self, state: TrainerState) -> None:
        if self.monitor not in state.metrics:
            return
        # Only react to *fresh* metric values. A metric written in step 8 must
        # not trigger another miss in step 9 just because it's still in the dict.
        metric_step = state.metric_steps.get(self.monitor, -1)
        if metric_step <= self._last_seen_step:
            return
        self._last_seen_step = metric_step
        value = state.metrics[self.monitor]
        delta = (value - self._best) if self.mode == "max" else (self._best - value)
        if delta > self.min_delta:
            self._best = value
            self._misses = 0
        else:
            self._misses += 1
            logger.info(
                "EarlyStopping: %s did not improve (%.4f vs best %.4f), "
                "miss %d/%d",
                self.monitor, value, self._best, self._misses, self.patience,
            )
            if self._misses >= self.patience:
                logger.info("EarlyStopping: stopping at epoch=%d step=%d", state.epoch, state.step)
                state.should_stop = True

    def on_step_end(self, state: TrainerState) -> None:
        self._check(state)

    def on_epoch_end(self, state: TrainerState) -> None:
        self._check(state)
