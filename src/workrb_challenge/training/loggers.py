"""Metrics loggers.

A logger is anything with one method:

    .log(metrics: dict[str, float], step: int) -> None

That tiny interface is the only contract. The validation callback in
``participant/validate.py`` builds whichever logger you want and calls
``.log(...)`` after each WorkRB evaluation. To plug in a new backend
(MLflow, Tensorboard, a custom dashboard), write a class with that one
method and use it instead.

Two shipped implementations:

* ``ConsoleLogger`` prints ``key=value`` lines via the standard logger.
  No setup, always works.

* ``WandbLogger`` logs to Weights and Biases. If ``wandb.init`` fails (no
  login, no network), it falls back to ``ConsoleLogger`` *at construction
  time*, so the failure is loud and early instead of silent halfway
  through training.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Logger(Protocol):
    """The minimal interface every metrics logger implements."""

    def log(self, metrics: dict[str, float], step: int) -> None: ...


class ConsoleLogger:
    """Prints ``key=value`` pairs through Python's logging module. Always works."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        parts = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("step=%d %s", step, parts)


class WandbLogger:
    """Logs to Weights and Biases.

    Falls back to :class:`ConsoleLogger` if ``wandb.init`` raises. The
    fallback decision happens at construction so you find out immediately
    that wandb is not wired up, not after 4 hours of training.
    """

    def __init__(
        self,
        project: str = "workrb-challenge-2026",
        run_name: str | None = None,
        config: dict | None = None,
    ):
        import wandb

        try:
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            self._wandb = wandb
            self._fallback: ConsoleLogger | None = None
        except Exception as e:
            logger.warning(
                "wandb.init failed (%s). Falling back to console logging. "
                "Run `wandb login` to enable wandb.",
                e,
            )
            self._run = None
            self._wandb = None
            self._fallback = ConsoleLogger()

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._fallback is not None:
            self._fallback.log(metrics, step)
            return
        self._wandb.log(metrics, step=step)
