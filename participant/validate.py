"""Validation during training: real workrb calls, logged locally by default.

This file defines `callback`, a training-loop callback that runs WorkRB
every `EVERY_STEPS` steps and logs the resulting metrics. Import it from
`participant/train.py`:

    from participant.validate import callback as validation_callback

There are two editable surfaces in this file:

  1. The constants at the top: wandb config and the validation cadence.
  2. The block marked `>>> WORKRB SWAP <<<` inside `ValidationCallback`,
     which is real WorkRB code (tasks + metrics + `workrb.evaluate`). Add
     tasks by uncommenting lines. Add metrics by editing the dict.

The rest is plumbing: a Callback subclass that fires the workrb block on
a schedule and forwards results to the configured logger.

Leaderboard alignment
----------------------
The official leaderboard score is the macro average of the ndcg@100 of the
test splits of five graded tasks: TechGraded, HouseGraded, TechWolfGraded,
SkillSkapeGraded and ESCOGradedSkillNormalisation.

TechWolfGraded ships a `test` split only (no `validation`), so it cannot be
scored during training. The most faithful validation proxy is therefore the
macro average of ndcg@100 over the four tasks that DO expose a `val` split:
TechGraded, HouseGraded, SkillSkapeGraded and ESCOGradedSkillNormalisation.
That is exactly what this callback computes by default and logs under the key
`ndcg@100_macro`, alongside the per-task `<task>/ndcg@100` entries.

"""

from __future__ import annotations

import workrb
from workrb.tasks import (
    ESCOGradedSkillNormRanking,
    HouseGradedSkillExtractRanking,
    SkillSkapeGradedSkillExtractRanking,
    TechGradedSkillExtractRanking,
)

from workrb_challenge.training import Callback, ConsoleLogger, TrainerState, WandbLogger

# ============================================================================
# Wandb. Edit these inline. Flip WANDB_ENABLED off to log to console only.
# To swap to a different backend (tensorboard, mlflow, ...), construct your
# own logger object below; the only contract is `.log(metrics: dict, step: int)`.
# ============================================================================
WANDB_PROJECT = "workrb-challenge-2026"
WANDB_RUN_NAME: str | None = None   # None -> wandb auto-names
# Default to local console logging so participants don't need a wandb account.
# Flip this to True (and run `wandb login`) to stream metrics to Weights & Biases.
WANDB_ENABLED = False

# How often to run validation. Set to 0 to disable mid-epoch validation
# (only fires at end of each epoch then).
EVERY_STEPS = 500


def _build_logger():
    if WANDB_ENABLED:
        return WandbLogger(project=WANDB_PROJECT, run_name=WANDB_RUN_NAME)
    return ConsoleLogger()


class ValidationCallback(Callback):
    """Runs WorkRB every `EVERY_STEPS` steps + at the end of each epoch."""

    def __init__(self):
        self._logger = _build_logger()

    def _validate(self, state: TrainerState) -> None:
        was_training = state.model.training
        try:
            metrics = _run_workrb_validation(state)
        finally:
            state.model.train(was_training)

        for k, v in metrics.items():
            state.log(k, v)
        self._logger.log(metrics, step=state.step)

    def on_step_end(self, state: TrainerState) -> None:
        if EVERY_STEPS and state.step > 0 and state.step % EVERY_STEPS == 0:
            self._validate(state)

    def on_epoch_end(self, state: TrainerState) -> None:
        self._validate(state)


# The four leaderboard tasks that expose a `val` split. TechWolfGraded is
# omitted on purpose: it has a `test` split only, so it cannot be validated.
# Macro-averaging ndcg@100 over exactly these four mirrors the leaderboard
# (which also macro-averages ndcg@100) minus the un-validatable TechWolf task.
VAL_TASKS = [
    TechGradedSkillExtractRanking,
    HouseGradedSkillExtractRanking,
    SkillSkapeGradedSkillExtractRanking,
    ESCOGradedSkillNormRanking,
]

# The leaderboard metric.
VAL_METRICS = ["ndcg@100"]

# The key under which the leaderboard-aligned macro average is logged.
MACRO_METRIC_KEY = "ndcg@100_macro"


def _run_workrb_validation(state: TrainerState) -> dict[str, float]:
    """The real WorkRB evaluation. Everything in this function is yours."""

    # ========================================================================
    # >>> WORKRB SWAP: which validation tasks and metrics to compute. <<<
    #
    # WorkRB tasks are plain classes under `workrb.tasks`; instantiate one
    # with a split and languages and it loads its own data. `workrb.evaluate`
    # runs your model on the chosen tasks and returns a `BenchmarkResults`
    # object; `.get_summary_metrics()` flattens it.
    #
    # The defaults below are the leaderboard-aligned validation set: the four
    # graded tasks that have a `val` split, scored with ndcg@100 and macro
    # averaged. Add tasks/metrics by editing `VAL_TASKS` / `VAL_METRICS`.
    #
    # See `workrb.list_available_tasks()` for the full task list.
    # ========================================================================
    tasks = [
        task_cls(split="val", languages=["en"]) for task_cls in VAL_TASKS
    ]
    # `workrb.evaluate` looks up per-task metrics by the task's display name
    # (`task.name`), NOT by the class name. Keying this dict by the class
    # name silently falls back to each task's default metrics (which do not
    # include ndcg@100), so build it from the instantiated tasks.
    metrics = {task.name: VAL_METRICS for task in tasks}

    results = workrb.evaluate(
        model=state.model,
        tasks=tasks,
        output_folder=str(state.run_dir / "workrb_val" / f"step-{state.step}"),
        metrics=metrics,
    )
    # ========================================================================

    summary = dict(results.get_summary_metrics())
    summary[MACRO_METRIC_KEY] = _macro_ndcg100(summary)
    return summary


def _macro_ndcg100(summary: dict[str, float]) -> float:
    """Macro average of per-task ndcg@100, mirroring the leaderboard.

    `get_summary_metrics()` exposes one `mean_per_task/<task>/ndcg@100/mean`
    entry per task (already averaged over that task's datasets). Each of the
    four val tasks has a single `en` dataset, so averaging these per-task
    values equals the macro average over datasets that the leaderboard uses.
    """
    per_task = [
        v
        for k, v in summary.items()
        if k.startswith("mean_per_task/") and k.endswith("/ndcg@100/mean")
    ]
    if not per_task:
        return 0.0
    return sum(per_task) / len(per_task)


# Exported for `participant/train.py` to drop into `TrainConfig.callbacks`.
callback = ValidationCallback()
