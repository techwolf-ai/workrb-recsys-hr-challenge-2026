"""Local test suite: run YOUR model + chosen baselines through real WorkRB.

This is your personal evaluation. It's not the hosted leaderboard, but the
defaults mirror it: the graded tasks on the validation split, scored with
nDCG@100, are exactly what CodaBench computes during the validation phase.
Pick tasks, pick metrics, pick which baselines to compare against, and you
get a model x task matrix printed to stdout.

Run with:

    uv run python participant/test.py

Or with a specific checkpoint:

    uv run python participant/test.py --checkpoint data/runs/MyModel-baseline/latest/best

There are two editable surfaces in this file:

  1. The constants below: which checkpoint to load, and which WorkRB
     baselines to compare against.
  2. The block marked `>>> WORKRB SWAP <<<` in `main()`. That block is real
     WorkRB code (tasks + metrics + `workrb.evaluate`). Add tasks by
     uncommenting lines. Add metrics by editing the dict.
"""

from __future__ import annotations

# Make ``participant.*`` and ``workrb_challenge.*`` importable even when the
# editable install is stale and even when this file is run as a path
# (``python participant/test.py``). Must run before any project import below.
# See ``participant/_bootstrap.py`` for the why.
import sys as _sys
from pathlib import Path as _Path

for _p in (_Path(__file__).resolve().parents[1], _Path(__file__).resolve().parents[1] / "src"):
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import logging
from pathlib import Path

import workrb
from workrb.models import (
    BiEncoderModel,
    ConTeXTMatchModel,
    CurriculumMatchModel,
    EditDistanceModel,
    JobBERTModel,
    ModelInterface,
    RandomRankingModel,
    RndESCOClassificationModel,
    TfIdfModel,
)

from workrb_challenge.models import WorkrbSaveable

# ============================================================================
# Default checkpoint. Override at the CLI:
#     uv run python participant/test.py --checkpoint <path>
# ============================================================================
DEFAULT_CHECKPOINT = "data/runs/MyModel-baseline/latest/last"

# ============================================================================
# WorkRB baselines to compare against. Comment out the ones you don't want.
# Every entry is a `workrb.models.ModelInterface` instance.
# ============================================================================
BASELINES: list[ModelInterface] = [
    TfIdfModel(),
    # EditDistanceModel(),
    # RandomRankingModel(),
    # BiEncoderModel(),                  # off-the-shelf MPNet (sentence-transformers/all-mpnet-base-v2)
    ConTeXTMatchModel(),
    CurriculumMatchModel(),
    # JobBERTModel(),
    # RndESCOClassificationModel(),
]


def main(checkpoint: str | Path) -> None:
    # ``WorkrbSaveable.from_pretrained`` reads the dotted-path target written
    # at save time and instantiates *that* subclass. The returned object
    # implements WorkRB's ``ModelInterface``, exactly like every baseline
    # below; ``workrb.evaluate`` sees no difference between them.
    my_model = WorkrbSaveable.from_pretrained(checkpoint)
    models: list[ModelInterface] = [my_model, *BASELINES]

    # ========================================================================
    # >>> WORKRB SWAP: which test tasks and metrics to compute. <<<
    #
    # WorkRB tasks are plain classes under `workrb.tasks`; instantiate one
    # with a split and languages and it loads its own data. `workrb.evaluate`
    # runs the model on the chosen tasks and returns a `BenchmarkResults`
    # object; `.get_summary_metrics()` flattens it.
    #
    # The defaults below mirror the leaderboard: the graded tasks on the
    # validation split, scored with nDCG@100. These are the numbers CodaBench
    # reports during the validation phase, so what you optimize here is what
    # ranks you there. (`TechWolfGradedSkillExtractRanking` is missing because
    # it only publishes a test split.)
    #
    # `metrics` is a dict from `task.name` (the display name, NOT the class
    # name) to the metric list to compute. A task whose name is missing falls
    # back to its own defaults (graded ranking tasks default to ["ndcg",
    # "ndcg@5", "ndcg@10", "map", "rp@10", "mrr"]). See
    # `workrb.list_available_tasks()` for everything available.
    # ========================================================================
    tasks = [
        workrb.tasks.HouseGradedSkillExtractRanking(split="val", languages=["en"]),
        workrb.tasks.TechGradedSkillExtractRanking(split="val", languages=["en"]),
        workrb.tasks.SkillSkapeGradedSkillExtractRanking(split="val", languages=["en"]),
        workrb.tasks.ESCOGradedSkillNormRanking(split="val", languages=["en"]),
        # Older binary-relevance tasks (faster, smaller target spaces; not on
        # the leaderboard):
        # workrb.tasks.HouseSkillExtractRanking(split="test", languages=["en"]),
        # workrb.tasks.TechSkillExtractRanking(split="test", languages=["en"]),
        # workrb.tasks.SkillSkapeExtractRanking(split="test", languages=["en"]),
        # workrb.tasks.ESCOJob2SkillRanking(split="test", languages=["en"]),
        # workrb.tasks.ESCOSkill2JobRanking(split="test", languages=["en"]),
        # workrb.tasks.ESCOSkillNormRanking(split="test", languages=["en"]),
        # workrb.tasks.SkillMatch1kSkillSimilarityRanking(split="test", languages=["en"]),
        # workrb.tasks.JobTitleSimilarityRanking(split="test", languages=["en"]),
    ]
    metrics = {
        "Skill Extraction House Graded":      ["ndcg@100"],
        "Skill Extraction Tech Graded":       ["ndcg@100"],
        "Skill Extraction SkillSkape Graded": ["ndcg@100"],
        "Skill Normalization ESCO Graded":    ["ndcg@100"],
        # "Skill Extraction House":     ["map", "rp@10", "mrr"],
        # "Skill Extraction Tech":      ["map", "rp@10", "mrr"],
        # "Skill Extraction SkillSkape": ["map", "rp@10", "mrr"],
    }

    rows: dict[str, dict[str, float]] = {}
    for model in models:
        results = workrb.evaluate(
            model=model,
            tasks=tasks,
            output_folder=f"data/test_results/{model.name}",
            metrics=metrics,
        )
        rows[model.name] = dict(results.get_summary_metrics())
    # ========================================================================

    _print_matrix(rows, [t.name for t in tasks])


def _print_matrix(rows: dict[str, dict[str, float]], task_names: list[str]) -> None:
    """Print a model x task matrix (uses each row's mean_per_task entries)."""
    print()
    name_w = max((len(m) for m in rows), default=20)
    task_w = max((len(t) for t in task_names), default=10)

    header = f"{'Model':<{name_w}}  " + "  ".join(f"{t:>{task_w}}" for t in task_names)
    print(header)
    print("-" * len(header))

    def cell(row: dict[str, float], task: str) -> str:
        for k, v in row.items():
            if k.startswith(f"mean_per_task/{task}/") and k.endswith("/mean"):
                return f"{v:.4f}"
        return "n/a"

    for model_name, row in rows.items():
        cells = "  ".join(f"{cell(row, t):>{task_w}}" for t in task_names)
        print(f"{model_name:<{name_w}}  {cells}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to a saved model folder.")
    args = parser.parse_args()
    main(args.checkpoint)
