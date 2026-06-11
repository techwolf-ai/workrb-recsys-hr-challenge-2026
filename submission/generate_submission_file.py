"""Generate the WorkRB submission file. Set SPLIT, MODEL_DEF_FILE, WEIGHTS_PATH below and run."""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

# Put the repo root + ``src/`` on the import path so ``workrb_challenge``
# resolves even when the editable install has gone stale (the same trap the
# entry points under ``participant/`` guard against). Idempotent; a no-op when
# the install is healthy. Must run before importing anything that pulls in
# ``workrb_challenge`` (done lazily inside ``_load_model``), hence the E402s
# on the third-party imports that follow.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import workrb  # noqa: E402
from workrb.models.base import ModelInterface  # noqa: E402
from workrb.tasks import (  # noqa: E402
    ESCOGradedSkillNormRanking,
    HouseGradedSkillExtractRanking,
    SkillSkapeGradedSkillExtractRanking,
    TechGradedSkillExtractRanking,
    TechWolfGradedSkillExtractRanking,
)
from workrb.tasks.abstract.base import DatasetSplit  # noqa: E402
from workrb.tasks.abstract.ranking_base import RankingTask  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ============================================================================
# >>> CONFIG <<<
# ============================================================================

SPLIT = "validation" # validation or test, check on which phase the challenge is currently in before submitting!

MODEL_DEF_FILE = "submission/_bm25_def.py" # Path to the WorkRB model interface defining your architecture
WEIGHTS_PATH = "" # Path to the weights that should be loaded in your model architecture

OUTPUT_FILE = "submission/submission.json" # Adjust if you would want to save your submission file somewhere else

# ============================================================================
# Evaluation : Do NOT EDIT BELOW THIS LINE!
# ============================================================================

TOP_K: int | None = 500
TARGET_ID_PREFIX = "http://data.europa.eu/esco/skill/"

TASKS: list[dict[str, Any]] = [
    {
        "task": TechGradedSkillExtractRanking,
        "metrics": ["ndcg@100"],
        "languages": ["en"],
    },
    {
        "task": HouseGradedSkillExtractRanking,
        "metrics": ["ndcg@100"],
        "languages": ["en"],
    },
    {
        "task": TechWolfGradedSkillExtractRanking,
        "metrics": ["ndcg@100"],
        "languages": ["en"],
    },
    {
        "task": SkillSkapeGradedSkillExtractRanking,
        "metrics": ["ndcg@100"],
        "languages": ["en"],
    },
    {
        "task": ESCOGradedSkillNormRanking,
        "metrics": ["ndcg@100"],
        "languages": ["en"],
    },
]

_SPLIT_ALIASES = {"validation": "val", "val": "val", "test": "test"}

def _resolve_split(raw: str) -> DatasetSplit:
    key = raw.strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(
            f"Unknown SPLIT '{raw}'. Use one of: 'validation' (or 'val'), 'test'."
        )
    return DatasetSplit(_SPLIT_ALIASES[key])


def _load_model(model_def_file: str, weights_path: str) -> ModelInterface:
    """Import the model definition file, then build the model.

    When ``weights_path`` is non-empty, the weights directory's
    ``workrb_model.json`` decides the exact subclass (via
    ``WorkrbModel.from_pretrained``). Otherwise the unique
    ``ModelInterface`` subclass found in ``model_def_file`` is instantiated
    with no args, which suits parameter-free baselines such as BM25.
    """
    def_path = Path(model_def_file).resolve()
    if not def_path.is_file():
        raise FileNotFoundError(f"MODEL_DEF_FILE not found: {def_path}")

    spec = importlib.util.spec_from_file_location(def_path.stem, def_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if weights_path:
        from workrb_challenge.models import WorkrbSaveable

        return WorkrbSaveable.from_pretrained(weights_path)

    exported = getattr(module, "__all__", None)
    candidates: list[type[ModelInterface]] = []
    for attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if not (issubclass(obj, ModelInterface) and obj is not ModelInterface):
            continue
        if exported is not None and attr_name not in exported:
            continue
        candidates.append(obj)
    # Drop duplicates while preserving order.
    seen: set[type] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    if len(candidates) != 1:
        names = [c.__name__ for c in candidates]
        raise RuntimeError(
            f"Expected exactly one ModelInterface subclass exposed in "
            f"{def_path} (got {len(candidates)}: {names}). Either narrow "
            f"the file (use __all__ to pin one class) or set WEIGHTS_PATH "
            f"so the saved config picks the class."
        )
    return candidates[0]()


def _matrix_to_score_dict(
    matrix: np.ndarray,
    query_ids: list[str],
    target_ids: list[str],
    top_k: int | None = TOP_K,
    target_id_prefix: str = TARGET_ID_PREFIX,
) -> dict[str, dict[str, float]]:
    """Convert (num_queries, num_targets) -> {query_id: {target_id: score}}.

    Keys are the source-dataset IDs (queries ``_id`` on the row axis, corpus
    ``_id`` on the column axis), aligned 1-to-1 with the matrix's rows and
    columns. Emitting explicit IDs instead of positional indices makes the
    submission immune to any reordering, deduplication or filtering the task
    applies when it builds its in-memory query/target arrays: the scorer maps
    each key straight back to the source row by its ID column, never by row
    position. Non-finite scores are rejected so the JSON stays loadable
    (matches the writer's ``allow_nan=False`` contract).

    When ``top_k`` is set, only each query's ``top_k`` highest-scoring targets
    are kept; the rest are dropped (the scorer ranks them below the cutoff).
    ``target_id_prefix`` is stripped from every target key to keep keys short;
    the scorer re-attaches it.
    """
    if len(query_ids) != matrix.shape[0]:
        raise ValueError(
            f"query_ids has {len(query_ids)} entries but matrix has "
            f"{matrix.shape[0]} rows"
        )
    if len(target_ids) != matrix.shape[1]:
        raise ValueError(
            f"target_ids has {len(target_ids)} entries but matrix has "
            f"{matrix.shape[1]} columns"
        )
    if not np.all(np.isfinite(matrix)):
        bad = np.argwhere(~np.isfinite(matrix))[0]
        raise ValueError(
            f"prediction matrix contains a non-finite score at "
            f"query_index={int(bad[0])}, target_index={int(bad[1])}"
        )

    # Precompute the short target keys once (same column order for every row).
    short_ids = [
        tid[len(target_id_prefix):] if tid.startswith(target_id_prefix) else tid
        for tid in target_ids
    ]

    scores: dict[str, dict[str, float]] = {}
    for q_idx in range(matrix.shape[0]):
        row = matrix[q_idx]
        if top_k is not None and top_k < row.shape[0]:
            # Indices of the top_k largest scores (unordered); the scorer
            # re-ranks by value, so order within the kept set does not matter.
            keep = np.argpartition(row, -top_k)[-top_k:]
        else:
            keep = range(row.shape[0])
        row_dict = {short_ids[t_idx]: float(row[t_idx]) for t_idx in keep}
        scores[query_ids[q_idx]] = row_dict
    return scores


def _resolve_axis_ids(task: RankingTask, dataset_id: str) -> tuple[list[str], list[str]]:
    """Recover source-dataset IDs aligned to the task's in-memory axes.

    Returns ``(query_ids, target_ids)`` where ``query_ids[i]`` is the queries
    ``_id`` for ``dataset.query_texts[i]`` and ``target_ids[j]`` is the corpus
    ``_id`` for ``dataset.target_space[j]``.

    The task builds its query/target arrays by grouping, deduplicating and
    filtering the raw data, so their positions do not line up with the source
    parquet row order. We recover the IDs by reloading the raw ``queries`` and
    ``corpus`` configs from the task's ``hf_name`` and matching on the
    (whitespace-stripped) text, which the task uses verbatim as its array
    entries. Query texts and corpus titles are unique in these datasets, so the
    text->ID map is unambiguous; any miss raises rather than silently
    misaligning.

    Only ``BaseGradedSkillExtractRanking`` tasks (the BEIR-layout graded skill
    extraction tasks) expose ``queries``/``corpus``/``qrels`` configs, so this
    is limited to tasks carrying an ``hf_name``.
    """
    from datasets import load_dataset

    hf_name = getattr(task, "hf_name", None)
    if hf_name is None:
        raise RuntimeError(
            f"Task '{task.name}' does not expose an 'hf_name'; cannot recover "
            f"source IDs for the ID-keyed submission schema."
        )

    # The queries config lives under the HF split backing the task's current
    # split: 'validation' for the validation phase, 'test' for the test phase.
    # Derive it from the task's own split mapping so test submissions recover
    # IDs from the test queries (not the validation ones). The corpus config
    # always lives under the 'corpus' split.
    split_to_hf = getattr(task, "split_to_hf_split", None)
    if split_to_hf and getattr(task, "split", None) in split_to_hf:
        queries_hf_split = split_to_hf[task.split]
    else:
        queries_hf_split = "validation"

    queries_df = load_dataset(hf_name, "queries", split=queries_hf_split).to_pandas()
    corpus_df = load_dataset(hf_name, "corpus", split="corpus").to_pandas()

    text_to_query_id: dict[str, str] = {
        str(text).strip(): str(qid)
        for qid, text in zip(queries_df["_id"], queries_df["text"], strict=True)
    }
    title_to_corpus_id: dict[str, str] = {
        str(title).strip(): str(cid)
        for cid, title in zip(corpus_df["_id"], corpus_df["title"], strict=True)
    }

    dataset = task.datasets[dataset_id]
    try:
        query_ids = [text_to_query_id[q] for q in dataset.query_texts]
    except KeyError as e:
        raise RuntimeError(
            f"Query text {e} (task '{task.name}', dataset '{dataset_id}') has no "
            f"match in the queries config of '{hf_name}'."
        ) from e
    try:
        target_ids = [title_to_corpus_id[t] for t in dataset.target_space]
    except KeyError as e:
        raise RuntimeError(
            f"Target title {e} (task '{task.name}', dataset '{dataset_id}') has no "
            f"match in the corpus config of '{hf_name}'."
        ) from e
    return query_ids, target_ids


def _instantiate_task(task_cfg: dict[str, Any], split: DatasetSplit) -> RankingTask | None:
    """Instantiate one task for the requested split, or return None to skip."""
    task_cls = task_cfg["task"]
    languages = task_cfg.get("languages")
    try:
        task = task_cls(split=split.value, languages=languages)
    except Exception as e:
        logger.warning(
            "Skipping task '%s': cannot instantiate for split='%s' (%s: %s)",
            task_cls.__name__,
            split.value,
            type(e).__name__,
            e,
        )
        return None
    if not isinstance(task, RankingTask):
        logger.warning(
            "Skipping task '%s': only ranking tasks are supported by this "
            "submission script (got %s).",
            task_cls.__name__,
            type(task).__name__,
        )
        return None
    if not task.dataset_ids:
        logger.warning(
            "Skipping task '%s': no datasets loaded for split='%s' and "
            "languages=%s.",
            task_cls.__name__,
            split.value,
            languages,
        )
        return None
    return task


def _evaluate_task(
    task: RankingTask,
    task_cfg: dict[str, Any],
    model: ModelInterface,
) -> dict[str, dict[str, Any]]:
    """Run the task across its dataset_ids; return ``{dataset_id: payload}``.

    ``payload`` contains ``num_queries``, ``num_targets``, and the per-row
    score dict, ready to plug into the submission schema's ``<language>``
    slot. Metrics are computed alongside and logged for the user.
    """
    out: dict[str, dict[str, Any]] = {}
    metrics = task_cfg.get("metrics")
    for dataset_id in task.dataset_ids:
        dataset = task.datasets[dataset_id]
        query_ids, target_ids = _resolve_axis_ids(task, dataset_id)
        logger.info(
            "[%s/%s] %d queries x %d targets",
            task.name,
            dataset_id,
            len(dataset.query_texts),
            len(dataset.target_space),
        )

        start = time.time()
        matrix = task.compute_prediction_matrix(model=model, dataset_id=dataset_id)
        matrix = np.asarray(matrix, dtype=np.float32)
        infer_secs = time.time() - start

        local_metrics = task.compute_metrics_from_prediction_matrix(
            prediction_matrix=matrix,
            dataset_id=dataset_id,
            metrics=metrics,
        )
        logger.info(
            "[%s/%s] inference %.1fs | metrics: %s",
            task.name,
            dataset_id,
            infer_secs,
            ", ".join(f"{k}={v:.4f}" for k, v in local_metrics.items()),
        )

        scores = _matrix_to_score_dict(matrix, query_ids, target_ids)
        del matrix
        out[dataset_id] = {
            "num_queries": len(dataset.query_texts),
            "num_targets": len(dataset.target_space),
            "scores": scores,
        }
    return out


def main() -> None:
    split = _resolve_split(SPLIT)
    logger.info("Loading model from %s (weights=%s)", MODEL_DEF_FILE, WEIGHTS_PATH or "<none>")
    model = _load_model(MODEL_DEF_FILE, WEIGHTS_PATH)
    logger.info("Model: %s", model.name)
    logger.info("Available registered tasks: %d", len(workrb.list_available_tasks()))

    submission: dict[str, dict[str, dict[str, Any]]] = {model.name: {}}
    for task_cfg in TASKS:
        task = _instantiate_task(task_cfg, split)
        if task is None:
            continue
        submission[model.name][task.name] = _evaluate_task(task, task_cfg, model)

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, allow_nan=False)
    logger.info("Wrote submission file: %s", out_path)

    # CodaBench only accepts a .zip bundle, never a bare .json, so emit the
    # zip the user actually uploads. The JSON is stored flat at the archive
    # root (no enclosing folder), which is the structure the scorer expects.
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=out_path.name)
    logger.info("Wrote submission archive (upload this to CodaBench): %s", zip_path)


if __name__ == "__main__":
    main()
