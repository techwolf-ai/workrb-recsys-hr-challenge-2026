# WorkRB Challenge 2026

A freely configurable training and evaluation environment for building
skill-extraction models against [WorkRB](https://github.com/techwolf-ai/workrb)
([PyPI](https://pypi.org/project/workrb/)).

> **TL;DR.** `uv sync`, then `uv run python participant/train.py` trains and
> validates a shipped baseline. Everything you edit lives in `participant/`.
> Any technique that produces rankings competes, training is optional; see
> [Ways onto the leaderboard](#ways-onto-the-leaderboard). A submission you can
> upload today, before training anything:
> `uv run python submission/generate_submission_file.py`.

Your model inherits from PyTorch's `nn.Module` and from WorkRB's
`workrb.models.ModelInterface` directly. WorkRB knows how to score it with
no adapter layer, no save-and-reload between training and validation. Each
participant model file lists those parents at the top, so the contract is
visible in the same file you're editing.

You do all your editing in `participant/`. The framework code under
`src/workrb_challenge/` is the training loop, the callback model, the config
dataclasses, the model base class, and the loggers. You only open it if you
want to understand or change the plumbing.

## Quick start

```bash
uv sync
uv run python participant/train.py                           # trains + validates
uv run python participant/test.py                            # local evaluation matrix
uv run python submission/generate_submission_file.py         # build CodaBench submission
```

No accounts needed: metrics log to the local console by default. To stream
them to Weights & Biases instead, flip `WANDB_ENABLED = True` at the top of
`participant/validate.py` and run `wandb login` once.

The submission script ships pointed at a parameter-free BM25 baseline, so
you can run it before you've trained anything and confirm the pipeline
end-to-end. See [Submitting to the leaderboard](#submitting-to-the-leaderboard).

### Sweep from the command line, no file edits

`participant/train.py` is still the single source of truth (one `TrainConfig`
literal), but you can override individual fields from the CLI, which is what
makes shell-loop sweeps practical:

```bash
uv run python participant/train.py --set optim.learning_rate=1e-4
uv run python participant/train.py --lr 5e-5 --batch-size 128 --epochs 3
uv run python participant/train.py --set loss.init.temperature=0.07   # into a component's init
uv run python participant/train.py --help                             # all --set targets + shortcuts

for lr in 1e-5 2e-5 5e-5; do
  uv run python participant/train.py --lr $lr --set model.init.leaderboard_name=sweep-lr-$lr
done
```

`--set dotted.key=value` reaches any field on `TrainConfig`, including a
component's `init` kwargs (`model.init.max_length`, `loss.init.temperature`).
Values parse as JSON (numbers, `true`/`false`/`null`, lists) and fall back to a
plain string. Every override is logged before training and written into the
run's `config.json`, so a swept run stays reproducible. With no flags the
recipe runs exactly as written.

> If `import workrb_challenge` ever fails (`ModuleNotFoundError`), the editable
> install has gone stale; `uv sync` occasionally leaves it that way. The
> scripts and tests already work around this (they put `src/` on the path
> themselves), so you usually won't notice. To heal the install itself, run
> `uv sync --reinstall-package workrb-challenge-2026`.


## Sanity-check your edits

You edit seven files, and the framework deliberately does not police the
seams between them (that freedom is the point). So before a slow training run
or a capped leaderboard submission, run the smoke tests:

```bash
uv run pytest
```

They run **offline in a few seconds**: no Hub download, no dataset, a tiny
stub backbone swapped in for `paraphrase-mpnet-base-v2`. They check contracts
and shapes, never model quality. Four things, mirroring the files you edit:

| Test file                       | Catches                                                              |
| ------------------------------- | ------------------------------------------------------------------- |
| `tests/test_model_contract.py`  | `_compute_rankings` shape, `name`/`description`/`label_space` types  |
| `tests/test_save_load.py`       | `save_pretrained` -> `from_pretrained` reloads identical scores      |
| `tests/test_loss.py`            | the loss returns a finite scalar that backprops into the model      |
| `tests/test_data.py`            | your collate builds a `Batch` with every declared field populated   |

The two failures most worth catching here are silent ones: a wrong
`_compute_rankings` shape (a zero on the leaderboard *after* you upload) and
trainable state you forgot to persist in `_save_extra` (training succeeds,
the reloaded model is random). If you change a participant file and a test
goes red, the assertion message points at the file and the fix. When you swap
the loss or batch shape, update the matching test to build the inputs your new
design expects.

The tests put `src/` on the path themselves (via `pythonpath` in
`pyproject.toml`), so they keep passing even when the editable install is
stale. See the note under [Quick start](#quick-start) if you hit a bare
`import workrb_challenge` failure outside the provided entry points.

Training writes to `data/runs/{model.name}/{timestamp}/`:

```
data/runs/MyModel-baseline/2026-05-27_14-32-01/
  config.json     <- the resolved TrainConfig that ran
  epoch-0/        <- model checkpoint (loaded via WorkrbSaveable.from_pretrained)
  last/
  best/           <- only when you set ModelCheckpoint(monitor=...)
  workrb_val/     <- per-step WorkRB output folders
data/runs/MyModel-baseline/latest -> 2026-05-27_14-32-01
```

Change `leaderboard_name` in `participant/train.py` and the next run gets
its own folder. Different recipes never overwrite each other.

## What the shipped baseline does

| Block       | Choice                                                         |
| ----------- | -------------------------------------------------------------- |
| Model       | `MyModel`: tied bi-encoder, mean pool, cosine                  |
| Backbone    | `sentence-transformers/paraphrase-mpnet-base-v2`               |
| Dataset     | [`TechWolf/Synthetic-ESCO-skill-sentences`][ds] (138K pairs)   |
| Sampler     | uniform random                                                 |
| Loss        | symmetric in-batch InfoNCE @ temperature=0.05                  |
| Optimizer   | AdamW, lr=2e-5, weight_decay=0.01                              |
| Schedule    | 1 epoch, seed=0                                                |
| Validation  | WorkRB every 500 steps + at epoch end, logged to console (wandb optional) |
| Checkpoints | every epoch + final, into the run folder                       |

[ds]: https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences

## Ways onto the leaderboard

Anything that produces rankings competes. The repo's file layout is built
around training an encoder, but that is a default, not a rule: WorkRB
scores any object implementing `ModelInterface`, and the leaderboard
scores the prediction file that object produces. The one expectation is
that you contribute something of your own; uploading an unmodified
open-source model is a baseline to beat, not an entry (see
[rules](#rules)).

Three well-trodden starting points, freely combinable:

1. **Train an encoder.** The seven-file path below: pick an architecture,
   a loss, a dataset, run `participant/train.py`. The shipped baseline and
   the cross-encoder / classifier-head examples live here.
2. **Build an inference pipeline.** No training at all. Retrieve with a
   bi-encoder, then re-rank the top candidates with a stronger scorer
   (a cross-encoder, or an LLM prompted per pair). The leaderboard metric
   is nDCG@100, so re-ordering the first ~100 candidates is exactly where
   the points are. `participant/examples/llm_reranker.py` is a worked,
   submittable example; it skips `train.py` entirely and goes straight to
   `test.py` and the submission script.
3. **Improve the data.** Keep the baseline architecture and attack the
   training signal instead: filter or re-weight the shipped synthetic
   dataset, generate harder pairs with an LLM, mine hard negatives. All
   of that lives in `participant/data.py` and `participant/sampler.py`.

These compose: a fine-tuned retriever from route 1 plugs into the
re-ranker of route 2 (point its `retriever_name` at your checkpoint
folder), and route 3 feeds route 1. And the design space is far larger
than these three: span taggers, late interaction, LLM distillation,
graph-aware losses, generative linkers, and combinations of all of the
above.
[`knowledge_sharing/skill_extraction_references.md`](knowledge_sharing/skill_extraction_references.md)
lists the published method families with the defining paper for each,
plus the vocabularies and training data; anything in there (or not yet
in there) is fair game under the [rules](#rules).

## The seven files you edit

Everything participant-facing lives in `participant/`:

```
participant/
  my_model.py     (1) architecture + inference
  loss.py         (2) the training objective
  data.py         (3) Dataset, Batch dataclass, collate function
  sampler.py      (4) which examples co-occur in a batch
  train.py        (5) the recipe: TrainConfig literal that points at the above
  validate.py     (6) WorkRB tasks + metrics during training, logged to wandb
  test.py         (7) WorkRB tasks + metrics + baselines for your local matrix
  examples/
    cross_encoder.py     alternative architecture (paired with hinge loss)
    classifier_head.py   alternative architecture (paired with cross entropy)
    llm_reranker.py      inference-only pipeline (bi-encoder retrieval + LLM re-ranking)
```

You do not need to (but of course you can) open `src/workrb_challenge/` to do research. The
framework discovers your model, dataset, sampler, and loss through dotted-path
strings on `TrainConfig`, so swapping any one of them is one string change
in `participant/train.py`.

## Where each knob lives

| To change                                | Edit                                              |
| ---------------------------------------- | ------------------------------------------------- |
| Architecture or inference scoring        | `participant/my_model.py`                         |
| The loss objective                       | `participant/loss.py` (or add a new class there)  |
| Batch sampler, hard negatives            | `participant/sampler.py`                          |
| Dataset, columns, filtering, mixing      | `participant/data.py`                             |
| Batch shape (extra fields)               | `participant/data.py` (the `Batch` dataclass)     |
| Validation tasks / metrics / logger      | `participant/validate.py`                         |
| Test tasks / WorkRB baselines / metrics  | `participant/test.py`                             |
| Learning rate, batch size, epochs, seed  | `participant/train.py` (`TrainConfig(...)`)       |
| Which checkpoints to save                | `participant/train.py` (`ModelCheckpoint(...)`)   |
| Where checkpoints go                     | `participant/train.py` (`output_dir=...`)         |
| Any of the above for one run, no edit    | CLI: `--set dotted.key=value` (see Quick start)   |

## The two big freedom dimensions

### (a) Swap any component without rewriting the loop

Every component is selected by a dotted-path `target` string. Examples:

```python
model=ModelConfig(target="participant.my_model:MyModel", init={...}),
loss=LossConfig(target="participant.loss:InfoNCELoss", init={"temperature": 0.05}),
data=DataConfig(
    dataset=TargetConfig(target="participant.data:SkillSentenceDataset", init={...}),
    sampler=SamplerConfig(target="participant.sampler:RandomBatchSampler", init={...}),
    collate="participant.data:default_collate",
    batch_size=64,
),
```

To try a cross-encoder, point `model.target` at
`participant.examples.cross_encoder:CrossEncoderModel` and `loss.target`
at the matching loss in the same file. To try a frozen-encoder classifier
head, do the same with `participant/examples/classifier_head.py`. The
third example, `participant/examples/llm_reranker.py`, is inference-only
(route 2 above): it never touches `TrainConfig`, so there is nothing to
point at; its bottom comment block shows how to evaluate and submit it
directly. All example files are end-to-end illustrations and run their
own dummy input when invoked directly:

```bash
uv run python -m participant.examples.cross_encoder
uv run python -m participant.examples.classifier_head
uv run python -m participant.examples.llm_reranker
```

### (b) Loss and model are co-designed

The training loop calls `loss_fn(model, batch)` and trusts the loss to
know which methods to read on the model. The default `InfoNCELoss` reads
`model.encode_query(...)` and `model.encode_target(...)`. A triplet loss
would read `model.score_pairs(...)`. A classifier CE loss reads
`model.classifier_logits(...)` plus `batch.labels`.

That means a loss swap is also a model swap (and sometimes a batch-shape
swap). The framework does not hide that pairing because it is the actual
research decision you are making. Worked examples of the three common
shapes:

| Model shape           | Reads on the model                  | Loss              |
| --------------------- | ----------------------------------- | ----------------- |
| Bi-encoder (default)  | `encode_query`, `encode_target`     | `InfoNCELoss`     |
| Cross-encoder         | `score_pairs`                       | `PairwiseHingeLoss` |
| Frozen + classifier   | `classifier_logits`                 | `ClassifierCELoss` |

The inference-only shape (`participant/examples/llm_reranker.py`) has no
row here on purpose: it has no loss and never enters the training loop.

## Validation and test

`participant/validate.py` runs WorkRB on a chosen task set every
`EVERY_STEPS` steps (and at epoch end), then forwards the metrics to a
logger. The logger is a one-line swap: `ConsoleLogger` (the default),
`WandbLogger`, or any class with a `.log(metrics: dict, step: int)` method.
Wandb config (project, run name, enabled flag) is at the top of the file.

`participant/test.py` is your local evaluation matrix: pick tasks, pick
metrics, pick which WorkRB baselines (`TfIdfModel`, `ConTeXTMatchModel`,
...) to score alongside your model. Its defaults mirror the leaderboard:
the graded tasks on the validation split, scored with nDCG@100, so the
number you optimize locally is the number CodaBench reports during the
validation phase. The hosted leaderboard is still a separate service;
`test.py` is just for your view.

```bash
uv run python participant/test.py
uv run python participant/test.py --checkpoint data/runs/MyModel-baseline/latest/best
```

## The WorkRB contract

[WorkRB](https://github.com/techwolf-ai/workrb) is a standalone benchmark
library, not challenge scaffolding. Tasks are plain classes you import
and instantiate; the whole evaluation loop is this:

```python
import workrb
from workrb.models import TfIdfModel                  # any ModelInterface works here
from workrb.tasks import ESCOGradedSkillNormRanking

task = ESCOGradedSkillNormRanking(split="val", languages=["en"])
results = workrb.evaluate(model=TfIdfModel(), tasks=[task], output_folder="data/demo")
print(results.get_summary_metrics())
```

Swap `TfIdfModel()` for your model and that is, modulo task choice, what
`participant/test.py`, `participant/validate.py`, and the leaderboard all do.
`workrb.list_available_tasks()` prints every task beyond the graded ones used
here (job-to-skill matching, job title similarity, multilingual splits, ...).

Every participant model inherits from three classes:

```python
from torch import nn
from workrb.models import ModelInterface          # the WorkRB contract
from workrb_challenge.models import WorkrbSaveable # project save/load mixin

class MyModel(nn.Module, ModelInterface, WorkrbSaveable):
    ...
```

`ModelInterface` (lives in the WorkRB library at
`.venv/lib/python3.12/site-packages/workrb/models/base.py`) declares the
five methods WorkRB calls when scoring your model:

| Method                                  | Returns                                       |
| --------------------------------------- | --------------------------------------------- |
| `_compute_rankings(queries, targets, ...)` | tensor `(Nq, Nt)`, higher = more relevant     |
| `_compute_classification(texts, targets, ...)` | tensor `(Nt, Nc)` over a fixed label space    |
| `name` (property)                       | leaderboard display name                      |
| `description` (property)                | leaderboard description                       |
| `classification_label_space` (property) | list of labels, or `None` for bi-encoder-style |

In practice only `_compute_rankings` does real work here: every
leaderboard task is a ranking task, and `name` and `description` are two
strings. The classification pair only fires on WorkRB's classification
tasks, which are not on the leaderboard; the shipped `MyModel` delegates
`_compute_classification` to `_compute_rankings` and returns `None` for
the label space, and you can leave both as they are.

`WorkrbSaveable` is the only piece this project adds. It is not part of
WorkRB: it captures the convention this challenge uses for serializing
models so `participant/test.py` and the submission script can reload
them.

## Saving and reloading a model

```python
from workrb_challenge.models import WorkrbSaveable
model = WorkrbSaveable.from_pretrained("data/runs/MyModel-baseline/latest/last")
```

`save_pretrained` writes `workrb_model.json` with the subclass's dotted
import path plus init kwargs. `from_pretrained` resolves that path and
returns the original subclass instance, not a `WorkrbSaveable`. Override
`_save_extra` if you need to persist trainable state beyond the backbone
(classifier heads, projection matrices, learned temperatures, etc.).

## Submitting to the leaderboard

The challenge is hosted on **[CodaBench][cb]**. Sign up there once, then
upload the `submission.zip` produced by the script in `submission/`.
CodaBench only accepts `.zip` archives, so the script zips the
`submission.json` for you; the JSON inside the zip is what gets scored.

[cb]: https://www.codabench.org/

### Phases

| Phase           | When                  | Cap                       |
| --------------- | --------------------- | ------------------------- |
| Validation      | open now              | unlimited submissions     |
| Test            | from 15 Jun 2026      | 20 submissions per team    |
| Submissions close | 31 Jul 2026         | n/a                       |

### Rules

The rules are deliberately short:

* **Any technique that produces rankings is allowed.** Trained
  bi-encoders or cross-encoders, off-the-shelf rerankers, LLMs (local or
  hosted APIs), retrieve-then-re-rank pipelines, ensembles, or no
  training at all. If it fills the `(num_queries, num_targets)` score
  matrix, it competes.
* **Bring something of your own.** Off-the-shelf models and APIs are
  welcome as components, but the submission as a whole should add a
  contribution of yours: fine-tuning, a pipeline around them, better
  training data, an ensemble. An existing open-source model submitted
  unchanged is a baseline, not an entry.
* **Any open-source data is allowed.** You are not limited to the shipped
  dataset; any publicly available data may be used for training, tuning,
  or prompt construction.
* **Everything goes through WorkRB.** Your model implements the real
  `workrb.models.ModelInterface`, and the submission file is produced by
  the real WorkRB tasks. That is the only hard interface, and it is the
  same one the baselines use.
* **Say what you did.** When you submit rankings you will be asked for a
  short sentence describing your method. This keeps the leaderboard
  interpretable for everyone.


### Three constants, one command

Open `submission/generate_submission_file.py` and set the three constants
at the top:

```python
SPLIT           = "validation"                  # or "test" once that phase opens
MODEL_DEF_FILE  = "submission/_bm25_def.py"     # the file defining your model class
WEIGHTS_PATH    = ""                            # path to a saved checkpoint (empty = no-weights model)
```

Then run:

```bash
uv run python submission/generate_submission_file.py
```

It writes `submission/submission.json` and zips it to
`submission/submission.zip`. Upload the `.zip` to CodaBench (it does not
accept a bare `.json`).

### What `MODEL_DEF_FILE` and `WEIGHTS_PATH` mean

The script supports two modes:

1. **No-weights model** (`WEIGHTS_PATH=""`). The script imports
   `MODEL_DEF_FILE`, finds the single `ModelInterface` subclass exported
   from it, and calls it with no arguments. Use this for parameter-free
   baselines. The shipped `submission/_bm25_def.py` re-exports
   `workrb.models.lexical_baselines.BM25Model`, so running the script
   out of the box produces a valid BM25 submission you can upload as
   your warm-up.

2. **Trained checkpoint** (`WEIGHTS_PATH` set). The script calls
   `WorkrbSaveable.from_pretrained(WEIGHTS_PATH)`, which reads
   `workrb_model.json` in the checkpoint folder and rebuilds the right
   subclass. `MODEL_DEF_FILE` still has to point at a file that
   *imports* the class (so the dotted path inside `workrb_model.json`
   resolves), but its job is just to make the class importable. The
   default workflow is to point it at your `participant/my_model.py`.

```python
SPLIT           = "validation"
MODEL_DEF_FILE  = "participant/my_model.py"
WEIGHTS_PATH    = "data/runs/MyModel-baseline/latest/best"
```

The script logs nDCG@100 for each task as it runs, so you see the same
numbers locally that CodaBench will score on the validation phase.

The "do not edit below" block is the WorkRB-side glue: task instantiation,
prediction-matrix serialization, JSON write. The submission schema is
fixed by WorkRB; the constants above are the only knobs.
