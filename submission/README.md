# submission/

Builds the `submission.json` file that you upload to
[CodaBench](https://www.codabench.org/) for the leaderboard, given a
model architecture and (optionally) a trained checkpoint.

Set the three constants at the top of `generate_submission_file.py`
(`SPLIT`, `MODEL_DEF_FILE`, `WEIGHTS_PATH`), then:

```bash
uv run python submission/generate_submission_file.py
```

The `name` property on your model class is what appears on the
leaderboard, so pick something recognizable. Out of the box this folder
ships a parameter-free BM25 baseline (`_bm25_def.py`) so you can produce
a valid submission before training anything.

See the [Submitting to the leaderboard](../README.md#submitting-to-the-leaderboard)
section of the top-level README for the full walkthrough.
