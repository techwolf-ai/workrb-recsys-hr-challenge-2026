# Sanity-check tests

Offline smoke tests for the seams the framework leaves to you. Run them after
every edit to `participant/`:

```bash
uv run pytest
```

A few seconds, no network: `conftest.py` swaps a tiny stub backbone in for
`paraphrase-mpnet-base-v2`, so nothing downloads from the Hub and no dataset
is loaded. The tests check **contracts and shapes, not model quality**.

| Test file                       | The seam it guards                                                  |
| ------------------------------- | ------------------------------------------------------------------- |
| `test_model_contract.py`        | The WorkRB contract on your model (ranking shape, metadata types)   |
| `test_save_load.py`             | `save_pretrained` -> `from_pretrained` rebuilds an identical model  |
| `test_loss.py`                  | The loss returns a finite scalar that backprops into the model      |
| `test_data.py`                  | Your collate produces a `Batch` with every field populated          |

When you swap a component, update the matching test to build the inputs your
new design expects. The loss/model/batch are co-designed; the tests are too.

If a run fails with `ModuleNotFoundError: No module named 'workrb_challenge'`,
the editable install went stale (`uv sync` does this occasionally). Fix it
with `uv sync --reinstall-package workrb-challenge-2026`.

These cover the four core seams only. They do not run the example
architectures (`participant/examples/`), and they do not run a real
end-to-end `train()`. That keeps them fast and offline. For a real run, use
`participant/train.py` and `participant/test.py`.
