# participant/

This is where you do all your editing. Define your model, loss, dataset,
sampler, and the training/validation/test recipes here. The framework
under `src/workrb_challenge/` discovers everything in this folder through
dotted-path strings on `TrainConfig`, so swapping a component is a
one-string change in `train.py`.

Run:

```bash
uv run python participant/train.py     # train + validate
uv run python participant/test.py      # local evaluation matrix
```

See the top-level [README](../README.md) for the full breakdown of which
file owns which knob.
