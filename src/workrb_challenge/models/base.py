"""``WorkrbSaveable``: a save/load mixin for the challenge.

What this is, what this is **not**
----------------------------------

This mixin does NOT define the model contract. The model contract is
``workrb.models.ModelInterface``, which lives in the WorkRB library and
declares the methods WorkRB calls when scoring you (``_compute_rankings``,
``_compute_classification``, ``name``, ``description``,
``classification_label_space``).

Participant models in ``participant/`` inherit from three classes directly,
so the lineage is visible at the top of every model file::

    from torch import nn
    from workrb.models import ModelInterface
    from workrb_challenge.models import WorkrbSaveable

    class MyModel(nn.Module, ModelInterface, WorkrbSaveable):
        ...

That ordering says exactly what your model is:

  * ``nn.Module``       so PyTorch optimizers and ``.to(device)`` work.
  * ``ModelInterface``  so WorkRB can score you with no adapter layer.
  * ``WorkrbSaveable``  so you can call ``save_pretrained`` / ``from_pretrained``
                        with the project's convention.

----------------------------------

WorkRB does not prescribe how a model is serialized; different backbones
serialize differently (HF, sentence-transformers, custom heads, ...).
This mixin captures the *convention this challenge uses*: write a
``workrb_model.json`` next to the backbone files, with the subclass's
dotted import path + init kwargs. ``from_pretrained`` reads the dotted
path back, so the same subclass that was saved is the same subclass that
gets reconstructed.

If your model has extra trainable state beyond ``self.backbone`` /
``self.tokenizer`` (a classifier head, projection matrices, learned
temperatures), override ``_save_extra`` to write that state and return
the init kwargs needed to reconstruct it.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any


class WorkrbSaveable:
    """Save/load mixin. Stores a ``workrb_model.json`` next to the backbone.

    The mixin assumes the host class is also an ``nn.Module`` (for state),
    but it does not require it to type-check that, so it composes cleanly
    with ``ModelInterface`` without forcing a particular metaclass.

    Default ``save_pretrained`` behavior
    ------------------------------------

    * If ``self.backbone`` exists and has ``save_pretrained`` (HF models do),
      the backbone is saved to ``path``.
    * If ``self.tokenizer`` exists and has ``save_pretrained``, same.
    * A ``workrb_model.json`` is written to ``path`` containing the
      subclass's dotted import path and whatever ``_save_extra`` returns
      as init kwargs.

    Default ``_save_extra`` returns ``{"model_name": str(path)}`` so the
    re-loaded model points its backbone init at the saved folder.
    """

    _CONFIG_FILE = "workrb_model.json"

    def save_pretrained(self, path: str | Path) -> Path:
        """Save the model under ``path``. Returns the path."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "backbone") and hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path)
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)

        config = {
            "target": f"{type(self).__module__}:{type(self).__name__}",
            "init": self._save_extra(path),
        }
        (path / self._CONFIG_FILE).write_text(json.dumps(config, indent=2))
        return path

    def _save_extra(self, path: Path) -> dict[str, Any]:
        """Return init kwargs ``from_pretrained`` will splat back into __init__.

        Default points ``model_name`` at the saved folder so the HF backbone
        re-init reads weights from disk instead of re-downloading.

        Override this to persist extra trainable state. Example::

            def _save_extra(self, path):
                import torch
                torch.save(self.head.state_dict(), path / "head.pt")
                return {"model_name": str(path), "labels": self.labels}
        """
        return {"model_name": str(path)}

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "WorkrbSaveable":
        """Reload a model previously saved with ``save_pretrained``.

        Resolves the dotted-path target written into ``workrb_model.json``
        and instantiates *that* subclass with the saved kwargs. Calling
        ``WorkrbSaveable.from_pretrained(path)`` therefore returns the
        original concrete subclass (``MyModel``, ``CrossEncoderModel``, ...),
        not a ``WorkrbSaveable``.
        """
        path = Path(path)
        config = json.loads((path / cls._CONFIG_FILE).read_text())
        module_name, _, class_name = config["target"].partition(":")
        target_cls = getattr(import_module(module_name), class_name)
        return target_cls(**config["init"])
