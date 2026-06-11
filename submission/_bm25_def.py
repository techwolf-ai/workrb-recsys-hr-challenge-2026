"""BM25 model definition for testing generate_submission_file.py.

Loaded when MODEL_DEF_FILE points here and WEIGHTS_PATH is empty.
"""

from workrb.models.lexical_baselines import BM25Model

__all__ = ["BM25Model"]
