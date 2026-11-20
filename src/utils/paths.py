"""The single source of truth for shared filesystem paths.

Every path to data, model weights, or checkpoints must come from this module.
Do not hard-code "/shared/..." anywhere else in the codebase.

The shared root is read from the ``FME_SHARED_DIR`` environment variable and
defaults to ``/shared`` (the shared folder on the team VM). To run the code on
another machine, point it somewhere else:

    export FME_SHARED_DIR=~/fme_shared
"""

import os
from pathlib import Path

DEFAULT_SHARED_DIR = "/shared"

SHARED_DIR = Path(os.environ.get("FME_SHARED_DIR", DEFAULT_SHARED_DIR)).expanduser()

# Base model weights. HF_HOME should point here so nobody re-downloads models.
MODELS_DIR = SHARED_DIR / "models"
# Raw downloads + processed splits.
DATA_DIR = SHARED_DIR / "data"
# Training run outputs.
CHECKPOINTS_DIR = SHARED_DIR / "checkpoints"

