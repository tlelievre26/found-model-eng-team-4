"""The single source of truth for shared filesystem paths.

Every path to data, model weights, or checkpoints must come from this module.
Do not hard-code "/shared/..." anywhere else in the codebase.

The shared root is read from the ``FME_SHARED_DIR`` environment variable and
defaults to ``/shared`` (the shared folder on the team VM). To run the code on
another machine, point it somewhere else:

    export FME_SHARED_DIR=~/fme_shared
"""

import os
from datetime import date
from pathlib import Path

DEFAULT_SHARED_DIR = "/shared"

SHARED_DIR = Path(os.environ.get("FME_SHARED_DIR", DEFAULT_SHARED_DIR)).expanduser()

# Base model weights. HF_HOME should point here so nobody re-downloads models.
MODELS_DIR = SHARED_DIR / "models"
# Raw downloads + processed splits.
DATA_DIR = SHARED_DIR / "data"
# One folder per training run: <your-name>-<short-description>-<date>/
CHECKPOINTS_DIR = SHARED_DIR / "checkpoints"


def run_checkpoint_dir(name: str, description: str, run_date: date | None = None) -> Path:
    """Return the checkpoint folder for a training run, following the team convention.

    Example: ``run_checkpoint_dir("taylor", "lora-r16")`` ->
    ``<CHECKPOINTS_DIR>/taylor-lora-r16-2026-11-15``.

    The folder is not created; call ``.mkdir(parents=True)`` when you are ready to write.
    """
    run_date = run_date or date.today()
    return CHECKPOINTS_DIR / f"{name}-{description}-{run_date.isoformat()}"
