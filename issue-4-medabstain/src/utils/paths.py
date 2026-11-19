"""Shared data, model, and checkpoint locations."""

import os
from pathlib import Path

DEFAULT_SHARED_DIR = "/shared"
SHARED_DIR = Path(os.environ.get("FME_SHARED_DIR", DEFAULT_SHARED_DIR)).expanduser()
MODELS_DIR = SHARED_DIR / "models"
DATA_DIR = Path(os.environ.get("DATA_DIR", SHARED_DIR / "data")).expanduser()
CHECKPOINTS_DIR = SHARED_DIR / "checkpoints"
MEDABSTAIN_DIR = DATA_DIR / "medabstain"
