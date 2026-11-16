import importlib
from pathlib import Path

from src.utils import paths


def test_default_shared_dir(monkeypatch):
    monkeypatch.delenv("FME_SHARED_DIR", raising=False)
    reloaded = importlib.reload(paths)
    assert reloaded.SHARED_DIR == Path("/shared")
    assert reloaded.MODELS_DIR == Path("/shared/models")
    assert reloaded.DATA_DIR == Path("/shared/data")
    assert reloaded.CHECKPOINTS_DIR == Path("/shared/checkpoints")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FME_SHARED_DIR", str(tmp_path))
    reloaded = importlib.reload(paths)
    assert reloaded.SHARED_DIR == tmp_path
    assert reloaded.MODELS_DIR == tmp_path / "models"

