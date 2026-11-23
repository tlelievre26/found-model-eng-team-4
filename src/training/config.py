"""Run configs: YAML files that may extend another (`base: base.yaml`), plus per-run `key=value` overrides.

configs/base.yaml defines every key. Other configs and overrides may only set keys that exist there, so a typo
fails instead of being silently ignored. Override values are parsed with yaml.safe_load (`lora_rank=32`,
`train_datasets=[medmcqa,medqa]`, `eval_split=null`) and coerced to float where the base value is a float, so
`learning_rate=5e-5` works although YAML reads `5e-5` as a string.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: Iterable[str] | Mapping | None = None) -> dict:
    """Load a config file (following its `base:` chain) and apply overrides."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    cfg = load_config(path.parent / base) if base else {}
    cfg = apply_overrides(cfg, raw) if cfg else raw
    return apply_overrides(cfg, overrides) if overrides else cfg


def parse_overrides(items: Iterable[str]) -> dict:
    """["key=value", ...] -> {key: yaml.safe_load(value)}."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Override {item!r} is not of the form key=value")
        out[key.strip()] = yaml.safe_load(value)
    return out


def apply_overrides(cfg: dict, overrides: Iterable[str] | Mapping) -> dict:
    """A copy of cfg with overrides applied. Unknown keys are an error."""
    if not isinstance(overrides, Mapping):
        overrides = parse_overrides(overrides)
    out = dict(cfg)
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"Unknown config key {key!r} (not in the base config)")
        if isinstance(cfg[key], float) and isinstance(value, (int, str)) and not isinstance(value, bool):
            value = float(value)
        out[key] = value
    return out
