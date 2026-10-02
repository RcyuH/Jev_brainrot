from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    _validate_config(config)
    return config


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    for section in ("paths", "data", "model", "analysis", "experiments"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    analysis = config["analysis"]
    if analysis["binning"] != "pooled_quantile":
        raise ValueError("This reproduction implements pooled_quantile binning only")
    if analysis["raw_reference"] != "full_sample":
        raise ValueError("This reproduction follows Appendix A.1: raw reference is full sample")
    if int(analysis["n_bins"]) < 2:
        raise ValueError("n_bins must be at least two")
    if int(analysis["min_per_group_per_bin"]) < 1:
        raise ValueError("min_per_group_per_bin must be positive")
